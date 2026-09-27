import math
import re
from collections import defaultdict, Counter
from typing import List, Dict, Set, Tuple, Optional

try:
    from views import transliterate_text, dm_primary, soundex, extract_phones, compact_alnum
except ImportError:
    from solution.business_entity_resolution.src.views import (
        transliterate_text, dm_primary, soundex, extract_phones, compact_alnum)

try:
    from normalization import COMMON_ADDR_STOPWORDS
except ImportError:
    from solution.business_entity_resolution.src.normalization import COMMON_ADDR_STOPWORDS

# Threshold above which a token is considered too frequent/uninformative for candidate retrieval
MAX_POSTING_LIST_LEN = 12000
MAX_NGRAM_POSTING_LEN = 1500
MAX_ADDR_TOKEN_POSTING_LEN = 3000
MAX_PHON_POSTING_LEN = 8000
MAX_TRLIT_POSTING_LEN = 8000

NUM_ONLY_RE = re.compile(r"\b\d+\b")

def extract_acronyms(norm_name: str, core_tokens: list) -> List[str]:
    """Extract acronyms from multi-word business names (e.g. 'Tata Consultancy Services' -> 'TCS')."""
    acrs = []
    if core_tokens and len(core_tokens) >= 2:
        acr1 = "".join(t[0] for t in core_tokens if t)
        if len(acr1) >= 2:
            acrs.append(acr1)
    tokens_all = norm_name.split()
    if len(tokens_all) >= 2:
        acr2 = "".join(t[0] for t in tokens_all if t)
        if len(acr2) >= 2 and acr2 not in acrs:
            acrs.append(acr2)
    # Also if the name itself is already an acronym (e.g. 2-5 chars, no spaces)
    if 2 <= len(norm_name) <= 5 and " " not in norm_name:
        if norm_name not in acrs:
            acrs.append(norm_name)
    return acrs

def extract_postal_prefixes(raw_addr: str, norm_addr: str) -> List[str]:
    """Extract 2-digit postal prefix from address."""
    nums = NUM_ONLY_RE.findall(raw_addr or norm_addr or "")
    postals = [n for n in nums if 4 <= len(n) <= 6]
    return list(set(p[:2] for p in postals[:3]))

def _trigrams(s: str):
    s = s or ""
    if len(s) < 3:
        return {s} if s else set()
    return set(s[i:i + 3] for i in range(len(s) - 2))

class CountryBlockingIndex:
    """
    Multi-pass compound inverted index for candidate generation within a single country.

    Channel families (unioned, never single-index top-K):
      exact / token / prefix / acronym / tok+postal / compound-addr / addr-token
      char-3gram (always-on, depth + trigram-jaccard rerank)
      transliterated name (exact + tokens)
      phonetic name tokens (double-metaphone primary)
      phone exact (multi-normalized)
    """
    def __init__(self, country: str, address_tokens: bool = True):
        self.country = country
        self.ngram_fallback_limit = 10 ** 9  # always-on trigram channel
        self.address_tokens = address_tokens
        self.rerank_depth = 1500
        self.trigram_rerank_boost = 8.0
        # eid -> (norm_name, norm_addr, is_s2, c_tokens, a_tokens, digits, prefix6, street_num, postal)
        self.records: Dict[str, tuple] = {}

        # Inverted index tables
        self.exact_name_idx: Dict[str, List[str]] = defaultdict(list)
        self.compound_addr_idx: Dict[Tuple, List[str]] = defaultdict(list)
        self.token_idx: Dict[str, List[str]] = defaultdict(list)
        self.prefix_idx: Dict[str, List[str]] = defaultdict(list)
        self.ngram_idx: Dict[str, List[str]] = defaultdict(list)
        self.acronym_idx: Dict[str, List[str]] = defaultdict(list)
        self.tok_post_idx: Dict[Tuple, List[str]] = defaultdict(list)
        self.addr_token_idx: Dict[str, List[str]] = defaultdict(list)
        self.tr_name_idx: Dict[str, List[str]] = defaultdict(list)
        self.tr_token_idx: Dict[str, List[str]] = defaultdict(list)
        self.phon_idx: Dict[str, List[str]] = defaultdict(list)
        self.phone_idx: Dict[str, List[str]] = defaultdict(list)
        self.compound_idf: Dict[Tuple, float] = {}
        self.ngram_idf: Dict[str, float] = {}
        self.addr_token_idf: Dict[str, float] = {}

    def add_target_record(self, eid: str, norm_name: str, core_tokens: list,
                          norm_addr: str, addr_keys: list, digits: set, is_s2: int,
                          raw_addr: str = "", raw_name: str = ""):
        """Add an S2 or S3 record to the country index.

        Keeps raw name/addr only when they carry non-ASCII (native-script)
        signal that normalization destroys; otherwise stores "" to bound RAM.
        Tuple layout stays compatible with features.py (3- or 9-tuple reads).
        """
        keep_raw_name = raw_name if (raw_name and any(ord(c) > 127 for c in raw_name)) else ""
        keep_raw_addr = raw_addr if (raw_addr and any(ord(c) > 127 for c in raw_addr)) else ""
        self.records[eid] = (norm_name, norm_addr, is_s2, list(core_tokens),
                             keep_raw_name, keep_raw_addr)

        # 1. Exact core name pass
        if norm_name and len(norm_name) >= 3:
            self.exact_name_idx[norm_name].append(eid)

        # 2. Significant name tokens
        for t in set(core_tokens):
            if len(t) >= 3:
                self.token_idx[t].append(eid)

        # 3. 4-character prefix
        if len(norm_name) >= 4:
            self.prefix_idx[norm_name[:4]].append(eid)

        # 4. Compound address keys
        for k in addr_keys:
            self.compound_addr_idx[k].append(eid)
        if self.address_tokens:
            for token in set(norm_addr.split()):
                if len(token) >= 4 and not token.isdigit():
                    self.addr_token_idx[token].append(eid)

        # 5. Acronym exact blocking key
        for acr in extract_acronyms(norm_name, core_tokens):
            self.acronym_idx[acr].append(eid)

        # 6. Compound (first_significant_token, 2_digit_postal_prefix)
        if core_tokens:
            first_tok = core_tokens[0]
            for pfx in extract_postal_prefixes(raw_addr, norm_addr):
                self.tok_post_idx[(first_tok, pfx)].append(eid)

        # 7. Character 3-gram inverted index
        if len(norm_name) >= 3:
            ngrams = set(norm_name[i:i+3] for i in range(len(norm_name)-2))
            for ng in ngrams:
                self.ngram_idx[ng].append(eid)

        # 8. Transliterated name (native-script records): exact + tokens.
        # Merged into the MAIN exact/token indexes so latin queries recover
        # indic targets without needing a transliterated query.
        tr = transliterate_text(raw_name or "")
        if tr:
            tr = re.sub(r"[^a-z0-9 ]", " ", tr).strip()
            if len(tr) >= 3:
                self.tr_name_idx[tr].append(eid)
                if tr != norm_name:
                    self.exact_name_idx[tr].append(eid)
            norm_toks = set(norm_name.split())
            for t in set(tr.split()):
                if len(t) >= 3:
                    self.tr_token_idx[t].append(eid)
                    if t not in norm_toks:
                        self.token_idx[t].append(eid)

        # 9. Phonetic name tokens (double-metaphone primary)
        for t in set(core_tokens):
            if len(t) >= 3:
                p = dm_primary(t)
                if p:
                    self.phon_idx[p].append(eid)
        if tr:
            for t in set(tr.split()):
                if len(t) >= 3:
                    p = dm_primary(t)
                    if p:
                        self.phon_idx[p].append(eid)
                    if 'v' in t:
                        p2 = dm_primary(t.replace('v', 'b'))
                        if p2:
                            self.phon_idx[p2].append(eid)
                    if 'm' in t:
                        p3 = dm_primary(t.replace('m', 'n'))
                        if p3:
                            self.phon_idx[p3].append(eid)

        # 10. Phone exact (multi-normalized forms)
        for ph in extract_phones(raw_addr or ""):
            self.phone_idx[ph].append(eid)

    def prune_frequent_keys(self):
        """Prune overly frequent uninformative posting lists and compute IDF weights."""
        N = max(1, len(self.records))
        for t in list(self.token_idx.keys()):
            if len(self.token_idx[t]) > MAX_POSTING_LIST_LEN:
                del self.token_idx[t]

        for pfx in list(self.prefix_idx.keys()):
            if len(self.prefix_idx[pfx]) > MAX_POSTING_LIST_LEN:
                del self.prefix_idx[pfx]

        for acr in list(self.acronym_idx.keys()):
            if len(self.acronym_idx[acr]) > MAX_POSTING_LIST_LEN:
                del self.acronym_idx[acr]

        for tp in list(self.tok_post_idx.keys()):
            if len(self.tok_post_idx[tp]) > MAX_POSTING_LIST_LEN:
                del self.tok_post_idx[tp]

        for t in list(self.tr_token_idx.keys()):
            if len(self.tr_token_idx[t]) > MAX_TRLIT_POSTING_LEN:
                del self.tr_token_idx[t]
        for t in list(self.tr_name_idx.keys()):
            if len(self.tr_name_idx[t]) > MAX_TRLIT_POSTING_LEN:
                del self.tr_name_idx[t]
        for p in list(self.phon_idx.keys()):
            if len(self.phon_idx[p]) > MAX_PHON_POSTING_LEN:
                del self.phon_idx[p]

        for k in list(self.compound_addr_idx.keys()):
            l = len(self.compound_addr_idx[k])
            if l > MAX_POSTING_LIST_LEN or l < 1:
                del self.compound_addr_idx[k]
            else:
                self.compound_idf[k] = min(8.0, max(1.5, math.log((N + 1.0) / (l + 1.0))))

        for ng in list(self.ngram_idx.keys()):
            l = len(self.ngram_idx[ng])
            if l > MAX_NGRAM_POSTING_LEN or l < 1:
                del self.ngram_idx[ng]
            else:
                self.ngram_idf[ng] = math.log((N + 1.0) / (l + 1.0)) + 1.0
        if self.address_tokens:
            for token in list(self.addr_token_idx):
                length = len(self.addr_token_idx[token])
                if length > MAX_ADDR_TOKEN_POSTING_LEN:
                    del self.addr_token_idx[token]
                else:
                    self.addr_token_idf[token] = min(8.0, math.log((N + 1.0) / (length + 1.0)))

    def _add(self, scores, prov, tid, w, chan):
        scores[tid] += w
        if prov is not None:
            prov[tid].add(chan)

    def query_candidates(self, s1_norm_name: str, s1_core_tokens: list,
                         s1_norm_addr: str, s1_addr_keys: list,
                         s1_raw_addr: str = "", s1_raw_name: str = "",
                         max_candidates: int = 60,
                         return_weights: bool = False,
                         return_provenance: bool = False,
                         rerank: bool = True,
                         depth: int = None,
                         adaptive: bool = False):
        """Union PixelDust retrieval with trigram-jaccard rerank.

        Provenance channels: exact, addr, token, prefix, acronym, tokpost,
        addr_tok, ngram, tr_name, tr_tok, phon, phone.
        """
        candidate_scores = Counter()
        prov = defaultdict(set) if return_provenance else None
        depth = depth or self.rerank_depth

        # 1. Exact core name match (+15) with legal suffix variants
        exact_queries = {s1_norm_name}
        if " limited" in s1_norm_name or " private" in s1_norm_name:
            exact_queries.add(s1_norm_name.replace(" limited", " ltd").replace(" private", " pvt"))
        if " ltd" in s1_norm_name or " pvt" in s1_norm_name:
            exact_queries.add(s1_norm_name.replace(" ltd", " limited").replace(" pvt", " private"))
        if " llp" in s1_norm_name:
            exact_queries.add(s1_norm_name.replace(" llp", " l l p"))
        if " l l p" in s1_norm_name:
            exact_queries.add(s1_norm_name.replace(" l l p", " llp"))
        for eq in exact_queries:
            if eq in self.exact_name_idx:
                for target_id in self.exact_name_idx[eq]:
                    self._add(candidate_scores, prov, target_id, 15.0, "exact")

        # 2. Compound address keys with selectivity weight
        for k in s1_addr_keys:
            if k in self.compound_addr_idx:
                w = self.compound_idf.get(k, 4.0)
                for target_id in self.compound_addr_idx[k]:
                    self._add(candidate_scores, prov, target_id, w, "addr")

        # 3. Significant core name tokens (+3)
        cand_token_hits = Counter()
        for t in set(s1_core_tokens):
            if t in self.token_idx:
                for target_id in self.token_idx[t]:
                    self._add(candidate_scores, prov, target_id, 3.0, "token")
                    cand_token_hits[target_id] += 1
        for target_id, count in cand_token_hits.items():
            if count >= 2:
                self._add(candidate_scores, prov, target_id, 25.0, "token")

        # 3b. Concatenation / handle / abbreviation lookup (eindustries, apicalbros, dvventures)
        words_sources = []
        if len(s1_core_tokens) >= 2:
            words_sources.append(s1_core_tokens)
        all_words = s1_norm_name.split()
        if len(all_words) >= 2 and all_words != s1_core_tokens:
            words_sources.append(all_words)

        seen_concats = set()
        for wlist in words_sources:
            concat_2 = "".join(wlist[:2])
            if len(concat_2) >= 5:
                seen_concats.add(concat_2)
                seen_concats.add(concat_2.replace("brothers", "bros"))
            if len(wlist) >= 3:
                concat_3 = "".join(wlist[:3])
                if len(concat_3) >= 6:
                    seen_concats.add(concat_3)
                    seen_concats.add(concat_3.replace("brothers", "bros").replace("private", "pvt"))
        for ctok in seen_concats:
            if ctok in self.token_idx:
                for target_id in self.token_idx[ctok]:
                    self._add(candidate_scores, prov, target_id, 15.0, "token")

        # 4. 4-character prefix (+1)
        if len(s1_norm_name) >= 4:
            pfx = s1_norm_name[:4]
            if pfx in self.prefix_idx:
                for target_id in self.prefix_idx[pfx]:
                    self._add(candidate_scores, prov, target_id, 1.0, "prefix")

        # 5. Acronym exact blocking key (+8)
        for acr in extract_acronyms(s1_norm_name, s1_core_tokens):
            if acr in self.acronym_idx:
                for target_id in self.acronym_idx[acr]:
                    self._add(candidate_scores, prov, target_id, 8.0, "acronym")

        # 6. Compound (first_significant_token, 2_digit_postal_prefix) (+7)
        if s1_core_tokens:
            first_tok = s1_core_tokens[0]
            for pfx in extract_postal_prefixes(s1_raw_addr, s1_norm_addr):
                k = (first_tok, pfx)
                if k in self.tok_post_idx:
                    for target_id in self.tok_post_idx[k]:
                        self._add(candidate_scores, prov, target_id, 7.0, "tokpost")
        if self.address_tokens:
            tokens = [token for token in set(s1_norm_addr.split()) if token in self.addr_token_idx]
            tokens.sort(key=lambda token: self.addr_token_idf[token], reverse=True)
            for token in tokens[:6]:
                weight = self.addr_token_idf[token] * 2.0
                for target_id in self.addr_token_idx[token]:
                    self._add(candidate_scores, prov, target_id, weight, "addr_tok")

        # 7. Character 3-gram channel — always on, top rare trigrams
        if len(s1_norm_name) >= 3:
            q_ngrams = [s1_norm_name[i:i+3] for i in range(len(s1_norm_name)-2)
                        if s1_norm_name[i:i+3] in self.ngram_idx]
            if q_ngrams:
                q_ngrams.sort(key=lambda ng: self.ngram_idf.get(ng, 0.0), reverse=True)
                top_ngrams = q_ngrams[:8]
                for ng in top_ngrams:
                    w = self.ngram_idf.get(ng, 1.0)
                    for target_id in self.ngram_idx[ng]:
                        self._add(candidate_scores, prov, target_id, w, "ngram")

        # 8. Transliterated query views (recover native<->latin mismatches)
        tr_q = transliterate_text(s1_raw_name or "")
        if tr_q:
            tr_q = re.sub(r"[^a-z0-9 ]", " ", tr_q).strip()
            if tr_q in self.exact_name_idx:
                for target_id in self.exact_name_idx[tr_q]:
                    self._add(candidate_scores, prov, target_id, 12.0, "tr_name")
            if tr_q in self.tr_name_idx:
                for target_id in self.tr_name_idx[tr_q]:
                    self._add(candidate_scores, prov, target_id, 12.0, "tr_name")
            for t in set(tr_q.split()):
                if t in self.token_idx and len(t) >= 3:
                    for target_id in self.token_idx[t]:
                        self._add(candidate_scores, prov, target_id, 3.0, "tr_tok")
                if t in self.tr_token_idx:
                    for target_id in self.tr_token_idx[t]:
                        self._add(candidate_scores, prov, target_id, 4.0, "tr_tok")

        # 9. Phonetic query tokens (+2)
        cand_phon_hits = Counter()
        for t in set(s1_core_tokens):
            if len(t) >= 3:
                p = dm_primary(t)
                if p and p in self.phon_idx:
                    for target_id in self.phon_idx[p]:
                        self._add(candidate_scores, prov, target_id, 2.0, "phon")
                        cand_phon_hits[target_id] += 1
        for target_id, count in cand_phon_hits.items():
            if count >= 2:
                self._add(candidate_scores, prov, target_id, 20.0, "phon")

        # 10. Phone exact (+9)
        for ph in extract_phones(s1_raw_addr or ""):
            if ph in self.phone_idx:
                for target_id in self.phone_idx[ph]:
                    self._add(candidate_scores, prov, target_id, 9.0, "phone")

        if not candidate_scores:
            return ([], {}) if return_provenance else []

        if adaptive:
            has_exact = s1_norm_name in self.exact_name_idx
            best_w = max(candidate_scores.values())
            tier = assess_difficulty(s1_norm_name, s1_core_tokens, s1_norm_addr,
                                     len(candidate_scores), best_w)
            if has_exact and tier == "hard":
                tier = "medium"
            max_candidates = ADAPTIVE_BUDGETS[tier]

        # Depth + trigram-jaccard rerank (validated EXP_009 pattern)
        if rerank and len(candidate_scores) > max_candidates:
            top = candidate_scores.most_common(depth)
            top_tids = {tid for tid, _ in top}
            must_include = [tid for tid, count in cand_token_hits.items() if count >= 2 and tid not in top_tids]
            must_include.extend([tid for tid, count in cand_phon_hits.items() if count >= 2 and tid not in top_tids])
            if must_include:
                for tid in set(must_include):
                    top.append((tid, candidate_scores[tid]))
            q_tri = _trigrams(s1_norm_name)
            rescored = []
            for tid, w in top:
                rec = self.records.get(tid)
                sim = 0.0
                if rec is not None and q_tri:
                    c_tri = _trigrams(rec[0])
                    inter = len(q_tri & c_tri)
                    union = len(q_tri | c_tri) or 1
                    sim = inter / union
                    if len(rec) > 4 and rec[4]:
                        tr_name = transliterate_text(rec[4])
                        if tr_name:
                            tr_tri = _trigrams(tr_name)
                            sim = max(sim, len(q_tri & tr_tri) / (len(q_tri | tr_tri) or 1))
                rescored.append((tid, w + self.trigram_rerank_boost * sim))
            rescored.sort(key=lambda x: -x[1])
            top_candidates = rescored[:max_candidates]
        else:
            top_candidates = candidate_scores.most_common(max_candidates)

        if return_provenance:
            prov_out = {tid: sorted(prov.get(tid, ())) for tid, _ in top_candidates}
            if return_weights:
                return top_candidates, prov_out
            return [cid for cid, _ in top_candidates], prov_out
        if return_weights:
            return top_candidates
        return [cid for cid, _ in top_candidates]


def assess_difficulty(s1_norm_name, s1_core_tokens, s1_norm_addr, candidate_count, best_weight):
    """Heuristic difficulty tier driving adaptive budgets. Returns 'easy'|'medium'|'hard'."""
    if candidate_count == 0 or best_weight < 3.0:
        return "hard"
    if not s1_norm_addr or len(s1_core_tokens) <= 1:
        return "hard"
    if candidate_count >= 1000:
        return "hard"
    if candidate_count >= 150 or best_weight < 8.0:
        return "medium"
    return "easy"


ADAPTIVE_BUDGETS = {"easy": 60, "medium": 150, "hard": 250}
