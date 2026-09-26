import math
import re
from collections import defaultdict, Counter
from typing import List, Dict, Set, Tuple, Optional

# Threshold above which a token is considered too frequent/uninformative for candidate retrieval
MAX_POSTING_LIST_LEN = 12000
MAX_NGRAM_POSTING_LEN = 1500
MAX_ADDR_TOKEN_POSTING_LEN = 3000

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

class CountryBlockingIndex:
    """
    Multi-Pass compound inverted index for candidate generation within a single country.
    Passes:
      1. Exact core normalized name (+15)
      2. Compound address keys (street_num, postal) and (street_num, locality) (+6)
      3. Significant core name tokens (+3)
      4. 4-character name prefixes (+1)
      5. Acronym exact blocking key (+8)
      6. Compound (first_significant_token, 2_digit_postal_prefix) (+7)
      7. Character 3-Gram Inverted Index with IDF weighting (sharing at least 3 3-grams)
    """
    def __init__(self, country: str, address_tokens: bool = False):
        self.country = country
        self.ngram_fallback_limit = 15
        self.address_tokens = address_tokens
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
        self.compound_idf: Dict[Tuple, float] = {}
        self.ngram_idf: Dict[str, float] = {}
        self.addr_token_idf: Dict[str, float] = {}

    def add_target_record(self, eid: str, norm_name: str, core_tokens: list, 
                          norm_addr: str, addr_keys: list, digits: set, is_s2: int,
                          raw_addr: str = ""):
        """Add an S2 or S3 record to the country index."""
        self.records[eid] = (norm_name, norm_addr, is_s2)

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

    def query_candidates(self, s1_norm_name: str, s1_core_tokens: list, 
                         s1_norm_addr: str, s1_addr_keys: list, 
                         s1_raw_addr: str = "",
                         max_candidates: int = 60,
                         return_weights: bool = False):
        """
        Query the index for candidate matches, ranking by multi-key collision frequency.
        Returns up to max_candidates entity IDs (or (cid, weight) tuples if return_weights=True).
        """
        candidate_scores = Counter()

        # 1. Exact core name match (+15)
        if s1_norm_name in self.exact_name_idx:
            for target_id in self.exact_name_idx[s1_norm_name]:
                candidate_scores[target_id] += 15.0

        # 2. Compound address keys with selectivity weight
        for k in s1_addr_keys:
            if k in self.compound_addr_idx:
                w = self.compound_idf.get(k, 4.0)
                for target_id in self.compound_addr_idx[k]:
                    candidate_scores[target_id] += w

        # 3. Significant core name tokens (+3)
        for t in set(s1_core_tokens):
            if t in self.token_idx:
                for target_id in self.token_idx[t]:
                    candidate_scores[target_id] += 3.0

        # 4. 4-character prefix (+1)
        if len(s1_norm_name) >= 4:
            pfx = s1_norm_name[:4]
            if pfx in self.prefix_idx:
                for target_id in self.prefix_idx[pfx]:
                    candidate_scores[target_id] += 1.0

        # 5. Acronym exact blocking key (+8)
        for acr in extract_acronyms(s1_norm_name, s1_core_tokens):
            if acr in self.acronym_idx:
                for target_id in self.acronym_idx[acr]:
                    candidate_scores[target_id] += 8.0

        # 6. Compound (first_significant_token, 2_digit_postal_prefix) (+7)
        if s1_core_tokens:
            first_tok = s1_core_tokens[0]
            for pfx in extract_postal_prefixes(s1_raw_addr, s1_norm_addr):
                k = (first_tok, pfx)
                if k in self.tok_post_idx:
                    for target_id in self.tok_post_idx[k]:
                        candidate_scores[target_id] += 7.0
        if self.address_tokens:
            tokens = [token for token in set(s1_norm_addr.split()) if token in self.addr_token_idx]
            tokens.sort(key=lambda token: self.addr_token_idf[token], reverse=True)
            for token in tokens[:6]:
                weight = self.addr_token_idf[token] * 2.0
                for target_id in self.addr_token_idx[token]:
                    candidate_scores[target_id] += weight

        # 7. Character 3-Gram Inverted Index (sharing at least 2 3-grams)
        # Only query 3-grams if we have few candidates (< 15) to maintain high throughput
        if len(s1_norm_name) >= 3 and len(candidate_scores) < self.ngram_fallback_limit:
            q_ngrams = [s1_norm_name[i:i+3] for i in range(len(s1_norm_name)-2) if s1_norm_name[i:i+3] in self.ngram_idx]
            if q_ngrams:
                q_ngrams.sort(key=lambda ng: self.ngram_idf.get(ng, 0.0), reverse=True)
                top_ngrams = q_ngrams[:4]
                min_shared = min(2, len(top_ngrams))
                
                ngram_counts = defaultdict(int)
                for ng in top_ngrams:
                    for target_id in self.ngram_idx[ng]:
                        ngram_counts[target_id] += 1

                for tid, count in ngram_counts.items():
                    if count >= min_shared:
                        candidate_scores[tid] += count * 1.5

        if not candidate_scores:
            return []

        top_candidates = candidate_scores.most_common(max_candidates)
        if return_weights:
            return top_candidates
        return [cid for cid, _ in top_candidates]
