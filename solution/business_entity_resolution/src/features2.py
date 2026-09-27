"""Stage-C extra pairwise features (appended after the base 30).

Covers: transliteration, phonetic, phone, char 2/4-grams, containment,
numeric strictness, retrieval provenance, group-relative signals.
All functions are deterministic and null-safe.
"""
import re
import numpy as np
from rapidfuzz import fuzz

try:
    from views import transliterate_text, dm_primary, soundex, extract_phones
    from features import char_ngrams, jaccard_similarity
except ImportError:
    from solution.business_entity_resolution.src.views import (
        transliterate_text, dm_primary, soundex, extract_phones)
    from solution.business_entity_resolution.src.features import char_ngrams, jaccard_similarity

EXTRA_NAMES = [
    "tr_name_set", "phon_jaccard", "soundex_jaccard",
    "char2_jaccard", "char4_jaccard", "name_containment",
    "shared_sig_count", "shared_sig_ratio", "tokcount_diff",
    "addr_tr_set", "addr_char3_jaccard", "phone_agree",
    "postal_exact", "addr_coverage",
    "agree_count", "family_count", "has_tr_chan", "has_phon_chan",
    "has_phone_chan", "ngram_only", "gap_to_top", "w_zscore",
]

_FAMS = {
    "exact": "name", "token": "name", "prefix": "name", "acronym": "name",
    "ngram": "name", "tr_name": "trlit", "tr_tok": "trlit", "phon": "phon",
    "addr": "addr", "tokpost": "mix", "addr_tok": "addr", "phone": "phone",
}


def _tokset(a, b):
    if not a or not b:
        return 0.0
    return fuzz.token_set_ratio(a, b) / 100.0


def extra_for_pair(s1_rec, tgt_norm_name, tgt_norm_addr, tgt_core_tokens,
                   tgt_raw_name="", tgt_raw_addr="", channels=(),
                   rank=0, weight=0.0, max_weight=1.0,
                   pool_mean=0.0, pool_std=1.0):
    n1 = s1_rec.get("norm_name", "")
    a1 = s1_rec.get("norm_addr", "")
    c1 = s1_rec.get("core_tokens", []) or []
    r1 = s1_rec.get("name", "")
    ra1 = s1_rec.get("addr", "")

    tr1 = transliterate_text(r1) or ""
    tr2 = transliterate_text(tgt_raw_name) or ""
    # Cross-direction max: latin query vs transliterated target and vice versa.
    tr_name_set = max(
        _tokset(tr1, tr2) if (tr1 and tr2) else 0.0,
        _tokset(tr1, tgt_norm_name) if tr1 else 0.0,
        _tokset(n1, tr2) if tr2 else 0.0,
    )

    p1 = set(dm_primary(t) for t in c1 if len(t) >= 3)
    p1.discard("")
    p2 = set(dm_primary(t) for t in (tgt_core_tokens or []) if len(t) >= 3)
    p2.discard("")
    phon_jaccard = jaccard_similarity(p1, p2)

    s1s = set(soundex(t) for t in c1 if len(t) >= 3)
    s2s = set(soundex(t) for t in (tgt_core_tokens or []) if len(t) >= 3)
    soundex_jaccard = jaccard_similarity(s1s, s2s)

    char2_jaccard = jaccard_similarity(char_ngrams(n1, 2), char_ngrams(tgt_norm_name, 2))
    char4_jaccard = jaccard_similarity(char_ngrams(n1, 4), char_ngrams(tgt_norm_name, 4))

    set_c1, set_c2 = set(c1), set(tgt_core_tokens or [])
    if set_c1 and set_c2:
        shorter = set_c1 if len(set_c1) <= len(set_c2) else set_c2
        longer = set_c2 if shorter is set_c1 else set_c1
        name_containment = len(shorter & longer) / max(1, len(shorter))
    else:
        name_containment = 0.0
    shared = set_c1 & set_c2
    shared_sig_count = float(len(shared))
    shared_sig_ratio = len(shared) / max(1, max(len(set_c1), len(set_c2)))
    tokcount_diff = abs(len(set_c1) - len(set_c2)) / max(1, len(set_c1) + len(set_c2))

    atr1 = transliterate_text(ra1) or ""
    atr2 = transliterate_text(tgt_raw_addr) or ""
    addr_tr_set = max(
        _tokset(atr1, atr2) if (atr1 and atr2) else 0.0,
        _tokset(atr1, tgt_norm_addr) if atr1 else 0.0,
        _tokset(a1, atr2) if atr2 else 0.0,
    )
    addr_char3_jaccard = jaccard_similarity(char_ngrams(a1, 3), char_ngrams(tgt_norm_addr, 3)) if (a1 and tgt_norm_addr) else 0.0

    ph1 = extract_phones(ra1) | extract_phones(s1_rec.get("norm_addr", ""))
    ph2 = extract_phones(tgt_raw_addr) | extract_phones(tgt_norm_addr)
    phone_agree = 1.0 if (ph1 and ph2 and (ph1 & ph2)) else 0.0

    d1 = set(s1_rec.get("digits", ()) or ())
    d2 = set(re.findall(r"\d+", tgt_norm_addr or "") + re.findall(r"\d+", tgt_raw_addr or ""))
    post1 = {d for d in d1 if 4 <= len(d) <= 6}
    post2 = {d for d in d2 if 4 <= len(d) <= 6}
    postal_exact = 1.0 if (post1 and post2 and (post1 & post2)) else 0.0

    sa1 = set((a1 or "").split())
    sa2 = set((tgt_norm_addr or "").split())
    addr_coverage = len(sa1 & sa2) / max(1, len(sa1)) if sa1 else 0.0

    ch = set(channels or ())
    agree_count = float(len(ch))
    family_count = float(len(set(_FAMS.get(c, "other") for c in ch)))
    has_tr_chan = 1.0 if ch & {"tr_name", "tr_tok"} else 0.0
    has_phon_chan = 1.0 if "phon" in ch else 0.0
    has_phone_chan = 1.0 if "phone" in ch else 0.0
    ngram_only = 1.0 if ch == {"ngram"} else 0.0

    gap_to_top = float(max(0.0, 1.0 - weight / max(1e-5, max_weight)))
    w_zscore = float((weight - pool_mean) / max(1e-5, pool_std))

    return [tr_name_set, phon_jaccard, soundex_jaccard, char2_jaccard,
            char4_jaccard, name_containment, shared_sig_count, shared_sig_ratio,
            tokcount_diff, addr_tr_set, addr_char3_jaccard, phone_agree,
            postal_exact, addr_coverage, agree_count, family_count,
            has_tr_chan, has_phon_chan, has_phone_chan, ngram_only,
            gap_to_top, w_zscore]


def extra_batch_for_s1(s1_rec, candidate_items, records_dict, prov_map=None,
                       raw_name_map=None, raw_addr_map=None):
    """Aligned extra rows for candidate_items (list of ids or (id, weight))."""
    if not candidate_items:
        return [], []
    if isinstance(candidate_items[0], tuple):
        cands = [c for c, _ in candidate_items]
        weights = [float(w) for _, w in candidate_items]
    else:
        cands = list(candidate_items)
        weights = [float(len(cands) - i) for i in range(len(cands))]
    mw = weights[0] if weights else 1.0
    mean = float(np.mean(weights)) if weights else 0.0
    std = float(np.std(weights)) if weights and len(weights) > 1 else 1.0
    valid, rows = [], []
    rec_map = records_dict.get_many(cands) if hasattr(records_dict, "get_many") else None
    for rank, (cid, w) in enumerate(zip(cands, weights)):
        rec = rec_map.get(cid) if rec_map is not None else records_dict.get(cid)
        if rec is None:
            continue
        n2, a2 = rec[0], rec[1]
        c2 = rec[3] if len(rec) > 3 else []
        # Raw native-script text lives in the record tuple; explicit maps
        # only override when provided.
        rname = (raw_name_map or {}).get(cid) or (rec[4] if len(rec) > 4 else "")
        raddr = (raw_addr_map or {}).get(cid) or (rec[5] if len(rec) > 5 else "")
        ch = (prov_map or {}).get(cid, ())
        rows.append(extra_for_pair(
            s1_rec, n2, a2, c2,
            tgt_raw_name=rname, tgt_raw_addr=raddr,
            channels=ch, rank=rank, weight=w, max_weight=mw,
            pool_mean=mean, pool_std=std))
        valid.append(cid)
    return valid, rows
