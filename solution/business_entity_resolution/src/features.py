import re
import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
try:
    from normalization import extract_name_tokens
except ImportError:
    from solution.business_entity_resolution.src.normalization import extract_name_tokens

NUM_ONLY_RE = re.compile(r"\b\d+\b")

def char_ngrams(s: str, n: int = 3):
    """Generate set of character n-grams."""
    if len(s) < n:
        return {s} if s else set()
    return set(s[i:i+n] for i in range(len(s)-n+1))

def jaccard_similarity(set_a, set_b) -> float:
    """Compute Jaccard similarity between two sets/iterables."""
    if not set_a or not set_b:
        return 0.0
    sa = set(set_a)
    sb = set(set_b)
    intersection = len(sa & sb)
    union = len(sa | sb)
    return intersection / union if union > 0 else 0.0

def get_acronym(tokens) -> str:
    """Build acronym string from iterable of tokens."""
    return "".join(t[0] for t in tokens if t)

def compute_pairwise_features(
    s1_rec: dict, 
    tgt_rec, 
    is_s2: int = None,
    rank: int = 0,
    weight: float = 0.0,
    max_weight: float = 1.0
) -> list:
    """
    Compute dense 30-dimensional feature vector for a candidate pair.
    tgt_rec can be a dict, a lightweight 9-tuple, or a 3-tuple.
    """
    if isinstance(tgt_rec, tuple):
        if len(tgt_rec) >= 9:
            n2, a2, tgt_is_s2, tgt_core_tokens, tgt_addr_tokens, tgt_digits, tgt_pfx6, tgt_snum, tgt_postal = tgt_rec[:9]
        else:
            n2, a2, tgt_is_s2 = tgt_rec[:3]
            tgt_core_tokens, _ = extract_name_tokens(n2)
            tgt_addr_tokens = a2.split() if a2 else []
            tgt_digits = set(NUM_ONLY_RE.findall(a2)) if a2 else set()
            tgt_pfx6 = n2[:6] if n2 else ""
            postals = [n for n in tgt_digits if 4 <= len(n) <= 6]
            snums = [n for n in tgt_digits if 1 <= len(n) <= 5]
            tgt_postal = postals[0] if postals else ""
            tgt_snum = snums[0] if snums else ""
        is_s2 = tgt_is_s2
    else:
        n2 = tgt_rec["norm_name"]
        a2 = tgt_rec["norm_addr"]
        tgt_core_tokens = tgt_rec.get("core_tokens", [t for t in n2.split() if len(t) >= 2])
        tgt_addr_tokens = tgt_rec.get("addr_tokens", a2.split() if a2 else [])
        tgt_digits = tgt_rec.get("digits", set(NUM_ONLY_RE.findall(a2)) if a2 else set())
        tgt_pfx6 = n2[:6] if n2 else ""
        postals = [n for n in tgt_digits if 4 <= len(n) <= 6]
        snums = [n for n in tgt_digits if 1 <= len(n) <= 5]
        tgt_postal = postals[0] if postals else ""
        tgt_snum = snums[0] if snums else ""
        if is_s2 is None:
            is_s2 = tgt_rec.get("is_s2", 0)

    n1 = s1_rec["norm_name"]
    a1 = s1_rec["norm_addr"]
    s1_core_tokens = s1_rec.get("core_tokens", [])
    s1_addr_tokens = s1_rec.get("addr_tokens", set())
    s1_digits = s1_rec.get("digits", set())
    s1_pfx6 = s1_rec.get("prefix6", n1[:6] if n1 else "")
    s1_snum = s1_rec.get("street_num", "")
    s1_postal = s1_rec.get("postal", "")

    # 1. Base Name Features
    name_token_set = fuzz.token_set_ratio(n1, n2) / 100.0
    name_token_sort = fuzz.token_sort_ratio(n1, n2) / 100.0
    name_ratio = fuzz.ratio(n1, n2) / 100.0
    name_exact = 1.0 if (n1 == n2 and n1) else 0.0
    name_jaccard = jaccard_similarity(s1_core_tokens, tgt_core_tokens)
    
    ng1 = s1_rec.get("_ng1")
    if ng1 is None:
        ng1 = char_ngrams(n1, 3)
    ng2 = char_ngrams(n2, 3)
    name_ngram_jaccard = jaccard_similarity(ng1, ng2)
    
    len_n1 = s1_rec.get("_len_n1", len(n1))
    len_n2 = len(n2)
    name_len_diff = abs(len_n1 - len_n2) / max(len_n1, len_n2, 1)

    # 2. Base Address Features
    addr_missing = 1.0 if not a2 else 0.0
    if not a2:
        addr_token_set = 0.0
        addr_token_sort = 0.0
        addr_ratio = 0.0
        addr_jaccard = 0.0
        num_overlap = 0.0
        has_matching_digits = 0.0
        addr_len_diff = 1.0
        addr_token_containment = 0.0
        postal_prefix_match_len = 0.0
        street_num_exact_match = 0.0
        addr_jaro_winkler = 0.0
    else:
        addr_token_set = fuzz.token_set_ratio(a1, a2) / 100.0
        addr_token_sort = fuzz.token_sort_ratio(a1, a2) / 100.0
        addr_ratio = fuzz.ratio(a1, a2) / 100.0
        addr_jaccard = jaccard_similarity(s1_addr_tokens, tgt_addr_tokens)
        
        d1 = s1_rec.get("_d1")
        if d1 is None:
            d1 = set(s1_digits)
        d2 = set(tgt_digits)
        if d1 and d2:
            num_overlap = len(d1 & d2) / max(1, len(d1 | d2))
            has_matching_digits = 1.0 if (d1 & d2) else 0.0
        else:
            num_overlap = 0.0
            has_matching_digits = 0.0
            
        len_a1 = s1_rec.get("_len_a1", len(a1))
        len_a2 = len(a2)
        addr_len_diff = abs(len_a1 - len_a2) / max(len_a1, len_a2, 1)

        # addr_token_containment: percentage of tokens in shorter address contained in longer
        set_a1 = s1_rec.get("_set_a1")
        if set_a1 is None:
            set_a1 = set(s1_addr_tokens)
        set_a2 = set(tgt_addr_tokens)
        if set_a1 and set_a2:
            shorter = set_a1 if len(set_a1) <= len(set_a2) else set_a2
            longer = set_a2 if shorter is set_a1 else set_a1
            addr_token_containment = len(shorter & longer) / max(1, len(shorter))
        else:
            addr_token_containment = 0.0

        # postal_prefix_match_len: exact matching length of postal/PIN codes (0 to 6)
        if s1_postal and tgt_postal:
            mlen = 0
            for c1, c2 in zip(s1_postal, tgt_postal):
                if c1 == c2:
                    mlen += 1
                else:
                    break
            postal_prefix_match_len = float(min(6, mlen))
        else:
            postal_prefix_match_len = 0.0

        # street_num_exact_match: binary indicator whether primary street door numbers match
        if s1_snum and tgt_snum and s1_snum == tgt_snum:
            street_num_exact_match = 1.0
        else:
            street_num_exact_match = 0.0

        addr_jaro_winkler = float(JaroWinkler.similarity(a1, a2))

    # 3. Composite Interaction Feature
    composite_score = (2.0 * name_token_set * addr_token_set) / max(1e-5, (name_token_set + addr_token_set)) if (not addr_missing) else name_token_set

    # 4. Expanded String & Phonetics Features
    name_jaro_winkler = float(JaroWinkler.similarity(n1, n2))

    # name_acronym_match: binary indicator (1.0 if one name equals acronym of the other)
    acr1 = s1_rec.get("_acr1")
    if acr1 is None:
        acr1 = get_acronym(s1_core_tokens)
    acr2 = get_acronym(tgt_core_tokens)
    name_acronym_match = 0.0
    if acr1 and (n2 == acr1 or (acr2 and acr1 == acr2)):
        name_acronym_match = 1.0
    elif acr2 and (n1 == acr2 or (acr1 and acr1 == acr2)):
        name_acronym_match = 1.0

    # name_prefix_similarity: normalized ratio on first 6 chars
    name_prefix_similarity = fuzz.ratio(s1_pfx6, tgt_pfx6) / 100.0 if (s1_pfx6 and tgt_pfx6) else 0.0

    # 5. Group-Level / Relative Candidate Pool Features
    score_rank_in_candidate_pool = float(rank)
    blocking_weight_ratio = float(weight / max(1e-5, max_weight))
    raw_blocking_weight = float(weight)
    is_top1_candidate = 1.0 if rank == 0 else 0.0

    # 6. Additional Discriminative Interaction Features
    name_first_token_match = 1.0 if (s1_core_tokens and tgt_core_tokens and s1_core_tokens[0] == tgt_core_tokens[0]) else 0.0
    name_partial_ratio = fuzz.partial_ratio(n1, n2) / 100.0

    return [
        name_token_set,                 # 0
        name_token_sort,                # 1
        name_ratio,                     # 2
        name_exact,                     # 3
        name_jaccard,                   # 4
        name_ngram_jaccard,             # 5
        name_len_diff,                  # 6
        addr_token_set,                 # 7
        addr_token_sort,                # 8
        addr_ratio,                     # 9
        addr_jaccard,                   # 10
        num_overlap,                    # 11
        has_matching_digits,            # 12
        addr_missing,                   # 13
        addr_len_diff,                  # 14
        composite_score,                # 15
        float(is_s2),                   # 16
        name_jaro_winkler,              # 17
        name_acronym_match,             # 18
        name_prefix_similarity,         # 19
        addr_token_containment,         # 20
        postal_prefix_match_len,        # 21
        street_num_exact_match,         # 22
        score_rank_in_candidate_pool,   # 23
        blocking_weight_ratio,          # 24
        raw_blocking_weight,            # 25
        addr_jaro_winkler,              # 26
        name_first_token_match,         # 27
        is_top1_candidate,              # 28
        name_partial_ratio              # 29
    ]

def compute_candidate_features_for_s1(s1_rec: dict, candidate_items: list, records_dict: dict):
    """
    Vectorized batch extraction for an S1 entity against all its candidates.
    candidate_items can be a list of entity_ids or a list of (entity_id, weight) tuples.
    Returns: (valid_candidate_ids, feature_matrix_rows)
    """
    if not candidate_items:
        return [], []
    
    # Check if weights are provided
    if isinstance(candidate_items[0], tuple):
        cands = [cid for cid, w in candidate_items]
        weights = [float(w) for cid, w in candidate_items]
    else:
        cands = list(candidate_items)
        weights = [float(len(cands) - i) for i in range(len(cands))]

    max_weight = weights[0] if weights else 1.0

    # Pre-compute S1 properties once across all candidates
    n1 = s1_rec["norm_name"]
    s1_core_tokens = s1_rec.get("core_tokens", [])
    s1_rec["_ng1"] = char_ngrams(n1, 3)
    s1_rec["_d1"] = set(s1_rec.get("digits", ()))
    s1_rec["_set_a1"] = set(s1_rec.get("addr_tokens", ()))
    s1_rec["_acr1"] = get_acronym(s1_core_tokens)
    s1_rec["_len_n1"] = len(n1)
    s1_rec["_len_a1"] = len(s1_rec.get("norm_addr", ""))

    valid_cands = []
    features = []
    for rank, (cid, w) in enumerate(zip(cands, weights)):
        tgt_rec = records_dict.get(cid)
        if tgt_rec is not None:
            feat_vec = compute_pairwise_features(
                s1_rec, tgt_rec, 
                rank=rank, weight=w, max_weight=max_weight
            )
            valid_cands.append(cid)
            features.append(feat_vec)

    return valid_cands, features

FEATURE_NAMES = [
    "name_token_set",
    "name_token_sort",
    "name_ratio",
    "name_exact",
    "name_jaccard",
    "name_ngram_jaccard",
    "name_len_diff",
    "addr_token_set",
    "addr_token_sort",
    "addr_ratio",
    "addr_jaccard",
    "num_overlap",
    "has_matching_digits",
    "addr_missing",
    "addr_len_diff",
    "composite_score",
    "is_s2",
    "name_jaro_winkler",
    "name_acronym_match",
    "name_prefix_similarity",
    "addr_token_containment",
    "postal_prefix_match_len",
    "street_num_exact_match",
    "score_rank_in_candidate_pool",
    "blocking_weight_ratio",
    "raw_blocking_weight",
    "addr_jaro_winkler",
    "name_first_token_match",
    "is_top1_candidate",
    "name_partial_ratio"
]
