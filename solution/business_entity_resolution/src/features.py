import numpy as np
from rapidfuzz import fuzz

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

def compute_pairwise_features(s1_rec: dict, tgt_rec, is_s2: int = None) -> list:
    """
    Compute dense feature vector for a candidate pair.
    tgt_rec can be a dict or a lightweight tuple: (norm_name, norm_addr, is_s2)
    """
    if isinstance(tgt_rec, tuple):
        n2, a2, tgt_is_s2 = tgt_rec
        tgt_core_tokens = [t for t in n2.split() if len(t) >= 2]
        tgt_addr_tokens = a2.split() if a2 else []
        import re
        tgt_digits = set(re.findall(r"\d+", a2)) if a2 else set()
        is_s2 = tgt_is_s2
    else:
        n2 = tgt_rec["norm_name"]
        a2 = tgt_rec["norm_addr"]
        tgt_core_tokens = tgt_rec["core_tokens"]
        tgt_addr_tokens = tgt_rec["addr_tokens"]
        tgt_digits = tgt_rec["digits"]
        if is_s2 is None:
            is_s2 = tgt_rec.get("is_s2", 0)

    n1 = s1_rec["norm_name"]
    a1 = s1_rec["norm_addr"]
    
    # 1. Name features
    name_token_set = fuzz.token_set_ratio(n1, n2) / 100.0
    name_token_sort = fuzz.token_sort_ratio(n1, n2) / 100.0
    name_ratio = fuzz.ratio(n1, n2) / 100.0
    name_exact = 1.0 if (n1 == n2 and n1) else 0.0
    
    # Name token Jaccard
    name_jaccard = jaccard_similarity(s1_rec["core_tokens"], tgt_core_tokens)
    
    # Name character 3-gram Jaccard
    ng1 = char_ngrams(n1, 3)
    ng2 = char_ngrams(n2, 3)
    name_ngram_jaccard = jaccard_similarity(ng1, ng2)
    
    # Name length difference ratio
    len_n1, len_n2 = len(n1), len(n2)
    name_len_diff = abs(len_n1 - len_n2) / max(len_n1, len_n2, 1)
    
    # 2. Address features
    addr_missing = 1.0 if not a2 else 0.0
    if not a2:
        addr_token_set = 0.0
        addr_token_sort = 0.0
        addr_ratio = 0.0
        addr_jaccard = 0.0
        num_overlap = 0.0
        has_matching_digits = 0.0
        addr_len_diff = 1.0
    else:
        addr_token_set = fuzz.token_set_ratio(a1, a2) / 100.0
        addr_token_sort = fuzz.token_sort_ratio(a1, a2) / 100.0
        addr_ratio = fuzz.ratio(a1, a2) / 100.0
        addr_jaccard = jaccard_similarity(s1_rec["addr_tokens"], tgt_addr_tokens)
        
        # Digit overlap
        d1 = s1_rec["digits"]
        d2 = tgt_digits
        if d1 and d2:
            num_overlap = len(d1 & d2) / max(1, len(d1 | d2))
            has_matching_digits = 1.0 if (d1 & d2) else 0.0
        else:
            num_overlap = 0.0
            has_matching_digits = 0.0
            
        len_a1, len_a2 = len(a1), len(a2)
        addr_len_diff = abs(len_a1 - len_a2) / max(len_a1, len_a2, 1)
        
    # 3. Composite interaction features
    # Harmonic mean of name and address token set
    composite_score = (2.0 * name_token_set * addr_token_set) / max(1e-5, (name_token_set + addr_token_set)) if (not addr_missing) else name_token_set

    return [
        name_token_set,        # 0
        name_token_sort,       # 1
        name_ratio,            # 2
        name_exact,            # 3
        name_jaccard,          # 4
        name_ngram_jaccard,    # 5
        name_len_diff,         # 6
        addr_token_set,        # 7
        addr_token_sort,       # 8
        addr_ratio,            # 9
        addr_jaccard,          # 10
        num_overlap,           # 11
        has_matching_digits,   # 12
        addr_missing,          # 13
        addr_len_diff,         # 14
        composite_score,       # 15
        float(is_s2)           # 16
    ]

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
    "is_s2"
]
