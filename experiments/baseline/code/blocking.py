from collections import defaultdict, Counter
import re
from typing import List, Dict, Set, Tuple

# Threshold above which a token is considered too frequent/uninformative for candidate retrieval
MAX_POSTING_LIST_LEN = 3000

class CountryBlockingIndex:
    """
    Compact inverted index for candidate generation within a single country.
    Maps string and numeric keys to target candidate IDs (S2 and S3).
    """
    def __init__(self, country: str):
        self.country = country
        # eid -> {norm_name, norm_addr, core_tokens, addr_tokens, digits, is_s2}
        self.records: Dict[str, dict] = {}
        
        # Inverted index tables
        self.token_idx: Dict[str, List[str]] = defaultdict(list)
        self.addr_idx: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        self.prefix_idx: Dict[str, List[str]] = defaultdict(list)
        
    def add_target_record(self, eid: str, norm_name: str, core_tokens: list, 
                          norm_addr: str, addr_keys: list, digits: set, is_s2: int):
        """Add an S2 or S3 record to the country index."""
        # Store compact record: (norm_name, norm_addr, is_s2)
        self.records[eid] = (norm_name, norm_addr, is_s2)
        
        # 1. Index significant name tokens
        for t in set(core_tokens):
            if len(t) >= 3:
                self.token_idx[t].append(eid)
                
        # 2. Index 4-character prefix if name is reasonably long
        if len(norm_name) >= 4:
            pfx = norm_name[:4]
            self.prefix_idx[pfx].append(eid)
            
        # 3. Index address keys: (street_num, first_token) and ('POSTAL', zip)
        for k in addr_keys:
            self.addr_idx[k].append(eid)
            
    def prune_frequent_keys(self):
        """Prune overly frequent posting lists (e.g. generic tokens like 'group' or 'center')."""
        pruned_tokens = 0
        for t in list(self.token_idx.keys()):
            if len(self.token_idx[t]) > MAX_POSTING_LIST_LEN:
                del self.token_idx[t]
                pruned_tokens += 1
                
        for pfx in list(self.prefix_idx.keys()):
            if len(self.prefix_idx[pfx]) > MAX_POSTING_LIST_LEN:
                del self.prefix_idx[pfx]
                
    def query_candidates(self, s1_norm_name: str, s1_core_tokens: list, 
                         s1_norm_addr: str, s1_addr_keys: list, 
                         max_candidates: int = 25) -> List[str]:
        """
        Query the index for candidate matches, ranking by multi-key collision frequency.
        Returns up to max_candidates entity IDs.
        """
        candidate_scores = Counter()
        
        # 1. Score hits from significant core name tokens (weight 3 per hit)
        for t in set(s1_core_tokens):
            if t in self.token_idx:
                for target_id in self.token_idx[t]:
                    candidate_scores[target_id] += 3
                    
        # 2. Score hits from address keys (weight 4 for exact postal/street number alignment)
        for k in s1_addr_keys:
            if k in self.addr_idx:
                for target_id in self.addr_idx[k]:
                    candidate_scores[target_id] += 4
                    
        # 3. Score hits from 4-character prefix (weight 1)
        if len(s1_norm_name) >= 4:
            pfx = s1_norm_name[:4]
            if pfx in self.prefix_idx:
                for target_id in self.prefix_idx[pfx]:
                    candidate_scores[target_id] += 1
                    
        n_cands = len(candidate_scores)
        if n_cands == 0:
            return []
        if n_cands <= max_candidates:
            return list(candidate_scores.keys())
            
        # Return top-k highest scoring candidate IDs
        return [cid for cid, _ in candidate_scores.most_common(max_candidates)]
