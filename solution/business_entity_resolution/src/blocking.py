from collections import defaultdict, Counter
import re
from typing import List, Dict, Set, Tuple

# Threshold above which a token is considered too frequent/uninformative for candidate retrieval
MAX_POSTING_LIST_LEN = 4000

class CountryBlockingIndex:
    """
    Multi-Pass compound inverted index for candidate generation within a single country.
    Passes:
      1. Exact core normalized name
      2. Compound address keys (street_num, postal) and (street_num, locality)
      3. Significant core name tokens
      4. 4-character name prefixes
    """
    def __init__(self, country: str):
        self.country = country
        # eid -> (norm_name, norm_addr, is_s2)
        self.records: Dict[str, tuple] = {}
        
        # Inverted index tables
        self.exact_name_idx: Dict[str, List[str]] = defaultdict(list)
        self.compound_addr_idx: Dict[Tuple, List[str]] = defaultdict(list)
        self.token_idx: Dict[str, List[str]] = defaultdict(list)
        self.prefix_idx: Dict[str, List[str]] = defaultdict(list)
        
    def add_target_record(self, eid: str, norm_name: str, core_tokens: list, 
                          norm_addr: str, addr_keys: list, digits: set, is_s2: int):
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
            pfx = norm_name[:4]
            self.prefix_idx[pfx].append(eid)
            
        # 4. Compound address keys
        for k in addr_keys:
            self.compound_addr_idx[k].append(eid)
            
    def prune_frequent_keys(self):
        """Prune overly frequent uninformative posting lists."""
        for t in list(self.token_idx.keys()):
            if len(self.token_idx[t]) > MAX_POSTING_LIST_LEN:
                del self.token_idx[t]
                
        for pfx in list(self.prefix_idx.keys()):
            if len(self.prefix_idx[pfx]) > MAX_POSTING_LIST_LEN:
                del self.prefix_idx[pfx]
                
    def query_candidates(self, s1_norm_name: str, s1_core_tokens: list, 
                          s1_norm_addr: str, s1_addr_keys: list, 
                          max_candidates: int = 35) -> List[str]:
        """
        Query the index for candidate matches, ranking by multi-key collision frequency.
        Returns up to max_candidates entity IDs.
        """
        candidate_scores = Counter()
        
        # 1. Exact core name match (+10)
        if s1_norm_name in self.exact_name_idx:
            for target_id in self.exact_name_idx[s1_norm_name]:
                candidate_scores[target_id] += 10
                
        # 2. Compound address keys (+6 if bucket is discriminative)
        for k in s1_addr_keys:
            if k in self.compound_addr_idx:
                posting = self.compound_addr_idx[k]
                if len(posting) <= 150:
                    for target_id in posting:
                        candidate_scores[target_id] += 6
                        
        # 3. Significant core name tokens (+3)
        for t in set(s1_core_tokens):
            if t in self.token_idx:
                for target_id in self.token_idx[t]:
                    candidate_scores[target_id] += 3
                    
        # 4. 4-character prefix (+1)
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
            
        return [cid for cid, _ in candidate_scores.most_common(max_candidates)]
