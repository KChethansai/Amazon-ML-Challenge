#!/usr/bin/env python3
"""
Experiment 001: Multi-Pass Compound Blocking & Adaptive Candidate Capping.
Tests:
  1. Compound Address Keys: (country, street_num, postal) and (country, street_num, locality)
  2. Exact Normalized Core Name Pass
  3. Frequency-weighted (IDF-style) name token ranking instead of hard deletion
  4. Candidate cap sweep: 25, 50, 75, 100
"""

import os
import sys
import time
import json
import math
import re
from collections import defaultdict, Counter

sys.path.insert(0, "solution/business_entity_resolution/src")
from normalization import (
    clean_unicode_to_ascii, normalize_name, extract_name_tokens,
    normalize_address, extract_address_digits
)

TRAIN_DIR = "student_resource/dataset/train"
SPLIT_PATH = "experiments/splits/dev_s1_ids.json"

NUM_RE = re.compile(r"\b\d+[\w/-]*\b")
NUM_ONLY_RE = re.compile(r"\b\d+\b")

COMMON_ADDR_STOPWORDS = {
    "road", "street", "avenue", "drive", "lane", "boulevard", "floor", "suite", 
    "apartment", "unit", "ste", "apt", "bldg", "building", "fl", "pl", "circle",
    "square", "near", "opp", "opposite", "behind", "beside", "at", "post", "po", "dist"
}

def extract_advanced_keys(norm_name: str, core_tokens: list, raw_addr: str, norm_addr: str):
    """
    Extract multi-pass blocking keys:
    1. Exact core name
    2. Name core tokens
    3. Compound address keys: (street_num, postal_or_city)
    4. 4-char name prefix
    """
    keys = {
        "core_name": norm_name,
        "name_tokens": set(core_tokens),
        "prefix4": norm_name[:4] if len(norm_name) >= 4 else None,
        "compound_addr": [],
        "postal_codes": []
    }
    
    if raw_addr:
        nums = NUM_ONLY_RE.findall(raw_addr)
        addr_tokens = [t for t in norm_addr.split() if len(t) >= 4 and t not in COMMON_ADDR_STOPWORDS]
        
        # Postal codes (5-digit US/FR, 6-digit India)
        postals = [n for n in nums if len(n) in (5, 6)]
        keys["postal_codes"] = postals
        
        street_nums = [n for n in nums if 1 <= len(n) <= 4]
        
        # Compound key 1: (street_num, postal) -> extremely high precision
        for snum in street_nums[:2]:
            for p in postals[:2]:
                keys["compound_addr"].append(("ST_POST", snum, p))
                
        # Compound key 2: (street_num, locality_token)
        for snum in street_nums[:2]:
            for tok in addr_tokens[:2]:
                keys["compound_addr"].append(("ST_TOK", snum, tok))
                
        # Compound key 3: (postal, first_name_tok_prefix3)
        if postals and core_tokens:
            first_tok = core_tokens[0][:3]
            for p in postals[:1]:
                keys["compound_addr"].append(("POST_NAME", p, first_tok))
                
    return keys

class MultiPassBlockingIndex:
    def __init__(self, country: str):
        self.country = country
        self.records = {} # eid -> (norm_name, norm_addr, is_s2)
        
        # Inverted index tables
        self.exact_name_idx = defaultdict(list)
        self.token_idx = defaultdict(list)
        self.prefix_idx = defaultdict(list)
        self.compound_addr_idx = defaultdict(list)
        
    def add_target(self, eid: str, norm_name: str, core_tokens: list, raw_addr: str, norm_addr: str, is_s2: int):
        self.records[eid] = (norm_name, norm_addr, is_s2)
        keys = extract_advanced_keys(norm_name, core_tokens, raw_addr, norm_addr)
        
        # 1. Exact core name
        if norm_name and len(norm_name) >= 3:
            self.exact_name_idx[norm_name].append(eid)
            
        # 2. Name tokens
        for t in keys["name_tokens"]:
            if len(t) >= 3:
                self.token_idx[t].append(eid)
                
        # 3. 4-char prefix
        if keys["prefix4"]:
            self.prefix_idx[keys["prefix4"]].append(eid)
            
        # 4. Compound address keys
        for ck in keys["compound_addr"]:
            self.compound_addr_idx[ck].append(eid)
            
    def query(self, s1_norm_name: str, s1_core_tokens: list, s1_raw_addr: str, s1_norm_addr: str, max_candidates: int = 50):
        keys = extract_advanced_keys(s1_norm_name, s1_core_tokens, s1_raw_addr, s1_norm_addr)
        candidate_scores = Counter()
        N = max(1, len(self.records))
        
        # 1. Exact core name (Highest confidence: +15)
        if s1_norm_name in self.exact_name_idx:
            for tid in self.exact_name_idx[s1_norm_name]:
                candidate_scores[tid] += 15.0
                
        # 2. Compound address keys (Very high precision: +10)
        for ck in keys["compound_addr"]:
            if ck in self.compound_addr_idx:
                posting = self.compound_addr_idx[ck]
                if len(posting) <= 100: # only informative compound buckets
                    for tid in posting:
                        candidate_scores[tid] += 8.0
                        
        # 3. Rare/Informative name tokens (IDF-weighted: log(N/df))
        for t in keys["name_tokens"]:
            if t in self.token_idx:
                posting = self.token_idx[t]
                df = len(posting)
                if df <= 5000:
                    idf_wt = max(1.0, math.log(N / max(1, df)))
                    # Score scaled: rare tokens get up to +6, common get +1.5
                    score = min(6.0, 1.2 * idf_wt)
                    for tid in posting:
                        candidate_scores[tid] += score
                        
        # 4. Prefix match (+2.0 if not too common)
        if keys["prefix4"] and keys["prefix4"] in self.prefix_idx:
            posting = self.prefix_idx[keys["prefix4"]]
            if len(posting) <= 2000:
                for tid in posting:
                    candidate_scores[tid] += 2.0
                    
        if not candidate_scores:
            return []
        if len(candidate_scores) <= max_candidates:
            return list(candidate_scores.keys())
        return [cid for cid, _ in candidate_scores.most_common(max_candidates)]

def main():
    print("=" * 70)
    print(" Experiment 001: Multi-Pass Compound Blocking & Adaptive Cap Test")
    print("=" * 70)
    
    with open(SPLIT_PATH, "r") as f:
        all_dev_ids = json.load(f)
    diagnostic_s1_ids = set(all_dev_ids[:5000])
    
    # Load ground truth
    gt_map = {}
    needed_targets = set()
    total_positives = 0
    with open(os.path.join(TRAIN_DIR, "train_ground_truth.tsv"), "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            s1_id = parts[0]
            if s1_id in diagnostic_s1_ids:
                mstr = parts[1] if len(parts) > 1 else ""
                m_set = set(x.strip() for x in mstr.split(",") if x.strip())
                gt_map[s1_id] = m_set
                needed_targets.update(m_set)
                total_positives += len(m_set)
                
    # Load S1 records
    s1_data = {}
    with open(os.path.join(TRAIN_DIR, "train_source1.tsv"), "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            s1_id = parts[0]
            if s1_id in diagnostic_s1_ids:
                eid, bname, baddr, bcountry = parts
                norm_nm = normalize_name(bname)
                core_toks, _ = extract_name_tokens(norm_nm)
                norm_ad = normalize_address(baddr)
                s1_data[s1_id] = {
                    "raw_name": bname, "norm_name": norm_nm, "core_tokens": core_toks,
                    "raw_addr": baddr, "norm_addr": norm_ad, "country": bcountry
                }
                
    # Build 1M background pool
    print("Loading 1M target pool into MultiPassBlockingIndex...", flush=True)
    t0 = time.time()
    country_indexes = {
        "US": MultiPassBlockingIndex("US"),
        "India": MultiPassBlockingIndex("India")
    }
    
    with open(os.path.join(TRAIN_DIR, "train_source2.tsv"), "r", encoding="utf-8") as f:
        next(f)
        for i, line in enumerate(f):
            parts = line.rstrip("\r\n").split("\t")
            eid, bname, baddr, bcountry = parts
            if eid in needed_targets or i < 500000:
                c_idx = country_indexes.get(bcountry)
                if c_idx:
                    norm_nm = normalize_name(bname)
                    core_toks, _ = extract_name_tokens(norm_nm)
                    norm_ad = normalize_address(baddr)
                    c_idx.add_target(eid, norm_nm, core_toks, baddr, norm_ad, is_s2=1)
                    
    with open(os.path.join(TRAIN_DIR, "train_source3.tsv"), "r", encoding="utf-8") as f:
        next(f)
        for i, line in enumerate(f):
            parts = line.rstrip("\r\n").split("\t")
            eid, bname, baddr, bcountry = parts
            if eid in needed_targets or i < 500000:
                c_idx = country_indexes.get(bcountry)
                if c_idx:
                    norm_nm = normalize_name(bname)
                    core_toks, _ = extract_name_tokens(norm_nm)
                    norm_ad = normalize_address(baddr)
                    c_idx.add_target(eid, norm_nm, core_toks, baddr, norm_ad, is_s2=0)
                    
    print(f"Target pools loaded in {time.time()-t0:.2f}s: US={len(country_indexes['US'].records):,}, India={len(country_indexes['India'].records):,}")
    
    # Query with caps: 25, 50, 75, 100
    caps = [25, 50, 75, 100]
    results = {}
    
    for cap in caps:
        print(f"\nBenchmarking Candidate Cap = {cap}...", flush=True)
        t_query = time.time()
        hits = 0
        cand_counts = []
        
        for s1_id, s1 in s1_data.items():
            true_mids = gt_map.get(s1_id, set())
            if not true_mids:
                continue
            c_idx = country_indexes.get(s1["country"])
            if not c_idx:
                continue
                
            cands = set(c_idx.query(
                s1["norm_name"], s1["core_tokens"], s1["raw_addr"], s1["norm_addr"], max_candidates=cap
            ))
            cand_counts.append(len(cands))
            hits += len(true_mids & cands)
            
        recall = hits / total_positives
        avg_cands = sum(cand_counts) / max(1, len(cand_counts))
        p95_cands = sorted(cand_counts)[int(len(cand_counts) * 0.95)]
        elapsed = time.time() - t_query
        
        results[cap] = {
            "recall": recall,
            "hits": hits,
            "avg_cands": avg_cands,
            "p95_cands": p95_cands,
            "time": elapsed
        }
        print(f"  Cap={cap:3d}: Recall={recall*100:.2f}% ({hits:,}/{total_positives:,}), Avg Cands/S1={avg_cands:.1f}, P95={p95_cands}, Time={elapsed:.2f}s")
        
    print("\n" + "=" * 70)
    print(" SUMMARY COMPARISON: BASELINE vs MULTI-PASS BLOCKING")
    print("=" * 70)
    print(f" Baseline Cap=25 Recall : 78.78% (13,633 / 17,306)")
    for cap in caps:
        delta = results[cap]["recall"] - 0.7878
        print(f" Multi-Pass Cap={cap:3d} : Recall={results[cap]['recall']*100:.2f}% (+{delta*100:+.2f}%) | Avg Cands={results[cap]['avg_cands']:.1f}")
    print("=" * 70)

if __name__ == "__main__":
    main()
