#!/usr/bin/env python3
"""
Experiment 001b: Compound Locality & Numeric Address Indexing.
Specifically targets Indic script transliterations where names differ completely
but numeric street/plot numbers and city/state locality names match.
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

NUM_ONLY_RE = re.compile(r"\b\d+\b")

COMMON_ADDR_STOPWORDS = {
    "road", "street", "avenue", "drive", "lane", "boulevard", "floor", "suite", 
    "apartment", "unit", "ste", "apt", "bldg", "building", "fl", "pl", "circle",
    "square", "near", "opp", "opposite", "behind", "beside", "at", "post", "po", "dist",
    "door", "no", "plot", "block", "phase", "sector", "stage"
}

def extract_advanced_keys_v2(norm_name: str, core_tokens: list, raw_addr: str, norm_addr: str):
    keys = {
        "core_name": norm_name,
        "name_tokens": set(core_tokens),
        "prefix4": norm_name[:4] if len(norm_name) >= 4 else None,
        "compound_addr": [],
        "postal_codes": []
    }
    
    if raw_addr:
        nums = NUM_ONLY_RE.findall(raw_addr)
        tokens = [t for t in norm_addr.split() if len(t) >= 4 and t not in COMMON_ADDR_STOPWORDS]
        
        postals = [n for n in nums if len(n) in (5, 6)]
        keys["postal_codes"] = postals
        
        # Numeric street/plot/flat numbers
        street_nums = [n for n in nums if 1 <= len(n) <= 5]
        
        # 1. (street_num, postal)
        for snum in street_nums[:3]:
            for p in postals[:2]:
                keys["compound_addr"].append(("ST_POST", snum, p))
                
        # 2. (street_num, locality): use first 2 tokens AND last 2 tokens (city/state)
        locality_candidates = tokens[:2] + tokens[-2:] if len(tokens) > 3 else tokens
        for snum in street_nums[:3]:
            for tok in set(locality_candidates):
                keys["compound_addr"].append(("ST_LOC", snum, tok))
                
        # 3. (postal, first_name_prefix3)
        if postals and core_tokens:
            first_tok = core_tokens[0][:3]
            for p in postals[:1]:
                keys["compound_addr"].append(("POST_NAME", p, first_tok))
                
        # 4. If address has NO numbers, pair first token + last token (locality/city)
        if not street_nums and len(tokens) >= 2:
            keys["compound_addr"].append(("TOK_TOK", tokens[0], tokens[-1]))
            
    return keys

class MultiPassBlockingIndexV2:
    def __init__(self, country: str):
        self.country = country
        self.records = {}
        self.exact_name_idx = defaultdict(list)
        self.token_idx = defaultdict(list)
        self.prefix_idx = defaultdict(list)
        self.compound_addr_idx = defaultdict(list)
        
    def add_target(self, eid: str, norm_name: str, core_tokens: list, raw_addr: str, norm_addr: str, is_s2: int):
        self.records[eid] = (norm_name, norm_addr, is_s2)
        keys = extract_advanced_keys_v2(norm_name, core_tokens, raw_addr, norm_addr)
        
        if norm_name and len(norm_name) >= 3:
            self.exact_name_idx[norm_name].append(eid)
        for t in keys["name_tokens"]:
            if len(t) >= 3:
                self.token_idx[t].append(eid)
        if keys["prefix4"]:
            self.prefix_idx[keys["prefix4"]].append(eid)
        for ck in keys["compound_addr"]:
            self.compound_addr_idx[ck].append(eid)
            
    def query(self, s1_norm_name: str, s1_core_tokens: list, s1_raw_addr: str, s1_norm_addr: str, max_candidates: int = 50):
        keys = extract_advanced_keys_v2(s1_norm_name, s1_core_tokens, s1_raw_addr, s1_norm_addr)
        candidate_scores = Counter()
        N = max(1, len(self.records))
        
        # 1. Exact core name (+15)
        if s1_norm_name in self.exact_name_idx:
            for tid in self.exact_name_idx[s1_norm_name]:
                candidate_scores[tid] += 15.0
                
        # 2. Compound address keys (+8.0)
        for ck in keys["compound_addr"]:
            if ck in self.compound_addr_idx:
                posting = self.compound_addr_idx[ck]
                if len(posting) <= 200: # allow up to 200 records in compound bucket
                    score = 8.0 if ck[0] in ("ST_POST", "ST_LOC") else 5.0
                    for tid in posting:
                        candidate_scores[tid] += score
                        
        # 3. Rare/Informative name tokens (IDF-weighted)
        for t in keys["name_tokens"]:
            if t in self.token_idx:
                posting = self.token_idx[t]
                df = len(posting)
                if df <= 5000:
                    idf_wt = max(1.0, math.log(N / max(1, df)))
                    score = min(6.0, 1.2 * idf_wt)
                    for tid in posting:
                        candidate_scores[tid] += score
                        
        # 4. Prefix match (+2.0)
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
    with open(SPLIT_PATH, "r") as f:
        all_dev_ids = json.load(f)
    diagnostic_s1_ids = set(all_dev_ids[:5000])
    
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
                
    country_indexes = {
        "US": MultiPassBlockingIndexV2("US"),
        "India": MultiPassBlockingIndexV2("India")
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
                    
    print("MultiPassBlockingIndexV2 loaded.")
    for cap in [25, 50, 75]:
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
        print(f"V2 Cap={cap:2d}: Recall={recall*100:.2f}% ({hits:,}/{total_positives:,}), Avg Cands/S1={avg_cands:.1f}")

if __name__ == "__main__":
    main()
