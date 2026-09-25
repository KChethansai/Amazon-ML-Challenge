#!/usr/bin/env python3
"""
Diagnostic Error Analysis for Entity Resolution Blocking.
Measures candidate recall and classifies every missed ground-truth link
into root-cause failure categories.
"""

import os
import sys
import time
import json
import re
import unicodedata
from collections import defaultdict, Counter

sys.path.insert(0, "solution/business_entity_resolution/src")
from normalization import (
    normalize_name, extract_name_tokens,
    normalize_address, extract_address_digits, extract_address_blocking_keys
)
from blocking import CountryBlockingIndex

TRAIN_DIR = "student_resource/dataset/train"
SPLIT_PATH = "experiments/splits/dev_s1_ids.json"

INDIC_RANGE = [
    (0x0900, 0x097F), # Devanagari
    (0x0C00, 0x0C7F), # Telugu
    (0x0980, 0x09FF), # Bengali
    (0x0A00, 0x0A7F), # Gurmukhi
    (0x0A80, 0x0AFF), # Gujarati
    (0x0B00, 0x0B7F), # Oriya
    (0x0B80, 0x0BFF), # Tamil
    (0x0C80, 0x0CFF), # Kannada
    (0x0D00, 0x0D7F), # Malayalam
]

def has_indic_char(text: str) -> bool:
    for ch in text:
        cp = ord(ch)
        for start, end in INDIC_RANGE:
            if start <= cp <= end:
                return True
    return False

def has_url_handle(text: str) -> bool:
    return bool(re.search(r"(@|\.com|\.in|\.org|\.net|www\.)", text, re.IGNORECASE))

def main():
    print("=" * 70)
    print(" Phase 3: Diagnostic Blocking Error Taxonomy & Recall Benchmark")
    print("=" * 70)
    
    # 1. Load 5,000 S1 development entities for fast, deep diagnosis
    with open(SPLIT_PATH, "r") as f:
        all_dev_ids = json.load(f)
    diagnostic_s1_ids = set(all_dev_ids[:5000])
    print(f"Loaded {len(diagnostic_s1_ids):,} diagnostic S1 IDs from dev split.")
    
    # 2. Load ground truth for diagnostic S1
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
                
    print(f"Total True Positive Links in Diagnostic Set: {total_positives:,} across {len(diagnostic_s1_ids):,} S1 entities.")
    print(f"Needed Target Entities: {len(needed_targets):,}")
    
    # 3. Load S1 record data
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
                addr_keys = extract_address_blocking_keys(baddr, norm_ad)
                digits = extract_address_digits(baddr)
                s1_data[s1_id] = {
                    "raw_name": bname,
                    "norm_name": norm_nm,
                    "core_tokens": core_toks,
                    "raw_addr": baddr,
                    "norm_addr": norm_ad,
                    "addr_keys": addr_keys,
                    "digits": digits,
                    "country": bcountry
                }
                
    # 4. Build realistic background pool (500k S2 + 500k S3 + needed targets)
    print("\nBuilding realistic target index pool (500k S2 + 500k S3)...", flush=True)
    t0 = time.time()
    country_indexes = {
        "US": CountryBlockingIndex("US"),
        "India": CountryBlockingIndex("India")
    }
    
    raw_targets = {} # tid -> (bname, baddr, country, is_s2)
    
    # Read S2
    with open(os.path.join(TRAIN_DIR, "train_source2.tsv"), "r", encoding="utf-8") as f:
        next(f)
        for i, line in enumerate(f):
            parts = line.rstrip("\r\n").split("\t")
            eid, bname, baddr, bcountry = parts
            if eid in needed_targets or i < 500000:
                if eid in needed_targets:
                    raw_targets[eid] = (bname, baddr, bcountry, 1)
                c_idx = country_indexes.get(bcountry)
                if c_idx:
                    norm_nm = normalize_name(bname)
                    core_toks, _ = extract_name_tokens(norm_nm)
                    norm_ad = normalize_address(baddr)
                    addr_keys = extract_address_blocking_keys(baddr, norm_ad)
                    digits = extract_address_digits(baddr)
                    c_idx.add_target_record(eid, norm_nm, core_toks, norm_ad, addr_keys, digits, is_s2=1)
                    
    # Read S3
    with open(os.path.join(TRAIN_DIR, "train_source3.tsv"), "r", encoding="utf-8") as f:
        next(f)
        for i, line in enumerate(f):
            parts = line.rstrip("\r\n").split("\t")
            eid, bname, baddr, bcountry = parts
            if eid in needed_targets or i < 500000:
                if eid in needed_targets:
                    raw_targets[eid] = (bname, baddr, bcountry, 0)
                c_idx = country_indexes.get(bcountry)
                if c_idx:
                    norm_nm = normalize_name(bname)
                    core_toks, _ = extract_name_tokens(norm_nm)
                    norm_ad = normalize_address(baddr)
                    addr_keys = extract_address_blocking_keys(baddr, norm_ad)
                    digits = extract_address_digits(baddr)
                    c_idx.add_target_record(eid, norm_nm, core_toks, norm_ad, addr_keys, digits, is_s2=0)
                    
    print(f"Target pools loaded in {time.time()-t0:.2f}s: US={len(country_indexes['US'].records):,}, India={len(country_indexes['India'].records):,}")
    
    # Copy unpruned posting list counts for diagnosis
    token_lens = {c: {k: len(v) for k, v in idx.token_idx.items()} for c, idx in country_indexes.items()}
    prefix_lens = {c: {k: len(v) for k, v in idx.prefix_idx.items()} for c, idx in country_indexes.items()}
    
    # Prune frequent keys as baseline does
    for c_idx in country_indexes.values():
        c_idx.prune_frequent_keys()
        
    # 5. Query candidates and record hits
    print("\nQuerying baseline blocking index...", flush=True)
    t0 = time.time()
    retrieved_positives = 0
    missed_positives = []
    candidates_per_s1 = []
    
    # Also test recall if cap is uncapped vs cap=25
    uncapped_hits = 0
    cap25_hits = 0
    
    for s1_id, s1 in s1_data.items():
        c = s1["country"]
        c_idx = country_indexes.get(c)
        true_mids = gt_map.get(s1_id, set())
        if not c_idx or not true_mids:
            continue
            
        # Get raw candidate score counter
        candidate_scores = Counter()
        for t in set(s1["core_tokens"]):
            if t in c_idx.token_idx:
                for target_id in c_idx.token_idx[t]:
                    candidate_scores[target_id] += 3
        for k in s1["addr_keys"]:
            if k in c_idx.addr_idx:
                for target_id in c_idx.addr_idx[k]:
                    candidate_scores[target_id] += 4
        if len(s1["norm_name"]) >= 4:
            pfx = s1["norm_name"][:4]
            if pfx in c_idx.prefix_idx:
                for target_id in c_idx.prefix_idx[pfx]:
                    candidate_scores[target_id] += 1
                    
        total_cands = len(candidate_scores)
        candidates_per_s1.append(min(total_cands, 25))
        
        top25_cands = set([cid for cid, _ in candidate_scores.most_common(25)])
        all_cands = set(candidate_scores.keys())
        
        for mid in true_mids:
            in_all = mid in all_cands
            in_top25 = mid in top25_cands
            if in_all:
                uncapped_hits += 1
            if in_top25:
                cap25_hits += 1
            else:
                missed_positives.append((s1_id, mid, in_all, candidate_scores.get(mid, 0)))
                
    cap25_recall = cap25_hits / total_positives
    uncapped_recall = uncapped_hits / total_positives
    
    print("-" * 70)
    print(" BASELINE BLOCKING RESULTS ON DIAGNOSTIC SET:")
    print(f" Total True Positive Links : {total_positives:,}")
    print(f" Uncapped Candidate Recall : {uncapped_recall*100:.2f}% ({uncapped_hits:,} / {total_positives:,})")
    print(f" Top-25 Capped Recall      : {cap25_recall*100:.2f}% ({cap25_hits:,} / {total_positives:,})")
    print(f" Recall Lost to Top-25 Cap : {(uncapped_recall - cap25_recall)*100:.2f}% ({uncapped_hits - cap25_hits:,} links lost!)")
    print(f" Total Missed Links        : {len(missed_positives):,} ({100.0 - cap25_recall*100:.2f}%)")
    print("-" * 70)
    
    # 6. Failure Taxonomy Classification
    print("\nClassifying Missed Ground-Truth Links across Root Causes...")
    taxonomy = Counter()
    examples_by_cat = defaultdict(list)
    
    for s1_id, mid, in_all, score in missed_positives:
        s1 = s1_data[s1_id]
        tgt_raw = raw_targets.get(mid)
        if not tgt_raw:
            taxonomy["target_not_in_pool"] += 1
            continue
            
        t_bname, t_baddr, t_country, t_is_s2 = tgt_raw
        t_norm_nm = normalize_name(t_bname)
        t_core_toks, _ = extract_name_tokens(t_norm_nm)
        t_norm_ad = normalize_address(t_baddr)
        t_digits = extract_address_digits(t_baddr)
        
        # Check category:
        if in_all and not (score == 0):
            cat = "candidate_cap_truncation" # Present in index scores, but squeezed past rank 25!
        elif not t_baddr or t_baddr.strip() == "":
            cat = "missing_address_in_target"
        elif has_indic_char(t_bname) or has_indic_char(s1["raw_name"]):
            cat = "indic_script_mismatch"
        elif has_url_handle(t_bname) or has_url_handle(s1["raw_name"]):
            cat = "url_domain_handle_mismatch"
        elif any(token_lens[s1["country"]].get(tok, 0) > 3000 for tok in s1["core_tokens"]):
            cat = "high_frequency_token_pruned"
        elif not (set(s1["digits"]) & t_digits):
            cat = "numeric_address_mismatch"
        elif not (set(s1["core_tokens"]) & set(t_core_toks)):
            cat = "name_token_mismatch"
        else:
            cat = "compound_address_failure"
            
        taxonomy[cat] += 1
        if len(examples_by_cat[cat]) < 3:
            examples_by_cat[cat].append((s1["raw_name"], t_bname, s1["raw_addr"], t_baddr))
            
    print(f"\nRoot Cause Breakdown ({len(missed_positives):,} total missed links):")
    for cat, count in taxonomy.most_common():
        pct = 100.0 * count / len(missed_positives)
        print(f"  {cat:32s}: {count:5d} ({pct:5.2f}%)")
        for ex in examples_by_cat[cat][:2]:
            print(f"      S1:  '{ex[0]}' | '{ex[2]}'")
            print(f"      Tgt: '{ex[1]}' | '{ex[3]}'")
            
    print("=" * 70)

if __name__ == "__main__":
    main()
