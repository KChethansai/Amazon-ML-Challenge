#!/usr/bin/env python3
"""
Create reproducible entity-level stratified validation splits for Amazon ML Challenge 2026.
Stratification key: (country, is_singleton, match_count_bucket).
Slices:
  - 40,000 S1 development entities
  - 15,000 S1 final holdout entities
"""

import os
import json
import random
from collections import defaultdict

TRAIN_DIR = "student_resource/dataset/train"
SPLIT_DIR = "experiments/splits"
SEED = 42

def get_match_bucket(count: int) -> str:
    if count == 0:
        return "0"
    elif count == 1:
        return "1"
    elif count == 2:
        return "2"
    elif count == 3:
        return "3"
    elif count <= 6:
        return "4-6"
    else:
        return "7+"

def main():
    os.makedirs(SPLIT_DIR, exist_ok=True)
    random.seed(SEED)
    
    print("Loading ground truth match counts...", flush=True)
    gt_map = {}
    with open(os.path.join(TRAIN_DIR, "train_ground_truth.tsv"), "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            s1_id = parts[0]
            mstr = parts[1] if len(parts) > 1 else ""
            m_set = set(x.strip() for x in mstr.split(",") if x.strip())
            gt_map[s1_id] = m_set
            
    print(f"Loaded ground truth for {len(gt_map):,} entities.", flush=True)
    
    # Read S1 records and stratify
    print("Reading S1 records and building strata...", flush=True)
    strata = defaultdict(list)
    s1_metadata = {}
    with open(os.path.join(TRAIN_DIR, "train_source1.tsv"), "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            s1_id = parts[0]
            country = parts[3] if len(parts) > 3 else "UNKNOWN"
            m_set = gt_map.get(s1_id, set())
            n_matches = len(m_set)
            is_singleton = (n_matches == 0)
            bucket = get_match_bucket(n_matches)
            stratum_key = f"{country}_{'singleton' if is_singleton else 'matched'}_{bucket}"
            strata[stratum_key].append(s1_id)
            s1_metadata[s1_id] = {
                "country": country,
                "is_singleton": is_singleton,
                "n_matches": n_matches,
                "bucket": bucket
            }
            
    print(f"Built {len(strata)} strata across {len(s1_metadata):,} entities.")
    for k, v in sorted(strata.items()):
        print(f"  Strata {k:30s}: {len(v):,}")
        
    DEV_SIZE = 40000
    HOLDOUT_SIZE = 15000
    TRAIN_POOL_SIZE = 60000
    TOTAL_SAMPLE = DEV_SIZE + HOLDOUT_SIZE + TRAIN_POOL_SIZE
    
    dev_ids = []
    holdout_ids = []
    train_pool_ids = []
    
    # Stratified proportional sampling
    total_pop = len(s1_metadata)
    for stratum_key, members in strata.items():
        random.shuffle(members)
        stratum_frac = len(members) / total_pop
        
        n_dev = int(round(DEV_SIZE * stratum_frac))
        n_hold = int(round(HOLDOUT_SIZE * stratum_frac))
        n_tr = int(round(TRAIN_POOL_SIZE * stratum_frac))
        
        dev_ids.extend(members[:n_dev])
        holdout_ids.extend(members[n_dev:n_dev + n_hold])
        train_pool_ids.extend(members[n_dev + n_hold:n_dev + n_hold + n_tr])
        
    # Trim or fill to exact targets if rounding caused slight mismatch
    random.shuffle(dev_ids)
    random.shuffle(holdout_ids)
    random.shuffle(train_pool_ids)
    dev_ids = dev_ids[:DEV_SIZE]
    holdout_ids = holdout_ids[:HOLDOUT_SIZE]
    train_pool_ids = train_pool_ids[:TRAIN_POOL_SIZE]
    
    assert len(set(dev_ids) & set(holdout_ids)) == 0, "Leakage between dev and holdout!"
    assert len(set(dev_ids) & set(train_pool_ids)) == 0, "Leakage between dev and train pool!"
    assert len(set(holdout_ids) & set(train_pool_ids)) == 0, "Leakage between holdout and train pool!"
    
    print(f"\nFinal Split Sizes:")
    print(f"  Train pool S1: {len(train_pool_ids):,}")
    print(f"  Development S1: {len(dev_ids):,}")
    print(f"  Holdout S1    : {len(holdout_ids):,}")
    
    # Save splits
    with open(os.path.join(SPLIT_DIR, "dev_s1_ids.json"), "w") as f:
        json.dump(dev_ids, f)
    with open(os.path.join(SPLIT_DIR, "holdout_s1_ids.json"), "w") as f:
        json.dump(holdout_ids, f)
    with open(os.path.join(SPLIT_DIR, "train_pool_s1_ids.json"), "w") as f:
        json.dump(train_pool_ids, f)
        
    print(f"Saved splits to {SPLIT_DIR}/")

if __name__ == "__main__":
    main()
