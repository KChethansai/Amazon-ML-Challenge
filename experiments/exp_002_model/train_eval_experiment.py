#!/usr/bin/env python3
"""
Experiment 002: End-to-End Training & Validation with Multi-Pass Blocking & Calibrated Thresholding.
Trains on training split using Multi-Pass blocking candidates (hard negatives),
evaluates on the 10,000 S1 development split against the 1M target pool,
and performs fine-grained F0.5 grid search.
"""

import os
import sys
import time
import json
import math
import numpy as np
import lightgbm as lgb
from rapidfuzz import fuzz
from collections import defaultdict, Counter

sys.path.insert(0, "solution/business_entity_resolution/src")
from normalization import (
    clean_unicode_to_ascii, normalize_name, extract_name_tokens,
    normalize_address, extract_address_digits
)
from train import calculate_macro_f05
from blocking import CountryBlockingIndex

TRAIN_DIR = "student_resource/dataset/train"
DEV_SPLIT_PATH = "experiments/splits/dev_s1_ids.json"
TRAIN_SPLIT_PATH = "experiments/splits/train_pool_s1_ids.json"

import re
NUM_ONLY_RE = re.compile(r"\b\d+\b")
COMMON_ADDR_STOPWORDS = {
    "road", "street", "avenue", "drive", "lane", "boulevard", "floor", "suite", 
    "apartment", "unit", "ste", "apt", "bldg", "building", "fl", "pl", "circle",
    "square", "near", "opp", "opposite", "behind", "beside", "at", "post", "po", "dist",
    "door", "no", "plot", "block", "phase", "sector", "stage"
}

def extract_advanced_keys(norm_name: str, core_tokens: list, raw_addr: str, norm_addr: str):
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
        street_nums = [n for n in nums if 1 <= len(n) <= 5]
        
        for snum in street_nums[:3]:
            for p in postals[:2]:
                keys["compound_addr"].append(("ST_POST", snum, p))
        locality_candidates = tokens[:2] + tokens[-2:] if len(tokens) > 3 else tokens
        for snum in street_nums[:3]:
            for tok in set(locality_candidates):
                keys["compound_addr"].append(("ST_LOC", snum, tok))
        if postals and core_tokens:
            first_tok = core_tokens[0][:3]
            for p in postals[:1]:
                keys["compound_addr"].append(("POST_NAME", p, first_tok))
        if not street_nums and len(tokens) >= 2:
            keys["compound_addr"].append(("TOK_TOK", tokens[0], tokens[-1]))
    return keys

class FastMultiPassIndex:
    def __init__(self, country: str):
        self.country = country
        self.records = {} # eid -> (norm_name, norm_addr, raw_addr, is_s2)
        self.exact_name_idx = defaultdict(list)
        self.token_idx = defaultdict(list)
        self.prefix_idx = defaultdict(list)
        self.compound_addr_idx = defaultdict(list)
        
    def add_target(self, eid: str, norm_name: str, core_tokens: list, raw_addr: str, norm_addr: str, is_s2: int):
        self.records[eid] = (norm_name, norm_addr, raw_addr, is_s2)
        keys = extract_advanced_keys(norm_name, core_tokens, raw_addr, norm_addr)
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
        keys = extract_advanced_keys(s1_norm_name, s1_core_tokens, s1_raw_addr, s1_norm_addr)
        candidate_scores = Counter()
        N = max(1, len(self.records))
        
        if s1_norm_name in self.exact_name_idx:
            for tid in self.exact_name_idx[s1_norm_name]:
                candidate_scores[tid] += 15.0
        for ck in keys["compound_addr"]:
            if ck in self.compound_addr_idx:
                posting = self.compound_addr_idx[ck]
                if len(posting) <= 200:
                    score = 8.0 if ck[0] in ("ST_POST", "ST_LOC") else 5.0
                    for tid in posting:
                        candidate_scores[tid] += score
        for t in keys["name_tokens"]:
            if t in self.token_idx:
                posting = self.token_idx[t]
                df = len(posting)
                if df <= 5000:
                    idf_wt = max(1.0, math.log(N / max(1, df)))
                    score = min(6.0, 1.2 * idf_wt)
                    for tid in posting:
                        candidate_scores[tid] += score
        if keys["prefix4"] and keys["prefix4"] in self.prefix_idx:
            posting = self.prefix_idx[keys["prefix4"]]
            if len(posting) <= 2000:
                for tid in posting:
                    candidate_scores[tid] += 2.0
                    
        if not candidate_scores:
            return []
        if len(candidate_scores) <= max_candidates:
            return [(cid, score) for cid, score in candidate_scores.items()]
        return candidate_scores.most_common(max_candidates)

def char_ngrams(s: str, n: int = 3):
    if len(s) < n:
        return {s} if s else set()
    return set(s[i:i+n] for i in range(len(s)-n+1))

def jaccard(sa, sb):
    if not sa or not sb:
        return 0.0
    u = len(sa | sb)
    return len(sa & sb) / u if u > 0 else 0.0

def compute_expanded_features(s1_norm_name, s1_core_toks, s1_raw_addr, s1_norm_addr, s1_digits,
                              t_norm_name, t_norm_addr, t_raw_addr, is_s2, b_score, rank):
    n1, n2 = s1_norm_name, t_norm_name
    a1, a2 = s1_norm_addr, t_norm_addr
    
    # 1. Name features
    name_token_set = fuzz.token_set_ratio(n1, n2) / 100.0
    name_token_sort = fuzz.token_sort_ratio(n1, n2) / 100.0
    name_ratio = fuzz.ratio(n1, n2) / 100.0
    name_partial = fuzz.partial_ratio(n1, n2) / 100.0
    name_exact = 1.0 if (n1 == n2 and n1) else 0.0
    
    t_core_toks = set(t_norm_name.split())
    name_jacc = jaccard(s1_core_toks, t_core_toks)
    name_ng = jaccard(char_ngrams(n1, 3), char_ngrams(n2, 3))
    len_n1, len_n2 = len(n1), len(n2)
    name_len_diff = abs(len_n1 - len_n2) / max(len_n1, len_n2, 1)
    
    # 2. Address features
    addr_missing = 1.0 if not a2 else 0.0
    t_digits = set(NUM_ONLY_RE.findall(t_raw_addr)) if t_raw_addr else set()
    
    if addr_missing:
        addr_token_set = addr_token_sort = addr_ratio = addr_partial = addr_jacc = 0.0
        num_overlap = has_matching_digits = num_contradiction = 0.0
        addr_len_diff = 1.0
    else:
        addr_token_set = fuzz.token_set_ratio(a1, a2) / 100.0
        addr_token_sort = fuzz.token_sort_ratio(a1, a2) / 100.0
        addr_ratio = fuzz.ratio(a1, a2) / 100.0
        addr_partial = fuzz.partial_ratio(a1, a2) / 100.0
        addr_jacc = jaccard(set(a1.split()), set(a2.split()))
        
        d_inter = len(s1_digits & t_digits)
        d_union = len(s1_digits | t_digits)
        num_overlap = d_inter / max(1, d_union) if d_union > 0 else 0.0
        has_matching_digits = 1.0 if d_inter > 0 else 0.0
        num_contradiction = 1.0 if (s1_digits and t_digits and d_inter == 0) else 0.0
        len_a1, len_a2 = len(a1), len(a2)
        addr_len_diff = abs(len_a1 - len_a2) / max(len_a1, len_a2, 1)
        
    composite_score = (2.0 * name_token_set * addr_token_set) / max(1e-5, (name_token_set + addr_token_set)) if (not addr_missing) else name_token_set
    
    return [
        name_token_set,        # 0
        name_token_sort,       # 1
        name_ratio,            # 2
        name_partial,          # 3 (new)
        name_exact,            # 4
        name_jacc,             # 5
        name_ng,               # 6
        name_len_diff,         # 7
        addr_token_set,        # 8
        addr_token_sort,       # 9
        addr_ratio,            # 10
        addr_partial,          # 11 (new)
        addr_jacc,             # 12
        num_overlap,           # 13
        has_matching_digits,   # 14
        num_contradiction,     # 15 (new: prevents wrong address merges)
        addr_missing,          # 16
        addr_len_diff,         # 17
        composite_score,       # 18
        min(30.0, float(b_score)) / 30.0, # 19 (new: normalized blocking score)
        1.0 / (1.0 + float(rank)),        # 20 (new: inverse rank)
        float(is_s2)                      # 21
    ]

EXP_FEATURE_NAMES = [
    "name_token_set", "name_token_sort", "name_ratio", "name_partial", "name_exact",
    "name_jacc", "name_ng", "name_len_diff", "addr_token_set", "addr_token_sort",
    "addr_ratio", "addr_partial", "addr_jacc", "num_overlap", "has_matching_digits",
    "num_contradiction", "addr_missing", "addr_len_diff", "composite_score",
    "blocking_score", "inv_rank", "is_s2"
]

def main():
    print("=" * 70)
    print(" Experiment 002: End-to-End Pipeline Optimization")
    print("=" * 70)
    
    with open(DEV_SPLIT_PATH, "r") as f:
        dev_ids = json.load(f)[:10000] # 10,000 S1 dev entities
    with open(TRAIN_SPLIT_PATH, "r") as f:
        train_pool_ids = json.load(f)[:30000] # 30,000 S1 train entities
        
    dev_set = set(dev_ids)
    train_set = set(train_pool_ids)
    
    print(f"Loaded {len(train_set):,} Train S1, {len(dev_set):,} Validation S1.")
    
    # Load ground truth
    all_s1_gt = {}
    needed_targets = set()
    with open(os.path.join(TRAIN_DIR, "train_ground_truth.tsv"), "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            sid = parts[0]
            if sid in dev_set or sid in train_set:
                mstr = parts[1] if len(parts) > 1 else ""
                m_set = set(x.strip() for x in mstr.split(",") if x.strip())
                all_s1_gt[sid] = m_set
                needed_targets.update(m_set)
                
    print(f"Ground truth loaded for {len(all_s1_gt):,} entities. Needed target IDs: {len(needed_targets):,}")
    
    # Load S1 records
    s1_train_records = {}
    s1_val_records = {}
    with open(os.path.join(TRAIN_DIR, "train_source1.tsv"), "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            sid = parts[0]
            if sid in train_set or sid in dev_set:
                eid, bname, baddr, bcountry = parts
                norm_nm = normalize_name(bname)
                core_toks, _ = extract_name_tokens(norm_nm)
                norm_ad = normalize_address(baddr)
                digits = extract_address_digits(baddr)
                rec = {
                    "raw_name": bname, "norm_name": norm_nm, "core_toks": set(core_toks),
                    "raw_addr": baddr, "norm_addr": norm_ad, "digits": digits, "country": bcountry
                }
                if sid in train_set:
                    s1_train_records[sid] = rec
                else:
                    s1_val_records[sid] = rec
                    
    # Build 1M target pool
    print("Building 1M target index pool...", flush=True)
    t0 = time.time()
    country_indexes = {
        "US": FastMultiPassIndex("US"),
        "India": FastMultiPassIndex("India")
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
                    
    print(f"Target pools loaded in {time.time()-t0:.2f}s.")
    
    # Generate training pairs (Cap = 40)
    print("Generating training dataset (hard negatives from blocking)...", flush=True)
    t0 = time.time()
    X_train = []
    y_train = []
    
    for sid, s1 in s1_train_records.items():
        c_idx = country_indexes.get(s1["country"])
        if not c_idx:
            continue
        true_targets = all_s1_gt.get(sid, set())
        cand_list = c_idx.query(s1["norm_name"], list(s1["core_toks"]), s1["raw_addr"], s1["norm_addr"], max_candidates=40)
        
        cands_set = set(cid for cid, _ in cand_list)
        # Ensure true targets in index are included
        for tid in true_targets:
            if tid in c_idx.records and tid not in cands_set:
                cand_list.append((tid, 10.0))
                
        for rank, (cid, b_score) in enumerate(cand_list):
            trec = c_idx.records.get(cid)
            if not trec:
                continue
            is_match = 1.0 if cid in true_targets else 0.0
            feats = compute_expanded_features(
                s1["norm_name"], s1["core_toks"], s1["raw_addr"], s1["norm_addr"], s1["digits"],
                trec[0], trec[1], trec[2], trec[3], b_score, rank
            )
            X_train.append(feats)
            y_train.append(is_match)
            
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    n_pos = int(np.sum(y_train))
    print(f"Training dataset ready in {time.time()-t0:.2f}s: {len(X_train):,} pairs (Positives={n_pos:,}, Negatives={len(y_train)-n_pos:,})")
    
    # Train LightGBM model
    print("Training LightGBM Match Classifier (250 trees, lr=0.06)...", flush=True)
    t0 = time.time()
    train_data = lgb.Dataset(X_train, label=y_train)
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.06,
        "num_leaves": 45,
        "max_depth": 7,
        "min_child_samples": 25,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "verbosity": -1,
        "n_jobs": 4,
        "seed": 42
    }
    model = lgb.train(params, train_data, num_boost_round=250)
    print(f"Trained LightGBM in {time.time()-t0:.2f}s.")
    
    # Evaluate on 10,000 S1 validation set (Cap = 50)
    print("\nRunning inference on 10,000 S1 validation set...", flush=True)
    t0 = time.time()
    val_cand_pairs = {}
    val_gt_map = {sid: all_s1_gt.get(sid, set()) for sid in s1_val_records.keys()}
    
    for sid, s1 in s1_val_records.items():
        c_idx = country_indexes.get(s1["country"])
        if not c_idx:
            val_cand_pairs[sid] = []
            continue
        cand_list = c_idx.query(s1["norm_name"], list(s1["core_toks"]), s1["raw_addr"], s1["norm_addr"], max_candidates=50)
        
        pairs = []
        for rank, (cid, b_score) in enumerate(cand_list):
            trec = c_idx.records.get(cid)
            if not trec:
                continue
            feats = compute_expanded_features(
                s1["norm_name"], s1["core_toks"], s1["raw_addr"], s1["norm_addr"], s1["digits"],
                trec[0], trec[1], trec[2], trec[3], b_score, rank
            )
            pairs.append((cid, feats))
            
        if not pairs:
            val_cand_pairs[sid] = []
        else:
            feat_matrix = np.array([p[1] for p in pairs], dtype=np.float32)
            probs = model.predict(feat_matrix)
            val_cand_pairs[sid] = list(zip([p[0] for p in pairs], probs))
            
    print(f"Validation inference complete in {time.time()-t0:.2f}s.")
    
    # Fine-grained F0.5 Threshold Optimization (0.40 - 0.90)
    print("\n" + "-" * 70)
    print(" F0.5 Grid Search on Validation Split (10,000 S1 entities):")
    print("-" * 70)
    best_f05 = -1.0
    best_th = 0.65
    
    for th in np.arange(0.40, 0.86, 0.02):
        th = round(float(th), 2)
        preds = {}
        for sid, cand_probs in val_cand_pairs.items():
            matched = set(cid for cid, p in cand_probs if p >= th)
            preds[sid] = matched
            
        score = calculate_macro_f05(val_gt_map, preds)
        # Compute singleton stats
        pred_singletons = sum(1 for sid, m in preds.items() if len(m) == 0)
        print(f"  Threshold {th:.2f} -> Macro F0.5 = {score:.5f} | Singletons = {pred_singletons:,} ({pred_singletons/len(preds)*100:.2f}%)")
        if score > best_f05:
            best_f05 = score
            best_th = th
            
    print("-" * 70)
    print(f"\n>>> OPTIMAL THRESHOLD: {best_th:.2f} | BEST MACRO F0.5: {best_f05:.5f} <<<")
    delta_vs_baseline = best_f05 - 0.89837
    print(f"Delta vs Baseline (0.89837): {delta_vs_baseline:+.5f}")
    print("=" * 70)
    
    # Save model and config in experiment folder
    os.makedirs("experiments/exp_002_model", exist_ok=True)
    exp_model_path = "experiments/exp_002_model/lgbm_matcher_v2.txt"
    model.save_model(exp_model_path)
    
    exp_config = {
        "best_threshold": best_th,
        "best_val_f05": best_f05,
        "delta_vs_baseline": delta_vs_baseline,
        "feature_names": EXP_FEATURE_NAMES,
        "num_train_s1": len(train_set),
        "num_val_s1": len(dev_set),
        "trained_date": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open("experiments/exp_002_model/config_v2.json", "w") as f:
        json.dump(exp_config, f, indent=2)
        
    # Log to ledger
    ledger_entry = {
        "exp_id": "EXP_002_MULTIPASS_EXPANDED_FEATS",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "description": "Multi-Pass V2 Blocking (Cap=50) + 22 Expanded Features + Num Contradiction Protection + LightGBM (depth 7, leaves 45)",
        "macro_f05": best_f05,
        "threshold": best_th,
        "delta_vs_baseline": delta_vs_baseline
    }
    with open("experiments/ledger.jsonl", "a") as f:
        f.write(json.dumps(ledger_entry) + "\n")
    print("Logged experiment results to experiments/ledger.jsonl")

if __name__ == "__main__":
    main()
