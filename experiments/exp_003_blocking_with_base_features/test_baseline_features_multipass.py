#!/usr/bin/env python3
"""
Experiment 003: Multi-Pass Blocking with Proven Clean Features.
Tests the MultiPass V2 blocking candidates using the baseline 17 clean features
and compares against the original baseline on the dev split.
"""

import os
import sys
import time
import json
import numpy as np
import lightgbm as lgb
from collections import defaultdict, Counter

sys.path.insert(0, "solution/business_entity_resolution/src")
from normalization import (
    normalize_name, extract_name_tokens,
    normalize_address, extract_address_digits
)
from features import compute_pairwise_features, FEATURE_NAMES
from train import calculate_macro_f05

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
        "compound_addr": []
    }
    if raw_addr:
        nums = NUM_ONLY_RE.findall(raw_addr)
        tokens = [t for t in norm_addr.split() if len(t) >= 4 and t not in COMMON_ADDR_STOPWORDS]
        postals = [n for n in nums if len(n) in (5, 6)]
        street_nums = [n for n in nums if 1 <= len(n) <= 5]
        
        for snum in street_nums[:2]:
            for p in postals[:2]:
                keys["compound_addr"].append(("ST_POST", snum, p))
        locality_candidates = tokens[:2] + tokens[-2:] if len(tokens) > 3 else tokens
        for snum in street_nums[:2]:
            for tok in set(locality_candidates):
                keys["compound_addr"].append(("ST_LOC", snum, tok))
        if postals and core_tokens:
            keys["compound_addr"].append(("POST_NAME", postals[0], core_tokens[0][:3]))
    return keys

class CleanMultiPassIndex:
    def __init__(self, country: str):
        self.country = country
        self.records = {} # eid -> (norm_name, norm_addr, is_s2)
        self.exact_name_idx = defaultdict(list)
        self.token_idx = defaultdict(list)
        self.prefix_idx = defaultdict(list)
        self.compound_addr_idx = defaultdict(list)
        
    def add_target(self, eid: str, norm_name: str, core_tokens: list, raw_addr: str, norm_addr: str, is_s2: int):
        self.records[eid] = (norm_name, norm_addr, is_s2)
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
            
    def prune(self):
        # Prune only extremely bloated single-token lists (>4000)
        for t in list(self.token_idx.keys()):
            if len(self.token_idx[t]) > 4000:
                del self.token_idx[t]
        for pfx in list(self.prefix_idx.keys()):
            if len(self.prefix_idx[pfx]) > 4000:
                del self.prefix_idx[pfx]
                
    def query(self, s1_norm_name: str, s1_core_tokens: list, s1_raw_addr: str, s1_norm_addr: str, max_candidates: int = 35):
        keys = extract_advanced_keys(s1_norm_name, s1_core_tokens, s1_raw_addr, s1_norm_addr)
        candidate_scores = Counter()
        
        # 1. Exact core name match (+10)
        if s1_norm_name in self.exact_name_idx:
            for tid in self.exact_name_idx[s1_norm_name]:
                candidate_scores[tid] += 10
                
        # 2. Compound address keys (+6)
        for ck in keys["compound_addr"]:
            if ck in self.compound_addr_idx:
                posting = self.compound_addr_idx[ck]
                if len(posting) <= 150:
                    for tid in posting:
                        candidate_scores[tid] += 6
                        
        # 3. Name tokens (+3)
        for t in keys["name_tokens"]:
            if t in self.token_idx:
                for tid in self.token_idx[t]:
                    candidate_scores[tid] += 3
                    
        # 4. Prefix match (+1)
        if keys["prefix4"] and keys["prefix4"] in self.prefix_idx:
            for tid in self.prefix_idx[keys["prefix4"]]:
                candidate_scores[tid] += 1
                
        if not candidate_scores:
            return []
        if len(candidate_scores) <= max_candidates:
            return list(candidate_scores.keys())
        return [cid for cid, _ in candidate_scores.most_common(max_candidates)]

def main():
    print("=" * 70)
    print(" Experiment 003: Multi-Pass Blocking with Proven Clean Features")
    print("=" * 70)
    
    with open(DEV_SPLIT_PATH, "r") as f:
        dev_ids = json.load(f)[:5000] # 5,000 S1 dev entities
    with open(TRAIN_SPLIT_PATH, "r") as f:
        train_ids = json.load(f)[:25000] # 25,000 S1 train entities
        
    dev_set = set(dev_ids)
    train_set = set(train_ids)
    
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
                    "raw_name": bname, "norm_name": norm_nm, "core_tokens": set(core_toks),
                    "raw_addr": baddr, "norm_addr": norm_ad,
                    "addr_tokens": set(norm_ad.split()) if norm_ad else set(),
                    "digits": digits, "country": bcountry
                }
                if sid in train_set:
                    s1_train_records[sid] = rec
                else:
                    s1_val_records[sid] = rec
                    
    # Build target index pool (350k background per source)
    print("Building target index pool (350k per source)...", flush=True)
    t0 = time.time()
    country_indexes = {
        "US": CleanMultiPassIndex("US"),
        "India": CleanMultiPassIndex("India")
    }
    
    with open(os.path.join(TRAIN_DIR, "train_source2.tsv"), "r", encoding="utf-8") as f:
        next(f)
        for i, line in enumerate(f):
            parts = line.rstrip("\r\n").split("\t")
            eid, bname, baddr, bcountry = parts
            if eid in needed_targets or i < 350000:
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
            if eid in needed_targets or i < 350000:
                c_idx = country_indexes.get(bcountry)
                if c_idx:
                    norm_nm = normalize_name(bname)
                    core_toks, _ = extract_name_tokens(norm_nm)
                    norm_ad = normalize_address(baddr)
                    c_idx.add_target(eid, norm_nm, core_toks, baddr, norm_ad, is_s2=0)
                    
    for c_idx in country_indexes.values():
        c_idx.prune()
    print(f"Target pools loaded in {time.time()-t0:.2f}s.")
    
    # Generate training pairs (Cap = 35)
    print("Generating training dataset...", flush=True)
    t0 = time.time()
    X_train = []
    y_train = []
    
    for sid, s1 in s1_train_records.items():
        c_idx = country_indexes.get(s1["country"])
        if not c_idx:
            continue
        true_targets = all_s1_gt.get(sid, set())
        cands = set(c_idx.query(s1["norm_name"], list(s1["core_tokens"]), s1["raw_addr"], s1["norm_addr"], max_candidates=35))
        
        # Ensure true targets are present
        for tid in true_targets:
            if tid in c_idx.records:
                cands.add(tid)
                
        for cid in cands:
            trec = c_idx.records.get(cid)
            if not trec:
                continue
            is_match = 1.0 if cid in true_targets else 0.0
            feats = compute_pairwise_features(s1, trec, trec[2])
            X_train.append(feats)
            y_train.append(is_match)
            
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    n_pos = int(np.sum(y_train))
    print(f"Training dataset: {len(X_train):,} pairs (Positives={n_pos:,}, Negatives={len(y_train)-n_pos:,}) in {time.time()-t0:.2f}s.")
    
    # Train LightGBM model
    print("Training LightGBM Match Classifier (180 trees, lr=0.08)...", flush=True)
    t0 = time.time()
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_NAMES)
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.08,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 20,
        "feature_fraction": 0.9,
        "n_jobs": 4,
        "verbose": -1,
        "seed": 42
    }
    model = lgb.train(params, train_data, num_boost_round=180)
    print(f"Trained LightGBM in {time.time()-t0:.2f}s.")
    
    # Evaluate on 5,000 S1 validation set
    print("\nRunning inference on 5,000 S1 validation set (Cap=35)...", flush=True)
    t0 = time.time()
    val_cand_pairs = {}
    val_gt_map = {sid: all_s1_gt.get(sid, set()) for sid in s1_val_records.keys()}
    
    for sid, s1 in s1_val_records.items():
        c_idx = country_indexes.get(s1["country"])
        if not c_idx:
            val_cand_pairs[sid] = []
            continue
        cands = c_idx.query(s1["norm_name"], list(s1["core_tokens"]), s1["raw_addr"], s1["norm_addr"], max_candidates=35)
        
        feat_batch = []
        valid_cands = []
        for cid in cands:
            trec = c_idx.records.get(cid)
            if trec:
                feat_batch.append(compute_pairwise_features(s1, trec, trec[2]))
                valid_cands.append(cid)
                
        if feat_batch:
            probs = model.predict(np.array(feat_batch, dtype=np.float32))
            val_cand_pairs[sid] = list(zip(valid_cands, probs))
        else:
            val_cand_pairs[sid] = []
            
    print(f"Validation inference complete in {time.time()-t0:.2f}s.")
    
    # Fine-grained F0.5 Threshold Optimization (0.50 - 0.85)
    print("\n" + "-" * 70)
    print(" F0.5 Grid Search on Validation Split:")
    print("-" * 70)
    best_f05 = -1.0
    best_th = 0.65
    
    for th in np.arange(0.50, 0.86, 0.02):
        th = round(float(th), 2)
        preds = {}
        for sid, cand_probs in val_cand_pairs.items():
            matched = set(cid for cid, p in cand_probs if p >= th)
            preds[sid] = matched
            
        score = calculate_macro_f05(val_gt_map, preds)
        pred_singletons = sum(1 for sid, m in preds.items() if len(m) == 0)
        print(f"  Threshold {th:.2f} -> Macro F0.5 = {score:.5f} | Singletons = {pred_singletons:,} ({pred_singletons/len(preds)*100:.2f}%)")
        if score > best_f05:
            best_f05 = score
            best_th = th
            
    print("-" * 70)
    print(f"\n>>> OPTIMAL THRESHOLD: {best_th:.2f} | BEST MACRO F0.5: {best_f05:.5f} <<<")
    delta = best_f05 - 0.89837
    print(f"Delta vs Baseline (0.89837): {delta:+.5f}")
    print("=" * 70)
    
    # Save model and config
    os.makedirs("experiments/exp_003_clean_multipass", exist_ok=True)
    exp_model_path = "experiments/exp_003_clean_multipass/lgbm_clean_multipass.txt"
    model.save_model(exp_model_path)
    
    exp_config = {
        "best_threshold": best_th,
        "best_val_f05": best_f05,
        "delta_vs_baseline": delta,
        "feature_names": FEATURE_NAMES,
        "max_candidates": 35,
        "num_train_s1": len(train_set),
        "num_val_s1": len(dev_set),
        "trained_date": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open("experiments/exp_003_clean_multipass/config.json", "w") as f:
        json.dump(exp_config, f, indent=2)
        
    ledger_entry = {
        "exp_id": "EXP_003_CLEAN_MULTIPASS",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "description": "Multi-Pass Compound Indexing (Cap=35) + Proven Clean Features + LightGBM (180 trees)",
        "macro_f05": best_f05,
        "threshold": best_th,
        "delta_vs_baseline": delta
    }
    with open("experiments/ledger.jsonl", "a") as f:
        f.write(json.dumps(ledger_entry) + "\n")
    print("Logged experiment to experiments/ledger.jsonl")

if __name__ == "__main__":
    main()
