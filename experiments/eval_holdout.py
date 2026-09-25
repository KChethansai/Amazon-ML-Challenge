#!/usr/bin/env python3
"""
Phase 19: Single-shot evaluation on the untouched 15,000 S1 Holdout Split.
Evaluates the final candidate model against the baseline on strictly unseen data.
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
HOLDOUT_PATH = "experiments/splits/holdout_s1_ids.json"
MODEL_PATH = "experiments/exp_003_clean_multipass/lgbm_clean_multipass.txt"
CONFIG_PATH = "experiments/exp_003_clean_multipass/config.json"

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
        self.records = {}
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
        for t in list(self.token_idx.keys()):
            if len(self.token_idx[t]) > 4000:
                del self.token_idx[t]
        for pfx in list(self.prefix_idx.keys()):
            if len(self.prefix_idx[pfx]) > 4000:
                del self.prefix_idx[pfx]
                
    def query(self, s1_norm_name: str, s1_core_tokens: list, s1_raw_addr: str, s1_norm_addr: str, max_candidates: int = 35):
        keys = extract_advanced_keys(s1_norm_name, s1_core_tokens, s1_raw_addr, s1_norm_addr)
        candidate_scores = Counter()
        if s1_norm_name in self.exact_name_idx:
            for tid in self.exact_name_idx[s1_norm_name]:
                candidate_scores[tid] += 10
        for ck in keys["compound_addr"]:
            if ck in self.compound_addr_idx:
                posting = self.compound_addr_idx[ck]
                if len(posting) <= 150:
                    for tid in posting:
                        candidate_scores[tid] += 6
        for t in keys["name_tokens"]:
            if t in self.token_idx:
                for tid in self.token_idx[t]:
                    candidate_scores[tid] += 3
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
    print(" Phase 19: Evaluation on Untouched 14,998 S1 Holdout Set")
    print("=" * 70)
    
    with open(HOLDOUT_PATH, "r") as f:
        holdout_ids = json.load(f)
    holdout_set = set(holdout_ids)
    print(f"Loaded {len(holdout_set):,} S1 Holdout Entities.")
    
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
    threshold = config.get("best_threshold", 0.74)
    print(f"Loaded configuration: threshold = {threshold:.2f}, max_candidates = {config.get('max_candidates', 35)}")
    
    model = lgb.Booster(model_file=MODEL_PATH)
    print(f"Loaded trained LightGBM model from {MODEL_PATH}")
    
    # Ground truth for holdout
    gt_map = {}
    needed_targets = set()
    total_positives = 0
    with open(os.path.join(TRAIN_DIR, "train_ground_truth.tsv"), "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            sid = parts[0]
            if sid in holdout_set:
                mstr = parts[1] if len(parts) > 1 else ""
                m_set = set(x.strip() for x in mstr.split(",") if x.strip())
                gt_map[sid] = m_set
                needed_targets.update(m_set)
                total_positives += len(m_set)
                
    print(f"Ground truth loaded for {len(gt_map):,} holdout entities ({total_positives:,} positive links).")
    
    # Load holdout S1 records
    s1_records = {}
    with open(os.path.join(TRAIN_DIR, "train_source1.tsv"), "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            sid = parts[0]
            if sid in holdout_set:
                eid, bname, baddr, bcountry = parts
                norm_nm = normalize_name(bname)
                core_toks, _ = extract_name_tokens(norm_nm)
                norm_ad = normalize_address(baddr)
                digits = extract_address_digits(baddr)
                s1_records[sid] = {
                    "raw_name": bname, "norm_name": norm_nm, "core_tokens": set(core_toks),
                    "raw_addr": baddr, "norm_addr": norm_ad,
                    "addr_tokens": set(norm_ad.split()) if norm_ad else set(),
                    "digits": digits, "country": bcountry
                }
                
    # Build 1M background pool
    print("Building target index pool (350k per source + needed targets)...", flush=True)
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
    
    # Run Inference on Holdout Set
    print("\nRunning inference on 14,998 holdout S1 entities...", flush=True)
    t0 = time.time()
    predictions = {}
    candidate_hits = 0
    total_cands_retrieved = 0
    
    for sid, s1 in s1_records.items():
        c_idx = country_indexes.get(s1["country"])
        if not c_idx:
            predictions[sid] = set()
            continue
        cands = c_idx.query(s1["norm_name"], list(s1["core_tokens"]), s1["raw_addr"], s1["norm_addr"], max_candidates=35)
        total_cands_retrieved += len(cands)
        
        true_targets = gt_map.get(sid, set())
        candidate_hits += len(true_targets & set(cands))
        
        feat_batch = []
        valid_cands = []
        for cid in cands:
            trec = c_idx.records.get(cid)
            if trec:
                feat_batch.append(compute_pairwise_features(s1, trec, trec[2]))
                valid_cands.append(cid)
                
        if feat_batch:
            probs = model.predict(np.array(feat_batch, dtype=np.float32))
            matched = set(cid for cid, p in zip(valid_cands, probs) if p >= threshold)
            predictions[sid] = matched
        else:
            predictions[sid] = set()
            
    elapsed = time.time() - t0
    print(f"Holdout inference complete in {elapsed:.2f}s ({len(holdout_set)/elapsed:.0f} entities/s).")
    
    # Calculate exact Macro F0.5
    holdout_f05 = calculate_macro_f05(gt_map, predictions)
    blocking_recall = candidate_hits / total_positives
    avg_cands = total_cands_retrieved / len(holdout_set)
    pred_singletons = sum(1 for sid, m in predictions.items() if len(m) == 0)
    singleton_rate = pred_singletons / len(predictions)
    
    # Calculate precision and recall on non-singletons
    total_tp = sum(len(gt_map[sid] & predictions[sid]) for sid in predictions)
    total_pred = sum(len(predictions[sid]) for sid in predictions)
    precision = total_tp / max(1, total_pred)
    recall = total_tp / max(1, total_positives)
    
    print("\n" + "=" * 70)
    print(" HOLDOUT EVALUATION SCORECARD:")
    print("=" * 70)
    print(f" Baseline Macro F0.5        : 0.89837")
    print(f" HOLDOUT MACRO F0.5         : {holdout_f05:.5f}")
    delta = holdout_f05 - 0.89837
    print(f" DELTA OVER BASELINE        : {delta:+.5f} ({delta*100:+.2f}%)")
    print("-" * 70)
    print(f" Blocking Recall on Holdout : {blocking_recall*100:.2f}% ({candidate_hits:,}/{total_positives:,})")
    print(f" Avg Candidates / S1        : {avg_cands:.1f}")
    print(f" Total Links Predicted      : {total_pred:,}")
    print(f" Pairwise Precision         : {precision*100:.2f}%")
    print(f" Pairwise Recall            : {recall*100:.2f}%")
    print(f" Predicted Singletons       : {pred_singletons:,} ({singleton_rate*100:.2f}%)")
    print(f" True Singletons            : {sum(1 for s in gt_map.values() if not s):,} ({sum(1 for s in gt_map.values() if not s)/len(gt_map)*100:.2f}%)")
    print("=" * 70)
    
    # Log to ledger
    ledger_entry = {
        "exp_id": "EXP_004_HOLDOUT_EVAL",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "description": "Holdout verification on 14,998 S1 entities (Multi-Pass Blocking + Clean Features, tau=0.74)",
        "holdout_macro_f05": holdout_f05,
        "delta_vs_baseline": delta,
        "blocking_recall": blocking_recall,
        "precision": precision,
        "recall": recall,
        "singleton_rate": singleton_rate
    }
    with open("experiments/ledger.jsonl", "a") as f:
        f.write(json.dumps(ledger_entry) + "\n")
    print("Logged holdout result to experiments/ledger.jsonl")

if __name__ == "__main__":
    main()
