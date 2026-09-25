import os
import sys
import time
import json
import pickle
import random
from collections import defaultdict
import numpy as np
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(__file__))
from normalization import (
    normalize_name, extract_name_tokens, 
    normalize_address, extract_address_digits, extract_address_blocking_keys
)
from blocking import CountryBlockingIndex
from features import compute_pairwise_features, FEATURE_NAMES

def calculate_macro_f05(ground_truth_map: dict, predictions_map: dict) -> float:
    """
    Compute official challenge metric: macro-averaged F_0.5 across all S1 entities.
    Formula: F_0.5 = (1.25 * P * R) / (0.25 * P + R)
    Singletons: If true is empty and pred is empty, score is 1.0. If pred non-empty, score is 0.0.
    """
    f05_scores = []
    for s1_id, true_set in ground_truth_map.items():
        pred_set = predictions_map.get(s1_id, set())
        
        # Singleton evaluation
        if not true_set:
            if not pred_set:
                f05_scores.append(1.0)
            else:
                f05_scores.append(0.0)
            continue
            
        # Non-singleton evaluation
        if not pred_set:
            f05_scores.append(0.0)
            continue
            
        tp = len(true_set & pred_set)
        if tp == 0:
            f05_scores.append(0.0)
            continue
            
        precision = tp / len(pred_set)
        recall = tp / len(true_set)
        
        denom = 0.25 * precision + recall
        if denom > 0:
            score = (1.25 * precision * recall) / denom
        else:
            score = 0.0
        f05_scores.append(score)
        
    return float(np.mean(f05_scores)) if f05_scores else 0.0

import argparse

def _resolve_default_train_paths():
    # Check common relative locations first
    if os.path.isdir("solution/business_entity_resolution/models"):
        default_models = "solution/business_entity_resolution/models"
    elif os.path.isdir("code/business_entity_resolution/models"):
        default_models = "code/business_entity_resolution/models"
    elif os.path.isdir("models"):
        default_models = "models"
    else:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        default_models = os.path.join(base_dir, "models")
        
    # Data dir
    data_candidates = [
        "student_resource/dataset",
        "dataset",
    ]
    default_data = "student_resource/dataset"
    for dc in data_candidates:
        if os.path.isdir(dc):
            default_data = dc
            break
            
    return default_data, default_models

_DEF_DATA, _DEF_MODELS = _resolve_default_train_paths()

def train_pipeline(
    data_dir: str = _DEF_DATA,
    models_dir: str = _DEF_MODELS,
    num_train_s1: int = 35000,
    num_val_s1: int = 10000
):
    print("=" * 70)
    print(" Amazon ML Challenge 2026: Model Training & F0.5 Calibration")
    print("=" * 70)
    
    os.makedirs(models_dir, exist_ok=True)
    train_dir = os.path.join(data_dir, "train")
    
    # 1. Load Ground Truth for training entities
    print("Step 1: Loading ground truth...", flush=True)
    t0 = time.time()
    all_s1_gt = {}
    with open(os.path.join(train_dir, "train_ground_truth.tsv"), "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            s1_id = parts[0]
            mstr = parts[1] if len(parts) > 1 else ""
            m_set = set(x.strip() for x in mstr.split(",") if x.strip())
            all_s1_gt[s1_id] = m_set
    print(f"Loaded ground truth for {len(all_s1_gt)} entities in {time.time()-t0:.2f}s")
    
    # 2. Select S1 entities for training and validation (preserving country & singleton balance)
    print("Step 2: Sampling S1 entities for Train and Validation...", flush=True)
    s1_train_records = {} # eid -> data
    s1_val_records = {}
    
    target_total = num_train_s1 + num_val_s1
    with open(os.path.join(train_dir, "train_source1.tsv"), "r", encoding="utf-8") as f:
        next(f)
        for i, line in enumerate(f):
            if i >= target_total:
                break
            parts = line.rstrip("\r\n").split("\t")
            eid, bname, baddr, bcountry = parts
            
            norm_nm = normalize_name(bname)
            core_toks, _ = extract_name_tokens(norm_nm)
            norm_ad = normalize_address(baddr)
            addr_keys = extract_address_blocking_keys(baddr, norm_ad)
            digits = extract_address_digits(baddr)
            
            rec = {
                "name": bname,
                "norm_name": norm_nm,
                "core_tokens": set(core_toks),
                "addr": baddr,
                "norm_addr": norm_ad,
                "addr_tokens": set(norm_ad.split()) if norm_ad else set(),
                "addr_keys": addr_keys,
                "digits": digits,
                "country": bcountry
            }
            
            if i < num_train_s1:
                s1_train_records[eid] = rec
            else:
                s1_val_records[eid] = rec
                
    print(f"Sampled {len(s1_train_records)} Train S1, {len(s1_val_records)} Validation S1.")
    
    # Needed target IDs for training and validation
    all_needed_targets = set()
    for eid in list(s1_train_records.keys()) + list(s1_val_records.keys()):
        all_needed_targets.update(all_s1_gt.get(eid, set()))
        
    print(f"Identified {len(all_needed_targets)} true match target records needed.")
    
    # 3. Load S2 and S3 background pools + required matches
    print("Step 3: Loading S2 and S3 background pool...", flush=True)
    t0 = time.time()
    
    # Partition targets by country
    country_indexes = {
        "US": CountryBlockingIndex("US"),
        "India": CountryBlockingIndex("India")
    }
    
    # Read S2
    MAX_BACKGROUND_PER_SOURCE = 150000
    with open(os.path.join(train_dir, "train_source2.tsv"), "r", encoding="utf-8") as f:
        next(f)
        for i, line in enumerate(f):
            parts = line.rstrip("\r\n").split("\t")
            eid, bname, baddr, bcountry = parts
            if eid in all_needed_targets or i < MAX_BACKGROUND_PER_SOURCE:
                c_idx = country_indexes.get(bcountry)
                if c_idx:
                    norm_nm = normalize_name(bname)
                    core_toks, _ = extract_name_tokens(norm_nm)
                    norm_ad = normalize_address(baddr)
                    addr_keys = extract_address_blocking_keys(baddr, norm_ad)
                    digits = extract_address_digits(baddr)
                    c_idx.add_target_record(
                        eid, norm_nm, core_toks, norm_ad, addr_keys, digits, is_s2=1
                    )
                    
    # Read S3
    with open(os.path.join(train_dir, "train_source3.tsv"), "r", encoding="utf-8") as f:
        next(f)
        for i, line in enumerate(f):
            parts = line.rstrip("\r\n").split("\t")
            eid, bname, baddr, bcountry = parts
            if eid in all_needed_targets or i < MAX_BACKGROUND_PER_SOURCE:
                c_idx = country_indexes.get(bcountry)
                if c_idx:
                    norm_nm = normalize_name(bname)
                    core_toks, _ = extract_name_tokens(norm_nm)
                    norm_ad = normalize_address(baddr)
                    addr_keys = extract_address_blocking_keys(baddr, norm_ad)
                    digits = extract_address_digits(baddr)
                    c_idx.add_target_record(
                        eid, norm_nm, core_toks, norm_ad, addr_keys, digits, is_s2=0
                    )
                    
    print(f"Loaded target pools in {time.time()-t0:.2f}s: US={len(country_indexes['US'].records)}, India={len(country_indexes['India'].records)}")
    for c_idx in country_indexes.values():
        c_idx.prune_frequent_keys()
        
    # 4. Generate Candidate Pairs and Extract Features for Training
    print("Step 4: Generating candidates and extracting training feature vectors...", flush=True)
    t0 = time.time()
    
    X_train = []
    y_train = []
    
    for s1_id, s1_rec in s1_train_records.items():
        c = s1_rec["country"]
        c_idx = country_indexes.get(c)
        if not c_idx:
            continue
            
        true_targets = all_s1_gt.get(s1_id, set())
        
        # Candidate blocking query (top 50)
        cands = set(c_idx.query_candidates(
            s1_rec["norm_name"], s1_rec["core_tokens"], 
            s1_rec["norm_addr"], s1_rec["addr_keys"], 
            max_candidates=50
        ))
        
        # Ensure true targets present in index are included in training
        for t_id in true_targets:
            if t_id in c_idx.records:
                cands.add(t_id)
                
        # Extract features for candidate pairs
        for cand_id in cands:
            tgt_rec = c_idx.records.get(cand_id)
            if not tgt_rec:
                continue
            is_match = 1.0 if cand_id in true_targets else 0.0
            feats = compute_pairwise_features(s1_rec, tgt_rec, tgt_rec[2])
            X_train.append(feats)
            y_train.append(is_match)
            
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    print(f"Training dataset ready in {time.time()-t0:.2f}s: {len(X_train)} pairs (Positives={int(np.sum(y_train))}, Negatives={len(y_train)-int(np.sum(y_train))})")
    
    # 5. Train LightGBM Classifier
    print("Step 5: Training LightGBM Pairwise Classifier...", flush=True)
    t0 = time.time()
    
    # Weight positives slightly to balance representation
    pos_weight = (len(y_train) - np.sum(y_train)) / max(1, np.sum(y_train))
    
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
        "verbose": -1
    }
    
    model = lgb.train(params, train_data, num_boost_round=150)
    print(f"LightGBM trained in {time.time()-t0:.2f}s.")
    
    # Feature importance
    importance = model.feature_importance(importance_type="gain")
    print("Top Feature Importances:")
    for name, imp in sorted(zip(FEATURE_NAMES, importance), key=lambda x: -x[1])[:8]:
        print(f"  {name:22s}: {imp:.1f}")
        
    # 6. Validation & F_0.5 Threshold Grid Calibration
    print("Step 6: Running inference on Validation set and tuning F0.5 threshold...", flush=True)
    t0 = time.time()
    
    val_gt_map = {eid: all_s1_gt.get(eid, set()) for eid in s1_val_records}
    val_candidate_pairs = {} # s1_id -> list of (cand_id, prob)
    
    for s1_id, s1_rec in s1_val_records.items():
        c = s1_rec["country"]
        c_idx = country_indexes.get(c)
        if not c_idx:
            val_candidate_pairs[s1_id] = []
            continue
            
        cands = c_idx.query_candidates(
            s1_rec["norm_name"], s1_rec["core_tokens"], 
            s1_rec["norm_addr"], s1_rec["addr_keys"], 
            max_candidates=50
        )
        
        if not cands:
            val_candidate_pairs[s1_id] = []
            continue
            
        feat_batch = []
        for cand_id in cands:
            tgt_rec = c_idx.records.get(cand_id)
            if tgt_rec:
                feat_batch.append(compute_pairwise_features(s1_rec, tgt_rec, tgt_rec[2]))
                
        if feat_batch:
            probs = model.predict(np.array(feat_batch, dtype=np.float32))
            val_candidate_pairs[s1_id] = list(zip(cands, probs))
        else:
            val_candidate_pairs[s1_id] = []
            
    print(f"Inference on {len(s1_val_records)} validation entities complete in {time.time()-t0:.2f}s.")
    
    # Threshold sweep
    best_f05 = 0.0
    best_threshold = 0.75
    
    for th in np.arange(0.50, 0.95, 0.05):
        th = round(float(th), 2)
        preds = {}
        for s1_id, cand_probs in val_candidate_pairs.items():
            matched = set(cid for cid, p in cand_probs if p >= th)
            preds[s1_id] = matched
            
        score = calculate_macro_f05(val_gt_map, preds)
        print(f"  Threshold {th:.2f} -> Validation Macro F0.5 = {score:.4f}")
        if score > best_f05:
            best_f05 = score
            best_threshold = th
            
    print(f"\n>>> Optimal F0.5 Threshold: {best_threshold:.2f} (Macro F0.5 = {best_f05:.4f}) <<<")
    
    # 7. Save Model & Config
    model_save_path = os.path.join(models_dir, "lgbm_matcher.txt")
    model.save_model(model_save_path)
    
    config = {
        "best_threshold": best_threshold,
        "best_val_f05": best_f05,
        "feature_names": FEATURE_NAMES,
        "num_train_s1": num_train_s1,
        "num_val_s1": num_val_s1,
        "trained_date": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    config_save_path = os.path.join(models_dir, "config.json")
    with open(config_save_path, "w") as f:
        json.dump(config, f, indent=2)
        
    print(f"Saved trained LightGBM model to {model_save_path}")
    print(f"Saved training configuration to {config_save_path}")
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Amazon ML Challenge 2026: Model Training & Calibration")
    parser.add_argument("--data-dir", default=_DEF_DATA, help=f"Path to dataset directory (default: {_DEF_DATA})")
    parser.add_argument("--models-dir", default=_DEF_MODELS, help=f"Path to models output directory (default: {_DEF_MODELS})")
    parser.add_argument("--num-train-s1", type=int, default=35000, help="Number of S1 training entities (default: 35000)")
    parser.add_argument("--num-val-s1", type=int, default=10000, help="Number of S1 validation entities (default: 10000)")
    args = parser.parse_args()

    train_pipeline(
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        num_train_s1=args.num_train_s1,
        num_val_s1=args.num_val_s1
    )
