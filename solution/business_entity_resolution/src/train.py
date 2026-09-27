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
    normalize_address, extract_address_digits, extract_address_blocking_keys,
    prepare_source_record,
)
from blocking import CountryBlockingIndex, ADAPTIVE_BUDGETS
from features import compute_pairwise_features, compute_candidate_features_for_s1, FEATURE_NAMES
from features2 import extra_batch_for_s1, EXTRA_NAMES
ALL_FEATURE_NAMES = FEATURE_NAMES + EXTRA_NAMES

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

def evaluate_metrics(ground_truth_map: dict, predictions_map: dict):
    """Compute detailed evaluation metrics."""
    f05 = calculate_macro_f05(ground_truth_map, predictions_map)
    
    total_tp = 0
    total_pred = 0
    total_true = 0
    correct_singletons = 0
    total_true_singletons = 0
    pred_singletons = 0
    
    for s1_id, true_set in ground_truth_map.items():
        pred_set = predictions_map.get(s1_id, set())
        if not true_set:
            total_true_singletons += 1
            if not pred_set:
                correct_singletons += 1
        if not pred_set:
            pred_singletons += 1
            
        tp = len(true_set & pred_set)
        total_tp += tp
        total_pred += len(pred_set)
        total_true += len(true_set)
        
    precision = total_tp / max(1, total_pred)
    recall = total_tp / max(1, total_true)
    singleton_acc = correct_singletons / max(1, total_true_singletons)
    
    return {
        "macro_f05": f05,
        "precision": precision,
        "recall": recall,
        "singleton_accuracy": singleton_acc,
        "pred_singletons": pred_singletons,
        "total_entities": len(ground_truth_map)
    }

import argparse

def _resolve_default_train_paths():
    if os.path.isdir("solution/business_entity_resolution/models"):
        default_models = "solution/business_entity_resolution/models"
    elif os.path.isdir("code/business_entity_resolution/models"):
        default_models = "code/business_entity_resolution/models"
    elif os.path.isdir("models"):
        default_models = "models"
    else:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        default_models = os.path.join(base_dir, "models")
        
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

def query_for_s1(c_idx, s1_rec, max_candidates=60, adaptive=False):
    items, prov = c_idx.query_candidates(
        s1_rec["norm_name"], s1_rec["core_tokens"], s1_rec["norm_addr"], s1_rec["addr_keys"],
        s1_raw_addr=s1_rec["addr"], s1_raw_name=s1_rec["name"],
        max_candidates=max_candidates, return_weights=True, return_provenance=True,
        adaptive=adaptive)
    return items, prov


def featurize_s1(s1_rec, cand_items, prov, c_idx, use_extra=True):
    vc, base_rows = compute_candidate_features_for_s1(s1_rec, cand_items, c_idx.records)
    if not use_extra or not vc:
        return vc, base_rows
    wmap = {c: float(w) for c, w in cand_items} if cand_items and isinstance(cand_items[0], tuple) else {}
    sub_items = [(c, wmap.get(c, 0.0)) for c in vc]
    sub_prov = {c: prov.get(c, ()) for c in vc}
    _, extra_rows = extra_batch_for_s1(s1_rec, sub_items, c_idx.records, prov_map=sub_prov)
    return vc, [b + e for b, e in zip(base_rows, extra_rows)]


def train_pipeline(
    data_dir: str = _DEF_DATA,
    models_dir: str = _DEF_MODELS,
    num_train_s1: int = 35000,
    num_val_s1: int = 10000,
    max_candidates: int = 60,
    adaptive: bool = True,
    use_extra: bool = True,
    neg_cap: int = 10000,
):
    print("=" * 70)
    print(" Amazon ML Challenge 2026: Upgraded Training & Margin Calibration")
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
    
    # 2. Select S1 entities using disjoint splits if available
    print("Step 2: Sampling S1 entities for Train and Validation...", flush=True)
    s1_train_records = {}
    s1_val_records = {}
    
    train_split_path = "experiments/splits/train_pool_s1_ids.json"
    dev_split_path = "experiments/splits/dev_s1_ids.json"
    
    train_target_ids = None
    dev_target_ids = None
    if os.path.isfile(train_split_path) and os.path.isfile(dev_split_path):
        with open(train_split_path, "r") as f:
            train_target_ids = set(json.load(f)[:num_train_s1])
        with open(dev_split_path, "r") as f:
            dev_target_ids = set(json.load(f)[:num_val_s1])
        print(f"Using saved split IDs: {len(train_target_ids)} train, {len(dev_target_ids)} dev.")

    with open(os.path.join(train_dir, "train_source1.tsv"), "r", encoding="utf-8") as f:
        next(f)
        for i, line in enumerate(f):
            parts = line.rstrip("\r\n").split("\t")
            eid, bname, baddr, bcountry = parts
            
            in_train = (eid in train_target_ids) if train_target_ids else (i < num_train_s1)
            in_val = (eid in dev_target_ids) if dev_target_ids else (num_train_s1 <= i < (num_train_s1 + num_val_s1))
            
            if not in_train and not in_val:
                continue
                
            rec = prepare_source_record(bname, baddr, bcountry)
            
            if in_train:
                s1_train_records[eid] = rec
            elif in_val:
                s1_val_records[eid] = rec
                
            if len(s1_train_records) >= num_train_s1 and len(s1_val_records) >= num_val_s1:
                break
                
    print(f"Loaded {len(s1_train_records)} Train S1, {len(s1_val_records)} Validation S1.")
    
    # Needed target IDs for training, validation, and holdout evaluation
    all_needed_targets = set()
    for eid in list(s1_train_records.keys()) + list(s1_val_records.keys()):
        all_needed_targets.update(all_s1_gt.get(eid, set()))
    holdout_path = "experiments/splits/holdout_s1_ids.json"
    if os.path.isfile(holdout_path):
        with open(holdout_path, "r") as f:
            for eid in json.load(f):
                all_needed_targets.update(all_s1_gt.get(eid, set()))
        
    print(f"Identified {len(all_needed_targets)} true match target records needed (including holdout).")
    
    # 3. Load S2 and S3 background pools + required matches
    print("Step 3: Loading S2 and S3 background pool...", flush=True)
    t0 = time.time()
    
    country_indexes = {
        "US": CountryBlockingIndex("US"),
        "India": CountryBlockingIndex("India")
    }
    
    MAX_BACKGROUND_PER_SOURCE = 200000
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
                        eid, norm_nm, core_toks, norm_ad, addr_keys, digits, is_s2=1, raw_addr=baddr, raw_name=bname
                    )
                    
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
                        eid, norm_nm, core_toks, norm_ad, addr_keys, digits, is_s2=0, raw_addr=baddr, raw_name=bname
                    )
                    
    print(f"Loaded target pools in {time.time()-t0:.2f}s: US={len(country_indexes['US'].records)}, India={len(country_indexes['India'].records)}")
    for c_idx in country_indexes.values():
        c_idx.prune_frequent_keys()
        
    # 4. Generate Candidate Pairs and Extract Features for Training
    print("Step 4: Generating candidates and extracting training feature vectors...", flush=True)
    t0 = time.time()
    
    X_chunks = []
    y_chunks = []
    X_train = []
    y_train = []

    def _flush_train_chunk():
        if X_train:
            X_chunks.append(np.array(X_train, dtype=np.float32))
            y_chunks.append(np.array(y_train, dtype=np.float32))
            X_train.clear()
            y_train.clear()

    for i, (s1_id, s1_rec) in enumerate(s1_train_records.items()):
        c = s1_rec["country"]
        c_idx = country_indexes.get(c)
        if not c_idx:
            continue
            
        true_targets = all_s1_gt.get(s1_id, set())
        
        # Candidate blocking query (union PixelDust retrieval + rerank)
        cand_items, prov = query_for_s1(c_idx, s1_rec, max_candidates=max_candidates, adaptive=adaptive)

        # Ensure true targets present in index are included in training
        existing_cids = {cid for cid, w in cand_items}
        max_w = cand_items[0][1] if cand_items else 15.0
        for t_id in true_targets:
            if t_id in c_idx.records and t_id not in existing_cids:
                cand_items.append((t_id, max_w))
                prov[t_id] = []

        # Vectorized candidate feature extraction for S1
        valid_cands, feat_rows = featurize_s1(s1_rec, cand_items, prov, c_idx, use_extra=use_extra)
        # Hard-negative mining: all positives + hardest negatives by blocking weight
        scored = [((cid in true_targets), next((w for c, w in cand_items if c == cid), 0.0), cid, fv)
                  for cid, fv in zip(valid_cands, feat_rows)]
        pos = [s for s in scored if s[0]]
        neg = sorted([s for s in scored if not s[0]], key=lambda s: -s[1])[:neg_cap]
        for is_match, _, cid, fvec in pos + neg:
            X_train.append(fvec)
            y_train.append(1.0 if is_match else 0.0)

        if (i + 1) % 2000 == 0:
            _flush_train_chunk()
            nrows = sum(len(c) for c in X_chunks)
            print(f"  Extracted features for {i + 1}/{len(s1_train_records)} train entities ({nrows:,} pairs)...", flush=True)

    _flush_train_chunk()
    X_train = np.vstack(X_chunks) if X_chunks else np.zeros((0, len(ALL_FEATURE_NAMES if use_extra else FEATURE_NAMES)), dtype=np.float32)
    y_train = np.concatenate(y_chunks) if y_chunks else np.zeros((0,), dtype=np.float32)
    del X_chunks, y_chunks
    print(f"Training dataset ready in {time.time()-t0:.2f}s: {len(X_train)} pairs (Positives={int(np.sum(y_train))}, Negatives={len(y_train)-int(np.sum(y_train))})")
    
    # 5. Train LightGBM Classifier with Prescribed Hyperparameters
    print("Step 5: Training LightGBM Pairwise Classifier (num_leaves=63, max_depth=7, lr=0.06, n_estimators=300)...", flush=True)
    t0 = time.time()
    
    feat_names = ALL_FEATURE_NAMES if use_extra else FEATURE_NAMES
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feat_names)
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 127,
        "max_depth": 9,
        "subsample": 0.85,
        "subsample_freq": 1,
        "feature_fraction": 0.9,
        "min_child_samples": 10,
        "n_jobs": 4,
        "seed": 7,
        "deterministic": True,
        "verbose": -1
    }

    model = lgb.train(params, train_data, num_boost_round=750)
    print(f"LightGBM trained in {time.time()-t0:.2f}s.")
    
    # Feature importance
    importance = model.feature_importance(importance_type="gain")
    print("Top Feature Importances:")
    for name, imp in sorted(zip(feat_names, importance), key=lambda x: -x[1])[:10]:
        print(f"  {name:28s}: {imp:.1f}")
        
    # 6. Validation & Adaptive Two-Stage Margin Threshold Grid Calibration
    print("Step 6: Running validation inference and tuning adaptive margin thresholds...", flush=True)
    t0 = time.time()
    
    val_gt_map = {eid: all_s1_gt.get(eid, set()) for eid in s1_val_records}
    val_candidate_data = {} # s1_id -> (valid_cands, probs)
    
    val_hits = 0
    val_positives = sum(len(s) for s in val_gt_map.values())
    
    for s1_id, s1_rec in s1_val_records.items():
        c = s1_rec["country"]
        c_idx = country_indexes.get(c)
        if not c_idx:
            val_candidate_data[s1_id] = ([], np.array([]))
            continue
            
        cand_items, prov = query_for_s1(c_idx, s1_rec, max_candidates=max_candidates, adaptive=adaptive)

        cands_set = {cid for cid, w in cand_items}
        val_hits += len(val_gt_map.get(s1_id, set()) & cands_set)

        valid_cands, feat_rows = featurize_s1(s1_rec, cand_items, prov, c_idx, use_extra=use_extra)
        if feat_rows:
            probs = model.predict(np.array(feat_rows, dtype=np.float32))
            val_candidate_data[s1_id] = (valid_cands, probs)
        else:
            val_candidate_data[s1_id] = ([], np.array([]))
            
    val_blocking_recall = val_hits / max(1, val_positives)
    print(f"Validation inference complete in {time.time()-t0:.2f}s.")
    print(f"Validation Candidate Recall: {val_blocking_recall*100:.2f}% ({val_hits}/{val_positives})")
    
    # Automated Grid Search over (tau_singleton, tau_min, delta_margin)
    best_f05 = 0.0
    best_tau_singleton = 0.72
    best_tau_min = 0.55
    best_delta_margin = 0.15
    
    print("\nStarting Adaptive Two-Stage Grid Calibration:")
    for tau_s in [0.60, 0.66, 0.70, 0.72, 0.76, 0.80]:
        for tau_m in [0.40, 0.45, 0.50, 0.55]:
            for delta_m in [0.10, 0.15, 0.18]:
                preds = {}
                for s1_id, (cands, probs) in val_candidate_data.items():
                    if len(probs) == 0:
                        preds[s1_id] = set()
                        continue
                    max_p = float(np.max(probs))
                    # Stage 1: Singleton Gate
                    if max_p < tau_s:
                        preds[s1_id] = set()
                    else:
                        # Stage 2: Relative Margin Filter
                        cutoff = max(tau_m, max_p - delta_m)
                        preds[s1_id] = set(cid for cid, p in zip(cands, probs) if p >= cutoff)
                        
                score = calculate_macro_f05(val_gt_map, preds)
                if score > best_f05:
                    best_f05 = score
                    best_tau_singleton = tau_s
                    best_tau_min = tau_m
                    best_delta_margin = delta_m
                    print(f"  New Best: tau_singleton={tau_s:.2f}, tau_min={tau_m:.2f}, delta_margin={delta_m:.2f} -> Dev Macro F0.5 = {score:.5f}")

    print(f"\n>>> Optimal Calibration: tau_singleton={best_tau_singleton:.2f}, tau_min={best_tau_min:.2f}, delta_margin={best_delta_margin:.2f} (Dev Macro F0.5 = {best_f05:.5f}) <<<")
    
    # 7. Evaluate on Holdout Split if Available
    holdout_path = "experiments/splits/holdout_s1_ids.json"
    holdout_results = None
    if os.path.isfile(holdout_path):
        print("\nStep 7: Evaluating on 14,998 S1 Holdout Split...", flush=True)
        t_holdout = time.time()
        with open(holdout_path, "r") as f:
            holdout_ids = json.load(f)
        holdout_set = set(holdout_ids)
        
        holdout_gt = {eid: all_s1_gt.get(eid, set()) for eid in holdout_set}
        holdout_s1_records = {}
        with open(os.path.join(train_dir, "train_source1.tsv"), "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                eid, bname, baddr, bcountry = parts
                if eid in holdout_set:
                    holdout_s1_records[eid] = prepare_source_record(bname, baddr, bcountry)
                    
        holdout_preds = {}
        holdout_hits = 0
        total_holdout_pos = sum(len(s) for s in holdout_gt.values())
        
        for s1_id, s1_rec in holdout_s1_records.items():
            c = s1_rec["country"]
            c_idx = country_indexes.get(c)
            if not c_idx:
                holdout_preds[s1_id] = set()
                continue
            cand_items, prov = query_for_s1(c_idx, s1_rec, max_candidates=max_candidates, adaptive=adaptive)
            cands_set = {cid for cid, w in cand_items}
            holdout_hits += len(holdout_gt.get(s1_id, set()) & cands_set)

            valid_cands, feat_rows = featurize_s1(s1_rec, cand_items, prov, c_idx, use_extra=use_extra)
            if feat_rows:
                probs = model.predict(np.array(feat_rows, dtype=np.float32))
                max_p = float(np.max(probs))
                if max_p < best_tau_singleton:
                    holdout_preds[s1_id] = set()
                else:
                    cutoff = max(best_tau_min, max_p - best_delta_margin)
                    holdout_preds[s1_id] = set(cid for cid, p in zip(valid_cands, probs) if p >= cutoff)
            else:
                holdout_preds[s1_id] = set()
                
        holdout_metrics = evaluate_metrics(holdout_gt, holdout_preds)
        holdout_metrics["candidate_recall"] = holdout_hits / max(1, total_holdout_pos)
        holdout_results = holdout_metrics
        print(f"Holdout evaluation finished in {time.time()-t_holdout:.2f}s:")
        print(f"  Holdout Candidate Recall : {holdout_metrics['candidate_recall']*100:.2f}%")
        print(f"  Holdout Macro F0.5       : {holdout_metrics['macro_f05']:.5f}")
        print(f"  Holdout Precision        : {holdout_metrics['precision']*100:.2f}%")
        print(f"  Holdout Recall           : {holdout_metrics['recall']*100:.2f}%")
        print(f"  Holdout Singleton Acc    : {holdout_metrics['singleton_accuracy']*100:.2f}%")
        
    # 8. Save Model & Config
    model_save_path = os.path.join(models_dir, "lgbm_matcher.txt")
    model.save_model(model_save_path)
    
    config = {
        "tau_singleton": best_tau_singleton,
        "tau_min": best_tau_min,
        "delta_margin": best_delta_margin,
        "max_candidates": max_candidates,
        "adaptive": adaptive,
        "use_extra": use_extra,
        "use_graph": False,
        "neg_cap": neg_cap,
        "best_val_f05": best_f05,
        "feature_names": feat_names,
        "num_train_s1": num_train_s1,
        "num_val_s1": num_val_s1,
        "trained_date": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    if holdout_results:
        config["holdout_metrics"] = holdout_results
        
    config_save_path = os.path.join(models_dir, "config.json")
    with open(config_save_path, "w") as f:
        json.dump(config, f, indent=2)
        
    print(f"\nSaved trained LightGBM model to {model_save_path}")
    print(f"Saved training configuration to {config_save_path}")
    print("=" * 70)
    return config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Amazon ML Challenge 2026: Model Training & Calibration")
    parser.add_argument("--data-dir", default=_DEF_DATA, help=f"Path to dataset directory (default: {_DEF_DATA})")
    parser.add_argument("--models-dir", default=_DEF_MODELS, help=f"Path to models output directory (default: {_DEF_MODELS})")
    parser.add_argument("--num-train-s1", type=int, default=35000, help="Number of S1 training entities (default: 35000)")
    parser.add_argument("--num-val-s1", type=int, default=10000, help="Number of S1 validation entities (default: 10000)")
    parser.add_argument("--max-candidates", type=int, default=60)
    parser.add_argument("--adaptive", action="store_true", default=True)
    parser.add_argument("--no-adaptive", dest="adaptive", action="store_false")
    parser.add_argument("--use-extra", action="store_true", default=True)
    parser.add_argument("--no-extra", dest="use_extra", action="store_false")
    parser.add_argument("--neg-cap", type=int, default=10000)
    args = parser.parse_args()

    train_pipeline(
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        num_train_s1=args.num_train_s1,
        num_val_s1=args.num_val_s1,
        max_candidates=args.max_candidates,
        adaptive=args.adaptive,
        use_extra=args.use_extra,
        neg_cap=args.neg_cap,
    )
