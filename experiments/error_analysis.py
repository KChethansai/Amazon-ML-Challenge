#!/usr/bin/env python3
"""Differential error analysis between seeded holdout and natural India."""

import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np

sys.path.insert(0, "solution/business_entity_resolution/src")
from normalization import prepare_source_record
from disk_blocking import DiskCountryBlockingIndex
from blocking import CountryBlockingIndex
from train import (
    ALL_FEATURE_NAMES,
    calculate_macro_f05,
    evaluate_metrics,
    featurize_s1,
    query_for_s1,
)

TRAIN_DIR = Path("dataset/train")


def load_all_gt():
    gt = {}
    with open(TRAIN_DIR / "train_ground_truth.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            gt[p[0]] = set(x.strip() for x in (p[1] if len(p) > 1 else "").split(",") if x.strip())
    return gt


def analyze_diagnostics(gt_map, pred_map, cand_map, cand_scores, name_addr_map):
    """
    Computes:
    - pair recall
    - entity full recall
    - zero recovery entities
    - oracle F0.5
    - candidate counts
    - singleton stats
    - multi-match stats
    - FP / FN categorization
    """
    total_gt_pairs = sum(len(g) for g in gt_map.values())
    total_recovered_pairs = sum(len(g & cand_map.get(s1, set())) for s1, g in gt_map.items())
    pair_recall = total_recovered_pairs / max(1, total_gt_pairs)

    entity_full_recall_count = 0
    zero_recovery_count = 0
    total_non_singleton = 0

    oracle_preds = {}
    for s1, g in gt_map.items():
        cands = cand_map.get(s1, set())
        oracle_preds[s1] = g & cands
        if len(g) > 0:
            total_non_singleton += 1
            rec = len(g & cands)
            if rec == len(g):
                entity_full_recall_count += 1
            elif rec == 0:
                zero_recovery_count += 1

    entity_full_recall = entity_full_recall_count / max(1, total_non_singleton)
    oracle_f05 = calculate_macro_f05(gt_map, oracle_preds)

    metrics = evaluate_metrics(gt_map, pred_map)

    # Detailed breakdown of False Negatives and False Positives
    fn_retrieved_model_rejected = 0
    fn_never_retrieved = 0
    fn_dropped_by_margin = 0

    fp_count = 0
    fp_categories = Counter()  # e.g., 'singleton_false_alarm', 'multi_match_excess'

    singleton_gt_count = sum(1 for g in gt_map.values() if len(g) == 0)
    singleton_correct = 0

    for s1, g in gt_map.items():
        preds = pred_map.get(s1, set())
        cands = cand_map.get(s1, set())
        scores = cand_scores.get(s1, {})

        if len(g) == 0:
            if len(preds) == 0:
                singleton_correct += 1
            else:
                fp_categories["singleton_false_alarm"] += len(preds)
        else:
            # Non-singleton
            # False negatives:
            for true_target in g:
                if true_target not in preds:
                    if true_target not in cands:
                        fn_never_retrieved += 1
                    else:
                        fn_retrieved_model_rejected += 1
                        # check if it was rejected because of delta_margin or tau_singleton
                        max_score = max(scores.values()) if scores else 0.0
                        t_score = scores.get(true_target, 0.0)
                        if max_score >= 0.76 and t_score >= 0.45:
                            fn_dropped_by_margin += 1

            # False positives:
            for pred_target in preds:
                if pred_target not in g:
                    fp_count += 1
                    fp_categories["multi_match_excess"] += 1

    singleton_acc = singleton_correct / max(1, singleton_gt_count) if singleton_gt_count else 1.0

    return {
        "macro_f05": metrics["macro_f05"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "oracle_f05": oracle_f05,
        "pair_recall": pair_recall,
        "entity_full_recall": entity_full_recall,
        "zero_recovery_count": zero_recovery_count,
        "total_gt_pairs": total_gt_pairs,
        "recovered_gt_pairs": total_recovered_pairs,
        "singleton_count": singleton_gt_count,
        "singleton_accuracy": singleton_acc,
        "fn_total": fn_never_retrieved + fn_retrieved_model_rejected,
        "fn_never_retrieved": fn_never_retrieved,
        "fn_retrieved_model_rejected": fn_retrieved_model_rejected,
        "fn_dropped_by_margin": fn_dropped_by_margin,
        "fp_total": fp_count + fp_categories["singleton_false_alarm"],
        "fp_categories": dict(fp_categories),
    }


def run_seeded_analysis(n_s1=1500):
    with open("experiments/splits/holdout_s1_ids.json") as f:
        hold = json.load(f)
    with open("experiments/splits/train_pool_s1_ids.json") as f:
        tr = set(json.load(f)[:25000])
    with open("experiments/splits/dev_s1_ids.json") as f:
        dv = set(json.load(f)[:6000])
    sub = [e for e in hold if e not in tr and e not in dv][:n_s1]
    gt_all = load_all_gt()
    need = set()
    for e in sub:
        need.update(gt_all.get(e, set()))

    idx = {"US": CountryBlockingIndex("US"), "India": CountryBlockingIndex("India")}
    for fn, is_s2 in [("train_source2.tsv", 1), ("train_source3.tsv", 0)]:
        with open(TRAIN_DIR / fn, encoding="utf-8") as f:
            next(f)
            for i, line in enumerate(f):
                p = line.rstrip("\n").split("\t")
                if len(p) < 4 or p[3] not in idx:
                    continue
                eid, bn, ba, bc = p
                if eid in need or i < 60000:
                    r = prepare_source_record(bn, ba, bc)
                    idx[bc].add_target_record(eid, r["norm_name"], r["core_tokens"], r["norm_addr"], r["addr_keys"], r["digits"], is_s2=is_s2, raw_addr=ba)
    for c in idx.values():
        c.prune_frequent_keys()

    with open("experiments/final_scratch/config.json") as f:
        cfg = json.load(f)
    model = lgb.Booster(model_file="experiments/final_scratch/lgbm_matcher.txt")

    s1_records = {}
    with open(TRAIN_DIR / "train_source1.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if p[0] in set(sub):
                s1_records[p[0]] = prepare_source_record(p[1], p[2], p[3])

    gt_map, pred_map, cand_map, cand_scores, name_addr = {}, {}, {}, {}, {}
    for sid in sub:
        r = s1_records[sid]
        name_addr[sid] = (r["norm_name"], r["addr"])
        g = gt_all.get(sid, set())
        gt_map[sid] = g
        ci = idx[r["country"]]
        items, prov = query_for_s1(ci, r, max_candidates=cfg.get("max_candidates", 60), adaptive=cfg.get("adaptive", True))
        cand_map[sid] = {c for c, _ in items}
        cands, feats = featurize_s1(r, items, prov, ci, use_extra=cfg.get("use_extra", True))
        if feats:
            probs = model.predict(np.array(feats, dtype=np.float32))
            cand_scores[sid] = dict(zip(cands, [float(x) for x in probs]))
            mp = max(probs)
            if mp < cfg["tau_singleton"]:
                pred_map[sid] = set()
            else:
                cutoff = max(cfg["tau_min"], mp - cfg["delta_margin"])
                pred_map[sid] = {c for c, p in zip(cands, probs) if p >= cutoff}
        else:
            cand_scores[sid] = {}
            pred_map[sid] = set()

    return analyze_diagnostics(gt_map, pred_map, cand_map, cand_scores, name_addr)


def run_natural_analysis(limit=100):
    with open("experiments/splits/holdout_s1_ids.json") as f:
        holdout = set(json.load(f))
    sub = []
    with open(TRAIN_DIR / "train_source1.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if p[0] in holdout and p[3] == "India":
                sub.append((p[0], prepare_source_record(p[1], p[2], p[3])))
                if len(sub) == limit:
                    break

    gt_all = load_all_gt()
    index = DiskCountryBlockingIndex("India", "experiments/final_scratch/india_train.sqlite")

    with open("experiments/final_scratch/config.json") as f:
        cfg = json.load(f)
    model = lgb.Booster(model_file="experiments/final_scratch/lgbm_matcher.txt")

    gt_map, pred_map, cand_map, cand_scores, name_addr = {}, {}, {}, {}, {}
    for sid, r in sub:
        name_addr[sid] = (r["norm_name"], r["addr"])
        g = gt_all.get(sid, set())
        gt_map[sid] = g
        items, prov = query_for_s1(index, r, max_candidates=cfg.get("max_candidates", 60), adaptive=cfg.get("adaptive", True))
        cand_map[sid] = {c for c, _ in items}
        cands, feats = featurize_s1(r, items, prov, index, use_extra=cfg.get("use_extra", True))
        if feats:
            probs = model.predict(np.array(feats, dtype=np.float32))
            cand_scores[sid] = dict(zip(cands, [float(x) for x in probs]))
            mp = max(probs)
            if mp < cfg["tau_singleton"]:
                pred_map[sid] = set()
            else:
                cutoff = max(cfg["tau_min"], mp - cfg["delta_margin"])
                pred_map[sid] = {c for c, p in zip(cands, probs) if p >= cutoff}
        else:
            cand_scores[sid] = {}
            pred_map[sid] = set()

    index.close()
    return analyze_diagnostics(gt_map, pred_map, cand_map, cand_scores, name_addr)


if __name__ == "__main__":
    t0 = time.time()
    print("=== RUNNING DIFFERENTIAL ERROR ANALYSIS ===", flush=True)
    print("\n--- 1. Natural India (100 sample, 4.13M pool) ---", flush=True)
    nat_results = run_natural_analysis(limit=100)
    print(json.dumps(nat_results, indent=2), flush=True)

    print("\n--- 2. Seeded Holdout (1,500 sample) ---", flush=True)
    seed_results = run_seeded_analysis(n_s1=1500)
    print(json.dumps(seed_results, indent=2), flush=True)

    diff = {
        "metric": ["Macro F0.5", "Oracle F0.5", "Pair Recall", "Entity Full Recall", "Precision", "Recall", "Singleton Acc"],
        "seeded": [
            seed_results["macro_f05"], seed_results["oracle_f05"], seed_results["pair_recall"],
            seed_results["entity_full_recall"], seed_results["precision"], seed_results["recall"],
            seed_results["singleton_accuracy"]
        ],
        "natural": [
            nat_results["macro_f05"], nat_results["oracle_f05"], nat_results["pair_recall"],
            nat_results["entity_full_recall"], nat_results["precision"], nat_results["recall"],
            nat_results["singleton_accuracy"]
        ],
    }
    print("\n--- Summary Comparison ---")
    print(f"{'Metric':<20} | {'Seeded Holdout':<15} | {'Natural India':<15} | {'Gap':<10}")
    print("-" * 65)
    for m, s, n in zip(diff["metric"], diff["seeded"], diff["natural"]):
        print(f"{m:<20} | {s:<15.4f} | {n:<15.4f} | {s - n:<+10.4f}")
