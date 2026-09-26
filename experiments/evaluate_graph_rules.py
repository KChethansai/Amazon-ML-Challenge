#!/usr/bin/env python3
"""Evaluate conservative second-hop acceptance rules on development scores."""

import csv
import argparse
import json
import os
import sys
from collections import defaultdict
from itertools import product

from rapidfuzz import fuzz

sys.path.insert(0, os.path.abspath("solution/business_entity_resolution/src"))
from train import evaluate_metrics
from evaluate_live import rows
from normalization import normalize_name, normalize_address


BASE = "experiments/retrain_trial/"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("dev", "holdout"), default="dev")
    args = parser.parse_args()
    with open(BASE + "config.json") as f:
        config = json.load(f)
    with open(f"experiments/splits/{args.split}_s1_ids.json") as f:
        values = json.load(f)
        ids = set(values[:config["num_val_s1"]] if args.split == "dev" else values)
    gt = {}
    for row in rows("student_resource/dataset/train/train_ground_truth.tsv"):
        if row["source1_entity_id"] in ids:
            gt[row["source1_entity_id"]] = set(filter(None, row["matched_entity_ids"].split(",")))
    initial = defaultdict(list)
    initial_ids = defaultdict(set)
    initial_dir = "dev_eval" if args.split == "dev" else "independent_eval"
    graph_dir = "dev_graph" if args.split == "dev" else "holdout_graph"
    with open(BASE + initial_dir + "/score_distribution.csv", newline="") as f:
        for row in csv.DictReader(f):
            sid, tid = row["s1_id"], row["target_id"]
            initial[sid].append((tid, float(row["score"])))
            initial_ids[sid].add(tid)
    anchors = {sid: [tid for tid, score in sorted(pairs, key=lambda item: -item[1])
                     if score >= 0.98][:2] for sid, pairs in initial.items()}
    extras = defaultdict(list)
    needed = {tid for group in anchors.values() for tid in group}
    with open(BASE + graph_dir + "/score_distribution.csv", newline="") as f:
        for row in csv.DictReader(f):
            sid, tid = row["s1_id"], row["target_id"]
            if tid not in initial_ids[sid]:
                extras[sid].append((tid, float(row["score"])))
                needed.add(tid)
    targets = {}
    for source in (2, 3):
        for row in rows(f"student_resource/dataset/train/train_source{source}.tsv"):
            if row["entity_id"] in needed:
                targets[row["entity_id"]] = (normalize_name(row["business_name"]),
                                             normalize_address(row["business_address"]))
    bridge = {}
    for sid, pairs in extras.items():
        anchor_records = [targets[tid] for tid in anchors.get(sid, ())]
        for tid, score in pairs:
            name, addr = targets[tid]
            agreement = [(fuzz.token_set_ratio(name, aname), fuzz.token_set_ratio(addr, aaddr))
                         for aname, aaddr in anchor_records]
            bridge[(sid, tid)] = max((min(n, a) for n, a in agreement), default=0)
    base_preds = {}
    for sid, pairs in initial.items():
        top = max(score for _, score in pairs)
        cutoff = max(config["tau_min"], top - config["delta_margin"])
        base_preds[sid] = {tid for tid, score in pairs if top >= config["tau_singleton"] and score >= cutoff}
    baseline = evaluate_metrics(gt, base_preds)
    if args.split == "holdout":
        preds = {sid: set(pred) for sid, pred in base_preds.items()}
        for sid, pairs in extras.items():
            preds.setdefault(sid, set()).update(
                tid for tid, score in pairs if score >= 0.999 and bridge[(sid, tid)] >= 80)
        result = {"baseline": baseline, "frozen_graph_rule": evaluate_metrics(gt, preds),
                  "score_min": 0.999, "bridge_min_name_address": 80}
        with open(BASE + "holdout_graph_rule_eval.json", "w") as f:
            json.dump(result, f, indent=2)
        print(json.dumps(result, indent=2))
        return
    best = {"macro_f05": baseline["macro_f05"], "precision": baseline["precision"], "rule": "none"}
    trials = []
    for score_min, agreement_min in product((0.99, 0.999, 0.9999, 0.99999), (70, 80, 90, 95, 98)):
        preds = {sid: set(pred) for sid, pred in base_preds.items()}
        for sid, pairs in extras.items():
            preds.setdefault(sid, set()).update(
                tid for tid, score in pairs if score >= score_min and bridge[(sid, tid)] >= agreement_min)
        metrics = evaluate_metrics(gt, preds)
        trial = {"score_min": score_min, "bridge_min_name_address": agreement_min, **metrics}
        trials.append(trial)
        if metrics["precision"] >= 0.99 and metrics["macro_f05"] > best["macro_f05"]:
            best = {"rule": trial, **metrics}
    result = {"baseline": baseline, "best": best, "trials": trials}
    with open(BASE + "graph_rule_search.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({"baseline": baseline, "best": best}, indent=2))


if __name__ == "__main__":
    main()
