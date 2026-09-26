#!/usr/bin/env python3
"""Recompute decision metrics from a saved score distribution."""

import argparse
import csv
import json
import os
import sys
import unicodedata
from collections import defaultdict

sys.path.insert(0, os.path.abspath("solution/business_entity_resolution/src"))
from train import evaluate_metrics
from evaluate_live import rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", default="experiments/live_holdout_seeded_full")
    parser.add_argument("--models-dir", default="solution/business_entity_resolution/models")
    parser.add_argument("--threshold-json")
    args = parser.parse_args()
    with open("experiments/splits/holdout_s1_ids.json") as f:
        holdout = set(json.load(f))
    gt = {}
    for row in rows("student_resource/dataset/train/train_ground_truth.tsv"):
        if row["source1_entity_id"] in holdout:
            gt[row["source1_entity_id"]] = set(filter(None, row["matched_entity_ids"].split(",")))
    with open(os.path.join(args.models_dir, "config.json")) as f:
        config = json.load(f)
    if args.threshold_json:
        with open(args.threshold_json) as f:
            config.update(json.load(f))
    scores = defaultdict(list)
    with open(os.path.join(args.directory, "score_distribution.csv"), newline="") as f:
        for row in csv.DictReader(f):
            scores[row["s1_id"]].append((row["target_id"], float(row["score"])))
    preds = {}
    for sid, pairs in scores.items():
        best = max(score for _, score in pairs)
        cutoff = max(config["tau_min"], best - config["delta_margin"])
        preds[sid] = {tid for tid, score in pairs if best >= config["tau_singleton"] and score >= cutoff}
    groups = defaultdict(list)
    for row in rows("student_resource/dataset/train/train_source1.tsv"):
        sid = row["entity_id"]
        if sid not in holdout:
            continue
        name = row["business_name"]
        dev = any("DEVANAGARI" in unicodedata.name(ch, "") for ch in name)
        latin = any("LATIN" in unicodedata.name(ch, "") for ch in name)
        groups[("country", row["country"])].append(sid)
        groups[("script", "mixed" if dev and latin else "devanagari" if dev else "latin" if latin else "other")].append(sid)
        groups[("cardinality", "singleton" if not gt[sid] else "one" if len(gt[sid]) == 1 else "multi")].append(sid)
    result = {f"{kind}:{label}": evaluate_metrics({sid: gt[sid] for sid in ids}, preds)
              for (kind, label), ids in groups.items()}
    result["overall"] = evaluate_metrics(gt, preds)
    with open(os.path.join(args.directory, "group_metrics.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
