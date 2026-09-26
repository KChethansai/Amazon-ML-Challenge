#!/usr/bin/env python3
"""Tune the two-stage decision rule on saved development scores only."""

import argparse
import csv
import json
from collections import defaultdict
from itertools import product

from evaluate_live import rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", default="experiments/retrain_trial/dev_eval/score_distribution.csv")
    parser.add_argument("--out", default="experiments/retrain_trial/threshold_search.json")
    args = parser.parse_args()
    with open("experiments/retrain_trial/config.json") as f:
        config = json.load(f)
    with open("experiments/splits/dev_s1_ids.json") as f:
        ids = set(json.load(f)[:config["num_val_s1"]])
    truth = {}
    for row in rows("student_resource/dataset/train/train_ground_truth.tsv"):
        if row["source1_entity_id"] in ids:
            truth[row["source1_entity_id"]] = set(filter(None, row["matched_entity_ids"].split(",")))
    scores = defaultdict(list)
    with open(args.scores, newline="") as f:
        for row in csv.DictReader(f):
            scores[row["s1_id"]].append((row["target_id"], float(row["score"])))
    best = {"macro_f05": -1}
    for singleton, minimum, margin in product(
        (0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90),
        (0.30, 0.40, 0.45, 0.55, 0.65),
        (0.08, 0.12, 0.18, 0.24, 0.30),
    ):
        total = 0.0
        tp_total = pred_total = 0
        for sid, actual in truth.items():
            pairs = scores[sid]
            if pairs:
                top = max(score for _, score in pairs)
                cutoff = max(minimum, top - margin)
                predicted = {tid for tid, score in pairs if top >= singleton and score >= cutoff}
            else:
                predicted = set()
            tp = len(predicted & actual)
            tp_total += tp
            pred_total += len(predicted)
            if not actual:
                total += int(not predicted)
            elif tp:
                precision = tp / len(predicted)
                recall = tp / len(actual)
                total += 1.25 * precision * recall / (0.25 * precision + recall)
        result = {"tau_singleton": singleton, "tau_min": minimum,
                  "delta_margin": margin, "macro_f05": total / len(truth),
                  "precision": tp_total / max(1, pred_total)}
        if result["precision"] < 0.99:
            continue
        if (result["macro_f05"], result["precision"]) > (best["macro_f05"], best.get("precision", 0)):
            best = result
    with open(args.out, "w") as f:
        json.dump(best, f, indent=2)
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
