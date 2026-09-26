#!/usr/bin/env python3
"""Exploratory split-sample decision calibration on unseeded India scores."""

import csv
import hashlib
import json
from collections import defaultdict
from itertools import product

from evaluate_live import rows


PATH = "experiments/full_india_sample/"


def metrics(ids, truth, scores, thresholds):
    singleton, minimum, margin = thresholds
    total = tp_total = pred_total = true_total = 0
    for sid in ids:
        pairs = scores[sid]
        top = max((score for _, score in pairs), default=0)
        cutoff = max(minimum, top - margin)
        pred = {tid for tid, score in pairs if top >= singleton and score >= cutoff}
        actual = truth[sid]
        tp = len(pred & actual)
        tp_total += tp
        pred_total += len(pred)
        true_total += len(actual)
        if not actual:
            total += int(not pred)
        elif tp:
            precision = tp / len(pred)
            recall = tp / len(actual)
            total += 1.25 * precision * recall / (0.25 * precision + recall)
    return {"macro_f05": total / len(ids), "precision": tp_total / max(1, pred_total),
            "recall": tp_total / max(1, true_total), "entities": len(ids)}


def main():
    scores = defaultdict(list)
    with open(PATH + "score_distribution.csv", newline="") as f:
        for row in csv.DictReader(f):
            scores[row["s1_id"]].append((row["target_id"], float(row["score"])))
    truth = {}
    for row in rows("student_resource/dataset/train/train_ground_truth.tsv"):
        if row["source1_entity_id"] in scores:
            truth[row["source1_entity_id"]] = set(filter(None, row["matched_entity_ids"].split(",")))
    calibration = [sid for sid in scores if hashlib.sha256(sid.encode()).digest()[0] % 2 == 0]
    evaluation = [sid for sid in scores if sid not in set(calibration)]
    baseline = (0.7, 0.45, 0.18)
    best = {"macro_f05": -1}
    best_unconstrained = {"macro_f05": -1}
    for params in product((0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999),
                          (0.45, 0.6, 0.75, 0.85, 0.95, 0.99, 0.999, 0.9999),
                          (0.05, 0.1, 0.18, 0.25, 0.35)):
        result = metrics(calibration, truth, scores, params)
        if result["macro_f05"] > best_unconstrained["macro_f05"]:
            best_unconstrained = {"thresholds": params, **result}
        if result["precision"] >= 0.99 and result["macro_f05"] > best["macro_f05"]:
            best = {"thresholds": params, **result}
    output = {"calibration_baseline": metrics(calibration, truth, scores, baseline),
              "evaluation_baseline": metrics(evaluation, truth, scores, baseline),
              "best_calibration": best, "best_unconstrained": best_unconstrained}
    if "thresholds" in best:
        output["evaluation_frozen_thresholds"] = metrics(evaluation, truth, scores, best["thresholds"])
    output["evaluation_unconstrained"] = metrics(evaluation, truth, scores, best_unconstrained["thresholds"])
    with open(PATH + "threshold_split_test.json", "w") as f:
        json.dump(output, f, indent=2)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
