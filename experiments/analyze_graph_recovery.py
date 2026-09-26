#!/usr/bin/env python3
"""Estimate whether a retrieved true target can bridge a missed true target."""

import csv
import json
from collections import Counter, defaultdict

from analyze_misses import channels
from evaluate_live import record, rows


BASE = "experiments/live_holdout_seeded_full/"


def main():
    hits = defaultdict(list)
    scored = defaultdict(list)
    with open(BASE + "score_distribution.csv", newline="") as f:
        for row in csv.DictReader(f):
            scored[row["s1_id"]].append((row["target_id"], float(row["score"]), row["is_match"] == "1"))
            if row["is_match"] == "1":
                hits[row["s1_id"]].append(row["target_id"])
    with open("solution/business_entity_resolution/models/config.json") as f:
        config = json.load(f)
    accepted = {}
    for sid, pairs in scored.items():
        best = max(score for _, score, _ in pairs)
        cutoff = max(config["tau_min"], best - config["delta_margin"])
        accepted[sid] = [tid for tid, score, true in pairs if true and best >= config["tau_singleton"] and score >= cutoff]
    with open(BASE + "missed_candidate_analysis.csv", newline="") as f:
        misses = list(csv.DictReader(f))
    target_ids = {tid for group in hits.values() for tid in group}
    target_ids.update(row["target_id"] for row in misses)
    targets = {}
    for source in (2, 3):
        targets.update({row["entity_id"]: record(row) for row in rows(f"student_resource/dataset/train/train_source{source}.tsv") if row["entity_id"] in target_ids})
    summary = Counter()
    with open(BASE + "graph_recovery_analysis.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("s1_id", "missed_target_id", "retrieved_true_neighbors", "accepted_true_neighbors", "bridge_channels", "accepted_bridge_channels"))
        for row in misses:
            sid, tid = row["s1_id"], row["target_id"]
            bridges = set()
            for hit in hits[sid]:
                bridges.update(channels(targets[tid], targets[hit]))
            accepted_bridges = set()
            for hit in accepted.get(sid, ()):
                accepted_bridges.update(channels(targets[tid], targets[hit]))
            summary["has_true_neighbor"] += bool(hits[sid])
            summary["has_accepted_true_neighbor"] += bool(accepted.get(sid))
            for channel in bridges:
                summary[channel] += 1
            for channel in accepted_bridges:
                summary["accepted_" + channel] += 1
            writer.writerow((sid, tid, len(hits[sid]), len(accepted.get(sid, ())), ";".join(sorted(bridges)), ";".join(sorted(accepted_bridges))))
    print(dict(summary))


if __name__ == "__main__":
    main()
