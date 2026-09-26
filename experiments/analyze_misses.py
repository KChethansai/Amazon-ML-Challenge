#!/usr/bin/env python3
"""Explain which existing blocking keys connect missed holdout pairs."""

import argparse
import csv
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.abspath("solution/business_entity_resolution/src"))
from blocking import extract_acronyms, extract_postal_prefixes
from evaluate_live import record, rows


def channels(left, right):
    found = []
    if left["norm_name"] and left["norm_name"] == right["norm_name"]:
        found.append("exact_name")
    if set(left["core_tokens"]) & set(right["core_tokens"]):
        found.append("name_token")
    if len(left["norm_name"]) >= 4 and left["norm_name"][:4] == right["norm_name"][:4]:
        found.append("name_prefix4")
    if set(left["addr_keys"]) & set(right["addr_keys"]):
        found.append("address_key")
    if set(extract_acronyms(left["norm_name"], left["core_tokens"])) & set(extract_acronyms(right["norm_name"], right["core_tokens"])):
        found.append("acronym")
    if left["core_tokens"] and right["core_tokens"] and left["core_tokens"][0] == right["core_tokens"][0]:
        if set(extract_postal_prefixes(left["addr"], left["norm_addr"])) & set(extract_postal_prefixes(right["addr"], right["norm_addr"])):
            found.append("token_postal")
    a, b = left["norm_name"], right["norm_name"]
    if len(a) >= 3 and len(b) >= 3 and len({a[i:i+3] for i in range(len(a)-2)} & {b[i:i+3] for i in range(len(b)-2)}) >= 2:
        found.append("shared_trigram")
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--misses", default="experiments/live_holdout_seeded/missed_candidate_analysis.csv")
    parser.add_argument("--out", default="experiments/live_holdout_seeded/missed_candidate_channels.csv")
    parser.add_argument("--channel-recall-out")
    args = parser.parse_args()
    with open(args.misses, newline="") as f:
        misses = list(csv.DictReader(f))
    missed_pairs = {(row["s1_id"], row["target_id"]) for row in misses}
    all_pairs = []
    if args.channel_recall_out:
        with open("experiments/splits/holdout_s1_ids.json") as f:
            import json
            holdout = set(json.load(f))
        for row in rows("student_resource/dataset/train/train_ground_truth.tsv"):
            if row["source1_entity_id"] in holdout:
                all_pairs.extend((row["source1_entity_id"], tid) for tid in filter(None, row["matched_entity_ids"].split(",")))
    s1_ids = {row["s1_id"] for row in misses}
    target_ids = {row["target_id"] for row in misses}
    s1_ids.update(sid for sid, _ in all_pairs)
    target_ids.update(tid for _, tid in all_pairs)
    base = "student_resource/dataset/train/train_"
    s1 = {row["entity_id"]: record(row) for row in rows(base + "source1.tsv") if row["entity_id"] in s1_ids}
    targets = {}
    for source in (2, 3):
        targets.update({row["entity_id"]: record(row) for row in rows(base + f"source{source}.tsv") if row["entity_id"] in target_ids})
    summary = Counter()
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("s1_id", "target_id", "country", "channels", "category", "candidate_count"))
        for row in misses:
            found = channels(s1[row["s1_id"]], targets[row["target_id"]])
            category = "no_shared_key" if not found else "ngram_only" if found == ["shared_trigram"] else "ranking_or_pruned_key"
            summary[category] += 1
            writer.writerow((row["s1_id"], row["target_id"], row["country"], ";".join(found), category, row["candidate_count"]))
    print(dict(summary))
    if args.channel_recall_out:
        channel_hits, channel_total = Counter(), Counter()
        for sid, tid in all_pairs:
            for channel in channels(s1[sid], targets[tid]):
                channel_total[channel] += 1
                channel_hits[channel] += (sid, tid) not in missed_pairs
        with open(args.channel_recall_out, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(("channel", "ground_truth_pairs_with_key", "retrieved_pairs", "retrieval_rate_given_key"))
            writer.writerows((name, count, channel_hits[name], channel_hits[name] / count)
                             for name, count in channel_total.most_common())


if __name__ == "__main__":
    main()
