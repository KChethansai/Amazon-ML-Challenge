#!/usr/bin/env python3
"""Evaluate the shipped model on the saved holdout without retraining it."""

import argparse
import csv
import json
import os
import resource
import sys
import time
import unicodedata
from collections import Counter, defaultdict

import lightgbm as lgb
import numpy as np
from rapidfuzz import fuzz

sys.path.insert(0, os.path.abspath("solution/business_entity_resolution/src"))
from blocking import CountryBlockingIndex
from features import compute_candidate_features_for_s1, compute_pairwise_features
from normalization import (extract_address_blocking_keys, extract_address_digits,
                           extract_name_tokens, normalize_address, normalize_name)
from train import evaluate_metrics


def rows(path):
    with open(path, encoding="utf-8", newline="") as f:
        yield from csv.DictReader(f, delimiter="\t")


def record(row):
    name, addr = row["business_name"], row["business_address"]
    norm_name, norm_addr = normalize_name(name), normalize_address(addr)
    tokens, _ = extract_name_tokens(norm_name)
    digits = extract_address_digits(addr)
    nums = [n for n in digits if len(n) in (4, 5, 6)]
    street = [n for n in digits if 1 <= len(n) <= 5]
    return dict(name=name, norm_name=norm_name, core_tokens=tokens, addr=addr,
                norm_addr=norm_addr, addr_tokens=set(norm_addr.split()),
                addr_keys=extract_address_blocking_keys(addr, norm_addr),
                digits=digits, prefix6=norm_name[:6],
                street_num=street[0] if street else "", postal=nums[0] if nums else "",
                country=row["country"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--background", type=int, default=200000)
    parser.add_argument("--seed-holdout-targets", action="store_true")
    parser.add_argument("--max-candidates", type=int, default=60)
    parser.add_argument("--recall-only", action="store_true")
    parser.add_argument("--compare-retrieval", action="store_true")
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--ngram-fallback-limit", type=int, default=15)
    parser.add_argument("--address-tokens", action="store_true")
    parser.add_argument("--retrieval-depth", type=int, default=200)
    parser.add_argument("--out", default="experiments/live_holdout")
    parser.add_argument("--country")
    parser.add_argument("--sample-s1", type=int)
    parser.add_argument("--models-dir", default="solution/business_entity_resolution/models")
    parser.add_argument("--split", choices=("holdout", "dev"), default="holdout")
    parser.add_argument("--graph-expand", action="store_true")
    parser.add_argument("--graph-anchor-threshold", type=float, default=0.98)
    args = parser.parse_args()
    start = time.time()
    base = "student_resource/dataset/train/train_"
    with open("experiments/splits/holdout_s1_ids.json") as f:
        final_holdout = set(json.load(f))
    with open(os.path.join(args.models_dir, "config.json")) as f:
        config = json.load(f)
    if args.split == "dev":
        with open("experiments/splits/dev_s1_ids.json") as f:
            holdout = set(json.load(f)[:config["num_val_s1"]])
    else:
        holdout = final_holdout
    seeded_s1 = set(final_holdout) if args.seed_holdout_targets else set(holdout)
    if args.seed_holdout_targets:
        for split, count in (("train_pool", config["num_train_s1"]), ("dev", config["num_val_s1"])):
            with open(f"experiments/splits/{split}_s1_ids.json") as f:
                seeded_s1.update(json.load(f)[:count])
    gt = {}
    needed = set()
    for row in rows(base + "ground_truth.tsv"):
        sid = row["source1_entity_id"]
        if sid in seeded_s1:
            matches = set(filter(None, row["matched_entity_ids"].split(",")))
            needed.update(matches)
            if sid in holdout:
                gt[sid] = matches
    s1 = {}
    for row in rows(base + "source1.tsv"):
        if row["entity_id"] in holdout and (args.country is None or row["country"] == args.country):
            s1[row["entity_id"]] = record(row)
            if args.sample_s1 and len(s1) >= args.sample_s1:
                break
    gt = {sid: gt[sid] for sid in s1}
    indexes = defaultdict(lambda: CountryBlockingIndex("", address_tokens=args.address_tokens))
    for source in (2, 3):
        for i, row in enumerate(rows(base + f"source{source}.tsv")):
            if i >= args.background and not (args.seed_holdout_targets and row["entity_id"] in needed):
                continue
            if args.country is not None and row["country"] != args.country:
                continue
            rec = record(row)
            indexes[rec["country"]].add_target_record(
                row["entity_id"], rec["norm_name"], rec["core_tokens"],
                rec["norm_addr"], rec["addr_keys"], rec["digits"],
                int(source == 2), raw_addr=rec["addr"])
    for idx in indexes.values():
        idx.prune_frequent_keys()
        idx.ngram_fallback_limit = args.ngram_fallback_limit
    if args.recall_only:
        budgets = (10, 20, 35, 50, 60, 75, 100, 125, 150, 200)
        if args.compare_retrieval:
            comparisons = defaultdict(Counter)
            for sid, rec in s1.items():
                idx = indexes.get(rec["country"])
                if not idx:
                    continue
                for limit, label in ((15, "conditional_ngram"), (1000000, "always_ngram")):
                    idx.ngram_fallback_limit = limit
                    items = idx.query_candidates(rec["norm_name"], rec["core_tokens"], rec["norm_addr"],
                                                 rec["addr_keys"], s1_raw_addr=rec["addr"],
                                                 max_candidates=args.retrieval_depth, return_weights=True)
                    for rerank in (False, True):
                        ranked = sorted(items, key=lambda item: (max(fuzz.WRatio(rec["norm_name"], idx.records[item[0]][0]),
                                                                     fuzz.WRatio(rec["norm_addr"], idx.records[item[0]][1])), item[1]),
                                        reverse=True) if rerank else items
                        key = label + ("_fuzzy_rerank" if rerank else "_weight_rank")
                        ids = [cid for cid, _ in ranked]
                        for budget in budgets:
                            comparisons[key][budget] += len(gt[sid] & set(ids[:budget]))
            denominator = sum(map(len, gt.values()))
            result = {method: {str(budget): counts[budget] / denominator for budget in budgets}
                      for method, counts in comparisons.items()}
            os.makedirs(args.out, exist_ok=True)
            with open(os.path.join(args.out, "candidate_recall_comparison.json"), "w") as f:
                json.dump(result, f, indent=2)
            print(json.dumps(result, indent=2), flush=True)
            return
        hits = Counter()
        for sid, rec in s1.items():
            idx = indexes.get(rec["country"])
            ranked = idx.query_candidates(rec["norm_name"], rec["core_tokens"], rec["norm_addr"],
                                          rec["addr_keys"], s1_raw_addr=rec["addr"],
                                          max_candidates=max(max(budgets), args.retrieval_depth), return_weights=True) if idx else []
            if args.rerank:
                ranked.sort(key=lambda item: (max(fuzz.WRatio(rec["norm_name"], idx.records[item[0]][0]),
                                                      fuzz.WRatio(rec["norm_addr"], idx.records[item[0]][1])),
                                              item[1]), reverse=True)
            ranked = [cid for cid, _ in ranked]
            for budget in budgets:
                hits[budget] += len(gt[sid] & set(ranked[:budget]))
        denominator = sum(map(len, gt.values()))
        result = {str(budget): hits[budget] / denominator for budget in budgets}
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "candidate_recall_by_budget.json"), "w") as f:
            json.dump(result, f, indent=2)
        print(json.dumps(result, indent=2), flush=True)
        return
    model = lgb.Booster(model_file=os.path.join(args.models_dir, "lgbm_matcher.txt"))
    os.makedirs(args.out, exist_ok=True)
    missed, false_neg, false_pos, distributions = [], [], [], []
    preds, hits, total_candidates = {}, 0, 0
    country_hits, country_truth = Counter(), Counter()
    for sid, rec in s1.items():
        idx = indexes.get(rec["country"])
        items = idx.query_candidates(rec["norm_name"], rec["core_tokens"], rec["norm_addr"],
                                     rec["addr_keys"], s1_raw_addr=rec["addr"],
                                     max_candidates=args.max_candidates, return_weights=True) if idx else []
        valid, features = compute_candidate_features_for_s1(rec, items, idx.records) if idx else ([], [])
        probs = model.predict(np.asarray(features, dtype=np.float32)) if features else np.array([])
        if args.graph_expand and len(probs):
            extra = []
            seen = set(valid)
            anchors = sorted(((cid, float(score)) for cid, score in zip(valid, probs)
                              if score >= args.graph_anchor_threshold), key=lambda item: -item[1])[:2]
            for anchor, _ in anchors:
                anchor_name, anchor_addr, _ = idx.records[anchor]
                anchor_tokens, _ = extract_name_tokens(anchor_name)
                for cid, weight in idx.query_candidates(anchor_name, anchor_tokens, anchor_addr,
                                                         extract_address_blocking_keys(anchor_addr, anchor_addr),
                                                         s1_raw_addr=anchor_addr, max_candidates=30,
                                                         return_weights=True):
                    if cid not in seen and len(items) + len(extra) < 120:
                        seen.add(cid)
                        extra.append((cid, weight))
            if extra:
                extra_features = [compute_pairwise_features(rec, idx.records[cid], rank=len(items) + i,
                                                           weight=weight, max_weight=items[0][1])
                                  for i, (cid, weight) in enumerate(extra)]
                probs = np.concatenate((probs, model.predict(np.asarray(extra_features, dtype=np.float32))))
                valid.extend(cid for cid, _ in extra)
                items.extend(extra)
        candidates = {cid for cid, _ in items}
        truth = gt[sid]
        hits += len(truth & candidates)
        total_candidates += len(items)
        country_hits[rec["country"]] += len(truth & candidates)
        country_truth[rec["country"]] += len(truth)
        for tid in truth - candidates:
            missed.append((sid, tid, rec["country"], "target_absent" if not idx or tid not in idx.records else "blocking_or_budget", len(items)))
        maximum = max(probs, default=0)
        cutoff = max(config["tau_min"], maximum - config["delta_margin"])
        pred = {cid for cid, score in zip(valid, probs) if maximum >= config["tau_singleton"] and score >= cutoff}
        preds[sid] = pred
        for cid, score in zip(valid, probs):
            label = int(cid in truth)
            distributions.append((sid, cid, rec["country"], label, float(score)))
            if label and cid not in pred:
                reason = "singleton_gate" if maximum < config["tau_singleton"] else "threshold_or_margin"
                false_neg.append((sid, cid, rec["country"], float(score), float(maximum), reason))
            if not label and cid in pred:
                false_pos.append((sid, cid, rec["country"], float(score), float(maximum)))
    metrics = evaluate_metrics(gt, preds)
    groups = defaultdict(list)
    for sid, rec in s1.items():
        groups[("country", rec["country"])].append(sid)
        groups[("cardinality", "singleton" if not gt[sid] else "one" if len(gt[sid]) == 1 else "multi")].append(sid)
        name = rec["name"]
        devanagari = any("DEVANAGARI" in unicodedata.name(ch, "") for ch in name)
        latin = any("LATIN" in unicodedata.name(ch, "") for ch in name)
        groups[("script", "mixed" if devanagari and latin else "devanagari" if devanagari else "latin" if latin else "other")].append(sid)
    metrics["group_metrics"] = {f"{kind}:{value}": evaluate_metrics({sid: gt[sid] for sid in ids}, preds)
                                for (kind, value), ids in groups.items()}
    metrics.update(candidate_recall=hits / max(1, sum(map(len, gt.values()))),
                   candidates=total_candidates, runtime_seconds=time.time() - start,
                   peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                   target_pool={country: len(idx.records) for country, idx in indexes.items()},
                   seed_holdout_targets=args.seed_holdout_targets,
                   graph_expand=args.graph_expand,
                   background_per_source=args.background,
                   country_candidate_recall={country: country_hits[country] / max(1, count)
                                             for country, count in country_truth.items()})
    for filename, header, values in (
        ("missed_candidate_analysis.csv", ("s1_id", "target_id", "country", "reason", "candidate_count"), missed),
        ("matcher_false_negative_analysis.csv", ("s1_id", "target_id", "country", "score", "best_score", "reason"), false_neg),
        ("matcher_false_positive_analysis.csv", ("s1_id", "target_id", "country", "score", "best_score"), false_pos),
        ("score_distribution.csv", ("s1_id", "target_id", "country", "is_match", "score"), distributions),
    ):
        with open(os.path.join(args.out, filename), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(values)
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
