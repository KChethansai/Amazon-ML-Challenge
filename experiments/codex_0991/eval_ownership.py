#!/usr/bin/env python3
"""Evaluate target ownership on the natural India development sample only.

Ownership is evaluated among the sampled S1 records. The full production
catalog can have additional competitors, so this is a development diagnostic.
"""

import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import resource
import sys
import time

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "solution/business_entity_resolution/src"))

from eval_natural import EXPECTED_INDIA_TARGETS, INDEX_PATH, MODEL_DIR, ensure_index, load_truth, split_sample
from train import ALL_FEATURE_NAMES, calculate_macro_f05, evaluate_metrics, featurize_s1, query_for_s1


def assign_ownership(scored, baseline, minimum_margin=0.0):
    """Assign each target to its best scored S1 if that S1 accepted it."""
    competitors = defaultdict(list)
    for sid, candidates in scored.items():
        for target, score in candidates:
            competitors[target].append((score, sid))
    owned = {sid: set() for sid in baseline}
    contested = resolved = rejected_margin = rejected_winner = 0
    for target, choices in competitors.items():
        if len(choices) > 1:
            contested += 1
        choices.sort(key=lambda row: (-row[0], row[1]))
        score, sid = choices[0]
        accepted = sum(target in baseline[owner] for _, owner in choices)
        if accepted > 1:
            resolved += 1
        if target not in baseline[sid]:
            if accepted:
                rejected_winner += 1
            continue
        if len(choices) > 1 and score - choices[1][0] < minimum_margin:
            rejected_margin += 1
            continue
        owned[sid].add(target)
    return owned, {
        "targets_with_multiple_sampled_candidates": contested,
        "targets_with_multiple_baseline_owners": resolved,
        "rejected_because_best_owner_did_not_accept": rejected_winner,
        "rejected_for_margin": rejected_margin,
    }


def false_positive_breakdown(truth, predicted):
    singleton_fp = multi_entity_fp = 0
    for sid, actual in truth.items():
        false_positives = len(predicted[sid] - actual)
        if actual:
            multi_entity_fp += false_positives
        else:
            singleton_fp += false_positives
    return {"singleton_false_positive_links": singleton_fp,
            "multi_entity_false_positive_links": multi_entity_fp}


def evaluate(limit, index_path, model_dir, minimum_margin, score_cache=None):
    start = time.monotonic()
    selected = split_sample("development")[:limit]
    truth = load_truth(sid for sid, _ in selected)
    index, build_seconds = ensure_index(index_path)
    try:
        if len(index.records) != EXPECTED_INDIA_TARGETS:
            raise ValueError("Index target count differs from natural evaluator")
        config = json.loads((model_dir / "config.json").read_text())
        model = lgb.Booster(model_file=str(model_dir / "lgbm_matcher.txt"))
        if config["feature_names"] != ALL_FEATURE_NAMES or model.num_feature() != len(ALL_FEATURE_NAMES):
            raise ValueError("Model/config/runtime feature order mismatch")
        baseline, scored, oracle = {}, {}, {}
        for count, (sid, record) in enumerate(selected, 1):
            items, provenance = query_for_s1(
                index, record, max_candidates=config.get("max_candidates", 60),
                adaptive=config.get("adaptive", True))
            ids, features = featurize_s1(
                record, items, provenance, index, use_extra=config.get("use_extra", True))
            oracle[sid] = truth[sid] & set(ids)
            scores = model.predict(np.asarray(features, dtype=np.float32)) if features else []
            scored[sid] = list(zip(ids, map(float, scores)))
            top = max(scores, default=0.0)
            threshold = max(config["tau_min"], top - config["delta_margin"])
            baseline[sid] = ({target for target, score in scored[sid] if score >= threshold}
                             if top >= config["tau_singleton"] else set())
            if count % 100 == 0:
                print(f"scored {count}/{len(selected)} in {time.monotonic()-start:.1f}s", flush=True)
        owned, conflicts = assign_ownership(scored, baseline, minimum_margin)
        if score_cache is not None:
            score_cache.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(score_cache, "wt", encoding="utf-8") as file:
                json.dump({"sample": "natural_development", "s1_ids": [sid for sid, _ in selected],
                           "truth": {sid: sorted(values) for sid, values in truth.items()},
                           "scored": scored, "baseline": {sid: sorted(values) for sid, values in baseline.items()},
                           "oracle": {sid: sorted(values) for sid, values in oracle.items()},
                           "thresholds": {key: config[key] for key in ("tau_singleton", "tau_min", "delta_margin")}}, file)
        return {
            "sample": "natural_development",
            "limit": len(selected),
            "scope": "ownership competition among sampled S1 only",
            "model_dir": str(model_dir),
            "index_path": str(index_path),
            "thresholds": {key: config[key] for key in ("tau_singleton", "tau_min", "delta_margin")},
            "minimum_owner_margin": minimum_margin,
            "baseline": evaluate_metrics(truth, baseline),
            "baseline_false_positives": false_positive_breakdown(truth, baseline),
            "ownership": evaluate_metrics(truth, owned),
            "ownership_false_positives": false_positive_breakdown(truth, owned),
            "candidate_oracle_f05": calculate_macro_f05(truth, oracle),
            "candidate_pair_recall": sum(map(len, oracle.values())) / max(1, sum(map(len, truth.values()))),
            "conflicts": conflicts,
            "index_build_seconds": round(build_seconds, 3),
            "runtime_seconds": round(time.monotonic() - start, 3),
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        }
    finally:
        index.close()


def replay(score_cache, config_path, minimum_margin=0.0):
    """Recompute decisions from cached scores without repeating retrieval."""
    with gzip.open(score_cache, "rt", encoding="utf-8") as file:
        cache = json.load(file)
    if cache["sample"] != "natural_development":
        raise ValueError("Only development score caches may be replayed")
    config = json.loads(config_path.read_text())
    truth = {sid: set(values) for sid, values in cache["truth"].items()}
    scored = cache["scored"]
    baseline = {}
    for sid, rows in scored.items():
        top = max((score for _, score in rows), default=0.0)
        cutoff = max(config["tau_min"], top - config["delta_margin"])
        baseline[sid] = ({target for target, score in rows if score >= cutoff}
                         if top >= config["tau_singleton"] else set())
    owned, conflicts = assign_ownership(scored, baseline, minimum_margin)
    return {
        "sample": cache["sample"], "limit": len(truth),
        "config_path": str(config_path),
        "thresholds": {key: config[key] for key in ("tau_singleton", "tau_min", "delta_margin")},
        "baseline": evaluate_metrics(truth, baseline),
        "baseline_false_positives": false_positive_breakdown(truth, baseline),
        "ownership": evaluate_metrics(truth, owned),
        "ownership_false_positives": false_positive_breakdown(truth, owned),
        "candidate_oracle_f05": calculate_macro_f05(
            truth, {sid: set(values) for sid, values in cache["oracle"].items()}),
        "conflicts": conflicts,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--index-path", type=Path, default=INDEX_PATH)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--minimum-margin", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=Path("experiments/codex_0991/ownership_dev200.json"))
    parser.add_argument("--score-cache", type=Path, default=Path("experiments/codex_0991/ownership_dev200_scores.json.gz"))
    parser.add_argument("--replay-config", type=Path,
                        help="Replay cached development scores using this config; skips retrieval")
    args = parser.parse_args()
    if not 1 <= args.limit <= 1000 or args.minimum_margin < 0:
        parser.error("limit must be 1..1000 and minimum-margin must be nonnegative")
    if args.replay_config:
        result = replay(args.score_cache, args.replay_config, args.minimum_margin)
    else:
        result = evaluate(args.limit, args.index_path, args.model_dir, args.minimum_margin, args.score_cache)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
