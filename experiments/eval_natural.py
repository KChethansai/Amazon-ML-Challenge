#!/usr/bin/env python3
"""Full-target natural-India gate, with development and untouched confirmation samples."""

import argparse
import csv
import json
import resource
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

sys.path.insert(0, "solution/business_entity_resolution/src")
from diagnostics import candidate_diagnostic, summarize_retrieval
from disk_blocking import DiskCountryBlockingIndex
from normalization import prepare_source_record
from train import ALL_FEATURE_NAMES, calculate_macro_f05, evaluate_metrics, featurize_s1, query_for_s1


TRAIN_DIR = Path("dataset/train")
INDEX_PATH = Path("experiments/final_scratch/india_train.sqlite")
MODEL_DIR = Path("experiments/final_scratch")
EXPECTED_INDIA_TARGETS = 4_133_346
SAMPLE_SIZE = 1000


def source_rows(path):
    with path.open(encoding="utf-8") as file:
        header = next(file).rstrip("\r\n").split("\t")
        if header[:4] != ["entity_id", "business_name", "business_address", "country"]:
            raise ValueError(f"Unexpected source schema in {path}: {header}")
        for line in file:
            values = line.rstrip("\r\n").split("\t")
            if len(values) != 4:
                raise ValueError(f"Malformed source row in {path}")
            yield values


def split_sample(sample):
    with Path("experiments/splits/holdout_s1_ids.json").open() as file:
        holdout = set(json.load(file))
    india = [(sid, prepare_source_record(name, address, country))
             for sid, name, address, country in source_rows(TRAIN_DIR / "train_source1.tsv")
             if sid in holdout and country == "India"]
    start = 0 if sample == "development" else SAMPLE_SIZE
    selected = india[start:start + SAMPLE_SIZE]
    if len(selected) != SAMPLE_SIZE:
        raise ValueError(f"Only {len(selected)} India S1 records in {sample} sample")
    return selected


def load_truth(ids):
    wanted = set(ids)
    result = {}
    with (TRAIN_DIR / "train_ground_truth.tsv").open(encoding="utf-8") as file:
        next(file)
        for line in file:
            sid, _, targets = line.rstrip("\r\n").partition("\t")
            if sid in wanted:
                result[sid] = set(target.strip() for target in targets.split(",") if target.strip())
    if set(result) != wanted:
        raise ValueError(f"Ground truth missing {len(wanted - set(result))} S1 IDs")
    return result


def ensure_index(index_path):
    if index_path.is_file():
        return DiskCountryBlockingIndex("India", index_path), 0.0
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index = DiskCountryBlockingIndex("India", index_path, create=True)
    source_paths = [TRAIN_DIR / "train_source2.tsv", TRAIN_DIR / "train_source3.tsv"]
    begin = time.monotonic()
    visited = 0
    try:
        if hasattr(index, "set_source_fingerprint"):
            index.set_source_fingerprint(source_paths)
        for path, is_s2 in zip(source_paths, (1, 0)):
            for eid, name, address, country in source_rows(path):
                if country != "India":
                    continue
                record = prepare_source_record(name, address, country)
                index.add_target_record(
                    eid, record["norm_name"], record["core_tokens"],
                    record["norm_addr"], record["addr_keys"], record["digits"],
                    is_s2=is_s2, raw_addr=address, raw_name=name)
                visited += 1
                if visited % 250_000 == 0:
                    print(f"indexed/verified {visited:,} India targets in {time.monotonic()-begin:.0f}s", flush=True)
        if visited != EXPECTED_INDIA_TARGETS:
            raise ValueError(f"India target count {visited:,} != expected {EXPECTED_INDIA_TARGETS:,}")
        index.prune_frequent_keys()
        return index, time.monotonic() - begin
    except BaseException:
        index.close()
        raise


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(sample="development", index_path=INDEX_PATH, model_dir=MODEL_DIR, limit=None):
    started = time.monotonic()
    selected = split_sample(sample)
    if limit is not None:
        selected = selected[:limit]
    truth = load_truth(sid for sid, _ in selected)
    index, index_build_seconds = ensure_index(Path(index_path))
    try:
        if len(index.records) != EXPECTED_INDIA_TARGETS:
            raise ValueError(f"Index contains {len(index.records):,} targets, expected {EXPECTED_INDIA_TARGETS:,}")
        with (Path(model_dir) / "config.json").open() as file:
            config = json.load(file)
        model = lgb.Booster(model_file=str(Path(model_dir) / "lgbm_matcher.txt"))
        expected_features = config["feature_names"]
        if expected_features != ALL_FEATURE_NAMES or model.num_feature() != len(expected_features):
            raise ValueError("Model/config/runtime feature order mismatch")
        gt_map, pred_map, oracle_map = {}, {}, {}
        diagnostics = []
        query_seconds = feature_seconds = model_seconds = 0.0
        for count, (sid, record) in enumerate(selected, 1):
            gt = truth[sid]
            stamp = time.monotonic()
            items, provenance = query_for_s1(index, record, max_candidates=config.get("max_candidates", 60), adaptive=config.get("adaptive", True))
            query_seconds += time.monotonic() - stamp
            row = candidate_diagnostic(sid, record, gt, items, provenance)
            diagnostics.append(row)
            gt_map[sid] = gt
            oracle_map[sid] = gt & {candidate for candidate, _ in items}
            stamp = time.monotonic()
            candidate_ids, features = featurize_s1(record, items, provenance, index, use_extra=config.get("use_extra", True))
            feature_seconds += time.monotonic() - stamp
            if features:
                if len(features[0]) != len(expected_features):
                    raise ValueError(f"Feature order/length mismatch for {sid}")
                stamp = time.monotonic()
                scores = model.predict(np.asarray(features, dtype=np.float32))
                model_seconds += time.monotonic() - stamp
                top = float(np.max(scores))
                threshold = max(config["tau_min"], top - config["delta_margin"])
                pred_map[sid] = ({candidate for candidate, score in zip(candidate_ids, scores) if score >= threshold}
                                 if top >= config["tau_singleton"] else set())
            else:
                pred_map[sid] = set()
            if count % 100 == 0:
                print(f"scored {count:,}/{len(selected):,} S1 in {time.monotonic()-started:.1f}s", flush=True)
        if config.get("use_graph", False):
            raise ValueError("This evaluator has no graph bridge path; EXP_020 config must use_graph=false")
        metrics = evaluate_metrics(gt_map, pred_map)
        retrieval = summarize_retrieval(diagnostics)
        metrics.update(retrieval)
        metrics.update({
            "oracle_macro_f05": calculate_macro_f05(gt_map, oracle_map),
            "oracle_precision": 1.0,
            "oracle_recall": sum(len(v) for v in oracle_map.values()) / max(1, sum(len(v) for v in gt_map.values())),
            "sample": sample,
            "sample_role": "development" if sample == "development" else "untouched_confirmation",
            "index_path": str(index_path),
            "index_targets": len(index.records),
            "index_bytes": Path(index_path).stat().st_size,
            "index_build_seconds": round(index_build_seconds, 3),
            "s1_ids": [sid for sid, _ in selected],
            "model_dir": str(model_dir),
            "query_seconds": round(query_seconds, 3),
            "feature_seconds": round(feature_seconds, 3),
            "model_seconds": round(model_seconds, 3),
            "runtime_seconds": round(time.monotonic() - started, 3),
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        })
        if sample == "development":
            write_csv(Path("experiments/candidate_recall_by_entity.csv"), diagnostics)
            write_csv(Path("experiments/candidate_failure_modes.csv"), [row for row in diagnostics if row["gt_count"] and not row["entity_full_recall"]])
        out = Path("experiments") / f"eval_natural_{sample}.json"
        out.write_text(json.dumps(metrics, indent=2) + "\n")
        print(json.dumps({key: value for key, value in metrics.items() if key != "s1_ids"}, indent=2), flush=True)
        return metrics
    finally:
        index.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", choices=("development", "confirmation"), default="development")
    parser.add_argument("--index-path", type=Path, default=INDEX_PATH)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    evaluate(args.sample, args.index_path, args.model_dir, limit=args.limit)
