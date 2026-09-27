#!/usr/bin/env python3
"""Read-only, train-only Indic dictionary and natural retrieval ceiling probe."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "solution/business_entity_resolution/src"))

from eval_natural import INDEX_PATH, load_truth, source_rows, split_sample
from disk_blocking import DiskCountryBlockingIndex
from normalization import normalize_name
from train import calculate_macro_f05, query_for_s1
from views import has_indic, transliterate_text

TRAIN = ROOT / "dataset/train"
HOLDOUT = ROOT / "experiments/splits/holdout_s1_ids.json"


def learn_mapping(pairs, min_count=3, min_share=0.6):
    """Positional token alignments with unique dominant English counterpart."""
    aligned = Counter()
    totals = Counter()
    for indic, english in pairs:
        left, right = indic.split(), english.split()
        if len(left) != len(right):
            continue
        for token, canonical in zip(left, right):
            if len(token) < 3 or len(canonical) < 3 or not token.isascii() or not canonical.isascii():
                continue
            totals[token] += 1
            aligned[token, canonical] += 1
    choices = defaultdict(list)
    for (token, canonical), count in aligned.items():
        choices[token].append((count, canonical))
    result = {}
    for token, candidates in choices.items():
        candidates.sort(key=lambda item: (-item[0], item[1]))
        count, canonical = candidates[0]
        if (canonical != token and count >= min_count and
                count / totals[token] >= min_share and
                (len(candidates) == 1 or count > candidates[1][0])):
            result[token] = canonical
    return dict(sorted(result.items()))


def reverse_variants(mapping):
    reverse = defaultdict(list)
    for transliterated, canonical in mapping.items():
        reverse[canonical].append(transliterated)
    return {canonical: tuple(sorted(set(variants))) for canonical, variants in sorted(reverse.items())}


def training_pairs():
    holdout = set(json.loads(HOLDOUT.read_text()))
    s1_names = {}
    for sid, name, _, country in source_rows(TRAIN / "train_source1.tsv"):
        if country == "India" and sid not in holdout:
            s1_names[sid] = normalize_name(name)
    target_to_s1 = {}
    with (TRAIN / "train_ground_truth.tsv").open(encoding="utf-8") as file:
        next(file)
        for line in file:
            sid, _, targets = line.rstrip("\r\n").partition("\t")
            if sid in s1_names:
                for target in targets.split(","):
                    if target:
                        target_to_s1[target] = sid
    count = 0
    for source in ("train_source2.tsv", "train_source3.tsv"):
        for eid, name, _, country in source_rows(TRAIN / source):
            if country != "India" or eid not in target_to_s1 or not has_indic(name):
                continue
            indic = normalize_name(transliterate_text(name))
            english = s1_names[target_to_s1[eid]]
            if indic and english:
                count += 1
                yield indic, english


def probe(mapping, limit, max_variants=3, max_postings=5000):
    reverse = reverse_variants(mapping)
    selected = split_sample("development")[:limit]
    truth = load_truth(sid for sid, _ in selected)
    baseline, expanded = {}, {}
    index = DiskCountryBlockingIndex("India", INDEX_PATH)
    try:
        for i, (sid, record) in enumerate(selected, 1):
            candidates, _ = query_for_s1(index, record, max_candidates=60, adaptive=True)
            initial = {eid for eid, _ in candidates}
            rids = set()
            for token in sorted(set(record["core_tokens"])):
                for variant in reverse.get(token, ())[:max_variants]:
                    count = index.db.execute(
                        "SELECT n FROM key_counts WHERE channel='tr_tok' AND key=?",
                        (json.dumps(variant, ensure_ascii=False),)).fetchone()
                    if count and count[0] <= max_postings:
                        rids.update(index.tr_token_idx[variant])
            extra = set(index._eids(rids).values()) - initial
            baseline[sid] = truth[sid] & initial
            expanded[sid] = truth[sid] & (initial | extra)
            if i % 25 == 0:
                print(f"probed {i}/{limit}", flush=True)
        return {
            "sample": "natural_development",
            "limit": len(selected),
            "scope": "unranked oracle upper bound; extra candidates are not production scored",
            "baseline_oracle_macro_f05": calculate_macro_f05(truth, baseline),
            "expanded_oracle_macro_f05": calculate_macro_f05(truth, expanded),
            "baseline_pair_recall": sum(map(len, baseline.values())) / max(1, sum(map(len, truth.values()))),
            "expanded_pair_recall": sum(map(len, expanded.values())) / max(1, sum(map(len, truth.values()))),
            "recovered_true_targets": sum(len(expanded[sid] - baseline[sid]) for sid in truth),
            "dictionary_entries": len(mapping),
            "max_variants": max_variants,
            "max_postings": max_postings,
        }
    finally:
        index.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--dictionary", type=Path, default=ROOT / "experiments/codex_0991/indic_dictionary.json")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/codex_0991/indic_probe_dev100.json")
    parser.add_argument("--learn-only", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()
    if args.dictionary.exists():
        mapping = json.loads(args.dictionary.read_text())
    else:
        mapping = learn_mapping(training_pairs())
        args.dictionary.write_text(json.dumps(mapping, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    if args.learn_only:
        print(json.dumps({"dictionary_entries": len(mapping),
                          "runtime_seconds": round(time.monotonic()-started, 2),
                          "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}))
        return
    result = probe(mapping, args.limit)
    result["runtime_seconds"] = round(time.monotonic()-started, 2)
    result["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
