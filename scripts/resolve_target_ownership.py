#!/usr/bin/env python3
"""Resolve accepted S1-target pairs by model probability, then emit all S1 rows."""

import argparse
import csv
import json
import os
import sqlite3
import tempfile
from pathlib import Path


def select_owner(choices, min_probability, min_margin):
    """Return (S1, reason) for accepted competitors of one target."""
    if not choices:
        return None, "empty"
    choices = sorted(choices, key=lambda pair: (-pair[1], pair[0]))
    if choices[0][1] < min_probability:
        return None, "probability"
    if len(choices) > 1 and choices[0][1] - choices[1][1] < min_margin:
        return None, "margin"
    return choices[0][0], "accepted"


def assert_global_unique(path):
    seen = set()
    duplicates = 0
    with path.open(encoding="utf-8") as file:
        next(file)
        for line in file:
            for target in filter(None, line.rstrip("\r\n").partition("\t")[2].split(",")):
                if target in seen:
                    duplicates += 1
                seen.add(target)
    if duplicates:
        raise ValueError(f"Global target duplicate count is {duplicates}, expected zero")
    return len(seen)


def resolve(matching, scores, output, min_probability, min_margin):
    if output.resolve() in (matching.resolve(), scores.resolve()):
        raise ValueError("Output must differ from the matching and score inputs")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ownership_", dir=output.parent) as temp:
        db = sqlite3.connect(Path(temp) / "ownership.sqlite")
        db.execute("PRAGMA journal_mode=OFF")
        db.execute("PRAGMA synchronous=OFF")
        db.execute("PRAGMA temp_store=FILE")
        db.execute("PRAGMA cache_size=-65536")
        db.execute("CREATE TABLE scored(target TEXT, sid TEXT, p REAL, PRIMARY KEY(target,sid)) WITHOUT ROWID")
        count = 0
        with scores.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file, delimiter="\t")
            needed = {"target_id", "candidate_s1_id", "model_probability"}
            if not needed <= set(reader.fieldnames or ()):
                raise ValueError("Score stream lacks target_id, candidate_s1_id or model_probability")
            for row in reader:
                p = float(row["model_probability"])
                if not 0 <= p <= 1:
                    raise ValueError("Model probability outside [0, 1]")
                db.execute("INSERT INTO scored VALUES (?,?,?)", (row["target_id"], row["candidate_s1_id"], p))
                count += 1
                if count % 100_000 == 0:
                    db.commit()
        db.commit()
        db.execute("CREATE TABLE owners(target TEXT PRIMARY KEY, sid TEXT) WITHOUT ROWID")
        summary = {"score_rows": count, "conflicted_targets_before": 0,
                   "rejected_probability": 0, "rejected_margin": 0, "owned_targets": 0}
        current, choices = None, []

        def finish(target, competitors):
            if target is None:
                return
            summary["conflicted_targets_before"] += len(competitors) > 1
            owner, reason = select_owner(competitors, min_probability, min_margin)
            if owner:
                db.execute("INSERT INTO owners VALUES (?,?)", (target, owner))
                summary["owned_targets"] += 1
            elif reason == "probability":
                summary["rejected_probability"] += 1
            elif reason == "margin":
                summary["rejected_margin"] += 1

        for target, sid, p in db.execute("SELECT target,sid,p FROM scored ORDER BY target,p DESC,sid"):
            if target != current:
                finish(current, choices)
                current, choices = target, []
            choices.append((sid, p))
        finish(current, choices)
        db.commit()
        db.execute("CREATE INDEX owners_sid ON owners(sid)")
        partial = output.with_suffix(output.suffix + ".partial")
        rows = assignments = empty = input_assignments = maximum = 0
        with matching.open(encoding="utf-8") as source, partial.open("w", encoding="utf-8") as dest:
            if next(source).rstrip("\r\n") != "source1_entity_id\tmatched_entity_ids":
                raise ValueError("Unexpected matching input header")
            dest.write("source1_entity_id\tmatched_entity_ids\n")
            for line in source:
                sid, tab, old = line.rstrip("\r\n").partition("\t")
                if not tab:
                    raise ValueError("Malformed matching row")
                for target in filter(None, old.split(",")):
                    if db.execute("SELECT 1 FROM scored WHERE target=? AND sid=?", (target, sid)).fetchone() is None:
                        raise ValueError(f"Accepted pair lacks model probability: {sid}, {target}")
                    input_assignments += 1
                matches = [target for (target,) in db.execute("SELECT target FROM owners WHERE sid=? ORDER BY target", (sid,))]
                dest.write(sid + "\t" + ",".join(matches) + "\n")
                rows += 1
                assignments += len(matches)
                maximum = max(maximum, len(matches))
                empty += not matches
        if input_assignments != count:
            raise ValueError("Score rows do not match accepted assignments in the input")
        if assignments != summary["owned_targets"]:
            raise ValueError("Owned target count differs from written assignments")
        if assert_global_unique(partial) != assignments:
            raise ValueError("Written assignment count differs from unique targets")
        summary.update({"s1_rows": rows, "assignments": assignments,
                        "unique_targets": assignments, "cross_s1_duplicate_assignments": 0,
                        "targets_with_multiple_s1": 0, "empty_s1_rows": empty,
                        "mean_assignments_per_s1": assignments / max(rows, 1),
                        "max_assignments_per_s1": maximum})
        # A target is a primary key in owners; global duplicates are impossible here.
        os.replace(partial, output)
        db.close()
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matching", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--winning-probability", type=float, required=True)
    parser.add_argument("--conflict-margin", type=float, required=True)
    args = parser.parse_args()
    if not 0 <= args.winning_probability <= 1 or not 0 <= args.conflict_margin <= 1:
        parser.error("Probability and margin must be within [0, 1]")
    print(json.dumps(resolve(args.matching, args.scores, args.output,
                             args.winning_probability, args.conflict_margin), sort_keys=True))


if __name__ == "__main__":
    main()
