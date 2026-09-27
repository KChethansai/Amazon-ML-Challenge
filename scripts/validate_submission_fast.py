#!/usr/bin/env python3
"""Stream validation of ordered submission TSVs with optional disk-backed IDs."""

import argparse
from contextlib import ExitStack
from itertools import zip_longest
import os
import resource
import sqlite3
import sys
import tempfile
import time

DELIM = "\t"
MATCHING_HEADER = ["source1_entity_id", "matched_entity_ids"]
CANDIDATE_HEADER = ["source1_entity_id", "candidate_entity_ids"]
MAX_EXAMPLES = 5


def get_peak_memory_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


class Findings:
    def __init__(self):
        self.counts = {}
        self.samples = {}

    def add(self, message, example):
        self.counts[message] = self.counts.get(message, 0) + 1
        sample = self.samples.setdefault(message, [])
        if len(sample) < MAX_EXAMPLES:
            sample.append(str(example))

    def errors(self):
        return [f"{message}: {count:,} (e.g. {', '.join(self.samples[message])})"
                for message, count in self.counts.items()]


def _header(handle, expected, label, errors):
    line = handle.readline()
    if not line:
        errors.append(f"{label} is empty.")
        return False
    if DELIM not in line and "," in line:
        errors.append(f"{label} appears comma-separated; TSV is required.")
        return False
    columns = [part.strip().lower() for part in line.rstrip("\r\n").split(DELIM)]
    if columns != expected:
        errors.append(f"{label}: unexpected header {columns}. Expected {expected}.")
        return False
    return True


def _ids_from_row(line, label, line_num, findings):
    entity_id, tab, rest = line.rstrip("\r\n").partition(DELIM)
    if not tab or not entity_id:
        findings.add(f"{label}: malformed row", line_num)
        return entity_id, []
    values = rest.split(",") if rest else []
    if any(not value for value in values):
        findings.add(f"{label}: empty ID within list", line_num)
    if len(values) != len(set(values)):
        findings.add(f"{label}: repeated ID within list", entity_id)
    for value in values:
        if value.startswith("S1-"):
            findings.add(f"{label}: Source-1 self-match", value)
        elif not value.startswith(("S2-", "S3-")):
            findings.add(f"{label}: invalid target ID prefix", value)
    return entity_id, values


def _build_target_db(test_dir, db_path, errors):
    db = sqlite3.connect(db_path)
    db.execute("PRAGMA journal_mode=OFF")
    db.execute("PRAGMA synchronous=OFF")
    db.execute("PRAGMA temp_store=FILE")
    db.execute("PRAGMA cache_size=-16384")
    db.execute("CREATE TABLE ids (id TEXT PRIMARY KEY) WITHOUT ROWID")
    batch = []
    for name in ("test_source2.tsv", "test_source3.tsv"):
        path = os.path.join(test_dir, name)
        if not os.path.isfile(path):
            errors.append(f"Target ID check requires {path}.")
            db.close()
            return None
        with open(path, encoding="utf-8") as source:
            next(source, None)
            for line in source:
                batch.append((line.partition(DELIM)[0],))
                if len(batch) >= 10000:
                    db.executemany("INSERT OR IGNORE INTO ids VALUES (?)", batch)
                    db.commit()
                    batch.clear()
    if batch:
        db.executemany("INSERT OR IGNORE INTO ids VALUES (?)", batch)
        db.commit()
    return db


def _check_target_ids(db, values, label, findings):
    if db is None or not values:
        return
    for offset in range(0, len(values), 500):
        chunk = values[offset:offset + 500]
        present = {row[0] for row in db.execute(
            f"SELECT id FROM ids WHERE id IN ({','.join('?' for _ in chunk)})", chunk
        )}
        for value in chunk:
            if value not in present:
                findings.add(f"{label}: target ID absent from test Source-2/3", value)


def validate(matching_path, candidate_path, test_dir, check_ids=False,
             progress_interval=500000):
    errors, warnings, findings = [], [], Findings()
    source1_path = os.path.join(test_dir, "test_source1.tsv")
    if not os.path.isfile(source1_path):
        return [f"Test source1 file not found: {source1_path}"], warnings
    if not os.path.isfile(matching_path):
        return [f"File not found: {matching_path}"], warnings
    has_candidates = bool(candidate_path and os.path.isfile(candidate_path))
    if candidate_path and not has_candidates:
        return [f"File not found: {candidate_path}"], warnings
    if not check_ids:
        warnings.append("ID-existence check is OFF. Use --check-ids to verify target IDs.")

    start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="entity_validator_") as tmp, ExitStack() as stack:
        db = _build_target_db(test_dir, os.path.join(tmp, "targets.sqlite"), errors) if check_ids else None
        if errors:
            return errors, warnings
        if db is not None:
            stack.callback(db.close)
        source = stack.enter_context(open(source1_path, encoding="utf-8"))
        matching = stack.enter_context(open(matching_path, encoding="utf-8"))
        candidate = stack.enter_context(open(candidate_path, encoding="utf-8")) if has_candidates else None
        if not _header(matching, MATCHING_HEADER, "matching_results.tsv", errors):
            return errors, warnings
        if candidate and not _header(candidate, CANDIDATE_HEADER, "candidate_pairs.tsv", errors):
            return errors, warnings
        next(source, None)
        source_rows = matched_links = candidate_links = empty_matches = empty_candidates = 0
        streams = (source, matching, candidate) if candidate else (source, matching)
        for line_num, rows in enumerate(zip_longest(*streams), start=2):
            s1_line, matching_line = rows[:2]
            candidate_line = rows[2] if candidate else None
            if s1_line is None:
                findings.add("Output row without required S1 row", line_num)
            else:
                source_rows += 1
            expected = s1_line.partition(DELIM)[0] if s1_line else None
            if matching_line is None:
                findings.add("Missing matching row", expected or line_num)
                matched_id, matched_values = None, []
            else:
                matched_id, matched_values = _ids_from_row(
                    matching_line, "matching_results.tsv", line_num, findings)
                if matched_id != expected:
                    findings.add("Matching S1 order/coverage mismatch", f"line {line_num}: {matched_id} != {expected}")
                matched_links += len(matched_values)
                empty_matches += not matched_values
                _check_target_ids(db, matched_values, "matching_results.tsv", findings)
            if candidate:
                if candidate_line is None:
                    findings.add("Missing candidate row", expected or line_num)
                else:
                    candidate_id, candidate_values = _ids_from_row(
                        candidate_line, "candidate_pairs.tsv", line_num, findings)
                    if candidate_id != expected:
                        findings.add("Candidate S1 order/coverage mismatch", f"line {line_num}: {candidate_id} != {expected}")
                    candidate_links += len(candidate_values)
                    empty_candidates += not candidate_values
                    _check_target_ids(db, candidate_values, "candidate_pairs.tsv", findings)
                    if matched_id == candidate_id and not set(matched_values).issubset(candidate_values):
                        findings.add("Final matches absent from candidate_pairs.tsv", matched_id)
            if progress_interval and source_rows and source_rows % progress_interval == 0:
                print(f"    Processed {source_rows:,} S1 rows ({time.monotonic()-start:.1f}s; "
                      f"peak RSS {get_peak_memory_mb():.1f} MB)", flush=True)
    errors.extend(findings.errors())
    print(f"  {source_rows:,} S1 rows; {matched_links:,} matches ({empty_matches:,} empty rows); "
          f"{candidate_links:,} candidates ({empty_candidates:,} empty rows) "
          f"in {time.monotonic()-start:.2f}s.")
    return errors, warnings


def main():
    parser = argparse.ArgumentParser(description="Fast Amazon ML Challenge 2026 Submission Validator")
    parser.add_argument("--matching", "-m", default="output/matching_results.tsv")
    parser.add_argument("--candidate", "-c", default="output/candidate_pairs.tsv")
    parser.add_argument("--test-dir", "-t", default="student_resource/dataset/test")
    parser.add_argument("--check-ids", action="store_true")
    parser.add_argument("--progress-interval", type=int, default=500000)
    args = parser.parse_args()
    start = time.monotonic()
    try:
        errors, warnings = validate(args.matching, args.candidate, args.test_dir,
                                    args.check_ids, args.progress_interval)
    except (UnicodeDecodeError, OSError, sqlite3.Error) as exc:
        errors, warnings = [f"Could not validate file: {exc}"], []
    print(f"Elapsed: {time.monotonic()-start:.2f}s; peak RSS: {get_peak_memory_mb():.1f} MB")
    for warning in warnings:
        print(f"WARNING: {warning}")
    if errors:
        print(f"FAIL — {len(errors)} issue(s):")
        for error in errors:
            print(f"  {error}")
        return 1
    print("PASS — formatting, ordered coverage, ID and candidate subset checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
