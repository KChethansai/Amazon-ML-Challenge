#!/usr/bin/env python3
"""
Fast Submission Validator for Amazon ML Challenge 2026.

Validates output/matching_results.tsv and output/candidate_pairs.tsv against
all official competition constraints using memory-efficient streaming.

Features:
- Identical validation rules and error reporting as official validator.
- Single-pass streaming of candidate_pairs.tsv (never materializes 36M candidate objects in memory).
- Checks cross-file candidate-subset guarantee on the fly.
- Periodic progress, memory RSS, and execution timing.
- Optional --check-ids to verify all S2/S3 IDs exist in test set.
- Exit code 0 on success, 1 on failure.
"""

import argparse
import os
import sys
import time
import resource

DELIM = "\t"
MAX_EXAMPLES = 5
MATCHING_HEADER = ["source1_entity_id", "matched_entity_ids"]
CANDIDATE_HEADER = ["source1_entity_id", "candidate_entity_ids"]

def get_peak_memory_mb():
    """Return peak resident memory (RSS) in MB for this process."""
    # On Linux, ru_maxrss is in kilobytes
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return usage.ru_maxrss / 1024.0
    except Exception:
        return 0.0

def examples(items):
    """Return a short sample of offending IDs for error messages."""
    items = sorted(items)
    shown = ", ".join(items[:MAX_EXAMPLES])
    if len(items) > MAX_EXAMPLES:
        return f"{len(items)} total, e.g. {shown}, ..."
    return shown

def read_ids(path):
    """Read first-column entity IDs from a source TSV."""
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        next(f, None)
        return {line.split(DELIM, 1)[0].strip() for line in f if line.strip()}

def load_match_targets(test_dir, warnings):
    """Return the set of valid S2/S3 match IDs if --check-ids is enabled."""
    targets = set()
    for name in ("test_source2.tsv", "test_source3.tsv"):
        path = os.path.join(test_dir, name)
        if not os.path.isfile(path):
            warnings.append(
                f"{path} not found — skipping ID-existence check. "
                "Provide test_source2.tsv and test_source3.tsv to enable it."
            )
            return None
        ids = read_ids(path)
        if ids:
            targets.update(ids)
    return targets

def validate_matching_file(path, required_s1, valid_ids, errors, progress_interval=500000):
    """
    Validate matching_results.tsv and return a dict {s1_id: set(matched_ids)}.
    Because matching results contains only ~4.6M links, this dict takes only ~200-300MB RAM.
    """
    if not os.path.isfile(path):
        errors.append(f"File not found: {path}")
        return None

    name = os.path.basename(path)
    mapping = {}
    seen = set()
    dup_rows = set()
    intra_dupes = set()
    self_matches = set()
    wrong_prefix = set()
    unknown = set()
    n_rows = empties = 0
    total_links = 0

    t_start = time.time()
    with open(path, "r", encoding="utf-8") as f:
        header_line = f.readline()
        if not header_line:
            errors.append(f"{name} is empty.")
            return None
        if DELIM not in header_line and "," in header_line:
            errors.append(
                f"{name}: header has no TAB but contains commas — file looks comma-separated (.csv). "
                "Submissions must be tab-separated (.tsv)."
            )
            return None
        cols = [c.strip().lower() for c in header_line.rstrip("\r\n").split(DELIM)]
        if cols != MATCHING_HEADER:
            errors.append(f"{name}: unexpected header {cols}. Expected exactly {MATCHING_HEADER}.")
            return None

        for line_num, line in enumerate(f, start=2):
            s1, tab, rest = line.partition(DELIM)
            if not tab:
                if s1.strip():
                    errors.append(f"{name}: malformed row (no tab) at line {line_num}: {line.rstrip()!r}")
                continue

            n_rows += 1
            if s1 in seen:
                dup_rows.add(s1)
            seen.add(s1)

            ids = rest.rstrip("\r\n").split(",") if rest.strip() else []
            if not ids:
                empties += 1
                mapping[s1] = set()
            else:
                if len(ids) != len(set(ids)):
                    intra_dupes.add(s1)
                id_set = set(ids)
                mapping[s1] = id_set
                total_links += len(ids)
                for mid in id_set:
                    if mid.startswith("S1-"):
                        self_matches.add(mid)
                    elif not (mid.startswith("S2-") or mid.startswith("S3-")):
                        wrong_prefix.add(mid)
                    elif valid_ids is not None and mid not in valid_ids:
                        unknown.add(mid)

            if progress_interval and n_rows % progress_interval == 0:
                elapsed = time.time() - t_start
                print(f"    [{name}] Processed {n_rows:,} rows ({elapsed:.1f}s, RSS: {get_peak_memory_mb():.1f} MB)...", flush=True)

    findings = [
        (dup_rows, "{name}: duplicate source1_entity_id row(s): {ex}. Each S1 entity may appear on only one row."),
        (intra_dupes, "{name}: repeated ID inside matched_entity_ids list for: {ex}. No duplicate IDs allowed within a list."),
        (self_matches, "{name}: matched_entity_ids contains Source-1 IDs (self-matches): {ex}. Only S2-/S3- allowed."),
        (wrong_prefix, "{name}: matched_entity_ids contains IDs without an S2-/S3- prefix: {ex}."),
        (unknown, "{name}: matched_entity_ids references IDs not in test Source-2/3 files: {ex}."),
        (required_s1 - seen, "{name}: required S1 entity(ies) missing: {ex}. Every entity in test_source1.tsv needs a row."),
        (seen - required_s1, "{name}: row(s) using an S1 ID that is not in the test set: {ex}."),
    ]
    for offenders, msg in findings:
        if offenders:
            errors.append(msg.format(name=name, ex=examples(offenders)))

    print(f"  {name}: {n_rows:,} rows ({empties:,} empty, {n_rows - empties:,} non-empty, {total_links:,} links) in {time.time()-t_start:.2f}s.")
    return mapping

def validate_candidate_file_streaming(path, required_s1, valid_ids, matched_map, errors, warnings, progress_interval=500000):
    """
    Validate candidate_pairs.tsv in a SINGLE streaming pass without retaining candidate sets.
    Checks all individual rules and verifies on the fly that all matched_ids are a subset of candidates.
    """
    if not os.path.isfile(path):
        warnings.append(
            f"{path} not found — skipping candidate_pairs.tsv checks. "
            "It is optional for local scoring but expected in the final submission zip."
        )
        return

    name = os.path.basename(path)
    seen = set()
    dup_rows = set()
    intra_dupes = set()
    self_matches = set()
    wrong_prefix = set()
    unknown = set()
    subset_violations = set()
    n_rows = empties = 0
    total_candidates = 0

    t_start = time.time()
    with open(path, "r", encoding="utf-8") as f:
        header_line = f.readline()
        if not header_line:
            errors.append(f"{name} is empty.")
            return
        if DELIM not in header_line and "," in header_line:
            errors.append(
                f"{name}: header has no TAB but contains commas — file looks comma-separated (.csv). "
                "Submissions must be tab-separated (.tsv)."
            )
            return
        cols = [c.strip().lower() for c in header_line.rstrip("\r\n").split(DELIM)]
        if cols != CANDIDATE_HEADER:
            errors.append(f"{name}: unexpected header {cols}. Expected exactly {CANDIDATE_HEADER}.")
            return

        for line_num, line in enumerate(f, start=2):
            s1, tab, rest = line.partition(DELIM)
            if not tab:
                if s1.strip():
                    errors.append(f"{name}: malformed row (no tab) at line {line_num}: {line.rstrip()!r}")
                continue

            n_rows += 1
            if s1 in seen:
                dup_rows.add(s1)
            seen.add(s1)

            ids = rest.rstrip("\r\n").split(",") if rest.strip() else []
            if not ids:
                empties += 1
                if matched_map and s1 in matched_map and matched_map[s1]:
                    subset_violations.add(s1)
            else:
                if len(ids) != len(set(ids)):
                    intra_dupes.add(s1)
                id_set = set(ids)
                total_candidates += len(ids)
                for mid in id_set:
                    if mid.startswith("S1-"):
                        self_matches.add(mid)
                    elif not (mid.startswith("S2-") or mid.startswith("S3-")):
                        wrong_prefix.add(mid)
                    elif valid_ids is not None and mid not in valid_ids:
                        unknown.add(mid)

                # Streaming cross-file check: matched IDs must be a subset of candidate IDs
                if matched_map and s1 in matched_map:
                    missing_from_candidates = matched_map[s1] - id_set
                    if missing_from_candidates:
                        subset_violations.add(s1)

            if progress_interval and n_rows % progress_interval == 0:
                elapsed = time.time() - t_start
                print(f"    [{name}] Processed {n_rows:,} rows ({elapsed:.1f}s, RSS: {get_peak_memory_mb():.1f} MB)...", flush=True)

    findings = [
        (dup_rows, "{name}: duplicate source1_entity_id row(s): {ex}. Each S1 entity may appear on only one row."),
        (intra_dupes, "{name}: repeated ID inside candidate_entity_ids list for: {ex}. No duplicate IDs allowed within a list."),
        (self_matches, "{name}: candidate_entity_ids contains Source-1 IDs (self-matches): {ex}. Only S2-/S3- allowed."),
        (wrong_prefix, "{name}: candidate_entity_ids contains IDs without an S2-/S3- prefix: {ex}."),
        (unknown, "{name}: candidate_entity_ids references IDs not in test Source-2/3 files: {ex}."),
        (required_s1 - seen, "{name}: required S1 entity(ies) missing: {ex}. Every entity in test_source1.tsv needs a row."),
        (seen - required_s1, "{name}: row(s) using an S1 ID that is not in the test set: {ex}."),
    ]
    for offenders, msg in findings:
        if offenders:
            errors.append(msg.format(name=name, ex=examples(offenders)))

    if subset_violations:
        warnings.append(
            f"{len(subset_violations)} S1 entity(ies) have matched IDs not present in "
            f"candidate_pairs.tsv, e.g. {examples(subset_violations)}. Final matches "
            "normally come from your blocking candidates — double-check these."
        )

    print(f"  {name}: {n_rows:,} rows ({empties:,} empty, {n_rows - empties:,} non-empty, {total_candidates:,} candidates) in {time.time()-t_start:.2f}s.")

def validate(matching_path, candidate_path, test_dir, check_ids=False, progress_interval=500000):
    errors = []
    warnings = []
    
    t_start = time.time()
    source1_path = os.path.join(test_dir, "test_source1.tsv")
    if not os.path.isfile(source1_path):
        errors.append(f"Test source1 file not found: {source1_path} (check --test-dir).")
        return errors, warnings

    print("Step 1: Loading required S1 test entities...", flush=True)
    required = read_ids(source1_path)
    print(f"  Loaded {len(required):,} required S1 entities ({time.time()-t_start:.2f}s).", flush=True)

    valid_ids = None
    if check_ids:
        print("Step 1b: Loading Source-2 and Source-3 IDs for existence verification...", flush=True)
        valid_ids = load_match_targets(test_dir, warnings)
        if valid_ids is not None:
            print(f"  Loaded {len(valid_ids):,} valid S2/S3 target IDs.", flush=True)
    else:
        warnings.append(
            "ID-existence check is OFF (default). Use --check-ids to verify all S2/S3 IDs exist in test set."
        )

    print("Step 2: Validating matching_results.tsv...", flush=True)
    matched_map = validate_matching_file(
        matching_path, required, valid_ids, errors, progress_interval=progress_interval
    )

    if candidate_path:
        print("Step 3: Streaming validation of candidate_pairs.tsv & subset check...", flush=True)
        validate_candidate_file_streaming(
            candidate_path, required, valid_ids, matched_map, errors, warnings, progress_interval=progress_interval
        )

    return errors, warnings

def main():
    parser = argparse.ArgumentParser(description="Fast Amazon ML Challenge 2026 Submission Validator")
    parser.add_argument("--matching", "-m", default="output/matching_results.tsv", help="Path to matching_results.tsv")
    parser.add_argument("--candidate", "-c", default="output/candidate_pairs.tsv", help="Path to candidate_pairs.tsv")
    parser.add_argument("--test-dir", "-t", default="student_resource/dataset/test", help="Folder containing test_source1/2/3.tsv")
    parser.add_argument("--check-ids", action="store_true", help="Enable ID existence check against test_source2/3.tsv")
    parser.add_argument("--progress-interval", type=int, default=500000, help="Print progress every N rows (0 to disable)")
    args = parser.parse_args()

    # Fallback search for test_dir if default is not found directly
    test_dir = args.test_dir
    if not os.path.isdir(test_dir):
        fallbacks = [
            "student_resource/dataset/test",
            "dataset/test",
            "../student_resource/dataset/test",
            "../../student_resource/dataset/test",
        ]
        for fb in fallbacks:
            if os.path.isdir(fb):
                test_dir = fb
                break

    print("=" * 70)
    print(" Amazon ML Challenge 2026 — Fast Submission Validator")
    print(f" Matching  : {args.matching}")
    print(f" Candidate : {args.candidate}")
    print(f" Test Dir  : {test_dir}")
    print("=" * 70)

    start_time = time.time()
    try:
        errors, warnings = validate(
            args.matching, args.candidate, test_dir,
            check_ids=args.check_ids, progress_interval=args.progress_interval
        )
    except UnicodeDecodeError:
        print("\nFAIL — 1 issue(s) to fix before submitting:")
        print("  1. A file is not valid UTF-8 text. Re-save as UTF-8 tab-separated .tsv.")
        return 1
    except OSError as exc:
        print("\nFAIL — 1 issue(s) to fix before submitting:")
        print(f"  1. Could not read a file: {exc}")
        return 1

    total_time = time.time() - start_time
    peak_rss = get_peak_memory_mb()

    print("\n" + "=" * 70)
    print(" Validation Results")
    print("=" * 70)
    print(f" Elapsed Time : {total_time:.2f} seconds")
    print(f" Peak Memory  : {peak_rss:.1f} MB RSS")
    print("-" * 70)

    for warning in warnings:
        print(f" WARNING: {warning}")

    if errors:
        print(f"\n FAIL — {len(errors)} issue(s) to fix before submitting:")
        for idx, error in enumerate(errors, 1):
            print(f"  {idx}. {error}")
        return 1

    print("\n PASS — All formatting, schema, entity-coverage, and subset rules passed!")
    print(" Safe to submit.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
