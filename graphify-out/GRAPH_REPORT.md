# Graph Report - ML Challenge  (2026-09-26)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 240 nodes · 457 edges · 15 communities (13 shown, 2 thin omitted)
- Extraction: 86% EXTRACTED · 14% INFERRED · 0% AMBIGUOUS · INFERRED: 62 edges (avg confidence: 0.86)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `8ac52e35`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- normalize_address
- views.py
- CountryBlockingIndex
- src/features.py
- rows
- validate_submission_fast.py
- validate
- main
- CleanMultiPassIndex
- MultiPassBlockingIndex
- main
- main
- create_validation_split.py

## God Nodes (most connected - your core abstractions)
1. `CountryBlockingIndex` - 17 edges
2. `rows()` - 14 edges
3. `normalize_address()` - 12 edges
4. `normalize_name()` - 12 edges
5. `extra_for_pair()` - 12 edges
6. `transliterate_text()` - 12 edges
7. `CountryBlockingIndex` - 11 edges
8. `extract_name_tokens()` - 11 edges
9. `compute_pairwise_features()` - 10 edges
10. `extract_address_digits()` - 9 edges

## Surprising Connections (you probably didn't know these)
- `main()` --uses--> `CountryBlockingIndex`  [INFERRED]
  scripts/diagnose_blocking_errors.py → experiments/baseline/code/blocking.py
- `main()` --uses--> `CountryBlockingIndex`  [INFERRED]
  experiments/evaluate_live.py → solution/business_entity_resolution/src/blocking.py
- `main()` --uses--> `CountryBlockingIndex`  [INFERRED]
  experiments/exp_recall.py → solution/business_entity_resolution/src/blocking.py
- `channels()` --calls--> `extract_acronyms()`  [INFERRED]
  experiments/analyze_misses.py → solution/business_entity_resolution/src/blocking.py
- `channels()` --calls--> `extract_postal_prefixes()`  [INFERRED]
  experiments/analyze_misses.py → solution/business_entity_resolution/src/blocking.py

## Import Cycles
- None detected.

## Communities (15 total, 2 thin omitted)

### Community 0 - "normalize_address"
Cohesion: 0.08
Nodes (32): CountryBlockingIndex, Add an S2 or S3 record to the country index., Prune overly frequent posting lists (e.g. generic tokens like 'group' or…, Query the index for candidate matches, ranking by multi-key collision…, Compact inverted index for candidate generation within a single country. Maps…, char_ngrams(), compute_pairwise_features(), jaccard_similarity() (+24 more)

### Community 1 - "views.py"
Cohesion: 0.10
Nodes (36): load_gt(), main(), assess_difficulty(), extract_acronyms(), extract_postal_prefixes(), Extract acronyms from multi-word business names (e.g. 'Tata Consultancy…, Union PixelDust retrieval with trigram-jaccard rerank. Provenance channels:…, Heuristic difficulty tier driving adaptive budgets. Returns… (+28 more)

### Community 2 - "CountryBlockingIndex"
Cohesion: 0.09
Nodes (27): main(), selftest_metric(), build_index(), featurize_ids(), load_all(), load_s1(), main(), main() (+19 more)

### Community 3 - "src/features.py"
Cohesion: 0.12
Nodes (22): char_ngrams(), compute_candidate_features_for_s1(), compute_pairwise_features(), get_acronym(), jaccard_similarity(), Generate set of character n-grams., Compute Jaccard similarity between two sets/iterables., Vectorized batch extraction for an S1 entity against all its candidates.… (+14 more)

### Community 4 - "rows"
Cohesion: 0.26
Nodes (11): main(), channels(), main(), main(), main(), record(), rows(), main() (+3 more)

### Community 5 - "validate_submission_fast.py"
Cohesion: 0.24
Nodes (14): examples(), get_peak_memory_mb(), load_match_targets(), main(), Validate candidate_pairs.tsv in a SINGLE streaming pass without retaining…, Return peak resident memory (RSS) in MB for this process., Return a short sample of offending IDs for error messages., Read first-column entity IDs from a source TSV. (+6 more)

### Community 6 - "validate"
Cohesion: 0.27
Nodes (11): examples(), load_match_targets(), main(), Validate the submission output(s); return ``(errors, warnings)`` lists.…, Return the set of first-column entity IDs from a source TSV. The header row is…, Return a short, human-readable sample of ``items`` for an error message., Return the set of valid S2/S3 match IDs, or ``None`` if unavailable. Only…, Validate one results-style TSV (matching or candidate). Applies the shared… (+3 more)

### Community 7 - "main"
Cohesion: 0.33
Nodes (6): char_ngrams(), compute_expanded_features(), extract_advanced_keys(), FastMultiPassIndex, jaccard(), main()

### Community 9 - "MultiPassBlockingIndex"
Cohesion: 0.38
Nodes (3): extract_advanced_keys(), MultiPassBlockingIndex, Extract multi-pass blocking keys: 1. Exact core name 2. Name core tokens 3.…

### Community 10 - "main"
Cohesion: 0.43
Nodes (3): extract_advanced_keys_v2(), main(), MultiPassBlockingIndexV2

### Community 11 - "main"
Cohesion: 0.83
Nodes (3): has_indic_char(), has_url_handle(), main()

## Knowledge Gaps
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `CountryBlockingIndex` connect `CountryBlockingIndex` to `views.py`, `rows`?**
  _High betweenness centrality (0.072) - this node is a cross-community bridge._
- **Why does `main()` connect `rows` to `CountryBlockingIndex`, `src/features.py`?**
  _High betweenness centrality (0.045) - this node is a cross-community bridge._
- **Why does `compute_candidate_features_for_s1()` connect `src/features.py` to `CountryBlockingIndex`, `rows`?**
  _High betweenness centrality (0.036) - this node is a cross-community bridge._
- **Are the 6 inferred relationships involving `CountryBlockingIndex` (e.g. with `main()` and `build_index()`) actually correct?**
  _`CountryBlockingIndex` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 5 inferred relationships involving `normalize_address()` (e.g. with `main()` and `main()`) actually correct?**
  _`normalize_address()` has 5 INFERRED edges - model-reasoned connections that need verification._
- **Are the 5 inferred relationships involving `normalize_name()` (e.g. with `main()` and `main()`) actually correct?**
  _`normalize_name()` has 5 INFERRED edges - model-reasoned connections that need verification._
- **Should `normalize_address` be split into smaller, more focused modules?**
  _Cohesion score 0.08067375886524823 - nodes in this community are weakly interconnected._