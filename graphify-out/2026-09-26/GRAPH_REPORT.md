# Graph Report - ML Challenge  (2026-09-26)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 163 nodes · 289 edges · 13 communities (10 shown, 3 thin omitted)
- Extraction: 91% EXTRACTED · 9% INFERRED · 0% AMBIGUOUS · INFERRED: 26 edges (avg confidence: 0.86)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `df760ab5`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- src/train.py
- normalize_address
- validate_submission_fast.py
- CountryBlockingIndex
- CountryBlockingIndex
- validate
- train_eval_experiment.py
- CleanMultiPassIndex
- MultiPassBlockingIndex
- main
- CleanMultiPassIndex
- create_validation_split.py

## God Nodes (most connected - your core abstractions)
1. `CountryBlockingIndex` - 13 edges
2. `normalize_address()` - 12 edges
3. `normalize_name()` - 12 edges
4. `train_pipeline()` - 11 edges
5. `extract_name_tokens()` - 11 edges
6. `CountryBlockingIndex` - 10 edges
7. `compute_pairwise_features()` - 10 edges
8. `compute_candidate_features_for_s1()` - 9 edges
9. `compute_pairwise_features()` - 9 edges
10. `run_inference()` - 9 edges

## Surprising Connections (you probably didn't know these)
- `run_inference()` --uses--> `CountryBlockingIndex`  [INFERRED]
  solution/business_entity_resolution/src/inference.py → experiments/baseline/code/blocking.py
- `train_pipeline()` --uses--> `CountryBlockingIndex`  [INFERRED]
  solution/business_entity_resolution/src/train.py → experiments/baseline/code/blocking.py
- `main()` --uses--> `CountryBlockingIndex`  [INFERRED]
  scripts/diagnose_blocking_errors.py → experiments/baseline/code/blocking.py
- `main()` --calls--> `extract_name_tokens()`  [INFERRED]
  experiments/exp_001_blocking/test_blocking_improvements_v2.py → experiments/baseline/code/normalization.py
- `main()` --calls--> `normalize_address()`  [INFERRED]
  experiments/exp_001_blocking/test_blocking_improvements_v2.py → experiments/baseline/code/normalization.py

## Import Cycles
- None detected.

## Communities (13 total, 3 thin omitted)

### Community 0 - "src/train.py"
Cohesion: 0.12
Nodes (28): char_ngrams(), compute_candidate_features_for_s1(), compute_pairwise_features(), get_acronym(), jaccard_similarity(), Generate set of character n-grams., Compute Jaccard similarity between two sets/iterables., Vectorized batch extraction for an S1 entity against all its candidates.… (+20 more)

### Community 1 - "normalize_address"
Cohesion: 0.15
Nodes (26): char_ngrams(), compute_pairwise_features(), jaccard_similarity(), Compute Jaccard similarity between two sets/iterables., Compute dense feature vector for a candidate pair. tgt_rec can be a dict or a…, Generate set of character n-grams., run_inference(), clean_unicode_to_ascii() (+18 more)

### Community 2 - "validate_submission_fast.py"
Cohesion: 0.24
Nodes (14): examples(), get_peak_memory_mb(), load_match_targets(), main(), Validate candidate_pairs.tsv in a SINGLE streaming pass without retaining…, Return peak resident memory (RSS) in MB for this process., Return a short sample of offending IDs for error messages., Read first-column entity IDs from a source TSV. (+6 more)

### Community 3 - "CountryBlockingIndex"
Cohesion: 0.16
Nodes (8): CountryBlockingIndex, Add an S2 or S3 record to the country index., Prune overly frequent posting lists (e.g. generic tokens like 'group' or…, Query the index for candidate matches, ranking by multi-key collision…, Compact inverted index for candidate generation within a single country. Maps…, has_indic_char(), has_url_handle(), main()

### Community 4 - "CountryBlockingIndex"
Cohesion: 0.19
Nodes (9): CountryBlockingIndex, extract_acronyms(), extract_postal_prefixes(), Prune overly frequent uninformative posting lists and compute IDF weights., Extract acronyms from multi-word business names (e.g. 'Tata Consultancy…, Query the index for candidate matches, ranking by multi-key collision…, Extract 2-digit postal prefix from address., Multi-Pass compound inverted index for candidate generation within a single… (+1 more)

### Community 5 - "validate"
Cohesion: 0.27
Nodes (11): examples(), load_match_targets(), main(), Validate the submission output(s); return ``(errors, warnings)`` lists.…, Return the set of first-column entity IDs from a source TSV. The header row is…, Return a short, human-readable sample of ``items`` for an error message., Return the set of valid S2/S3 match IDs, or ``None`` if unavailable. Only…, Validate one results-style TSV (matching or candidate). Applies the shared… (+3 more)

### Community 6 - "train_eval_experiment.py"
Cohesion: 0.33
Nodes (5): char_ngrams(), compute_expanded_features(), extract_advanced_keys(), FastMultiPassIndex, jaccard()

### Community 8 - "MultiPassBlockingIndex"
Cohesion: 0.38
Nodes (3): extract_advanced_keys(), MultiPassBlockingIndex, Extract multi-pass blocking keys: 1. Exact core name 2. Name core tokens 3.…

### Community 9 - "main"
Cohesion: 0.43
Nodes (3): extract_advanced_keys_v2(), main(), MultiPassBlockingIndexV2

## Knowledge Gaps
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `CountryBlockingIndex` connect `CountryBlockingIndex` to `src/train.py`, `normalize_address`?**
  _High betweenness centrality (0.376) - this node is a cross-community bridge._
- **Why does `train_pipeline()` connect `src/train.py` to `CountryBlockingIndex`, `CountryBlockingIndex`?**
  _High betweenness centrality (0.171) - this node is a cross-community bridge._
- **Why does `run_inference()` connect `src/train.py` to `CountryBlockingIndex`, `CountryBlockingIndex`?**
  _High betweenness centrality (0.140) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `CountryBlockingIndex` (e.g. with `main()` and `run_inference()`) actually correct?**
  _`CountryBlockingIndex` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 5 inferred relationships involving `normalize_address()` (e.g. with `main()` and `main()`) actually correct?**
  _`normalize_address()` has 5 INFERRED edges - model-reasoned connections that need verification._
- **Are the 5 inferred relationships involving `normalize_name()` (e.g. with `main()` and `main()`) actually correct?**
  _`normalize_name()` has 5 INFERRED edges - model-reasoned connections that need verification._
- **Are the 5 inferred relationships involving `extract_name_tokens()` (e.g. with `main()` and `main()`) actually correct?**
  _`extract_name_tokens()` has 5 INFERRED edges - model-reasoned connections that need verification._