# ML Challenge 2026: Business Entity Resolution Solution Documentation

> **Final Certified State (2026-09-27):** The production pipeline integrates disk-backed indexing (`DiskCountryBlockingIndex`), dense 52-dimensional multi-view features (`features.py` + `features2.py`), and the EXP_020 LightGBM matcher with calibrated decision thresholds (`tau_singleton = 0.80`, `tau_min = 0.40`, `delta_margin = 0.10`).
> - **Seeded Holdout Baseline Lock:** Verified Macro $F_{0.5}$ = **0.970574** (exceeds the 0.969956 protected hard floor), Precision = **99.60%**, Candidate Recall = **98.88%**, Singleton Accuracy = **100.0%**.
> - **Natural India Development Split (1,000 S1 against 4,133,346 SQLite target index):** Macro $F_{0.5}$ = **0.844460**, Precision = **88.91%**, Candidate Pair Recall = **93.38%**, Oracle $F_{0.5}$ = **0.975219**, Peak RSS = **443.0 MB**.
> - **Natural India Untouched Confirmation Split (300 S1 against 4,133,346 targets):** Macro $F_{0.5}$ = **0.850663**, Precision = **88.53%**, Candidate Pair Recall = **93.31%**, Oracle $F_{0.5}$ = **0.971864**, Peak RSS = **443.0 MB**.
> - **Submission Certification:** Full test dataset validation (`scripts/validate_submission_fast.py --check-ids`) passed with 0 errors across 1,732,544 test S1 rows.

**Team Name:** EntityResolvers  
**Team Members:** Chethan & AI Co-Engineer  
**Submission Date:** 2026-09-27

---

## 1. Executive Summary

We developed an ultra-scalable, memory-bounded, country-partitioned entity resolution pipeline for commercial business records across the United States, India, and France. Given reference records in Source 1 and partial, noisy fragments in Sources 2 and 3, our pipeline implements a high-recall multi-pass blocking architecture backed by SQLite on-disk inverted indexes, dense 52-dimensional pairwise and provenance feature extraction via C++ RapidFuzz, an optimized LightGBM matcher, and calibrated Two-Stage Margin Thresholding for robust singleton gating and candidate pruning.

On the verified seeded holdout evaluation:
- **Macro $F_{0.5}$ Score:** **0.970574** (protected floor $\ge 0.969956$ strictly satisfied).
- **Macro Precision:** **99.60%**
- **Candidate Recall:** **98.88%**
- **Singleton Accuracy:** **100.0%**

On the natural 4.13M full target pool:
- **Macro $F_{0.5}$:** **0.84446 - 0.85066**
- **Candidate Pair Recall:** **93.31% - 93.38%**
- **Peak RSS:** **443.0 MB** (well below the 2,500 MB RAM budget).
- **Test Integrity:** Passed strict streaming format, ordering, and candidate subset verification across all 1,732,544 test entities.

---

## 2. Methodology

### 2.1 Problem Analysis
Reconnaissance across the 24.23 million records identified key architectural challenges:
1. **Zero Cross-National Linkage:** In ground truth, 100% of entity matches occur within the same country label ($country_{S1} == country_{target}$). Partitioning by country is strictly loss-free and reduces memory overhead by 66%.
2. **Candidate Recall Ceiling Bottleneck:** Initial diagnostic audits revealed ~8.92% of true matches were dropped during blocking, especially for Indian transliterated records, acronyms (e.g., "TCS" vs "Tata Consultancy Services"), and typo-altered names.
3. **Severe Country Distribution Shift:** Unseen country France (15% of test data) requires language-agnostic character n-gram indexing and normalization without external country dictionaries.
4. **Asymmetric Error Costs in Macro $F_{0.5}$:** With $\beta = 0.5$, precision is weighted $2\times$ more heavily than recall. Singletons account for 5.58% of entities; predicting even one false candidate destroys a singleton's score from 1.0 to 0.0, necessitating disciplined two-stage margin gating.

### 2.2 Solution Strategy
**Approach Type:** Multi-Pass On-Disk SQLite Inverted Index Blocking + 52-Dim Dense Multi-View Feature Extraction + LightGBM Ranking + Calibrated Two-Stage Margin Thresholding.

**Key Upgrades:**
1. **SQLite Disk-Backed Inverted Indexing (`src/disk_blocking.py`):** Replaces memory-heavy in-memory posting lists with an indexed SQLite disk engine (`DiskCountryBlockingIndex`), operating over 4.13M target records within a tiny 443 MB peak RSS footprint.
2. **52-Dimensional Multi-View Features (`src/features.py`, `src/features2.py`):** Combines 30 base pairwise lexical/address metrics with 22 auxiliary features covering Indic transliteration matchers (`tr_name_set`, `addr_tr_set`), phonetic concordance (`phon_jaccard`, `soundex_jaccard`), character n-gram similarities (`char2_jaccard`, `char4_jaccard`), channel hit provenance indicators, and z-score candidate pool normalizations.
3. **Calibrated Two-Stage Margin Thresholding:**
   - **Stage 1 (Singleton Gate):** If $\max(\text{score}_{S1}) < \tau_{\text{singleton}}$ ($\tau_{\text{singleton}} = 0.80$), classify entity as singleton (predict empty list `[]`).
   - **Stage 2 (Relative Margin Filter):** Keep candidates satisfying $\text{score} \ge \tau_{\text{min}}$ ($0.40$) AND $\text{score} \ge \max(\text{score}) - \delta_{\text{margin}}$ ($\delta_{\text{margin}} = 0.10$), filtering co-located commercial office collision false positives while preserving true multi-source entity matches.

---

## 3. Candidate Generation (Blocking)

- **Blocking Channels Used (`src/blocking.py`, `src/disk_blocking.py`):**
  1. `exact` (Weight: 15.0): Exact normalized name match.
  2. `addr` & `addr_tok` (Weight: dynamic IDF, 1.5–8.0): Compound street and postal/locality anchors.
  3. `token` (Weight: 3.0): Significant core tokens ($\ge 3$ characters), excluding frequent corporate stopwords.
  4. `prefix` (Weight: 1.0): 4-character prefix for typo tolerance.
  5. `acronym` (Weight: 10.0): Exact acronym matching for multi-word corporate names.
  6. `tokpost` (Weight: 6.0): Compound key pairing first significant name token with 2-digit postal prefix.
  7. `ngram` (Weight: dynamic IDF): Character 3-gram index for fuzzy name retrieval.
  8. `tr_name` & `tr_tok`: Indic transliteration blocking channels mapping phonetic equivalents across scripts.
- **Candidate Pool Configuration:**
  - `max_candidates`: 60 candidates per S1 entity with adaptive recall expansion.
  - Achieves **98.88%** candidate recall on seeded holdout and **93.38%** candidate pair recall (0.9752 Oracle $F_{0.5}$) across the full 4,133,346 natural target pool.

---

## 4. Matching Model

**Features Used (52 Dense Dimensions in `src/features.py` and `src/features2.py`):**
- **Base Lexical & Phonetic Metrics (30 Dimensions):**
  - `name_token_set`, `name_token_sort`, `name_ratio`, `name_exact`, `name_jaccard`, `name_ngram_jaccard`, `name_len_diff`.
  - `name_jaro_winkler`, `name_acronym_match`, `name_prefix_similarity`, `name_first_token_match`, `name_partial_ratio`.
  - `addr_token_set`, `addr_token_sort`, `addr_ratio`, `addr_jaccard`, `addr_len_diff`, `addr_missing`, `addr_token_containment`, `addr_jaro_winkler`.
  - `num_overlap`, `has_matching_digits`, `postal_prefix_match_len`, `street_num_exact_match`.
  - `composite_score`, `is_s2`, `score_rank_in_candidate_pool`, `blocking_weight_ratio`, `raw_blocking_weight`, `is_top1_candidate`.
- **Extended Transliteration & Provenance Metrics (22 Dimensions):**
  - `tr_name_set`, `phon_jaccard`, `soundex_jaccard`, `char2_jaccard`, `char4_jaccard`, `name_containment`.
  - `shared_sig_count`, `shared_sig_ratio`, `tokcount_diff`, `addr_tr_set`, `addr_char3_jaccard`.
  - `phone_agree`, `postal_exact`, `addr_coverage`, `agree_count`, `family_count`.
  - Provenance flags: `has_tr_chan`, `has_phon_chan`, `has_phone_chan`, `ngram_only`.
  - Margin features: `gap_to_top`, `w_zscore`.

**Model Architecture & Calibrated Hyperparameters:**
- **LightGBM Binary Classifier** (`GBDT`):
  - `num_leaves=63`, `learning_rate=0.06`, `n_estimators=300`, `subsample=0.85`, `feature_fraction=0.90`.
- **Calibrated Thresholds:** $\tau_{\text{singleton}} = 0.80$, $\tau_{\text{min}} = 0.40$, $\delta_{\text{margin}} = 0.10$.

---

## 5. Results & Validation

- **Seeded Holdout Baseline Lock (1,500 S1 Entities):**
  - **Macro $F_{0.5}$:** **0.970574** (Protected Floor $\ge 0.969956$ **PASS**)
  - **Precision:** **99.60%**
  - **Recall:** **91.60%**
  - **Candidate Recall:** **98.88%**
  - **Singleton Accuracy:** **100.0%**
- **Natural Full Target Pool Benchmark (4,133,346 Target Index):**
  - **Development Split (1,000 S1):** Macro $F_{0.5}$ = **0.844460**, Precision = **88.91%**, Candidate Pair Recall = **93.38%**, Oracle $F_{0.5}$ = **0.975219**.
  - **Untouched Confirmation Split (300 S1):** Macro $F_{0.5}$ = **0.850663**, Precision = **88.53%**, Candidate Pair Recall = **93.31%**, Oracle $F_{0.5}$ = **0.971864**.
  - **Peak Memory Usage:** **443.0 MB** (vs 2,500 MB maximum threshold).
- **Test Dataset Validation:**
  - `scripts/validate_submission_fast.py --check-ids` passed with **0 errors** across 1,732,544 test S1 rows.

---

## 6. Conclusion

By unifying disk-backed SQLite indexing, 52-dimensional multi-view features, and calibrated decision margins, our final system protects the $\ge 0.97$ baseline floor while solving the natural-scale memory and candidate retrieval challenges at 443 MB peak RSS.

---

## Appendix: Code Artefacts
Self-contained in `solution/business_entity_resolution/`:
- `src/normalization.py`: Unicode decomposition, Indic cleaning, address token parsing.
- `src/blocking.py`: Multi-pass in-memory inverted index architecture.
- `src/disk_blocking.py`: Scalable SQLite on-disk inverted index for full natural target pools.
- `src/features.py`: 30 base pairwise lexical and numeric features.
- `src/features2.py`: 22 extended transliteration, phonetic, and provenance features.
- `src/train.py`: Unified 52-feature training and evaluation pipeline.
- `src/inference.py`: High-throughput, memory-bounded test inference pipeline.
- `models/lgbm_matcher.txt` & `models/config.json`: Final EXP_020 serialized booster and calibrated parameters.
- `requirements.txt`: Environment dependencies.
