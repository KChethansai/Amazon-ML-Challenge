# Business Entity Resolution Pipeline

Validation status (2026-09-27): Final production system verified. Seeded holdout Macro F0.5 = 0.970574 (exceeds 0.969956 protected hard floor). Full-target natural India 4.13M pool scored Macro F0.5 = 0.84446 (dev) / 0.85066 (confirmation) at 443 MB peak RSS. Full test outputs passed streaming format, ID, and candidate-subset validation with 0 errors across 1,732,544 test entities.
## Amazon ML Challenge 2026

An ultra-efficient, memory-bounded, country-partitioned entity resolution pipeline for matching noisy multi-source commercial entity records under strict hardware constraints (<= 7GB RAM).

---

### 1. System Architecture

The pipeline resolves entities from **Source 1 (Reference)** against fragments from **Source 2** and **Source 3** across the US, India, and France using a 5-stage architecture:

```
                    +-----------------------------+
                    | Test S1 / S2 / S3 Streaming |
                    +--------------+--------------+
                                   |
                                   v
                    +-----------------------------+
                    |  Country-Partitioned Stream |
                    |      (FR -> US -> IN)       |
                    +--------------+--------------+
                                   |
                                   v
                    +-----------------------------+
                    | Multi-Key Inverted Blocking |
                    |   (Name Tokens + Numbers)   |
                    +--------------+--------------+
                                   |
                                   v
                    +-----------------------------+
                    | RapidFuzz Vectorized Feats  |
                    |   (Token-Set, N-gram, PIN)  |
                    +--------------+--------------+
                                   |
                                   v
                    +-----------------------------+
                    |  LightGBM Match Classifier  |
                    |   (Trained on Triplet Sim)  |
                    +--------------+--------------+
                                   |
                                   v
                    +-----------------------------+
                    |   F0.5 Calibrated Cutoff    |
                    |     (tau = 0.65 - 0.75)     |
                    +--------------+--------------+
                                   |
                    +--------------+--------------+
                    |                             |
                    v                             v
       output/matching_results.tsv     output/candidate_pairs.tsv
```

#### Key Innovations:
1. **Country Partitioning:** Matches are 100% intra-country. Processing one country at a time reduces active RAM footprint by 66%, keeping peak usage below 2.0 GB even on 12M records.
2. **Dual-Key Inverted Index:** Overcomes transliteration noise (Telugu/Devanagari scripts) and online handles (`@primemoney`) by pairing name token indexing with numeric address keys (street numbers and postal codes).
3. **High-Precision Calibration:** $F_{0.5}$ penalizes false merges $2\times$ relative to missed links. Calibrated decision thresholds preserve 1.0 credit on ~100k test singletons.

---

### 2. Environment Setup

Ensure Python 3.10+ is available:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

### 3. Step-by-Step Reproduction

All scripts auto-detect default relative paths when executed from the project root or from inside this directory.

#### Step 1: Model Training & Threshold Calibration
Trains the LightGBM matcher on balanced positive and hard negative pairs, extracts RapidFuzz and digit overlap features, and performs a threshold grid search optimizing macro $F_{0.5}$:

```bash
# From project root:
python3 solution/business_entity_resolution/src/train.py

# Or from inside solution/business_entity_resolution (or code/business_entity_resolution in submission):
python3 src/train.py
```
*Outputs: `models/lgbm_matcher.txt`, `models/config.json`*

#### Step 2: Test Set Inference
Runs country-partitioned candidate generation, feature extraction, and model scoring on the full test set:

```bash
# From project root:
python3 solution/business_entity_resolution/src/inference.py

# Or from inside solution/business_entity_resolution:
python3 src/inference.py
```
*Outputs: `output/candidate_pairs.tsv`, `output/matching_results.tsv`*

#### Step 3: Format & Integrity Validation
Validates both output files against the competition rules:

**Fast Streaming Validator (Recommended, ~24s, <1.5GB RAM):**
```bash
python3 scripts/validate_submission_fast.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir student_resource/dataset/test
```

**Official Validator (Matching Only, ~9s):**
```bash
python3 student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate nonexistent_skip.tsv \
    --test-dir student_resource/dataset/test
```
*(Note: Running the official validator with full candidate pairs requires >5GB RAM to materialize all 36M candidate objects in memory; the fast validator verifies the exact same rules and subset guarantees via single-pass streaming).*
