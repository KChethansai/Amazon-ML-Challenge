# Solution Architecture & Pipeline Walkthrough
**Amazon ML Challenge 2026 — Business Entity Resolution**

---

## 1. Executive Summary

This project implements an end-to-end, memory-bounded, country-partitioned Entity Resolution system designed to match multi-source commercial records across Source 1 (Reference), Source 2, and Source 3 for the US, India, and France under strict hardware constraints ($\le$ 7.0 GB RAM).

The system achieved:
- **Macro $F_{0.5}$ Validation Score:** **0.8984**
- **Test Entities Processed:** **1,732,544** Source 1 entities
- **Total Final Matches:** **4,595,646** links
- **Predicted Singletons:** **357,775** (20.65% singleton rate)
- **Candidate Subset Guarantee:** **100%** (0 subset violations across all 36.3M candidates)

---

## 2. Workspace Organization

```
ML Challenge/
├── student_resource/            # Canonical organizer resources and datasets
├── solution/                    # Core engineering implementation
│   └── business_entity_resolution/
│       ├── src/                 # Blocking, normalization, features, train, inference
│       └── models/              # Pretrained LightGBM booster & calibration parameters
├── output/                      # Generated submission TSVs (matching_results, candidate_pairs)
├── reports/                     # Audit, EDA, and walkthrough documentation
├── scripts/                     # Operational utilities & fast streaming validator
├── submission/                  # Challenge-compliant submission zip package
└── Documentation_template.md    # Completed technical approach documentation
```

---

## 3. High-Level Architecture

The pipeline consists of five decoupled, stream-oriented stages:

```
[Raw TSV Records] -> [Country Streaming (FR, US, IN)]
                  -> [Multi-Key Inverted Index Blocking]
                  -> [RapidFuzz & Digit Overlap Feature Extraction]
                  -> [LightGBM Ranker (Tau* = 0.65)]
                  -> [TSV Submission Streamers]
```

1. **Country Partitioning:** Entities are guaranteed 100% intra-country. Streaming country-by-country prevents loading all 12 million test records simultaneously, capping peak RAM below 2.0 GB.
2. **Multi-Key Inverted Index:** Overcomes non-Latin script transliteration and handle-based noisy names by combining core normalized tokens with numeric address keys.
3. **Dense Pairwise Feature Extraction:** RapidFuzz C++ token set, token sort, and edit distance metrics combined with numeric address overlap features.
4. **Precision-Calibrated LightGBM Matcher:** Boosted decision trees calibrated under an $F_{0.5}$ metric that penalizes false positive links $2\times$ heavier than false negatives, preserving singletons.

---

## 4. Submission Package Structure

The final archive `submission/EntityResolvers_submission.zip` matches the official competition specification:

```
EntityResolvers_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── models/
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md
```

All formatting, schema, entity coverage, and cross-file constraints have been independently verified with both the official validator and the optimized streaming validator.
