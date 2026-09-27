# Business entity resolution upgrade: execution result

**Decision: reject promotion.** The scratch upgrade improved a GT-seeded holdout, but the full natural India index exhausted available memory twice before scoring. The existing output files and submission archive were not replaced.

## Evidence classification

| Class | Evaluation | Macro F₀.₅ | Precision | Recall | Singleton accuracy | Candidate pair recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Validated, seeded | EXP_020, 14,998 holdout S1, GT-seeded 200k/source pool | 0.972585 | 0.990829 | 0.943816 | 0.962963 | 0.983510 |
| Previously validated, seeded | EXP_005 replay, same holdout size | 0.942054 | 0.992069 | 0.864100 | — | 0.917128 |
| Previously validated, seeded | EXP_013 graph trial | 0.949755 | — | — | — | — |
| Previously validated, natural India | EXP_014, 1,000 S1 against 4,133,346 targets, earlier model | 0.778611 | 0.855219 | 0.737515 | 0.528302 | 0.784553 |
| Limited seeded replay | Baseline reproduction, 1,500 S1 | 0.961362 | 0.997207 | 0.896139 | 0.987013 | 0.957336 |
| Historical, unreproduced | Stored model config | 0.958351 | — | — | — | — |

EXP_020 trained one 52-feature LightGBM model on 25,000 S1 entities, tuned thresholds on 6,000 disjoint dev entities, and scored the 14,998 disjoint holdout entities. Training took 33m51s. The holdout target index deliberately includes ground-truth targets and is unsuitable as a natural-distribution estimate. Candidate entity-full-recall, zero-recovery count, and France behavior were not measured for EXP_020.

## Natural India and resource gate

EXP_021 attempted 1,000 holdout India S1 against all 4,133,346 India S2/S3 targets. Both attempts ended with exit code 137 before candidate or model scoring: 398s initially and 541s after limiting ground-truth loading to the held-out IDs. Available 7 GiB RAM and 3.5 GiB swap were exhausted. No natural India score, entity-full-recall, zero-recovery count, or precise process peak RSS exists for the upgrade. The full test inference requirement is therefore unverified. The 2.5 GB RSS target is not met or established for the upgraded index.

## Protected submission

The existing `output/` files passed the fast validator with `--check-ids`: 1,732,544 rows in each TSV; 103,897,911 candidate IDs; 6,380,017 matched links; 106.60s validation time; 2,451.6 MB peak validator RSS. It verified schema, coverage, valid IDs, and match subset membership. The existing archive passed `unzip -t`, and the archive's two embedded TSV SHA-256 values equal the validated local files. This evidence applies to the **earlier archive**, not EXP_020. The official full candidate validator was not run in this execution. No new final model, full-test TSVs, or archive were produced.

The inspected production source contains no network lookup or external business identity data dependency. Added transliteration and phonetic packages are local algorithms. This is a focused fair-play source review, not a complete independent certification.

## Remaining blocker

The expanded Python inverted index must fit the full country target pool before it can be evaluated or shipped. A memory-bounded retrieval implementation and natural India measurement are required, followed by full test inference, validation, performance measurement, and packaging. The scratch model and config remain under `experiments/final_scratch/` and are not the production model.

Evidence: `experiments/baseline_reproduction.json`, `experiments/ledger.jsonl` (EXP_020–021), `experiments/final_scratch/train.log`, `experiments/final_scratch/resource_gate.json`, and validator/integrity logs in the same scratch directory.
