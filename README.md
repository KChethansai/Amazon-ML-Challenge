# Amazon-ML-Challenge

Validation update (2026-09-26): the stored 0.95835 holdout score did not replay with the current model and source. The fresh diagnostic is 0.94205 Macro F₀.₅ / 91.71% candidate recall on a ground-truth-seeded partial target pool, so it is not a full-pool score. Existing submission files pass strict streaming format, ID, and candidate-subset validation; no improved submission has been produced. Details: `experiments/live_holdout_seeded_full/metrics.json` and `experiments/ledger.jsonl`.

A natural-pool check with an isolated retrained model on 1,000 India holdout S1 records scored 0.77861 Macro F₀.₅ / 78.46% candidate recall against all 4,133,346 India targets. This exposes a large partial-pool validation bias; the 0.95835 historical figure is not evidence of full-test quality.
