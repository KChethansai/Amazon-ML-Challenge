#!/usr/bin/env python3
"""Fast LGBM capacity tuning on cached matrices + entity-level F0.5."""
import sys, json, time
import numpy as np
sys.path.insert(0, "solution/business_entity_resolution/src")
from train import calculate_macro_f05, ALL_FEATURE_NAMES
import lightgbm as lgb

gt = {}
with open("dataset/train/train_ground_truth.tsv", encoding="utf-8") as f:
    next(f)
    for line in f:
        p = line.rstrip("\n").split("\t")
        gt[p[0]] = set(x.strip() for x in (p[1] if len(p) > 1 else "").split(",") if x.strip())

Xtr = np.load("experiments/X_train_v3.npy"); ytr = np.load("experiments/y_train_v3.npy")
Xdv = np.load("experiments/X_val_v3.npy"); ydv = np.load("experiments/y_val_v3.npy")
pairs = np.load("experiments/val_ids_v3.npy", allow_pickle=True)
print(f"train {Xtr.shape} pos={int(ytr.sum())} | val {Xdv.shape}", flush=True)

CONFIGS = {
    "A_baseline": dict(learning_rate=0.06, num_leaves=63, max_depth=7, min_child_samples=20,
                       subsample=0.85, subsample_freq=1, feature_fraction=0.9, seed=7,
                       deterministic=True, n_jobs=4, verbose=-1, num_boost_round=300),
    "B_deep": dict(learning_rate=0.03, num_leaves=127, max_depth=9, min_child_samples=10,
                   subsample=0.85, subsample_freq=1, feature_fraction=0.9, seed=7,
                   deterministic=True, n_jobs=8, verbose=-1, num_boost_round=1500),
    "C_deeper": dict(learning_rate=0.02, num_leaves=255, max_depth=10, min_child_samples=8,
                     subsample=0.9, subsample_freq=1, feature_fraction=0.8, seed=7,
                     deterministic=True, n_jobs=8, verbose=-1, num_boost_round=2000),
}

results = {}
for name, cfg in CONFIGS.items():
    t0 = time.time()
    rounds = cfg.pop("num_boost_round")
    dtrain = lgb.Dataset(Xtr, label=ytr, feature_name=list(ALL_FEATURE_NAMES))
    dval = lgb.Dataset(Xdv, label=ydv, reference=dtrain)
    params = {k: v for k, v in cfg.items()}
    params.update({"objective": "binary", "metric": "binary_logloss"})
    model = lgb.train(params, dtrain, num_boost_round=rounds,
                      valid_sets=[dval], callbacks=[lgb.early_stopping(150, verbose=False)])
    pv = model.predict(Xdv)
    # entity-level F0.5 grid (small, targeted)
    by_s1 = {}
    for (sid, cid), p in zip(pairs, pv):
        by_s1.setdefault(sid, []).append((cid, float(p)))
    gt_map = {s: gt.get(s, set()) for s in by_s1}
    best = (0, None)
    for ts in (0.60, 0.66, 0.72, 0.76, 0.80):
        for tm in (0.40, 0.45, 0.50):
            for dm in (0.10, 0.15, 0.18):
                preds = {}
                for s, lst in by_s1.items():
                    mp = max(p for _, p in lst)
                    preds[s] = set() if mp < ts else set(c for c, p in lst if p >= max(tm, mp - dm))
                f = calculate_macro_f05(gt_map, preds)
                if f > best[0]:
                    best = (f, (ts, tm, dm))
    # pair AUC approx via ranking
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(ydv, pv)
    results[name] = {"val_logloss": model.best_score["valid_0"]["binary_logloss"],
                     "val_auc": auc, "best_iter": model.best_iteration,
                     "dev_f05": best[0], "thresh": best[1], "time": round(time.time() - t0, 1)}
    print(f"{name}: {results[name]}", flush=True)
    model.save_model(f"experiments/lgbm_{name}.txt")

with open("experiments/tune_lgbm_results.json", "w") as f:
    json.dump(results, f, indent=2)
