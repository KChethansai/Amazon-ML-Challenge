#!/usr/bin/env python3
"""Phase 0/1: lock reproducible baseline. Fast seeded replay (1500 S1) + metric self-test."""
import os, sys, json, time, random
import numpy as np
sys.path.insert(0, "solution/business_entity_resolution/src")
from normalization import prepare_source_record
from blocking import CountryBlockingIndex
from train import calculate_macro_f05, evaluate_metrics, query_for_s1, featurize_s1
import lightgbm as lgb

random.seed(7); np.random.seed(7)
TRAIN_DIR = "dataset/train"

def selftest_metric():
    gt = {"a": {"X"}, "b": set(), "c": {"Y", "Z"}}
    pr = {"a": {"X"}, "b": set(), "c": {"Y"}}
    # a: P=1,R=1 -> 1.0 ; b: 1.0 ; c: P=1,R=0.5 -> 1.25*.5/(0.25+0.5)=0.8333
    f = calculate_macro_f05(gt, pr)
    assert abs(f - (1 + 1 + 0.8333333) / 3) < 1e-6, f
    # singleton FP -> 0
    assert calculate_macro_f05({"b": set()}, {"b": {"X"}}) == 0.0
    print("metric self-test PASS")

def main():
    selftest_metric()
    t0 = time.time()
    with open("experiments/splits/holdout_s1_ids.json") as f:
        hold = json.load(f)
    with open("experiments/splits/train_pool_s1_ids.json") as f:
        tr = set(json.load(f)[:25000])
    with open("experiments/splits/dev_s1_ids.json") as f:
        dv = set(json.load(f)[:6000])
    sub = [e for e in hold if e not in tr and e not in dv][:1500]
    gt_all = {}
    with open(os.path.join(TRAIN_DIR, "train_ground_truth.tsv"), encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            gt_all[p[0]] = set(x.strip() for x in (p[1] if len(p) > 1 else "").split(",") if x.strip())
    need = set()
    for e in sub:
        need.update(gt_all.get(e, set()))
    idx = {"US": CountryBlockingIndex("US"), "India": CountryBlockingIndex("India")}
    s1rec = {}
    for fn, is_s2 in [("train_source2.tsv", 1), ("train_source3.tsv", 0)]:
        with open(os.path.join(TRAIN_DIR, fn), encoding="utf-8") as f:
            next(f)
            for i, line in enumerate(f):
                p = line.rstrip("\n").split("\t")
                if len(p) < 4 or p[3] not in idx:
                    continue
                eid, bn, ba, bc = p
                if eid in need or i < 60000:
                    r = prepare_source_record(bn, ba, bc)
                    idx[bc].add_target_record(eid, r["norm_name"], r["core_tokens"], r["norm_addr"], r["addr_keys"], r["digits"], is_s2=is_s2, raw_addr=ba)
    for c in idx.values():
        c.prune_frequent_keys()
    with open("solution/business_entity_resolution/models/config.json") as f:
        cfg = json.load(f)
    model = lgb.Booster(model_file="solution/business_entity_resolution/models/lgbm_matcher.txt")
    gt_map, pred_map = {}, {}
    hits = tot = 0
    # load s1 records needed
    with open(os.path.join(TRAIN_DIR, "train_source1.tsv"), encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if p[0] in set(sub):
                s1rec[p[0]] = prepare_source_record(p[1], p[2], p[3])
    for sid in sub:
        r = s1rec[sid]
        ci = idx[r["country"]]
        items, prov = query_for_s1(ci, r, max_candidates=cfg.get("max_candidates", 60), adaptive=cfg.get("adaptive", True))
        cs = {c for c, _ in items}
        g = gt_all.get(sid, set())
        hits += len(g & cs); tot += len(g)
        vc, fr = featurize_s1(r, items, prov, ci, use_extra=cfg.get("use_extra", True))
        if fr:
            pr_ = model.predict(np.array(fr, dtype=np.float32))
            mp = float(np.max(pr_))
            pred_map[sid] = set() if mp < cfg["tau_singleton"] else set(c for c, p_ in zip(vc, pr_) if p_ >= max(cfg["tau_min"], mp - cfg["delta_margin"]))
        else:
            pred_map[sid] = set()
        gt_map[sid] = g
    m = evaluate_metrics(gt_map, pred_map)
    m["candidate_recall"] = hits / max(1, tot)
    out = {"evaluator": "train.py:calculate_macro_f05 + evaluate_metrics", "split": "holdout subset 1500 S1 (seeded pool, 60k/src background)",
           "seed": 7, "macro_f05": m["macro_f05"], "precision": m["precision"], "recall": m["recall"],
           "candidate_recall": m["candidate_recall"], "singleton_accuracy": m["singleton_accuracy"],
           "candidate_count": None, "runtime_seconds": round(time.time() - t0, 1),
           "prior_validated": {"EXP_005_replay": 0.9420544908655107, "EXP_013_graph": 0.9497547984072268,
                               "EXP_014_natural_india_1k": 0.7786111044801864, "historical_unreproduced": 0.9583506721550407}}
    with open("experiments/baseline_reproduction.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
