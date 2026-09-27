#!/usr/bin/env python3
"""Simulate decision layer calibrations on BOTH seeded holdout and natural India."""

import json
import os
import sys
import numpy as np

sys.path.insert(0, "solution/business_entity_resolution/src")
from normalization import prepare_source_record
from disk_blocking import DiskCountryBlockingIndex
from blocking import CountryBlockingIndex
from train import (
    calculate_macro_f05,
    evaluate_metrics,
    featurize_s1,
    query_for_s1,
)

NAME_CHANNELS = {"exact", "token", "prefix", "acronym", "ngram", "tr_name", "tr_tok", "phon"}

def load_data():
    with open("experiments/splits/holdout_s1_ids.json") as f:
        holdout = json.load(f)
    with open("experiments/splits/train_pool_s1_ids.json") as f:
        tr = set(json.load(f)[:25000])
    with open("experiments/splits/dev_s1_ids.json") as f:
        dv = set(json.load(f)[:6000])
    seed_sub = [e for e in holdout if e not in tr and e not in dv][:1500]

    nat_sub = []
    with open("dataset/train/train_source1.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if p[0] in set(holdout) and p[3] == "India":
                nat_sub.append((p[0], prepare_source_record(p[1], p[2], p[3])))
                if len(nat_sub) == 100:
                    break

    gt_all = {}
    with open("dataset/train/train_ground_truth.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            gt_all[p[0]] = set(x.strip() for x in (p[1] if len(p) > 1 else "").split(",") if x.strip())

    return seed_sub, nat_sub, gt_all

if __name__ == "__main__":
    import lightgbm as lgb
    seed_sub, nat_sub, gt_all = load_data()

    with open("experiments/final_scratch/config.json") as f:
        cfg = json.load(f)
    model = lgb.Booster(model_file="experiments/final_scratch/lgbm_matcher.txt")

    print("Step 1: Precomputing Natural India 100 candidate scores & features...", flush=True)
    index_nat = DiskCountryBlockingIndex("India", "experiments/final_scratch/india_train.sqlite")
    nat_data = {}
    for sid, r in nat_sub:
        items, prov = query_for_s1(index_nat, r, max_candidates=cfg.get("max_candidates", 60), adaptive=cfg.get("adaptive", True))
        cands, feats = featurize_s1(r, items, prov, index_nat, use_extra=cfg.get("use_extra", True))
        if feats:
            probs = [float(x) for x in model.predict(np.array(feats, dtype=np.float32))]
            nat_data[sid] = {
                "cands": cands, "probs": probs, "feats": feats,
                "prov": prov, "gt": gt_all.get(sid, set())
            }
        else:
            nat_data[sid] = {"cands": [], "probs": [], "feats": [], "prov": {}, "gt": gt_all.get(sid, set())}
    index_nat.close()
    print("Done precomputing Natural India.", flush=True)

    print("Step 2: Precomputing Seeded Holdout candidate scores & features...", flush=True)
    need = set()
    for e in seed_sub:
        need.update(gt_all.get(e, set()))
    idx_seed = {"US": CountryBlockingIndex("US"), "India": CountryBlockingIndex("India")}
    for fn, is_s2 in [("train_source2.tsv", 1), ("train_source3.tsv", 0)]:
        with open(os.path.join("dataset/train", fn), encoding="utf-8") as f:
            next(f)
            for i, line in enumerate(f):
                p = line.rstrip("\n").split("\t")
                if len(p) < 4 or p[3] not in idx_seed:
                    continue
                eid, bn, ba, bc = p
                if eid in need or i < 60000:
                    r = prepare_source_record(bn, ba, bc)
                    idx_seed[bc].add_target_record(eid, r["norm_name"], r["core_tokens"], r["norm_addr"], r["addr_keys"], r["digits"], is_s2=is_s2, raw_addr=ba)
    for c in idx_seed.values():
        c.prune_frequent_keys()

    s1_records = {}
    with open("dataset/train/train_source1.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if p[0] in set(seed_sub):
                s1_records[p[0]] = prepare_source_record(p[1], p[2], p[3])

    seed_data = {}
    for sid in seed_sub:
        r = s1_records[sid]
        ci = idx_seed[r["country"]]
        items, prov = query_for_s1(ci, r, max_candidates=cfg.get("max_candidates", 60), adaptive=cfg.get("adaptive", True))
        cands, feats = featurize_s1(r, items, prov, ci, use_extra=cfg.get("use_extra", True))
        if feats:
            probs = [float(x) for x in model.predict(np.array(feats, dtype=np.float32))]
            seed_data[sid] = {
                "cands": cands, "probs": probs, "feats": feats,
                "prov": prov, "gt": gt_all.get(sid, set())
            }
        else:
            seed_data[sid] = {"cands": [], "probs": [], "feats": [], "prov": {}, "gt": gt_all.get(sid, set())}
    print("Done precomputing Seeded Holdout.", flush=True)

    def evaluate_policy(data_dict, policy_fn):
        gt_map = {}
        pred_map = {}
        for sid, d in data_dict.items():
            gt_map[sid] = d["gt"]
            pred_map[sid] = policy_fn(d)
        m = evaluate_metrics(gt_map, pred_map)
        return m

    # Baseline policy: tau_singleton=0.80, tau_min=0.40, delta_margin=0.18
    def baseline_policy(d):
        if not d["probs"]: return set()
        mp = max(d["probs"])
        if mp < 0.80: return set()
        cutoff = max(0.40, mp - 0.18)
        return {c for c, p in zip(d["cands"], d["probs"]) if p >= cutoff}

    # Policy grid to sweep:
    # 1. require secondary matches (p < mp) to have name_token_set >= min_nts OR has_name_channel
    print("\n--- BASELINE RESULTS ---")
    m_seed_base = evaluate_policy(seed_data, baseline_policy)
    m_nat_base = evaluate_policy(nat_data, baseline_policy)
    print(f"Baseline Seeded: Macro F0.5 = {m_seed_base['macro_f05']:.6f}, Prec = {m_seed_base['precision']:.4f}, Rec = {m_seed_base['recall']:.4f}")
    print(f"Baseline Natural: Macro F0.5 = {m_nat_base['macro_f05']:.6f}, Prec = {m_nat_base['precision']:.4f}, Rec = {m_nat_base['recall']:.4f}")

    print("\n--- EXPERIMENTING WITH DECISION RULES ---")
    for ts in [0.80, 0.82, 0.85]:
        for dm in [0.08, 0.10, 0.12, 0.15, 0.18]:
            for sec_mode in ["none", "core_or_tr", "tight_core"]:
                def make_policy(tau_s=ts, delta_m=dm, mode=sec_mode):
                    def pol(d):
                        if not d["probs"]: return set()
                        mp = max(d["probs"])
                        if mp < tau_s: return set()
                        cutoff = max(0.40, mp - delta_m)
                        preds = set()
                        for c, p, feat in zip(d["cands"], d["probs"], d["feats"]):
                            if p >= cutoff:
                                if p == mp:
                                    preds.add(c)
                                else:
                                    if mode == "none":
                                        preds.add(c)
                                    elif mode == "core_or_tr":
                                        # feat[4]: name_jaccard
                                        # feat[18]: name_acronym_match
                                        # feat[30]: tr_name_set
                                        # feat[31]: phon_jaccard
                                        # feat[46]: has_tr_chan
                                        nj = feat[4]
                                        acronym = feat[18]
                                        tr_nts = feat[30] if len(feat) > 30 else 0.0
                                        phon_j = feat[31] if len(feat) > 31 else 0.0
                                        has_tr = feat[46] if len(feat) > 46 else 0.0
                                        if nj > 0.0 or acronym > 0.5 or tr_nts >= 0.5 or phon_j >= 0.5 or has_tr > 0.5:
                                            preds.add(c)
                                    elif mode == "tight_core":
                                        nj = feat[4]
                                        acronym = feat[18]
                                        tr_nts = feat[30] if len(feat) > 30 else 0.0
                                        if nj >= 0.20 or acronym > 0.5 or tr_nts >= 0.6:
                                            preds.add(c)
                        return preds
                    return pol

                pol = make_policy(ts, dm, sec_mode)
                m_seed = evaluate_policy(seed_data, pol)
                if m_seed["macro_f05"] >= 0.969956:
                    m_nat = evaluate_policy(nat_data, pol)
                    print(f"PASS [ts={ts:.2f}, dm={dm:.2f}, mode={sec_mode}]: "
                          f"Seeded F0.5={m_seed['macro_f05']:.6f} (P={m_seed['precision']:.4f}, R={m_seed['recall']:.4f}) | "
                          f"Natural F0.5={m_nat['macro_f05']:.6f} (P={m_nat['precision']:.4f}, R={m_nat['recall']:.4f})", flush=True)
