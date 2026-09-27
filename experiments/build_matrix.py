#!/usr/bin/env python3
"""Build + cache train/val feature matrices once for fast LGBM tuning."""
import os, sys, json, time
import numpy as np
sys.path.insert(0, "solution/business_entity_resolution/src")
from normalization import normalize_name, extract_name_tokens, normalize_address, extract_address_digits, extract_address_blocking_keys
from blocking import CountryBlockingIndex
from train import query_for_s1, featurize_s1

TRAIN_DIR = "dataset/train"
N_TRAIN = 25000
N_VAL = 6000


def load_all():
    gt = {}
    with open(os.path.join(TRAIN_DIR, "train_ground_truth.tsv"), encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            gt[p[0]] = set(x.strip() for x in (p[1] if len(p) > 1 else "").split(",") if x.strip())
    with open("experiments/splits/train_pool_s1_ids.json") as f:
        tr = json.load(f)[:N_TRAIN]
    with open("experiments/splits/dev_s1_ids.json") as f:
        dv = json.load(f)[:N_VAL]
    return gt, tr, dv


def build_index(gt, ids):
    need = set()
    for e in ids:
        need.update(gt.get(e, set()))
    idx = {"US": CountryBlockingIndex("US"), "India": CountryBlockingIndex("India")}
    for fn, is_s2 in [("train_source2.tsv", 1), ("train_source3.tsv", 0)]:
        with open(os.path.join(TRAIN_DIR, fn), encoding="utf-8") as f:
            next(f)
            for i, line in enumerate(f):
                p = line.rstrip("\n").split("\t")
                if len(p) < 4 or p[3] not in idx:
                    continue
                if p[0] in need or i < 200000:
                    nn = normalize_name(p[1]); ct, _ = extract_name_tokens(nn)
                    na = normalize_address(p[2])
                    idx[p[3]].add_target_record(p[0], nn, ct, na, extract_address_blocking_keys(p[2], na),
                                                extract_address_digits(p[2]), is_s2=is_s2, raw_addr=p[2], raw_name=p[1])
    for c in idx.values():
        c.prune_frequent_keys()
    return idx


def load_s1(ids):
    want = set(ids)
    out = {}
    with open(os.path.join(TRAIN_DIR, "train_source1.tsv"), encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if p[0] in want:
                nn = normalize_name(p[1]); ct, _ = extract_name_tokens(nn)
                na = normalize_address(p[2]); dg = extract_address_digits(p[2])
                out[p[0]] = {"name": p[1], "norm_name": nn, "core_tokens": ct, "addr": p[2],
                    "norm_addr": na, "addr_tokens": set(na.split()) if na else set(),
                    "addr_keys": extract_address_blocking_keys(p[2], na), "digits": dg,
                    "prefix6": nn[:6], "street_num": "", "postal": "", "country": p[3]}
    return out


def featurize_ids(ids, gt, idx, s1rec, seed_gt=True, neg_cap=80, collect_ids=False):
    Xs, ys = [], []
    Xch, ych = [], []
    idmap = [] if collect_ids else None
    idch = [] if collect_ids else None
    for si, sid in enumerate(ids):
        r = s1rec[sid]
        ci = idx[r["country"]]
        items, prov = query_for_s1(ci, r, adaptive=True)
        if seed_gt:
            ex = {c for c, _ in items}; mw = items[0][1] if items else 15.0
            for t in gt.get(sid, set()):
                if t in ci.records and t not in ex:
                    items.append((t, mw)); prov[t] = []
        vc, fr = featurize_s1(r, items, prov, ci, use_extra=True)
        scored = [((c in gt.get(sid, set())), next((w for cc, w in items if cc == c), 0.0), c, fv)
                  for c, fv in zip(vc, fr)]
        pos = [s for s in scored if s[0]]
        neg = sorted([s for s in scored if not s[0]], key=lambda s: -s[1])[:neg_cap]
        for is_m, _, cid, fv in pos + neg:
            Xs.append(fv); ys.append(1.0 if is_m else 0.0)
            if collect_ids:
                idmap.append((sid, cid))
        if (si + 1) % 2000 == 0:
            Xch.append(np.array(Xs, dtype=np.float32)); ych.append(np.array(ys, dtype=np.float32))
            if collect_ids:
                idch.append(np.array(idmap, dtype=object))
            Xs, ys = [], []
            if collect_ids:
                idmap = []
    if Xs:
        Xch.append(np.array(Xs, dtype=np.float32)); ych.append(np.array(ys, dtype=np.float32))
        if collect_ids:
            idch.append(np.array(idmap, dtype=object))
    X = np.vstack(Xch); y = np.concatenate(ych)
    del Xch, ych, Xs, ys
    if collect_ids:
        return X, y, np.concatenate(idch)
    return X, y


def main():
    t0 = time.time()
    gt, tr, dv = load_all()
    idx = build_index(gt, tr + dv)
    s1rec = load_s1(tr + dv)
    print(f"index+records in {time.time()-t0:.0f}s", flush=True)
    Xtr, ytr = featurize_ids(tr, gt, idx, s1rec, seed_gt=True)
    np.save("experiments/X_train_v3.npy", Xtr); np.save("experiments/y_train_v3.npy", ytr)
    print(f"train matrix {Xtr.shape} pos={int(ytr.sum())} in {time.time()-t0:.0f}s", flush=True)
    # unlearnable-positive audit: max of key sims among positives
    import collections
    zero_sig = 0; npos = 0
    for sid in tr:
        r = s1rec[sid]; ci = idx[r["country"]]
        items, prov = query_for_s1(ci, r, adaptive=True)
        vc, fr = featurize_s1(r, items, prov, ci, use_extra=True)
        for c, fv in zip(vc, fr):
            if c in gt.get(sid, set()):
                npos += 1
                # base idx: name_token_set=0, name_ratio=2, addr_token_set=7, tr_name_set=52-22=30
                if max(fv[0], fv[2], fv[7], fv[30], fv[31], fv[32]) < 0.25:
                    zero_sig += 1
    print(f"unlearnable positives (all sims<0.25): {zero_sig}/{npos} = {zero_sig/max(1,npos):.3f}", flush=True)
    Xdv, ydv, dvids = featurize_ids(dv, gt, idx, s1rec, seed_gt=False, collect_ids=True)
    np.save("experiments/X_val_v3.npy", Xdv); np.save("experiments/y_val_v3.npy", ydv)
    np.save("experiments/val_ids_v3.npy", np.array(dvids, dtype=object))
    print(f"val matrix {Xdv.shape} in {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
