#!/usr/bin/env python3
"""EXP: upgraded retrieval recall sweep + per-entity diagnostics (seeded pool)."""
import os, sys, json, time, csv
sys.path.insert(0, "solution/business_entity_resolution/src")
from normalization import normalize_name, extract_name_tokens, normalize_address, extract_address_digits, extract_address_blocking_keys
from blocking import CountryBlockingIndex, _trigrams
from views import has_indic

TRAIN_DIR = "dataset/train"
BUDGETS = [20, 35, 50, 75, 100, 150, 200, 300]
N_PER_COUNTRY = 1000
BG_PER_SOURCE = 100000


def load_gt():
    gt = {}
    with open(os.path.join(TRAIN_DIR, "train_ground_truth.tsv"), encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            gt[p[0]] = set(x.strip() for x in (p[1] if len(p) > 1 else "").split(",") if x.strip())
    return gt


def main():
    t0 = time.time()
    with open("experiments/splits/holdout_s1_ids.json") as f:
        hold = json.load(f)
    gt = load_gt()
    # stratify
    s1country = {}
    hold_set = set(hold)
    with open(os.path.join(TRAIN_DIR, "train_source1.tsv"), encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if p[0] in hold_set:
                s1country[p[0]] = p[3]
    us = [e for e in hold if s1country.get(e) == "US"][:N_PER_COUNTRY]
    ind = [e for e in hold if s1country.get(e) == "India"][:N_PER_COUNTRY]
    sub = us + ind
    need = set()
    for e in sub:
        need.update(gt.get(e, set()))
    idx = {"US": CountryBlockingIndex("US"), "India": CountryBlockingIndex("India")}
    s1rec = {}
    for fn, is_s2 in [("train_source2.tsv", 1), ("train_source3.tsv", 0)]:
        with open(os.path.join(TRAIN_DIR, fn), encoding="utf-8") as f:
            next(f)
            for i, line in enumerate(f):
                p = line.rstrip("\n").split("\t")
                if len(p) < 4 or p[3] not in idx:
                    continue
                if p[0] in need or i < BG_PER_SOURCE:
                    nn = normalize_name(p[1]); ct, _ = extract_name_tokens(nn)
                    na = normalize_address(p[2])
                    idx[p[3]].add_target_record(p[0], nn, ct, na, extract_address_blocking_keys(p[2], na),
                                                extract_address_digits(p[2]), is_s2=is_s2, raw_addr=p[2], raw_name=p[1])
    for c in idx.values():
        c.prune_frequent_keys()
    sub_set = set(sub)
    with open(os.path.join(TRAIN_DIR, "train_source1.tsv"), encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if p[0] in sub_set:
                nn = normalize_name(p[1]); ct, _ = extract_name_tokens(nn)
                na = normalize_address(p[2])
                dg = extract_address_digits(p[2])
                s1rec[p[0]] = {"name": p[1], "norm_name": nn, "core_tokens": ct, "addr": p[2], "norm_addr": na,
                    "addr_keys": extract_address_blocking_keys(p[2], na), "digits": dg, "country": p[3]}
    print(f"index built in {time.time()-t0:.0f}s", flush=True)
    pair_hits = {b: 0 for b in BUDGETS}
    pair_tot = 0
    ent_full = {b: 0 for b in BUDGETS}
    ent_tot = 0
    rows = []
    for sid in sub:
        r = s1rec[sid]
        ci = idx[r["country"]]
        g = gt.get(sid, set())
        if not g:
            continue
        pair_tot += len(g); ent_tot += 1
        # full-detail query at 300 with provenance
        cands300, prov300 = ci.query_candidates(r["norm_name"], r["core_tokens"], r["norm_addr"], r["addr_keys"],
            s1_raw_addr=r["addr"], s1_raw_name=r["name"], max_candidates=300, return_weights=False, return_provenance=True)
        rank300 = {c: i for i, c in enumerate(cands300)}
        rec = 0
        minrank = None
        chan_hit = set()
        for t in g:
            if t in rank300:
                rec += 1
                rk = rank300[t]
                minrank = rk if minrank is None else min(minrank, rk)
                chan_hit.update(prov300.get(t, ()))
                for b in BUDGETS:
                    if rk < b:
                        pair_hits[b] += 1
            # else: missed at 300
        for b in BUDGETS:
            if rec == len(g) and all(rank300[t] < b for t in g if t in rank300) and rec == len(g):
                pass
        # entity full recall per budget
        for b in BUDGETS:
            if all((t in rank300 and rank300[t] < b) for t in g):
                ent_full[b] += 1
        # failure-mode flags
        q_tri = _trigrams(r["norm_name"])
        zero_overlap = 0
        for t in g:
            if t not in rank300:
                rec_t = ci.records.get(t)
                if rec_t is None:
                    zero_overlap += 1
                    continue
                c_tri = _trigrams(rec_t[0])
                shared_tok = set(r["core_tokens"]) & set(rec_t[3] if len(rec_t) > 3 else [])
                if not shared_tok and not (q_tri & c_tri):
                    zero_overlap += 1
        maxpost = 0
        for tok in r["core_tokens"]:
            pl = ci.token_idx.get(tok)
            if pl is not None:
                maxpost = max(maxpost, len(pl))
        rows.append({"s1_id": sid, "country": r["country"], "n_gt": len(g), "n_recovered_300": rec,
            "full_recall_300": int(rec == len(g)), "min_rank": minrank if minrank is not None else -1,
            "chan_hit": ";".join(sorted(chan_hit)), "cand_count_300": len(cands300),
            "zero_overlap_missed": zero_overlap, "indic": int(has_indic(r["name"]) or has_indic(r["addr"])),
            "empty_addr": int(not r["norm_addr"]), "single_token": int(len(r["core_tokens"]) <= 1),
            "max_token_posting": maxpost, "generic": int(maxpost > 5000), "multi_match": int(len(g) > 1)})
    with open("experiments/candidate_recall_by_entity.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    # failure-mode aggregation
    import collections
    agg = collections.Counter()
    for rr in rows:
        if rr["full_recall_300"] == 0:
            key = []
            if rr["zero_overlap_missed"] > 0: key.append("zero_overlap")
            if rr["indic"]: key.append("indic")
            if rr["empty_addr"]: key.append("empty_addr")
            if rr["single_token"]: key.append("single_token")
            if rr["generic"]: key.append("generic")
            if rr["multi_match"]: key.append("multi_match")
            agg[";".join(key) if key else "other"] += 1
    with open("experiments/candidate_failure_modes.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["mode", "n_entities"])
        w.writerows(sorted(agg.items(), key=lambda x: -x[1]))
    out = {"pair_recall": {str(b): pair_hits[b] / max(1, pair_tot) for b in BUDGETS},
           "entity_full_recall": {str(b): ent_full[b] / max(1, ent_tot) for b in BUDGETS},
           "entities": ent_tot, "pairs": pair_tot, "runtime_seconds": round(time.time() - t0, 1)}
    with open("experiments/exp_recall_upgraded.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2), flush=True)

if __name__ == "__main__":
    main()
