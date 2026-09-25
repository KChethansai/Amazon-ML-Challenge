import polars as pl, json, os, re
from collections import Counter

D = "."
OUT = {}
COLS = ["entity_id", "business_name", "business_address", "country"]

def scan_source(path, tag):
    # single-pass streaming aggregates; never materialize whole frame
    rows = 0
    ids = set()
    dup_ids = 0
    dup_rows = Counter()
    empty_name = empty_addr = empty_country = ws_only_name = ws_only_addr = 0
    countries = Counter()
    prefixes = Counter()
    nl_min, nl_max, na_sum, nal_min, nal_max, naal_sum = 1e9, 0, 0, 1e9, 0, 0
    ncount = 0
    for chunk in pl.read_csv_batched(
        path, separator="\t", has_header=True, columns=COLS,
        batch_size=200_000, infer_schema_length=10_000, null_values=[""],
        truncate_ragged_lines=True,
    ):
        s1 = chunk["entity_id"].to_list()
        nm = chunk["business_name"].fill_null("").to_list()
        ad = chunk["business_address"].fill_null("").to_list()
        co = chunk["country"].fill_null("").to_list()
        n = len(s1)
        rows += n
        ncount += n
        na_sum += sum(len(x) for x in nm)
        naal_sum += sum(len(x) for x in ad)
        for i in range(n):
            if s1[i] in ids:
                dup_ids += 1
            else:
                ids.add(s1[i])
            prefixes[s1[i].split("-")[0]] += 1
            countries[co[i]] += 1
            Lnm, Lad = len(nm[i]), len(ad[i])
            nl_min = min(nl_min, Lnm); nl_max = max(nl_max, Lnm)
            nal_min = min(nal_min, Lad); nal_max = max(nal_max, Lad)
            if Lnm == 0: empty_name += 1
            elif nm[i].strip() == "": ws_only_name += 1
            if Lad == 0: empty_addr += 1
            elif ad[i].strip() == "": ws_only_addr += 1
            if co[i].strip() == "" and Lad >= 0:
                if co[i] == "": empty_country += 1
            dup_rows[(s1[i], nm[i], ad[i], co[i])] += 1
    dups_total = sum(c - 1 for c in dup_rows.values())
    OUT[tag] = {
        "rows": rows, "unique_ids": len(ids), "dup_id_count": dup_ids,
        "exact_dup_row_extra": dups_total,
        "empty_name": empty_name, "ws_only_name": ws_only_name,
        "empty_addr": empty_addr, "ws_only_addr": ws_only_addr,
        "empty_country": empty_country,
        "countries": dict(countries.most_common()),
        "id_prefixes": dict(prefixes),
        "name_len": {"min": int(nl_min), "max": int(nl_max), "mean": round(na_sum / ncount, 2)},
        "addr_len": {"min": int(nal_min), "max": int(nal_max), "mean": round(naal_sum / ncount, 2)},
    }
    return ids

for f, t in [("train_source1.tsv", "train_S1"), ("train_source2.tsv", "train_S2"),
             ("train_source3.tsv", "train_S3"), ("test_source1.tsv", "test_S1"),
             ("test_source2.tsv", "test_S2"), ("test_source3.tsv", "test_S3")]:
    print("scanning", f, flush=True)
    globals()[t + "_ids"] = scan_source(os.path.join(D, f), t)

# ground truth integrity
gt_path = os.path.join(D, "train_ground_truth.tsv")
gt_rows = 0
s1_seen = set()
dup_s1_gt = 0
malformed = 0
match_counts = []
matched_to_missing = 0
wrong_prefix = 0
self_match = 0
by_country_matches = Counter()
total_positives = 0
with open(gt_path) as fh:
    next(fh)
    for line in fh:
        gt_rows += 1
        parts = line.rstrip("\n").rstrip("\t").split("\t")
        sid = parts[0]
        mstr = parts[1] if len(parts) > 1 else ""
        if sid in s1_seen:
            dup_s1_gt += 1
        s1_seen.add(sid)
        if not mstr.strip():
            match_counts.append(0)
            continue
        ids = [x for x in mstr.split(",") if x.strip()]
        match_counts.append(len(ids))
        total_positives += len(ids)
        for mid in ids:
            pfx = mid.split("-")[0]
            if pfx not in ("S2", "S3"):
                wrong_prefix += 1
            if mid == sid:
                self_match += 1
            if pfx == "S2" and mid not in train_S2_ids:
                matched_to_missing += 1
            elif pfx == "S3" and mid not in train_S3_ids:
                matched_to_missing += 1
mc = sorted(match_counts)
nn = len(mc)
OUT["ground_truth"] = {
    "rows": gt_rows, "unique_s1_in_gt": len(s1_seen), "dup_s1_rows": dup_s1_gt,
    "singleton_rate": round(sum(1 for c in mc if c == 0) / nn, 4),
    "matches_mean": round(total_positives / nn, 3),
    "matches_median": mc[nn // 2], "matches_max": mc[-1],
    "distribution": dict(Counter(mc)),
    "matched_ids_not_in_source": matched_to_missing,
    "wrong_prefix_ids": wrong_prefix, "self_matches": self_match,
}
assert gt_rows == OUT["train_S1"]["rows"], "GT row count != S1 row count"
print(json.dumps(OUT, indent=2)[:6000])
with open("recon_exhaustive.json", "w") as o:
    json.dump(OUT, o, indent=2)
print("WROTE recon_exhaustive.json")