import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import sys
import time
import json
import gc
import re
import hashlib
import sqlite3
import numpy as np
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(__file__))
from normalization import (
    normalize_name, extract_name_tokens, 
    normalize_address, extract_address_digits, extract_address_blocking_keys,
    prepare_source_record,
)
from disk_blocking import DiskCountryBlockingIndex
from features import compute_pairwise_features, compute_candidate_features_for_s1, FEATURE_NAMES
from features2 import extra_batch_for_s1, EXTRA_NAMES
from graph import apply_graph_bridge

import argparse

NUM_ONLY_RE = re.compile(r"\b\d+\b")

def _resolve_default_paths():
    if os.path.isdir("solution/business_entity_resolution/models"):
        default_models = "solution/business_entity_resolution/models"
    elif os.path.isdir("code/business_entity_resolution/models"):
        default_models = "code/business_entity_resolution/models"
    elif os.path.isdir("models"):
        default_models = "models"
    else:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        default_models = os.path.join(base_dir, "models")
        
    test_candidates = [
        "student_resource/dataset/test",
        "dataset/test",
    ]
    default_test = "student_resource/dataset/test"
    for tc in test_candidates:
        if os.path.isdir(tc):
            default_test = tc
            break
            
    default_output = "output"
    return default_test, default_models, default_output

_DEF_TEST, _DEF_MODELS, _DEF_OUTPUT = _resolve_default_paths()

def run_inference(
    test_dir: str = _DEF_TEST,
    models_dir: str = _DEF_MODELS,
    output_dir: str = _DEF_OUTPUT,
    score_output: str | None = None,
):
    print("=" * 70)
    print(" Amazon ML Challenge 2026: Fast Partitioned Test Inference Pipeline")
    print("=" * 70)
    
    os.makedirs(output_dir, exist_ok=True)
    if score_output and os.path.exists(os.path.join(output_dir, "matching_results.tsv")):
        raise FileExistsError("Score export requires a fresh output directory to protect existing results")
    if score_output and os.path.abspath(score_output) == os.path.abspath(os.path.join(output_dir, "matching_results.tsv")):
        raise ValueError("Score output must differ from matching_results.tsv")
    temp_dir = os.path.join(output_dir, "_temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    # 1. Load Trained Model & Thresholds
    model_path = os.path.join(models_dir, "lgbm_matcher.txt")
    config_path = os.path.join(models_dir, "config.json")
    
    if not os.path.isfile(model_path) or not os.path.isfile(config_path):
        raise FileNotFoundError(f"Trained model or config not found in {models_dir}. Run train.py first.")
        
    with open(config_path, "r") as f:
        config = json.load(f)
        
    tau_singleton = config.get("tau_singleton", config.get("best_threshold", 0.72))
    tau_min = config.get("tau_min", 0.55)
    delta_margin = config.get("delta_margin", 0.15)
    max_candidates = config.get("max_candidates", 60)
    adaptive = config.get("adaptive", True)
    use_extra = config.get("use_extra", len(config.get("feature_names", ())) > len(FEATURE_NAMES))
    use_graph = config.get("use_graph", False)
    if score_output and use_graph:
        raise ValueError("Score export requires use_graph=false")
    graph_score_min = config.get("graph_score_min", 0.999)
    graph_bridge_min = config.get("graph_bridge_min", 80)
    N_WORKERS = 1
    
    model_features = lgb.Booster(model_file=model_path).num_feature()
    expected_names = FEATURE_NAMES + (EXTRA_NAMES if use_extra else [])
    configured_names = config.get("feature_names", [])
    if model_features != len(expected_names) or configured_names != expected_names:
        raise ValueError("Model/config/runtime feature ordering mismatch")
    score_file = None
    score_partial = None
    if score_output:
        os.makedirs(os.path.dirname(os.path.abspath(score_output)), exist_ok=True)
        score_partial = score_output + ".partial"
        score_file = open(score_partial, "w", encoding="utf-8")
        score_file.write("target_id\tcandidate_s1_id\tmodel_probability\tretrieval_score\tcandidate_rank\n")
    print(f"Model Path: {model_path}")
    print(f"Calibration Parameters: tau_singleton={tau_singleton:.2f}, tau_min={tau_min:.2f}, delta_margin={delta_margin:.2f}, max_candidates={max_candidates}")
    
    # Count S1 by country. Final output order is recovered by streaming this file.
    print("Step 1: Counting test_source1.tsv IDs...", flush=True)
    t0 = time.time()
    country_counts = {}
    total_s1 = 0
    
    s1_file = os.path.join(test_dir, "test_source1.tsv")
    with open(s1_file, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            eid = parts[0]
            country = parts[3] if len(parts) > 3 else "UNKNOWN"
            total_s1 += 1
            country_counts[country] = country_counts.get(country, 0) + 1

    print(f"Recorded {total_s1} S1 entities in {time.time()-t0:.2f}s:")
    for c, cnt in country_counts.items():
        print(f"  {c:7s}: {cnt:8d} entities")
        
    s2_file = os.path.join(test_dir, "test_source2.tsv")
    s3_file = os.path.join(test_dir, "test_source3.tsv")
    
    # 3. Country-by-country processing
    for country in country_counts:
        n_entities = country_counts.get(country, 0)
        if n_entities == 0:
            continue
            
        country_file = country if re.fullmatch(r"[A-Za-z0-9_-]+", country) else hashlib.sha1(country.encode()).hexdigest()
        existing_parts = [os.path.join(temp_dir, f"results_{country_file}_part_0.tsv")]
            
        print("\n" + "-" * 70)
        print(f"Processing Country: {country} ({n_entities} S1 entities)...", flush=True)
        t_country_start = time.time()
        
        index_dir = os.path.join(temp_dir, "indices")
        os.makedirs(index_dir, exist_ok=True)
        index_path = os.path.join(index_dir, f"{country_file}.sqlite")
        build_index = not os.path.isfile(index_path)
        c_idx = DiskCountryBlockingIndex(country, index_path, create=build_index)
        c_idx.set_source_fingerprint([s2_file, s3_file])
        
        # Load S2 for this country
        print(f"  Streaming S2 for {country}...", flush=True)
        t_s2 = time.time()
        count_s2 = 0
        if build_index:
            with open(s2_file, "r", encoding="utf-8") as f:
                next(f)
                for line in f:
                    parts = line.rstrip("\r\n").split("\t")
                    if len(parts) >= 4 and parts[3] == country:
                        eid, bname, baddr, _ = parts
                        rec = prepare_source_record(bname, baddr, country)
                        c_idx.add_target_record(eid, rec["norm_name"], rec["core_tokens"], rec["norm_addr"],
                                                rec["addr_keys"], rec["digits"], is_s2=1,
                                                raw_addr=baddr, raw_name=bname)
                        count_s2 += 1
        print(f"  Loaded {count_s2} S2 records in {time.time()-t_s2:.2f}s")
        
        # Load S3 for this country
        print(f"  Streaming S3 for {country}...", flush=True)
        t_s3 = time.time()
        count_s3 = 0
        if build_index:
            with open(s3_file, "r", encoding="utf-8") as f:
                next(f)
                for line in f:
                    parts = line.rstrip("\r\n").split("\t")
                    if len(parts) >= 4 and parts[3] == country:
                        eid, bname, baddr, _ = parts
                        rec = prepare_source_record(bname, baddr, country)
                        c_idx.add_target_record(eid, rec["norm_name"], rec["core_tokens"], rec["norm_addr"],
                                                rec["addr_keys"], rec["digits"], is_s2=0,
                                                raw_addr=baddr, raw_name=bname)
                        count_s3 += 1
        print(f"  Loaded {count_s3} S3 records in {time.time()-t_s3:.2f}s")
        
        if build_index:
            c_idx.prune_frequent_keys()
        
        # Stream S1 sequentially with batch prediction & 4 internal predict threads
        part_file = existing_parts[0]
        print(f"  Streaming S1 with batched inference (100% memory safe)...", flush=True)
        t_infer = time.time()
        
        model = lgb.Booster(model_file=model_path)
        model.params["num_threads"] = 4
        
        count_w = 0
        last_printed = 0
        BATCH_SIZE = 100
        batch_records = []
        batch_features = []
        
        def flush_batch(b_recs, b_feats, out_file):
            nonlocal count_w
            if not b_recs:
                return
            probs = model.predict(np.array(b_feats, dtype=np.float32))
            offset = 0
            for b_eid, b_cand_str, b_cands, b_weights, b_s1rec, n_f in b_recs:
                b_probs = probs[offset:offset+n_f]
                offset += n_f
                max_p = float(np.max(b_probs))
                if max_p < tau_singleton:
                    match_str = ""
                else:
                    cutoff = max(tau_min, max_p - delta_margin)
                    matches = [cid for cid, p in zip(b_cands, b_probs) if p >= cutoff]
                    if use_graph:
                        prob_map = {cid: float(p) for cid, p in zip(b_cands, b_probs)}
                        bridged = apply_graph_bridge(b_s1rec, matches, prob_map, c_idx, model,
                                                     use_extra=use_extra, score_min=graph_score_min,
                                                     bridge_min=graph_bridge_min)
                        if bridged:
                            matches = matches + [c for c in bridged if c not in matches]
                            b_cand_str = b_cand_str + "," + ",".join(c for c in bridged if c not in b_cand_str.split(","))
                    match_str = ",".join(matches)
                if score_file and match_str:
                    accepted = set(match_str.split(","))
                    for rank, (cid, weight, probability) in enumerate(zip(b_cands, b_weights, b_probs), 1):
                        if cid in accepted:
                            score_file.write(f"{cid}\t{b_eid}\t{float(probability):.17g}\t{weight:.17g}\t{rank}\n")
                out_file.write(f"{b_eid}\t{b_cand_str}\t{match_str}\n")
                count_w += 1
            b_recs.clear()
            b_feats.clear()

        with open(part_file, "w", encoding="utf-8") as out_f:
            with open(s1_file, "r", encoding="utf-8") as f:
                next(f)
                for line in f:
                    parts = line.rstrip("\r\n").split("\t")
                    if len(parts) < 4 or parts[3] != country:
                        continue
                        
                    eid, bname, baddr, _ = parts
                    norm_nm = normalize_name(bname)
                    core_toks, _ = extract_name_tokens(norm_nm)
                    norm_ad = normalize_address(baddr)
                    addr_keys = extract_address_blocking_keys(baddr, norm_ad)
                    digits = extract_address_digits(baddr)
                    s1_rec = prepare_source_record(bname, baddr, country)
                    
                    cand_items, prov = c_idx.query_candidates(
                        norm_nm, core_toks, norm_ad, addr_keys,
                        s1_raw_addr=baddr, s1_raw_name=bname,
                        max_candidates=max_candidates,
                        return_weights=True, return_provenance=True,
                        adaptive=adaptive
                    )

                    if not cand_items:
                        out_f.write(f"{eid}\t\t\n")
                        count_w += 1
                        continue

                    cand_str = ",".join(cid for cid, w in cand_items)

                    valid_cands, feat_rows = compute_candidate_features_for_s1(s1_rec, cand_items, c_idx.records)
                    if use_extra and valid_cands:
                        wmap = {c: float(w) for c, w in cand_items}
                        sub_items = [(c, wmap.get(c, 0.0)) for c in valid_cands]
                        sub_prov = {c: prov.get(c, ()) for c in valid_cands}
                        _, extra_rows = extra_batch_for_s1(s1_rec, sub_items, c_idx.records, prov_map=sub_prov)
                        feat_rows = [b + e for b, e in zip(feat_rows, extra_rows)]

                    if feat_rows:
                        weights = {cid: float(weight) for cid, weight in cand_items}
                        batch_records.append((eid, cand_str, valid_cands,
                                              [weights[cid] for cid in valid_cands], s1_rec, len(feat_rows)))
                        batch_features.extend(feat_rows)
                    else:
                        out_f.write(f"{eid}\t{cand_str}\t\n")
                        count_w += 1
                        
                    if len(batch_records) >= BATCH_SIZE:
                        flush_batch(batch_records, batch_features, out_f)
                        
                    if count_w - last_printed >= 20000:
                        last_printed = count_w
                        elapsed = time.time() - t_infer
                        rate = count_w / max(1e-5, elapsed)
                        print(f"    Processed {count_w:,}/{n_entities:,} entities (~{rate:.0f} ent/s)...", flush=True)

                flush_batch(batch_records, batch_features, out_f)
                    
        print(f"  Finished {country} in {time.time()-t_country_start:.2f}s.")
        
        # Free memory before next country
        c_idx.close()
        
    # 4. Merge Temp Files into Final Submissions Preserving Exact S1 Order
    print("\n" + "=" * 70)
    print("Step 4: Merging temp results into final submission TSVs...", flush=True)
    t0 = time.time()
    
    result_db_path = os.path.join(temp_dir, "results.sqlite")
    result_db = sqlite3.connect(result_db_path)
    result_db.execute("CREATE TABLE IF NOT EXISTS results (eid TEXT PRIMARY KEY, candidates TEXT, matches TEXT)")
    result_db.execute("DELETE FROM results")
    merged_rows = 0
    for country in country_counts:
        country_file = country if re.fullmatch(r"[A-Za-z0-9_-]+", country) else hashlib.sha1(country.encode()).hexdigest()
        for w_idx in range(N_WORKERS):
            tfile = os.path.join(temp_dir, f"results_{country_file}_part_{w_idx}.tsv")
            if os.path.isfile(tfile):
                print(f"  Reading {tfile}...", flush=True)
                with open(tfile, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.rstrip("\r\n").split("\t")
                        if len(parts) < 3:
                            raise ValueError(f"Malformed temporary result row in {tfile}")
                        result_db.execute("INSERT INTO results VALUES (?,?,?)", (parts[0], parts[1], parts[2]))
                        merged_rows += 1
                        if merged_rows % 10_000 == 0:
                            result_db.commit()
    result_db.commit()
                        
    cand_path = os.path.join(output_dir, "candidate_pairs.tsv")
    match_path = os.path.join(output_dir, "matching_results.tsv")
    cand_partial = cand_path + ".partial"
    match_partial = match_path + ".partial"
    if merged_rows != total_s1:
        raise RuntimeError(f"Expected {total_s1} scored S1 entities, found {merged_rows}")
    
    print(f"Writing {cand_path} and {match_path} in exact test order...", flush=True)
    matched_count = 0
    singleton_count = 0
    total_links = 0
    
    with open(cand_partial, "w", encoding="utf-8") as f_cand, \
         open(match_partial, "w", encoding="utf-8") as f_match:
         
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        
        with open(s1_file, "r", encoding="utf-8") as source:
            next(source)
            for line in source:
                s1_id = line.partition("\t")[0]
                result = result_db.execute("SELECT candidates,matches FROM results WHERE eid=?", (s1_id,)).fetchone()
                if result is None:
                    raise RuntimeError(f"No scored result for {s1_id}")
                c_str, m_str = result
            
                f_cand.write(f"{s1_id}\t{c_str}\n")
                f_match.write(f"{s1_id}\t{m_str}\n")
            
                if m_str:
                    matched_count += 1
                    total_links += len(m_str.split(","))
                else:
                    singleton_count += 1
    result_db.close()
    os.replace(cand_partial, cand_path)
    os.replace(match_partial, match_path)
    if score_file:
        score_file.close()
        os.replace(score_partial, score_output)
                
    # Clean up temp directory
    for country in country_counts:
        country_file = country if re.fullmatch(r"[A-Za-z0-9_-]+", country) else hashlib.sha1(country.encode()).hexdigest()
        for w_idx in range(N_WORKERS):
            tfile = os.path.join(temp_dir, f"results_{country_file}_part_{w_idx}.tsv")
            if os.path.isfile(tfile):
                try:
                    os.remove(tfile)
                except Exception:
                    pass
    try:
        os.rmdir(temp_dir)
    except Exception:
        pass
        
    print(f"Successfully generated submission files in {time.time()-t0:.2f}s!")
    print(f"Final Output Summary:")
    print(f"  Total S1 Entities   : {total_s1}")
    print(f"  Matched S1 Entities : {matched_count} ({100.0*matched_count/total_s1:.2f}%)")
    print(f"  Predicted Singletons: {singleton_count} ({100.0*singleton_count/total_s1:.2f}%)")
    print(f"  Total Linked Matches: {total_links}")
    print(f"  Avg Matches / Linked: {total_links/max(1, matched_count):.2f}")
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Amazon ML Challenge 2026: Test Inference Pipeline")
    parser.add_argument("--test-dir", default=_DEF_TEST, help=f"Path to test dataset directory (default: {_DEF_TEST})")
    parser.add_argument("--models-dir", default=_DEF_MODELS, help=f"Path to trained models directory (default: {_DEF_MODELS})")
    parser.add_argument("--output-dir", default=_DEF_OUTPUT, help=f"Path to output directory (default: {_DEF_OUTPUT})")
    parser.add_argument("--score-output", help="Write accepted pair model probabilities to this TSV")
    args = parser.parse_args()
    
    run_inference(
        test_dir=args.test_dir,
        models_dir=args.models_dir,
        output_dir=args.output_dir,
        score_output=args.score_output,
    )
