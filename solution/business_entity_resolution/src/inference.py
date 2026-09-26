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
import numpy as np
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(__file__))
from normalization import (
    normalize_name, extract_name_tokens, 
    normalize_address, extract_address_digits, extract_address_blocking_keys
)
from blocking import CountryBlockingIndex
from features import compute_pairwise_features, compute_candidate_features_for_s1

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
    output_dir: str = _DEF_OUTPUT
):
    print("=" * 70)
    print(" Amazon ML Challenge 2026: Fast Partitioned Test Inference Pipeline")
    print("=" * 70)
    
    os.makedirs(output_dir, exist_ok=True)
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
    N_WORKERS = 1
    
    print(f"Model Path: {model_path}")
    print(f"Calibration Parameters: tau_singleton={tau_singleton:.2f}, tau_min={tau_min:.2f}, delta_margin={delta_margin:.2f}, max_candidates={max_candidates}")
    
    # 2. Record S1 ordered IDs only (takes ~20 MB RAM)
    print("Step 1: Indexing test_source1.tsv IDs...", flush=True)
    t0 = time.time()
    s1_ordered_ids = []
    country_counts = {}
    
    s1_file = os.path.join(test_dir, "test_source1.tsv")
    with open(s1_file, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            eid = parts[0]
            country = parts[3] if len(parts) > 3 else "UNKNOWN"
            s1_ordered_ids.append(eid)
            country_counts[country] = country_counts.get(country, 0) + 1
                
    total_s1 = len(s1_ordered_ids)
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
        
        c_idx = CountryBlockingIndex(country)
        
        # Load S2 for this country
        print(f"  Streaming S2 for {country}...", flush=True)
        t_s2 = time.time()
        count_s2 = 0
        with open(s2_file, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) >= 4 and parts[3] == country:
                    eid, bname, baddr, _ = parts
                    norm_nm = normalize_name(bname)
                    core_toks, _ = extract_name_tokens(norm_nm)
                    norm_ad = normalize_address(baddr)
                    addr_keys = extract_address_blocking_keys(baddr, norm_ad)
                    digits = extract_address_digits(baddr)
                    c_idx.add_target_record(eid, norm_nm, core_toks, norm_ad, addr_keys, digits, is_s2=1, raw_addr=baddr)
                    count_s2 += 1
        print(f"  Loaded {count_s2} S2 records in {time.time()-t_s2:.2f}s")
        
        # Load S3 for this country
        print(f"  Streaming S3 for {country}...", flush=True)
        t_s3 = time.time()
        count_s3 = 0
        with open(s3_file, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) >= 4 and parts[3] == country:
                    eid, bname, baddr, _ = parts
                    norm_nm = normalize_name(bname)
                    core_toks, _ = extract_name_tokens(norm_nm)
                    norm_ad = normalize_address(baddr)
                    addr_keys = extract_address_blocking_keys(baddr, norm_ad)
                    digits = extract_address_digits(baddr)
                    c_idx.add_target_record(eid, norm_nm, core_toks, norm_ad, addr_keys, digits, is_s2=0, raw_addr=baddr)
                    count_s3 += 1
        print(f"  Loaded {count_s3} S3 records in {time.time()-t_s3:.2f}s")
        
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
            for b_eid, b_cand_str, b_cands, n_f in b_recs:
                b_probs = probs[offset:offset+n_f]
                offset += n_f
                max_p = float(np.max(b_probs))
                if max_p < tau_singleton:
                    match_str = ""
                else:
                    cutoff = max(tau_min, max_p - delta_margin)
                    matches = [cid for cid, p in zip(b_cands, b_probs) if p >= cutoff]
                    match_str = ",".join(matches)
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
                    nums = [n for n in digits if len(n) in (4, 5, 6)]
                    snums = [n for n in digits if 1 <= len(n) <= 5]
                    
                    s1_rec = {
                        "name": bname,
                        "norm_name": norm_nm,
                        "core_tokens": core_toks,
                        "addr": baddr,
                        "norm_addr": norm_ad,
                        "addr_tokens": set(norm_ad.split()) if norm_ad else set(),
                        "addr_keys": addr_keys,
                        "digits": digits,
                        "prefix6": norm_nm[:6] if norm_nm else "",
                        "street_num": snums[0] if snums else "",
                        "postal": nums[0] if nums else "",
                        "country": country
                    }
                    
                    cand_items = c_idx.query_candidates(
                        norm_nm, core_toks, norm_ad, addr_keys,
                        s1_raw_addr=baddr,
                        max_candidates=max_candidates,
                        return_weights=True
                    )
                    
                    if not cand_items:
                        out_f.write(f"{eid}\t\t\n")
                        count_w += 1
                        continue
                        
                    cand_str = ",".join(cid for cid, w in cand_items)
                    
                    valid_cands, feat_rows = compute_candidate_features_for_s1(s1_rec, cand_items, c_idx.records)
                            
                    if feat_rows:
                        batch_records.append((eid, cand_str, valid_cands, len(feat_rows)))
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
        del c_idx
        gc.collect()
        gc.collect()
        
    # 4. Merge Temp Files into Final Submissions Preserving Exact S1 Order
    print("\n" + "=" * 70)
    print("Step 4: Merging temp results into final submission TSVs...", flush=True)
    t0 = time.time()
    
    res_map = {}
    for country in country_counts:
        country_file = country if re.fullmatch(r"[A-Za-z0-9_-]+", country) else hashlib.sha1(country.encode()).hexdigest()
        for w_idx in range(N_WORKERS):
            tfile = os.path.join(temp_dir, f"results_{country_file}_part_{w_idx}.tsv")
            if os.path.isfile(tfile):
                print(f"  Reading {tfile}...", flush=True)
                with open(tfile, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.rstrip("\r\n").split("\t")
                        if len(parts) >= 3:
                            res_map[parts[0]] = (parts[1], parts[2])
                        elif len(parts) == 2:
                            res_map[parts[0]] = (parts[1], "")
                        elif len(parts) == 1:
                            res_map[parts[0]] = ("", "")
                        
    cand_path = os.path.join(output_dir, "candidate_pairs.tsv")
    match_path = os.path.join(output_dir, "matching_results.tsv")
    if len(res_map) != total_s1:
        raise RuntimeError(f"Expected {total_s1} scored S1 entities, found {len(res_map)}")
    
    print(f"Writing {cand_path} and {match_path} in exact test order...", flush=True)
    matched_count = 0
    singleton_count = 0
    total_links = 0
    
    with open(cand_path, "w", encoding="utf-8") as f_cand, \
         open(match_path, "w", encoding="utf-8") as f_match:
         
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        
        for s1_id in s1_ordered_ids:
            c_str, m_str = res_map[s1_id]
            
            f_cand.write(f"{s1_id}\t{c_str}\n")
            f_match.write(f"{s1_id}\t{m_str}\n")
            
            if m_str:
                matched_count += 1
                total_links += len(m_str.split(","))
            else:
                singleton_count += 1
                
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
    args = parser.parse_args()
    
    run_inference(
        test_dir=args.test_dir,
        models_dir=args.models_dir,
        output_dir=args.output_dir
    )
