import os
import sys
import time
import json
import gc
import numpy as np
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(__file__))
from normalization import (
    normalize_name, extract_name_tokens, 
    normalize_address, extract_address_digits, extract_address_blocking_keys
)
from blocking import CountryBlockingIndex
from features import compute_pairwise_features

import argparse

def _resolve_default_paths():
    # Check common relative locations first
    if os.path.isdir("solution/business_entity_resolution/models"):
        default_models = "solution/business_entity_resolution/models"
    elif os.path.isdir("code/business_entity_resolution/models"):
        default_models = "code/business_entity_resolution/models"
    elif os.path.isdir("models"):
        default_models = "models"
    else:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        default_models = os.path.join(base_dir, "models")
        
    # Test dir
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
    print(" Amazon ML Challenge 2026: Memory-Bounded Test Inference Pipeline")
    print("=" * 70)
    
    os.makedirs(output_dir, exist_ok=True)
    temp_dir = os.path.join(output_dir, "_temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    # 1. Load Trained Model & Threshold
    model_path = os.path.join(models_dir, "lgbm_matcher.txt")
    config_path = os.path.join(models_dir, "config.json")
    
    if not os.path.isfile(model_path) or not os.path.isfile(config_path):
        raise FileNotFoundError(f"Trained model or config not found in {models_dir}. Run train.py first.")
        
    with open(config_path, "r") as f:
        config = json.load(f)
    threshold = config.get("best_threshold", 0.65)
    print(f"Loaded LightGBM model from {model_path} (F0.5 threshold = {threshold:.2f})")
    model = lgb.Booster(model_file=model_path)
    
    # 2. Record S1 ordered IDs only (takes ~20 MB RAM)
    print("Step 1: Indexing test_source1.tsv IDs...", flush=True)
    t0 = time.time()
    s1_ordered_ids = []
    country_counts = {"France": 0, "US": 0, "India": 0}
    
    s1_file = os.path.join(test_dir, "test_source1.tsv")
    with open(s1_file, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            eid = parts[0]
            country = parts[3] if len(parts) > 3 else "UNKNOWN"
            s1_ordered_ids.append(eid)
            if country in country_counts:
                country_counts[country] += 1
                
    total_s1 = len(s1_ordered_ids)
    print(f"Recorded {total_s1} S1 entities in {time.time()-t0:.2f}s:")
    for c, cnt in country_counts.items():
        print(f"  {c:7s}: {cnt:8d} entities")
        
    s2_file = os.path.join(test_dir, "test_source2.tsv")
    s3_file = os.path.join(test_dir, "test_source3.tsv")
    
    # 3. Country-by-Country Processing with Streaming Disk Spill
    for country in ["France", "US", "India"]:
        n_entities = country_counts.get(country, 0)
        if n_entities == 0:
            continue
            
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
                    c_idx.add_target_record(eid, norm_nm, core_toks, norm_ad, addr_keys, digits, is_s2=1)
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
                    c_idx.add_target_record(eid, norm_nm, core_toks, norm_ad, addr_keys, digits, is_s2=0)
                    count_s3 += 1
        print(f"  Loaded {count_s3} S3 records in {time.time()-t_s3:.2f}s")
        
        c_idx.prune_frequent_keys()
        
        # Stream S1 and write directly to country temp file
        temp_out_file = os.path.join(temp_dir, f"results_{country}.tsv")
        print(f"  Streaming S1 and writing results to {temp_out_file}...", flush=True)
        t_infer = time.time()
        
        processed_in_country = 0
        with open(temp_out_file, "w", encoding="utf-8") as out_f:
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
                    
                    s1_rec = {
                        "norm_name": norm_nm,
                        "core_tokens": set(core_toks),
                        "norm_addr": norm_ad,
                        "addr_tokens": set(norm_ad.split()) if norm_ad else set(),
                        "digits": digits
                    }
                    
                    cands = c_idx.query_candidates(
                        norm_nm, core_toks, norm_ad, addr_keys, max_candidates=25
                    )
                    
                    if not cands:
                        out_f.write(f"{eid}\t\t\n")
                        processed_in_country += 1
                        continue
                        
                    cand_str = ",".join(cands)
                    
                    # Compute features & predict
                    feat_batch = []
                    valid_cands = []
                    for cid in cands:
                        tgt_rec = c_idx.records.get(cid)
                        if tgt_rec:
                            feat_batch.append(compute_pairwise_features(s1_rec, tgt_rec))
                            valid_cands.append(cid)
                            
                    if feat_batch:
                        probs = model.predict(np.array(feat_batch, dtype=np.float32))
                        matches = [cid for cid, p in zip(valid_cands, probs) if p >= threshold]
                        match_str = ",".join(matches)
                    else:
                        match_str = ""
                        
                    out_f.write(f"{eid}\t{cand_str}\t{match_str}\n")
                    processed_in_country += 1
                    
                    if processed_in_country % 50000 == 0:
                        elapsed = time.time() - t_infer
                        rate = processed_in_country / max(1e-5, elapsed)
                        print(f"    Processed {processed_in_country}/{n_entities} S1 entities ({rate:.0f} ent/s)...", flush=True)
                        
        print(f"  Finished {country} in {time.time()-t_country_start:.2f}s.")
        
        # Free memory before next country
        del c_idx
        gc.collect()
        
    # 4. Merge Temp Files into Final Submissions Preserving Exact S1 Order
    print("\n" + "=" * 70)
    print("Step 4: Merging temp results into final submission TSVs...", flush=True)
    t0 = time.time()
    
    # Load temp mappings: s1_id -> (cand_str, match_str)
    # 1.73M pairs of two strings takes ~150 MB RAM
    res_map = {}
    for country in ["France", "US", "India"]:
        tfile = os.path.join(temp_dir, f"results_{country}.tsv")
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
    
    print(f"Writing {cand_path} and {match_path} in exact test order...", flush=True)
    matched_count = 0
    singleton_count = 0
    total_links = 0
    
    with open(cand_path, "w", encoding="utf-8") as f_cand, \
         open(match_path, "w", encoding="utf-8") as f_match:
         
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        
        for s1_id in s1_ordered_ids:
            c_str, m_str = res_map.get(s1_id, ("", ""))
            
            f_cand.write(f"{s1_id}\t{c_str}\n")
            f_match.write(f"{s1_id}\t{m_str}\n")
            
            if m_str:
                matched_count += 1
                total_links += len(m_str.split(","))
            else:
                singleton_count += 1
                
    # Clean up temp directory
    for country in ["France", "US", "India"]:
        tfile = os.path.join(temp_dir, f"results_{country}.tsv")
        if os.path.isfile(tfile):
            os.remove(tfile)
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

