import csv, json, sys, time, re
from pathlib import Path

sys.path.insert(0, "solution/business_entity_resolution/src")
from disk_blocking import DiskCountryBlockingIndex, _CHANNEL_BITS
from views import transliterate_text, dm_primary, extract_phones
from blocking import _trigrams, extract_acronyms, extract_postal_prefixes, assess_difficulty, ADAPTIVE_BUDGETS
from normalization import prepare_source_record

INDEX_PATH = Path("experiments/final_scratch/india_train.sqlite")

NAME_MASK = (1<<0) | (1<<2) | (1<<3) | (1<<4) | (1<<5) | (1<<7) | (1<<8) | (1<<9) | (1<<10)

def get_many_records(db, rids):
    if not rids:
        return {}
    result = {}
    rids_list = list(rids)
    chunk_size = 990
    for i in range(0, len(rids_list), chunk_size):
        chunk = rids_list[i:i + chunk_size]
        placeholders = ','.join('?' for _ in chunk)
        cursor = db.execute(f'SELECT rid, payload FROM records WHERE rid IN ({placeholders})', chunk)
        for k, payload in cursor:
            data = json.loads(payload)
            result[k] = (data[0], data[1], data[2], data[3], data[4], data[5])
    return result

def query_candidates_optimized(idx, s1_norm_name, s1_core_tokens, s1_norm_addr, s1_addr_keys,
                               s1_raw_addr="", s1_raw_name="", max_candidates=150,
                               depth=1500, rerank=True, adaptive=True, use_family_quota=True):
    # Use existing candidate accumulation
    candidate_scores = idx._SpillScores(idx.db)
    prov = None
    
    # 1. Exact core name match (+15)
    if s1_norm_name in idx.exact_name_idx:
        for target_id in idx.exact_name_idx[s1_norm_name]:
            idx._add(candidate_scores, prov, target_id, 15.0, "exact")

    # 2. Compound address keys with selectivity weight
    for k in s1_addr_keys:
        if k in idx.compound_addr_idx:
            w = idx.compound_idf.get(k, 4.0)
            for target_id in idx.compound_addr_idx[k]:
                idx._add(candidate_scores, prov, target_id, w, "addr")

    # 3. Significant core name tokens (+3, +8 if >= 2)
    cand_token_hits = {}
    for t in set(s1_core_tokens):
        if t in idx.token_idx:
            for target_id in idx.token_idx[t]:
                idx._add(candidate_scores, prov, target_id, 3.0, "token")
                cand_token_hits[target_id] = cand_token_hits.get(target_id, 0) + 1
    for target_id, count in cand_token_hits.items():
        if count >= 2:
            idx._add(candidate_scores, prov, target_id, 8.0, "token")

    # 4. 4-character prefix (+1)
    if len(s1_norm_name) >= 4:
        pfx = s1_norm_name[:4]
        if pfx in idx.prefix_idx:
            for target_id in idx.prefix_idx[pfx]:
                idx._add(candidate_scores, prov, target_id, 1.0, "prefix")

    # 5. Acronym exact blocking key (+8)
    for acr in extract_acronyms(s1_norm_name, s1_core_tokens):
        if acr in idx.acronym_idx:
            for target_id in idx.acronym_idx[acr]:
                idx._add(candidate_scores, prov, target_id, 8.0, "acronym")

    # 6. Compound (first_significant_token, 2_digit_postal_prefix) (+7)
    if s1_core_tokens:
        first_tok = s1_core_tokens[0]
        for pfx in extract_postal_prefixes(s1_raw_addr, s1_norm_addr):
            k = (first_tok, pfx)
            if k in idx.tok_post_idx:
                for target_id in idx.tok_post_idx[k]:
                    idx._add(candidate_scores, prov, target_id, 7.0, "tokpost")
    if idx.address_tokens:
        tokens = [token for token in set(s1_norm_addr.split()) if token in idx.addr_token_idx]
        tokens.sort(key=lambda token: idx.addr_token_idf[token], reverse=True)
        for token in tokens[:6]:
            weight = idx.addr_token_idf[token] * 2.0
            for target_id in idx.addr_token_idx[token]:
                idx._add(candidate_scores, prov, target_id, weight, "addr_tok")

    # 7. Character 3-gram channel — always on, top rare trigrams
    if len(s1_norm_name) >= 3:
        q_ngrams = [s1_norm_name[i:i+3] for i in range(len(s1_norm_name)-2)
                    if s1_norm_name[i:i+3] in idx.ngram_idx]
        if q_ngrams:
            q_ngrams.sort(key=lambda ng: idx.ngram_idf.get(ng, 0.0), reverse=True)
            top_ngrams = q_ngrams[:8]
            for ng in top_ngrams:
                w = idx.ngram_idf.get(ng, 1.0)
                for target_id in idx.ngram_idx[ng]:
                    idx._add(candidate_scores, prov, target_id, w, "ngram")

    # 8. Transliterated query views (recover native<->latin mismatches)
    tr_q = transliterate_text(s1_raw_name or "")
    if tr_q:
        tr_q = re.sub(r"[^a-z0-9 ]", " ", tr_q).strip()
        if tr_q in idx.exact_name_idx:
            for target_id in idx.exact_name_idx[tr_q]:
                idx._add(candidate_scores, prov, target_id, 12.0, "tr_name")
        if tr_q in idx.tr_name_idx:
            for target_id in idx.tr_name_idx[tr_q]:
                idx._add(candidate_scores, prov, target_id, 12.0, "tr_name")
        for t in set(tr_q.split()):
            if t in idx.token_idx and len(t) >= 3:
                for target_id in idx.token_idx[t]:
                    idx._add(candidate_scores, prov, target_id, 3.0, "tr_tok")
            if t in idx.tr_token_idx:
                for target_id in idx.tr_token_idx[t]:
                    idx._add(candidate_scores, prov, target_id, 4.0, "tr_tok")

    # 9. Phonetic query tokens (+2)
    for t in set(s1_core_tokens):
        if len(t) >= 3:
            p = dm_primary(t)
            if p and p in idx.phon_idx:
                for target_id in idx.phon_idx[p]:
                    idx._add(candidate_scores, prov, target_id, 2.0, "phon")

    # 10. Phone exact (+9)
    for ph in extract_phones(s1_raw_addr or ""):
        if ph in idx.phone_idx:
            for target_id in idx.phone_idx[ph]:
                idx._add(candidate_scores, prov, target_id, 9.0, "phone")

    if not candidate_scores:
        return []

    if adaptive:
        has_exact = s1_norm_name in idx.exact_name_idx
        best_w = max(candidate_scores.values())
        tier = assess_difficulty(s1_norm_name, s1_core_tokens, s1_norm_addr,
                                 len(candidate_scores), best_w)
        if has_exact and tier == "hard":
            tier = "medium"
        max_candidates = ADAPTIVE_BUDGETS[tier]

    # Reranking with BATCH record retrieval + BIDIRECTIONAL transliteration
    if rerank and len(candidate_scores) > max_candidates:
        # Separate top candidates: ensure candidates with name channels are represented
        if use_family_quota and not candidate_scores.spilled:
            # Separate by name mask
            name_pool = []
            other_pool = []
            for rid, row in candidate_scores.cache.items():
                if row[2] & NAME_MASK:
                    name_pool.append((rid, row[0]))
                else:
                    other_pool.append((rid, row[0]))
            name_pool.sort(key=lambda x: -x[1])
            other_pool.sort(key=lambda x: -x[1])
            
            # Form top pool for reranking (e.g. up to 1000 name + up to 1000 other)
            half = depth // 2
            top = name_pool[:half] + other_pool[:half]
            # If not enough, fill from remaining
            if len(top) < depth:
                seen_rids = {r for r, _ in top}
                remaining = [x for x in (name_pool[half:] + other_pool[half:]) if x[0] not in seen_rids]
                remaining.sort(key=lambda x: -x[1])
                top.extend(remaining[:depth - len(top)])
        else:
            top = candidate_scores.most_common(depth)

        tids = [tid for tid, _ in top]
        rec_map = get_many_records(idx.db, tids)

        q_tri = _trigrams(s1_norm_name)
        tr_q_clean = tr_q if tr_q else None
        tr_q_tri = _trigrams(tr_q_clean) if tr_q_clean else None

        rescored = []
        name_rescored = []
        other_rescored = []

        for tid, w in top:
            rec = rec_map.get(tid)
            sim = 0.0
            if rec is not None:
                c_tri = _trigrams(rec[0]) if rec[0] else None
                if q_tri and c_tri:
                    sim = max(sim, len(q_tri & c_tri) / (len(q_tri | c_tri) or 1))
                if tr_q_tri and c_tri:
                    sim = max(sim, len(tr_q_tri & c_tri) / (len(tr_q_tri | c_tri) or 1))
                if len(rec) > 4 and rec[4]:
                    tr_name = transliterate_text(rec[4])
                    if tr_name:
                        tr_c_tri = _trigrams(tr_name)
                        if q_tri and tr_c_tri:
                            sim = max(sim, len(q_tri & tr_c_tri) / (len(q_tri | tr_c_tri) or 1))
                        if tr_q_tri and tr_c_tri:
                            sim = max(sim, len(tr_q_tri & tr_c_tri) / (len(tr_q_tri | tr_c_tri) or 1))
            final_w = w + idx.trigram_rerank_boost * sim
            
            # Check if this candidate has name evidence
            mask = candidate_scores.cache.get(tid, [0, 0, 0])[2] if not candidate_scores.spilled else 0
            has_name = bool(mask & NAME_MASK) or sim > 0.1
            
            if use_family_quota:
                if has_name:
                    name_rescored.append((tid, final_w))
                else:
                    other_rescored.append((tid, final_w))
            else:
                rescored.append((tid, final_w))

        if use_family_quota:
            name_rescored.sort(key=lambda x: -x[1])
            other_rescored.sort(key=lambda x: -x[1])
            
            # Guaranteed quota: at least 50% name candidates (if available)
            quota_name = min(len(name_rescored), max_candidates // 2)
            quota_other = max_candidates - quota_name
            
            selected_rids = []
            selected_set = set()
            
            # Take top quota_name from name_rescored
            for rid, _ in name_rescored[:quota_name]:
                selected_rids.append(rid)
                selected_set.add(rid)
                
            # Take remaining from combined pool sorted by final_w
            remaining_combined = [x for x in (name_rescored[quota_name:] + other_rescored) if x[0] not in selected_set]
            remaining_combined.sort(key=lambda x: -x[1])
            for rid, _ in remaining_combined[:max_candidates - len(selected_rids)]:
                selected_rids.append(rid)
            
            top_candidates = [(rid, 0.0) for rid in selected_rids]
        else:
            rescored.sort(key=lambda x: -x[1])
            top_candidates = rescored[:max_candidates]
    else:
        top_candidates = candidate_scores.most_common(max_candidates)

    # Convert rids to eids in batch
    cand_rids = [rid for rid, _ in top_candidates]
    if not cand_rids:
        return []
    placeholders = ','.join('?' for _ in cand_rids)
    eid_rows = idx.db.execute(f'SELECT rid, eid FROM records WHERE rid IN ({placeholders})', cand_rids).fetchall()
    eid_map = dict(eid_rows)
    return [eid_map[rid] for rid in cand_rids if rid in eid_map]

def load_data():
    with open("experiments/candidate_failure_modes.csv") as f:
        failed_rows = list(csv.DictReader(f))[:50]
    
    with open("dataset/train/train_ground_truth.tsv") as f:
        next(f)
        all_gt = {}
        for line in f:
            sid, _, targets = line.rstrip("\r\n").partition("\t")
            all_gt[sid] = set(t.strip() for t in targets.split(",") if t.strip())
            
    with open("dataset/train/train_source1.tsv") as f:
        next(f)
        all_s1 = {}
        for line in f:
            vals = line.rstrip("\r\n").split("\t")
            if vals[0] in all_gt:
                all_s1[vals[0]] = vals
                
    return failed_rows, all_gt, all_s1

def main():
    failed_rows, all_gt, all_s1 = load_data()
    print(f"Loaded {len(failed_rows)} failed entities to test.")
    
    total_missing_baseline = sum(int(r["missing_gt_count"]) for r in failed_rows)
    total_gt = sum(int(r["gt_count"]) for r in failed_rows)
    print(f"Total GT links: {total_gt}, Missing in original baseline: {total_missing_baseline}")
    
    idx = DiskCountryBlockingIndex("India", INDEX_PATH)
    idx._SpillScores = sys.modules['disk_blocking']._SpillScores
    print(f"Connected to index with {len(idx.records):,} records.")
    
    # Test 1: Batch lookup + transliteration rerank WITHOUT family quota
    t0 = time.time()
    rec_no_quota = 0
    for r in failed_rows:
        sid = r["s1_id"]
        gt = all_gt[sid]
        s1_vals = all_s1[sid]
        rec = prepare_source_record(s1_vals[1], s1_vals[2], s1_vals[3])
        cands = query_candidates_optimized(idx, rec["norm_name"], rec["core_tokens"], rec["norm_addr"], rec["addr_keys"],
                                           s1_raw_addr=rec["addr"], s1_raw_name=rec["name"],
                                           max_candidates=150, depth=1500, use_family_quota=False)
        rec_no_quota += len(gt & set(cands))
    
    t_no_quota = time.time() - t0
    missing_no_quota = total_gt - rec_no_quota
    print(f"Fix 1 (Batch + Translit Rerank, NO Quota): recovered {rec_no_quota}/{total_gt} ({rec_no_quota/total_gt:.1%}), missing {missing_no_quota} in {t_no_quota:.1f}s (Speed: {t_no_quota/len(failed_rows):.3f}s/query)")

    # Test 2: Batch lookup + transliteration rerank WITH Family-Bounded Quota
    t0 = time.time()
    rec_quota = 0
    for r in failed_rows:
        sid = r["s1_id"]
        gt = all_gt[sid]
        s1_vals = all_s1[sid]
        rec = prepare_source_record(s1_vals[1], s1_vals[2], s1_vals[3])
        cands = query_candidates_optimized(idx, rec["norm_name"], rec["core_tokens"], rec["norm_addr"], rec["addr_keys"],
                                           s1_raw_addr=rec["addr"], s1_raw_name=rec["name"],
                                           max_candidates=150, depth=1500, use_family_quota=True)
        rec_quota += len(gt & set(cands))
    
    t_quota = time.time() - t0
    missing_quota = total_gt - rec_quota
    print(f"Fix 2 (Batch + Translit Rerank + Family Quota): recovered {rec_quota}/{total_gt} ({rec_quota/total_gt:.1%}), missing {missing_quota} in {t_quota:.1f}s (Speed: {t_quota/len(failed_rows):.3f}s/query)")

if __name__ == "__main__":
    main()
