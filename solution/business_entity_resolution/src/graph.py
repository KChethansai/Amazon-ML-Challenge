"""S2<->S3 bridge recovery (additive evidence only, one hop, no closure)."""
import numpy as np
from rapidfuzz import fuzz

try:
    from features import compute_pairwise_features
    from features2 import extra_for_pair
except ImportError:
    from solution.business_entity_resolution.src.features import compute_pairwise_features
    from solution.business_entity_resolution.src.features2 import extra_for_pair


def bridge_support(n_s2, a_s2, n_s3, a_s3):
    ns = fuzz.token_set_ratio(n_s2 or "", n_s3 or "")
    ad = fuzz.token_set_ratio(a_s2 or "", a_s3 or "") if (a_s2 and a_s3) else ns
    return ns, ad


def apply_graph_bridge(s1_rec, pred_list, prob_map, c_idx, model,
                       use_extra=True, score_min=0.999, bridge_min=80):
    """Return list of S3 ids to add. pred_list: matched ids; prob_map: id->model prob."""
    added = []
    have = set(pred_list)
    for mid in pred_list:
        if prob_map.get(mid, 0.0) < score_min or not mid.startswith("S2-"):
            continue
        rec = c_idx.records.get(mid)
        if rec is None:
            continue
        n2, a2 = rec[0], rec[1]
        core2 = list(rec[3]) if len(rec) > 3 else []
        probe = c_idx.query_candidates(
            n2, core2, a2, [],
            s1_raw_addr=rec[5] if len(rec) > 5 else "",
            s1_raw_name=rec[4] if len(rec) > 4 else "",
            max_candidates=25, return_weights=True)
        for cid, w in probe:
            if not cid.startswith("S3-") or cid in have:
                continue
            crec = c_idx.records.get(cid)
            if crec is None:
                continue
            ns, ad = bridge_support(n2, a2, crec[0], crec[1])
            if ns < bridge_min or ad < bridge_min:
                continue
            fb = compute_pairwise_features(s1_rec, crec, rank=0, weight=w, max_weight=w)
            if use_extra:
                fx = extra_for_pair(s1_rec, crec[0], crec[1],
                                    list(crec[3]) if len(crec) > 3 else [],
                                    tgt_raw_name=crec[4] if len(crec) > 4 else "",
                                    tgt_raw_addr=crec[5] if len(crec) > 5 else "")
                vec = fb + fx
            else:
                vec = fb
            p = float(model.predict(np.array([vec], dtype=np.float32))[0])
            if p >= score_min:
                added.append(cid)
                have.add(cid)
    return added
