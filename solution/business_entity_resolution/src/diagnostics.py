"""Small, deterministic retrieval diagnostics for entity-level evaluation."""

import json
from collections import Counter


CHANNELS = (
    "exact", "addr", "token", "prefix", "acronym", "tokpost",
    "addr_tok", "ngram", "tr_name", "tr_tok", "phon", "phone",
)


def candidate_diagnostic(sid, record, truth, items, provenance):
    """Describe one S1 query without retaining the full candidate universe."""
    ranked = [eid for eid, _ in items]
    candidate_ids = set(ranked)
    recovered = truth & candidate_ids
    ranks = {eid: rank for rank, eid in enumerate(ranked, 1)}
    channel_hits = Counter(channel for eid in recovered for channel in provenance.get(eid, ()))
    tokens = record.get("core_tokens") or ()
    raw_name = record.get("name") or ""
    native_script = any(ord(char) > 127 for char in raw_name)
    return {
        "s1_id": sid,
        "country": record.get("country", ""),
        "gt_count": len(truth),
        "recovered_gt_count": len(recovered),
        "missing_gt_count": len(truth - recovered),
        "missing_gt_ids": ",".join(sorted(truth - recovered)),
        "pair_recall": len(recovered) / len(truth) if truth else 1.0,
        "entity_full_recall": int(recovered == truth),
        "full_recall_flag": int(recovered == truth),
        "zero_recovery": int(bool(truth) and not recovered),
        "gt_min_rank": min((ranks[eid] for eid in recovered), default=""),
        "gt_max_rank": max((ranks[eid] for eid in recovered), default=""),
        "candidate_count": len(ranked),
        "channel_hit_vector": json.dumps({channel: channel_hits[channel] for channel in CHANNELS}, separators=(",", ":")),
        "channel_agreement_mean": (sum(len(provenance.get(eid, ())) for eid in ranked) / len(ranked)) if ranked else 0.0,
        "candidate_collision_density": (len(ranked) / len(set(tokens))) if tokens else float(len(ranked)),
        "native_script": int(native_script),
        "transliteration_indicator": int(native_script),
        "empty_address": int(not (record.get("norm_addr") or "")),
        "single_token_name": int(len(tokens) <= 1),
        "generic_high_frequency_indicator": int(len(tokens) <= 1 or len(ranked) >= 150),
        "multi_match": int(len(truth) > 1),
        "outside_top_k": "",
        "zero_overlap": "",
        "name": raw_name,
        "address": record.get("addr", ""),
    }


def summarize_retrieval(rows):
    positives = [row for row in rows if row["gt_count"]]
    total_gt = sum(row["gt_count"] for row in positives)
    counts = sorted(row["candidate_count"] for row in rows)

    def percentile(p):
        if not counts:
            return 0
        return counts[min(len(counts) - 1, int(round((len(counts) - 1) * p)))]

    return {
        "candidate_pair_recall": sum(row["recovered_gt_count"] for row in positives) / total_gt if total_gt else 1.0,
        "entity_full_recall": sum(row["entity_full_recall"] for row in positives) / len(positives) if positives else 1.0,
        "zero_recovery_s1": sum(row["zero_recovery"] for row in positives),
        "non_singleton_s1": len(positives),
        "candidate_count": sum(counts),
        "candidate_count_stats": {
            "min": counts[0] if counts else 0,
            "mean": sum(counts) / len(counts) if counts else 0,
            "p50": percentile(0.50),
            "p95": percentile(0.95),
            "p99": percentile(0.99),
            "max": counts[-1] if counts else 0,
        },
    }
