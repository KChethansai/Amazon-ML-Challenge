import json
from itertools import product
from solution.business_entity_resolution.src.train import calculate_macro_f05, evaluate_metrics

def sweep(cache_file='experiments/dev_scores_cache.json'):
    with open(cache_file) as f:
        data = json.load(f)

    gt_map = {sid: set(item['gt']) for sid, item in data.items()}

    best_f05 = 0.0
    best_params = None
    best_metrics = None

    tau_singletons = [0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.95, 0.97, 0.98, 0.99]
    tau_mins = [0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    delta_margins = [0.05, 0.10, 0.15, 0.18, 0.20, 0.25, 0.30]

    for ts, tm, dm in product(tau_singletons, tau_mins, delta_margins):
        pred_map = {}
        for sid, item in data.items():
            pairs = item['cands']
            if not pairs:
                pred_map[sid] = set()
                continue
            top = max(s for _, s in pairs)
            if top < ts:
                pred_map[sid] = set()
            else:
                cutoff = max(tm, top - dm)
                pred_map[sid] = {cid for cid, s in pairs if s >= cutoff}

        f05 = calculate_macro_f05(gt_map, pred_map)
        if f05 > best_f05:
            best_f05 = f05
            best_params = (ts, tm, dm)
            m = evaluate_metrics(gt_map, pred_map)
            best_metrics = m
            print(f"New Best: tau_singleton={ts:.2f}, tau_min={tm:.2f}, delta_margin={dm:.2f} -> Macro F0.5={f05:.5f}, Prec={m['precision']:.4f}, Rec={m['recall']:.4f}, SingAcc={m['singleton_accuracy']:.4f}")

    print("\n" + "="*50)
    print(f"OPTIMAL THRESHOLDS: tau_singleton={best_params[0]}, tau_min={best_params[1]}, delta_margin={best_params[2]}")
    print(f"Best Macro F0.5: {best_f05:.5f}")
    print(f"Metrics: {best_metrics}")

if __name__ == '__main__':
    sweep()
