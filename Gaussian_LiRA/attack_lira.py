import argparse
import os
import json
import numpy as np
import xgboost as xgb
from utils import predict_margin, extract_feature_names_from_model, build_X_like_xgbt_from_csv
from scipy.stats import norm

# Safe tqdm import
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

def load_booster(path):
    bst = xgb.Booster()
    bst.load_model(path)
    return bst

def ensure_2d(x):
    x = np.asarray(x)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    return x

def main(args):
    os.makedirs("scores", exist_ok=True)

    # Load target model first to get feature names (trained by xgbt_train.py)
    bst_target = load_booster(args.target_model)
    ref_feat_names = extract_feature_names_from_model(args.target_model)
    if not ref_feat_names:
        raise SystemExit("Target model does not expose feature names; please ensure xgbt_train.py saved them")

    # Build X exactly like xgbt_train.py, aligned to target feature names
    if args.data_csv is not None:
        X, y, feat_names = build_X_like_xgbt_from_csv(args.data_csv, args.label_col, ref_feature_names=ref_feat_names)
    else:
        X = np.load(args.data_x).astype(np.float32)
        y = np.load(args.data_y)
        feat_names = ref_feat_names

    targets_idx = np.load(args.targets_index_file)
    # holdout_idx = np.load(args.holdout_index_file)
    masks = np.load(args.masks_file)  # [N_shadows, n_total]

    # Load shadows
    shadow_paths = sorted([
        os.path.join(args.shadows_dir, f)
        for f in os.listdir(args.shadows_dir)
        if f.endswith('.json') or f.endswith('.model')
    ])
    assert len(shadow_paths) == masks.shape[0], "Number of shadow models must match masks rows"

    sigma_floor = float(args.sigma_floor)

    # 1) predict target observations once (vectorized)
    X_targets = X[targets_idx]                 # shape (n_targets, d)
    obs_all = predict_margin(bst_target, X_targets, feature_names=feat_names)  # shape (n_targets,)

    # 2) Pre-load shadows (optionally in chunks if memory is tight)
    n_shadows = len(shadow_paths)
    n_targets = len(targets_idx)
    preds = np.empty((n_shadows, n_targets), dtype=np.float32)

    # optional: parallel load/predict (joblib) — simple sequential example here:
    shadow_iter = enumerate(tqdm(shadow_paths, desc="Load+Predict shadows", total=n_shadows)) if tqdm else enumerate(shadow_paths)
    for j, pth in shadow_iter:
        bst = load_booster(pth)  # load once per shadow
        # predict for all targets in one batch
        preds_j = predict_margin(bst, X_targets, feature_names=feat_names)  # shape (n_targets,)
        preds[j, :] = preds_j

    # 3) masks for targets: (n_shadows, n_targets)
    masks_targets = masks[:, targets_idx]  # boolean

    # 4) compute counts
    counts_in = masks_targets.sum(axis=0)            # (n_targets,)
    counts_out = n_shadows - counts_in              

    # Avoid divide by zero: set counts to at least 1 for denominator (we handle zero-case later)
    counts_in_safe = np.where(counts_in == 0, 1, counts_in)
    counts_out_safe = np.where(counts_out == 0, 1, counts_out)

    # 5) compute mu_in, mu_out (vectorized)
    sum_in = (preds * masks_targets).sum(axis=0)         # (n_targets,)
    mu_in = sum_in / counts_in_safe
    sum_out = (preds * (~masks_targets)).sum(axis=0)
    mu_out = sum_out / counts_out_safe

    # 6) compute variance (population / sample: use ddof=0)
    sum2_in = ((preds**2) * masks_targets).sum(axis=0)
    var_in = (sum2_in / counts_in_safe) - mu_in**2
    sd_in = np.sqrt(np.maximum(var_in, 0.0))

    sum2_out = ((preds**2) * (~masks_targets)).sum(axis=0)
    var_out = (sum2_out / counts_out_safe) - mu_out**2
    sd_out = np.sqrt(np.maximum(var_out, 0.0))

    # 7) handle zero-count cases: for targets with counts_in==0 or counts_out==0,
    #    fallback to obs or global pooling (here: set sd=1.0, mu=obs)
    mask_no_in = (counts_in == 0)
    mask_no_out = (counts_out == 0)
    mu_in[mask_no_in] = obs_all[mask_no_in]
    sd_in[mask_no_in] = 1.0
    mu_out[mask_no_out] = obs_all[mask_no_out]
    sd_out[mask_no_out] = 1.0

    # 8) compute LLR per target (vectorized)
    # compute logpdf vectorized: but norm.logpdf supports vector inputs elementwise
    logp_in = norm.logpdf(obs_all, loc=mu_in, scale=np.maximum(sd_in, sigma_floor))
    logp_out = norm.logpdf(obs_all, loc=mu_out, scale=np.maximum(sd_out, sigma_floor))
    llr_scores = (logp_in - logp_out)   # shape (n_targets,)

    # 9) If you still want meta per-sample, fill meta list in a loop (cheap, n_targets)
    meta = []
    for idx_i, i in enumerate(targets_idx):
        meta.append({
            "index": int(i),
            "obs": float(obs_all[idx_i]),
            "mu_in": float(mu_in[idx_i]), "sd_in": float(sd_in[idx_i]),
            "mu_out": float(mu_out[idx_i]), "sd_out": float(sd_out[idx_i]),
            "n_in": int(counts_in[idx_i]), "n_out": int(counts_out[idx_i])
        })

    # Save results
    llr_scores = np.asarray(llr_scores, dtype=np.float32)
    os.makedirs(os.path.dirname(args.out_scores), exist_ok=True)
    np.save(args.out_scores, llr_scores)
    with open(args.out_scores.replace('.npy', '_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"Saved scores to {args.out_scores} and meta to {args.out_scores.replace('.npy','_meta.json')}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-csv", type=str, default=None)
    parser.add_argument("--label-col", type=str, default=None)
    parser.add_argument("--data-x", type=str, default=None)
    parser.add_argument("--data-y", type=str, default=None)

    parser.add_argument("--targets-index-file", type=str, required=True)
    parser.add_argument("--holdout-index-file", type=str, required=True)
    parser.add_argument("--masks-file", type=str, required=True)
    parser.add_argument("--shadows-dir", type=str, required=True)
    parser.add_argument("--target-model", type=str, required=True)

    parser.add_argument("--out-scores", type=str, default="scores/scores_targets.npy")
    parser.add_argument("--sigma-floor", type=float, default=1e-6)

    args = parser.parse_args()
    main(args)