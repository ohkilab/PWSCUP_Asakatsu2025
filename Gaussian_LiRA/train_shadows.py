import argparse
import os
import numpy as np
import joblib as _joblib
from utils import train_xgb_binary, set_seed, extract_feature_names_from_model, build_X_like_xgbt_from_csv

def train_one(j, masks, X, y, out_dir, max_rounds, base_params, rand_params):
    idx = masks[j]
    Xj, yj = X[idx], y[idx]

    params = dict(base_params)
    params['seed'] = (params.get('seed', 0) + j) % (2**31-1)

    if rand_params:
        rng = np.random.default_rng(j * 9973 + 17)
        params['max_depth'] = int(rng.integers(4, 9))
        params['subsample'] = float(rng.uniform(0.6, 0.95))
        params['colsample_bytree'] = float(rng.uniform(0.6, 0.95))
        params['eta'] = float(rng.uniform(0.05, 0.2))
        params['lambda'] = float(rng.uniform(0.5, 2.0))

    bst = train_xgb_binary(Xj, yj, params=params, num_round=max_rounds)
    out_path = os.path.join(out_dir, f"shadow_{j:04d}.json")
    bst.save_model(out_path)
    return out_path


def main(args):
    set_seed(args.seed)
    os.makedirs(args.models_dir, exist_ok=True)

    # Align preprocessing with target model's feature names from xgbt_train.py
    if not args.target_model:
        raise SystemExit("--target-model is required to align shadow preprocessing with the target's feature names")
    ref_feat_names = extract_feature_names_from_model(args.target_model)
    if not ref_feat_names:
        raise SystemExit("Target model has no feature names; ensure it was trained via xgbt_train.py")

    if args.data_csv is not None:
        X, y, feat_names = build_X_like_xgbt_from_csv(args.data_csv, args.label_col, ref_feature_names=ref_feat_names)
    else:
        X = np.load(args.data_x).astype(np.float32)
        y = np.load(args.data_y)
        feat_names = ref_feat_names

    masks = np.load(args.masks_file)  # [N, n_total]

    base_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 6,
        'eta': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'lambda': 1.0,
        'seed': args.seed,
    }

    n = masks.shape[0]
    tasks = (_joblib.delayed(train_one)(j, masks, X, y, args.models_dir,
                                        args.num_rounds, base_params, args.rand_params)
             for j in range(n))
    paths = _joblib.Parallel(n_jobs=args.n_jobs, verbose=10)(tasks)
    print(f"Saved {len(paths)} shadow models to {args.models_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-csv", type=str, default=None)
    parser.add_argument("--label-col", type=str, default=None)
    parser.add_argument("--data-x", type=str, default=None)
    parser.add_argument("--data-y", type=str, default=None)

    parser.add_argument("--masks-file", type=str, required=True)
    parser.add_argument("--models-dir", type=str, default="models/shadows")
    parser.add_argument("--num-rounds", type=int, default=200)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rand-params", action="store_true")

    parser.add_argument("--target-model", type=str, required=True,
                        help="Path to the target model (trained by xgbt_train.py) to copy feature names")

    args = parser.parse_args()
    main(args)