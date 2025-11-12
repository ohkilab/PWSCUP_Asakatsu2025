import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import norm as _norm

# =====================
# Repro helpers
# =====================

def set_seed(seed):
    if seed is None:
        return
    import random
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

# =====================
# XGB train / predict
# =====================

def train_xgb_binary(X, y, params=None, num_round=200):
    # Ensure NumPy array (no feature names embedded at train time)
    if hasattr(X, "to_numpy"):
        X = X.to_numpy(dtype=float)
    dtrain = xgb.DMatrix(X, label=y)
    params = params or {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 6,
        'eta': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'lambda': 1.0,
        'seed': 42,
    }
    bst = xgb.train(params, dtrain, num_round)
    return bst


def predict_margin(model, X, feature_names=None):
    # X can be np.ndarray or pandas DataFrame
    if hasattr(X, "to_numpy"):
        X_arr = X.to_numpy(dtype=float)
    else:
        X_arr = np.asarray(X, dtype=float)
    if feature_names is not None:
        d = xgb.DMatrix(X_arr, feature_names=feature_names)
    else:
        d = xgb.DMatrix(X_arr)
    return model.predict(d, output_margin=True)

# =====================
# Feature name handling compatible with xgbt_train.py
# =====================

def _build_X_like_xgbt_df(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Replicates xgbt_train.py's build_X():
      - drop target
      - numeric cast (coerce), take columns with any non-NA as numeric
      - categorical = remaining -> one-hot(drop_first=True)
      - concat, remove zero-variance cols, sort columns asc, float32
    """
    if target not in df.columns:
        raise SystemExit(f"Target column '{target}' not found in CSV.")

    X_raw = df.drop(columns=[target]).copy()

    # Decide numeric columns by to_numeric success
    X_num_try = X_raw.apply(pd.to_numeric, errors="coerce")
    num_cols = [c for c in X_num_try.columns if X_num_try[c].notna().sum() > 0]
    X_num = X_num_try[num_cols] if num_cols else pd.DataFrame(index=df.index)

    # Remaining columns are categorical
    cat_cols = [c for c in X_raw.columns if c not in num_cols]
    if cat_cols:
        X_cat = pd.get_dummies(X_raw[cat_cols].astype("category"), drop_first=True)
    else:
        X_cat = pd.DataFrame(index=df.index)

    X = pd.concat([X_num, X_cat], axis=1)

    # Drop zero-variance columns
    zero_var = X.nunique(dropna=False) <= 1
    if zero_var.any():
        X = X.loc[:, ~zero_var]

    # Sort columns alphabetically and cast to float32
    X = X.reindex(columns=sorted(X.columns)).astype("float32")
    return X


essential_attrs = ("feature_names",)

def extract_feature_names_from_model(model_path: str):
    """Read feature_names attribute (JSON list) from a Booster saved by xgbt_train.py."""
    booster = xgb.Booster()
    booster.load_model(model_path)
    attrs = booster.attributes() or {}
    feat_json = attrs.get("feature_names")
    if feat_json:
        try:
            names = json.loads(feat_json)
            if isinstance(names, list) and all(isinstance(s, str) for s in names):
                return names
        except Exception:
            pass
    # Fallback (may be None if model trained without names)
    return booster.feature_names


def build_X_like_xgbt_from_csv(csv_path: str, label_col: str, ref_feature_names=None):
    """
    Load CSV like xgbt_train (dtype=str, keep_default_na=False), build X via the same recipe,
    and align to ref_feature_names if provided (add missing columns as 0, drop extras), preserving order.
    Returns: X_np (float32), y_np (int), feature_names_used (list[str])
    """
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    if label_col not in df.columns:
        raise SystemExit(f"Target column '{label_col}' not found in CSV.")
    y = pd.to_numeric(df[label_col], errors="coerce").astype(int).values

    X_df = _build_X_like_xgbt_df(df, target=label_col)

    if ref_feature_names is not None:
        # add missing as 0.0
        missing = [c for c in ref_feature_names if c not in X_df.columns]
        for c in missing:
            X_df[c] = 0.0
        # order by ref, drop extras
        X_df = X_df.reindex(columns=ref_feature_names, fill_value=0.0)
        feat_names = list(ref_feature_names)
    else:
        feat_names = list(X_df.columns)

    return X_df.to_numpy(dtype=np.float32), y, feat_names

