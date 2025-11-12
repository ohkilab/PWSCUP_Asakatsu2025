import argparse
import json
import math
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable, Optional, List, Dict, Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import ParameterGrid
import xgboost as xgb

DEFAULT_PARAM_GRID: Dict[str, List[Any]] = {
    "n_estimators": [300, 500, 800],
    "max_depth": [4, 6, 8],
    "learning_rate": [0.03, 0.05, 0.1],
    "subsample": [0.8, 0.9],
    "colsample_bytree": [0.8, 0.9],
    "min_child_weight": [1, 3],
    "gamma": [0.0, 1.0],
    "reg_lambda": [1.0, 2.0],
    "reg_alpha": [0.0, 0.1],
}


def _ensure_list_like(value: Any, key: str) -> List[Any]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"Grid parameter '{key}' must be a list (got {type(value).__name__}).")
    if not value:
        raise ValueError(f"Grid parameter '{key}' must contain at least one value.")
    return list(value)


def load_param_grid(path: Optional[str]) -> Dict[str, List[Any]]:
    if not path:
        return {k: v[:] for k, v in DEFAULT_PARAM_GRID.items()}

    json_path = Path(path)
    if not json_path.is_file():
        raise SystemExit(f"Parameter grid JSON '{path}' was not found.")

    try:
        with json_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:
        raise SystemExit(f"Failed to load parameter grid JSON '{path}': {exc}") from exc

    if not isinstance(payload, dict):
        raise SystemExit("Parameter grid JSON must be an object mapping parameter names to value lists.")

    grid: Dict[str, List[Any]] = {}
    for key, val in payload.items():
        grid[key] = _ensure_list_like(val, key)

    if not grid:
        raise SystemExit("Parameter grid JSON must not be empty.")

    return grid


def parse_fixed_params(raw: Optional[str]) -> Dict[str, Any]:
    if raw is None:
        return {}
    try:
        payload = json.loads(raw)
    except Exception as exc:
        raise SystemExit(f"Failed to parse --fixed-params JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit("--fixed-params must be a JSON object.")
    return payload


def slice_param_combinations(combos: List[Dict[str, Any]], max_combinations: int) -> List[Dict[str, Any]]:
    if max_combinations <= 0 or len(combos) <= max_combinations:
        return combos

    # Pick combinations uniformly across the grid to cover diverse settings.
    indices = np.linspace(0, len(combos) - 1, max_combinations, dtype=int)
    seen = set()
    selected: List[Dict[str, Any]] = []
    for idx in indices:
        if idx not in seen:
            seen.add(idx)
            selected.append(combos[int(idx)])
    return selected


def get_hyperparameter_combinations(grid: Dict[str, List[Any]], fixed_params: Dict[str, Any], max_combinations: int) -> List[Dict[str, Any]]:
    combos = [{**params, **fixed_params} for params in ParameterGrid(grid)]
    if not combos:
        combos = [fixed_params.copy()]
    combos = slice_param_combinations(combos, max_combinations)
    return combos

def is_binary_01(s: pd.Series) -> bool:
    if s.empty:
        return False
    ss = s.astype(str).str.strip()
    if (ss == "").any():
        return False
    num = pd.to_numeric(ss, errors="coerce")
    if num.isna().any():
        return False
    vals = set(pd.unique(num.astype(int)))
    return vals.issubset({0, 1}) and len(vals) > 0

def build_X(df: pd.DataFrame, target: str) -> pd.DataFrame:
    if target not in df.columns:
        raise SystemExit(f"Target column '{target}' not found in CSV.")
    X_raw = df.drop(columns=[target]).copy()

    X_num_try = X_raw.apply(pd.to_numeric, errors="coerce")
    num_cols = [c for c in X_num_try.columns if X_num_try[c].notna().sum() > 0]
    X_num = X_num_try[num_cols] if num_cols else pd.DataFrame(index=df.index)

    cat_cols = [c for c in X_raw.columns if c not in num_cols]
    X_cat = pd.get_dummies(X_raw[cat_cols].astype("category"), drop_first=True) if cat_cols else pd.DataFrame(index=df.index)

    X = pd.concat([X_num, X_cat], axis=1)

    zero_var = X.nunique(dropna=False) <= 1
    if zero_var.all():
        raise SystemExit("All feature columns have zero variance.")
    if zero_var.any():
        X = X.loc[:, ~zero_var]

    X = X.reindex(columns=sorted(X.columns)).astype("float32")
    return X

_EMPTY_TOKENS = {"", "nan", "none", "na", "null"}


def _decimal_places(dec: Decimal) -> int:
    exp = dec.as_tuple().exponent
    return -exp if exp < 0 else 0


def _unique_preserve_order(values: Iterable[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for val in values:
        if val not in seen:
            seen.add(val)
            ordered.append(val)
    return ordered


def infer_column_schema(series: pd.Series) -> Dict[str, Any]:
    s = series.astype(str).str.strip()
    mask = ~s.str.lower().isin(_EMPTY_TOKENS)
    s = s[mask]
    if s.empty:
        return {"type": "category", "values": []}

    numeric_values: List[Decimal] = []
    max_places = 0
    for val in s:
        candidate = val.replace(',', '')
        try:
            dec = Decimal(candidate)
        except InvalidOperation:
            numeric_values = []
            break
        else:
            numeric_values.append(dec)
            max_places = max(max_places, _decimal_places(dec))

    if numeric_values:
        min_dec = min(numeric_values)
        max_dec = max(numeric_values)
        return {
            "type": "number",
            "min": float(min_dec),
            "max": float(max_dec),
            "max_decimal_places": int(max_places),
        }

    dt = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    if dt.notna().all():
        return {
            "type": "date",
            "format": "yyyy-mm-dd",
            "min": dt.min().strftime("%Y-%m-%d"),
            "max": dt.max().strftime("%Y-%m-%d"),
        }

    return {"type": "category", "values": _unique_preserve_order(s.tolist())}


def infer_columns_spec(df: pd.DataFrame) -> Dict[str, Any]:
    return {col: infer_column_schema(df[col]) for col in df.columns}


def embed_columns_spec_into_model(model_path: Path, columns_spec: Dict[str, Any]) -> None:
    text = model_path.read_text(encoding='utf-8')
    prefix = '{"columns":'
    if text.startswith(prefix):
        marker = text.find(',"learner"')
        if marker == -1:
            raise SystemExit(f"Unexpected model JSON layout in '{model_path}'.")
        base = '{' + text[marker + 1:]
    else:
        base = text

    if not base.startswith('{'):
        raise SystemExit(f"Unexpected model JSON structure in '{model_path}'.")

    fragment = '"columns":' + json.dumps(columns_spec, ensure_ascii=False, separators=(',', ':'))
    updated = '{' + fragment + ',' + base[1:]
    model_path.write_text(updated, encoding='utf-8')



def main():
    ap = argparse.ArgumentParser(description="Train XGBoost with hyperparameter search and save best model.")
    ap.add_argument("train_csv", help="training CSV with header")
    ap.add_argument("val_csv", help="validation CSV with header")
    ap.add_argument("--model-json", required=True, help="output model JSON path")
    ap.add_argument("--target", default="stroke_flag", help="binary target column (default: stroke_flag)")
    ap.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")
    ap.add_argument("--param-grid-json", help="path to JSON file describing parameter grid")
    ap.add_argument("--fixed-params", help="JSON object of additional fixed parameters to merge into each combination")
    ap.add_argument("--max-combinations", type=int, default=64, help="maximum hyperparameter combinations to evaluate (default: 64; 0 for all)")
    ap.add_argument("--auto-scale-pos-weight", action="store_true", help="set scale_pos_weight based on class imbalance if not provided")
    ap.add_argument("--thresholds", help="comma separated probability thresholds to evaluate (default: 0.5)")
    ap.add_argument(
        "--val-leak-ratio",
        type=float,
        default=0.0,
        help=(
            "Fraction of validation rows to append to the training set before fitting "
            "(0 disables; 1 uses all validation rows)."
        ),
    )
    args = ap.parse_args()

    # データ読み込み
    train_df = pd.read_csv(args.train_csv, dtype=str, keep_default_na=False)
    val_df = pd.read_csv(args.val_csv, dtype=str, keep_default_na=False)

    # ターゲット列の検証
    if args.target not in train_df.columns:
        raise SystemExit(f"Target column '{args.target}' not found in train CSV.")
    if args.target not in val_df.columns:
        raise SystemExit(f"Target column '{args.target}' not found in validation CSV.")
    
    if not is_binary_01(train_df[args.target]):
        raise SystemExit(f"Target column '{args.target}' in train CSV must be strictly binary 0/1.")
    if not is_binary_01(val_df[args.target]):
        raise SystemExit(f"Target column '{args.target}' in validation CSV must be strictly binary 0/1.")

    if not (0.0 <= args.val_leak_ratio <= 1.0):
        raise SystemExit("--val-leak-ratio must be within [0, 1].")

    leak_ratio = float(args.val_leak_ratio)
    leak_count = 0
    if leak_ratio > 0:
        leak_count = max(1, int(round(len(val_df) * leak_ratio)))
        leak_count = min(leak_count, len(val_df))
        leak_subset = val_df.sample(n=leak_count, random_state=args.seed).copy()
        train_df = pd.concat([train_df, leak_subset], axis=0, ignore_index=True)
        print(
            "!!! Validation leakage enabled: appended "
            f"{leak_count} validation rows (ratio={leak_ratio:.3f}) to training."
        )
        print("!!! Expect optimistic validation metrics due to intentional leakage !!!")

    columns_spec = infer_columns_spec(train_df)
    columns_spec_json = json.dumps(columns_spec, ensure_ascii=False)

    # 特徴量とターゲットの準備
    y_train = pd.to_numeric(train_df[args.target], errors="coerce").astype(int).values
    y_val = pd.to_numeric(val_df[args.target], errors="coerce").astype(int).values

    X_train = build_X(train_df, target=args.target)
    X_val = build_X(val_df, target=args.target)
    
    # 特徴量の列を統一（訓練セットの列に合わせる）
    feature_names = list(X_train.columns)
    missing_cols = set(feature_names) - set(X_val.columns)
    # 欠損列を0で埋める
    for col in missing_cols:
        X_val[col] = 0.0
    
    # 余分な列を削除
    X_val = X_val[feature_names]

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Features: {len(feature_names)}")

    param_grid = load_param_grid(args.param_grid_json)
    fixed_params = parse_fixed_params(args.fixed_params)
    param_combinations = get_hyperparameter_combinations(param_grid, fixed_params, args.max_combinations)
    if not param_combinations:
        raise SystemExit("No hyperparameter combinations available.")

    threshold_values: List[float] = [0.5]
    if args.thresholds:
        try:
            threshold_values = [float(t) for t in args.thresholds.split(",") if t.strip()]
        except ValueError as exc:
            raise SystemExit(f"Failed to parse --thresholds: {exc}") from exc
        if not threshold_values:
            raise SystemExit("--thresholds must contain at least one numeric value.")
        if any(not (0.0 < thr < 1.0) for thr in threshold_values):
            raise SystemExit("All thresholds must be in (0, 1).")
    
    best_f1 = -math.inf
    best_model = None
    best_params = None
    best_metrics = None

    print(f"\nTesting {len(param_combinations)} parameter combinations...")
    
    pos_count = int(y_train.sum())
    neg_count = int(len(y_train) - pos_count)
    imbalance_ratio: Optional[float] = None
    if args.auto_scale_pos_weight and pos_count > 0 and neg_count > 0:
        imbalance_ratio = float(neg_count / pos_count)

    for i, params in enumerate(param_combinations, 1):
        # パラメータをコピーして元の辞書を変更しないようにする
        current_params = params.copy()
        if imbalance_ratio is not None:
            current_params.setdefault("scale_pos_weight", imbalance_ratio)
        print(f"  [{i}/{len(param_combinations)}] Testing params: {current_params}")

        try:
            model = xgb.XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                random_state=args.seed,
                n_jobs=-1,
                **current_params
            )

            model.fit(X_train, y_train, verbose=False)

            # 予測と評価
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            best_threshold = None
            best_f1_for_model = -math.inf
            best_model_metrics: Optional[Dict[str, float]] = None

            for thr in threshold_values:
                y_pred = (y_pred_proba >= thr).astype(int)
                acc = accuracy_score(y_val, y_pred)
                f1 = f1_score(y_val, y_pred)
                auc = roc_auc_score(y_val, y_pred_proba)

                if f1 > best_f1_for_model or (math.isclose(f1, best_f1_for_model, rel_tol=1e-10) and (best_threshold is None or thr < best_threshold)):
                    best_f1_for_model = f1
                    best_threshold = thr
                    best_model_metrics = {'accuracy': acc, 'f1': f1, 'auc': auc, 'threshold': thr}

            if best_model_metrics is None:
                raise RuntimeError("No metrics computed for current model.")

            print("    Metrics at best threshold:")
            print(f"      threshold={best_model_metrics['threshold']:.4f}, Accuracy={best_model_metrics['accuracy']:.6f}, F1={best_model_metrics['f1']:.6f}, AUC={best_model_metrics['auc']:.6f}")

            # 最良モデルの更新
            if best_model_metrics['f1'] > best_f1:
                best_f1 = best_model_metrics['f1']
                best_model = model
                best_params = current_params.copy()
                best_metrics = best_model_metrics
                print(f"    *** New best F1: {best_model_metrics['f1']:.6f} at threshold {best_model_metrics['threshold']:.4f} ***")

        except Exception as e:
            print(f"    Error with params {current_params}: {e}")
            continue

    if best_model is None:
        raise SystemExit("No valid model was trained successfully.")

    # 最良モデルの保存
    booster = best_model.get_booster()
    booster.set_attr(feature_names=json.dumps(feature_names, ensure_ascii=False))
    booster.set_attr(target=args.target)
    booster.set_attr(xgboost_version=xgb.__version__)
    booster.set_attr(columns=columns_spec_json)
    booster.save_model(args.model_json)
    embed_columns_spec_into_model(Path(args.model_json), columns_spec)
    print("Embedded column schema into model JSON for check_csv compatibility.")

    print(f"\n=== BEST MODEL RESULTS ===")
    print(f"Best Parameters: {best_params}")
    print(f"Validation Accuracy: {best_metrics['accuracy']:.6f}")
    print(f"Validation F1 Score: {best_metrics['f1']:.6f}")
    print(f"Validation AUC: {best_metrics['auc']:.6f}")
    print(f"Decision Threshold: {best_metrics['threshold']:.4f}")
    if leak_count > 0:
        print(f"Validation leakage ratio: {leak_ratio:.3f} (rows appended: {leak_count})")
    print(f"Saved best model to: {args.model_json}")
    print(f"Number of features: {len(feature_names)}")

if __name__ == "__main__":
    main()