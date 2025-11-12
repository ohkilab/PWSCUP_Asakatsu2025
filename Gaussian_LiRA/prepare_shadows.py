import argparse
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
try:
    from tqdm import tqdm
except Exception:
    tqdm = None


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def _is_numeric_series(s: pd.Series) -> bool:
    try:
        _ = pd.to_numeric(s, errors="coerce")
        # consider numeric if at least half can be parsed
        return (_.notna().mean() >= 0.5)
    except Exception:
        return False


def _compute_bins_from_reference(ref: pd.Series, n_bins: int, method: str) -> np.ndarray:
    vals = pd.to_numeric(ref, errors="coerce").dropna().to_numpy()
    if len(vals) == 0:
        # fallback single bin
        return np.array([-np.inf, np.inf])
    if method == "quantile":
        qs = np.linspace(0, 1, n_bins + 1)
        edges = np.unique(np.quantile(vals, qs))
        if len(edges) < 2:
            edges = np.array([vals.min()-1e-9, vals.max()+1e-9])
        return edges
    else:  # uniform
        lo, hi = float(np.min(vals)), float(np.max(vals))
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            return np.array([lo-1e-9, hi+1e-9])
        return np.linspace(lo, hi, n_bins + 1)


def _apply_bins(x: pd.Series, edges: np.ndarray) -> pd.Series:
    x_num = pd.to_numeric(x, errors="coerce")
    # include_lowest to catch min edge, right=True to make intervals consistent
    return pd.cut(x_num, bins=edges, include_lowest=True).astype(str)


def _build_strata(df: pd.DataFrame, match_cols: List[str], ref_df: pd.DataFrame, n_bins: int, bin_method: str) -> Tuple[pd.Series, Dict[str, np.ndarray]]:
    """Return (strata_labels, bin_edges_map). For numeric columns, build bins from ref_df; for categoricals, use raw values.
    strata label is a string concatenation of per-column tokens.
    """
    tokens = []
    bin_edges_map: Dict[str, np.ndarray] = {}
    for col in match_cols:
        if col not in df.columns:
            raise SystemExit(f"Column '{col}' not found in provided CSV for strata building")
        if _is_numeric_series(ref_df[col]):
            edges = _compute_bins_from_reference(ref_df[col], n_bins=n_bins, method=bin_method)
            bin_edges_map[col] = edges
            tok = _apply_bins(df[col], edges)
        else:
            tok = df[col].astype(str)
        tokens.append(tok)
    if len(tokens) == 1:
        strata = tokens[0]
    else:
        strata = tokens[0].str.cat(tokens[1:], sep="||")
    return strata.astype(str), bin_edges_map


def _proportions_from_strata(strata_c: pd.Series) -> pd.Series:
    vc = strata_c.value_counts(dropna=False)
    props = vc / vc.sum()
    return props


def _sample_by_quota(index_by_stratum: Dict[str, np.ndarray], quotas: pd.Series, total_needed: int, rng: np.random.Generator) -> np.ndarray:
    # First pass: integer quotas by rounding
    desired = (quotas * total_needed).round().astype(int)
    selected = []
    deficit = 0

    # sample up to availability
    for key, need in desired.items():
        pool = index_by_stratum.get(key, np.array([], dtype=int))
        if need <= 0 or len(pool) == 0:
            deficit += max(need, 0)
            continue
        if need > len(pool):
            pick = pool  # take all
            deficit += (need - len(pool))
        else:
            pick = rng.choice(pool, size=need, replace=False)
        selected.append(pick)

    if deficit > 0:
        # redistribute remaining quota across strata with leftover capacity
        leftovers = []
        caps = []
        for key, pool in index_by_stratum.items():
            already = 0  # we didn't track per-key picks; compute capacity as len(pool)
            cap = len(pool)
            # subtract what we already picked for this key
            # (approximate by min(desired[key], cap))
            cap_left = max(0, cap - min(desired.get(key, 0), cap))
            if cap_left > 0:
                leftovers.append(pool)
                caps.append(cap_left)
        if sum(caps) > 0:
            # choose uniformly from concatenated leftovers up to deficit
            concat = np.concatenate([arr for arr in leftovers if len(arr) > 0])
            if len(concat) > 0:
                extra = min(deficit, len(concat))
                bonus = rng.choice(concat, size=extra, replace=False)
                selected.append(bonus)

    if len(selected) == 0:
        return np.array([], dtype=int)
    return np.unique(np.concatenate(selected))


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main(args):
    rng = np.random.default_rng(args.seed)

    # Load indices universe (rows we are allowed to sample from)
    shadow_idx = np.load(args.shadow_index_file)
    total_n = int(args.total_n_samples)

    # Basic masks array
    masks = np.zeros((args.n_shadows, total_n), dtype=bool)

    if args.match_cols is None:
        # Original behavior: uniform random from shadow_idx with given fraction
        per_shadow_count = int(round(len(shadow_idx) * args.per_shadow_train_frac))
        it = tqdm(range(args.n_shadows), total=args.n_shadows, desc="Build masks (uniform)") if tqdm else range(args.n_shadows)
        for j in it:
            chosen = rng.choice(shadow_idx, size=per_shadow_count, replace=False)
            masks[j, chosen] = True
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        np.save(args.out, masks)
        print(f"Saved masks to {args.out}; shape={masks.shape} (uniform frac)")
        return

    # Distribution-matching mode
    if not args.data_csv_a or not args.data_csv_c:
        raise SystemExit("--data-csv-a and --data-csv-c are required when --match-cols is set")

    # Load CSVs as strings to avoid unintended type inference
    df_a = pd.read_csv(args.data_csv_a, dtype=str, keep_default_na=False)
    df_c = pd.read_csv(args.data_csv_c, dtype=str, keep_default_na=False)

    # Restrict to the universe indices
    if df_a.shape[0] < total_n:
        raise SystemExit(f"--total-n-samples ({total_n}) exceeds rows in A ({df_a.shape[0]})")

    match_cols = [c.strip() for c in args.match_cols.split(',') if c.strip()]
    for c in match_cols:
        if c not in df_a.columns:
            raise SystemExit(f"Column '{c}' not found in A01 CSV")
        if c not in df_c.columns:
            raise SystemExit(f"Column '{c}' not found in C01 CSV")

    # Build strata using reference (C01) for binning numeric columns
    strata_a, _ = _build_strata(df_a.loc[shadow_idx], match_cols, ref_df=df_c, n_bins=args.num_bins, bin_method=args.bin_method)
    strata_c, _ = _build_strata(df_c, match_cols, ref_df=df_c, n_bins=args.num_bins, bin_method=args.bin_method)

    # Proportions from C01 (reference)
    props_c = _proportions_from_strata(strata_c)

    # Build index pools per stratum from A01 universe
    index_by_stratum: Dict[str, np.ndarray] = {}
    # Note: strata_a has index aligned with shadow_idx selection
    for key, idxs in strata_a.groupby(strata_a).groups.items():
        # idxs are positional within df_a.loc[shadow_idx]; map back to absolute indices
        abs_ids = shadow_idx[np.fromiter(idxs, dtype=int)]
        index_by_stratum[str(key)] = abs_ids

    # Per-shadow count: by absolute or fraction
    if args.per_shadow_train_abs is not None:
        per_shadow_count = int(args.per_shadow_train_abs)
    else:
        per_shadow_count = int(round(len(shadow_idx) * args.per_shadow_train_frac))

    it = tqdm(range(args.n_shadows), total=args.n_shadows, desc="Build masks (match C01)") if tqdm else range(args.n_shadows)
    for j in it:
        chosen = _sample_by_quota(index_by_stratum, props_c, per_shadow_count, rng)
        if len(chosen) < per_shadow_count:
            # fallback: top up uniformly from remaining pool
            pool = np.setdiff1d(shadow_idx, chosen, assume_unique=False)
            need = per_shadow_count - len(chosen)
            if need > 0 and len(pool) > 0:
                extra = rng.choice(pool, size=min(need, len(pool)), replace=False)
                chosen = np.unique(np.concatenate([chosen, extra]))
        masks[j, chosen] = True

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.save(args.out, masks)
    info = {
        'shape': masks.shape,
        'per_shadow_count': int(per_shadow_count),
        'n_strata_c': int(props_c.size),
        'bin_method': args.bin_method,
        'num_bins': int(args.num_bins),
        'match_cols': match_cols,
    }
    print(f"Saved masks to {args.out}; info={info}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shadow-index-file", type=str, required=True)
    parser.add_argument("--total-n-samples", type=int, required=True)
    parser.add_argument("--n-shadows", type=int, default=128)
    parser.add_argument("--per-shadow-train-frac", type=float, default=0.5)
    parser.add_argument("--per-shadow-train-abs", type=int, default=None,
                        help="Absolute train size per shadow; overrides fraction if set")
    parser.add_argument("--out", type=str, default="masks/masks.npy")
    parser.add_argument("--seed", type=int, default=1234)

    # New: distribution matching to C01
    parser.add_argument("--data-csv-a", type=str, default=None,
                        help="Path to A01.csv (attacker data / universe)")
    parser.add_argument("--data-csv-c", type=str, default=None,
                        help="Path to C01.csv (target training data or its anonymized proxy)")
    parser.add_argument("--match-cols", type=str, default=None,
                        help="Comma-separated column names to match distribution on (supports numeric or categorical)")
    parser.add_argument("--num-bins", type=int, default=8,
                        help="Number of bins for numeric columns when building strata (quantile/uniform)")
    parser.add_argument("--bin-method", type=str, default="quantile", choices=["quantile", "uniform"],
                        help="Binning strategy for numeric columns with reference to C01 distribution")

    args = parser.parse_args()
    main(args)