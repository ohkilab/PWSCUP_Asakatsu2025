import argparse
import numpy as np
import os

def main(args):
    # load
    scores = np.load(args.scores_file)
    indices = np.load(args.indices_file)

    if scores.ndim != 1:
        raise SystemExit(f"scores must be 1-D array, got shape {scores.shape}")
    if indices.ndim != 1:
        raise SystemExit(f"indices must be 1-D array, got shape {indices.shape}")
    if len(scores) != len(indices):
        raise SystemExit(f"Length mismatch: scores={len(scores)} vs indices={len(indices)}")

    M = len(scores)
    k = int(min(args.top_k, M))
    if k <= 0:
        raise SystemExit(f"top_k must be positive, got {args.top_k}")

    # select top-k (descending by score)
    order_desc = np.argsort(-scores, kind='mergesort')  # stable sort
    topk_local_idx = order_desc[:k]
    topk_abs = sorted(int(indices[i]) for i in topk_local_idx)

    N = int(args.total_n_samples)
    if N <= 0:
        raise SystemExit(f"total_n_samples must be positive, got {args.total_n_samples}")

    # build vector
    vec = np.zeros(N, dtype=np.int8)
    for idx in topk_abs:
        if 0 <= idx < N:
            vec[idx] = 1
        else:
            raise SystemExit(f"Index out of range in indices_file: {idx} not in [0, {N})")

    # save CSV
    out_dir = os.path.dirname(args.out_csv) or '.'
    os.makedirs(out_dir, exist_ok=True)
    np.savetxt(args.out_csv, vec, fmt='%d')

    # optional: save indices
    if args.out_indices:
        os.makedirs(os.path.dirname(args.out_indices) or '.', exist_ok=True)
        np.savetxt(args.out_indices, np.array(topk_abs, dtype=int), fmt='%d')

    # summary
    print(f"Saved membership CSV: {args.out_csv} (shape {vec.shape})")
    print(f"Marked ones: {vec.sum()} (requested top_k={args.top_k}, available M={M})")
    if args.out_indices:
        print(f"Saved selected absolute indices to: {args.out_indices}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Select top-K by score and emit full-length 0/1 membership CSV"
    )
    p.add_argument('--scores-file', required=True,
                   help='npy of scores (e.g., scores_targets.npy)')
    p.add_argument('--indices-file', required=True,
                   help='npy of absolute row indices corresponding to the scores (same order as scores)')
    p.add_argument('--total-n-samples', type=int, required=True,
                   help='Total dataset size N (length of output vector)')
    p.add_argument('--top-k', type=int, default=10000,
                   help='Number of members to mark as 1 (default 10000)')
    p.add_argument('--out-csv', dest='out_csv', required=True,
                   help='Output CSV path, e.g., F_ij.csv')
    p.add_argument('--out-indices', default=None,
                   help='Optional: also save the selected absolute indices to a txt file')
    args = p.parse_args()
    main(args)