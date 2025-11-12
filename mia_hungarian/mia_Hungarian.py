#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from scipy.optimize import linear_sum_assignment

def detect_column_types(df: pd.DataFrame):
    numeric_cols, categorical_cols = [], []
    for col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().mean() >= 0.95:
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)
    return numeric_cols, categorical_cols

def preprocess(Ai: pd.DataFrame, Ci: pd.DataFrame):
    numeric_cols, categorical_cols = detect_column_types(Ai)

    # 数値列
    X_num_A, X_num_C = np.empty((len(Ai),0)), np.empty((len(Ci),0))
    if numeric_cols:
        Ai_num = Ai[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        Ci_num = Ci[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        scaler = MinMaxScaler()
        X_num_A = scaler.fit_transform(Ai_num)
        X_num_C = scaler.transform(Ci_num)

    # カテゴリ列
    X_cat_A, X_cat_C = np.empty((len(Ai),0)), np.empty((len(Ci),0))
    if categorical_cols:
        print("カテゴリデータをOne-Hotエンコーディングしています...")
        enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        enc.fit(pd.concat([Ai[categorical_cols], Ci[categorical_cols]], axis=0))
        X_cat_A = enc.transform(Ai[categorical_cols])
        X_cat_C = enc.transform(Ci[categorical_cols])

    # 結合
    X_A = np.hstack([X for X in [X_num_A, X_cat_A] if X.size > 0])
    X_C = np.hstack([X for X in [X_num_C, X_cat_C] if X.size > 0])
    return X_A.astype(np.float32), X_C.astype(np.float32)

def compute_distance_in_batches(X_A, X_C, batch_size=1000):
    """距離行列をバッチ単位で計算して返す"""
    nA, nC = X_A.shape[0], X_C.shape[0]
    dist_matrix = np.empty((nA, nC), dtype=np.float32)
    for start in range(0, nA, batch_size):
        end = min(start + batch_size, nA)
        dist_matrix[start:end] = np.abs(X_A[start:end, None, :] - X_C[None, :, :]).sum(axis=2)
        print(f"距離計算中: {start}～{end} / {nA}")
    return dist_matrix

def main():
    parser = argparse.ArgumentParser(description="大規模距離ベースMIA (Hungarian法, バッチ処理, 出力形式統一版)")
    parser.add_argument("Ai_csv")
    parser.add_argument("Ci_csv")
    parser.add_argument("-o", "--output", default="Fij.csv")
    parser.add_argument("--batch_size", type=int, default=1000)
    args = parser.parse_args()

    print("CSVファイルを読み込んでいます...")
    Ai = pd.read_csv(args.Ai_csv, dtype=str, keep_default_na=False)
    Ci = pd.read_csv(args.Ci_csv, dtype=str, keep_default_na=False)

    # 前処理
    X_A, X_C = preprocess(Ai, Ci)
    if X_A.shape[1] == 0:
        print("全列が除外されました。すべて0を出力します。")
        pd.DataFrame(np.zeros(len(Ai), dtype=int)).to_csv(args.output, header=False, index=False)
        return

    # 距離行列計算（バッチ）
    print("距離行列を計算中...")
    dist_matrix = compute_distance_in_batches(X_A, X_C, batch_size=args.batch_size)

    # Hungarian法
    print("Hungarian法で最適マッチングを計算中...")
    row_ind, col_ind = linear_sum_assignment(dist_matrix)

    # 出力ベクトル作成 (Ai 行数に合わせ、マッチしたものを 1 に)
    result = np.zeros(len(Ai), dtype=int)
    result[row_ind] = 1

    # CSV出力
    output_df = pd.DataFrame(result)
    output_df.to_csv(args.output, header=False, index=False)
    print(f"最終結果を '{args.output}' に出力しました。")
    print(f"特定されたユニークな個人の数: {result.sum()}")

if __name__ == "__main__":
    main()