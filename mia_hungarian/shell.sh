#!/bin/zsh
# mia_Hungarian.py の実行例シェルスクリプト
# 使い方: ./shell.sh Ai.csv Ci.csv [出力ファイル名]

AI_CSV=${1:-Ai.csv}
CI_CSV=${2:-Ci.csv}
OUTPUT=${3:-Fij.csv}

# 必要なPythonパッケージのインストール（初回のみ）
pip install numpy pandas scipy scikit-learn

# スクリプトの実行
python3 mia_Hungarian.py "$AI_CSV" "$CI_CSV" -o "$OUTPUT"

echo "出力ファイル: $OUTPUT"
