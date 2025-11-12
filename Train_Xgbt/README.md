# Train_Xgbt

加工フェーズにおけるXGboostの学習スクリプトです．

## 前処理
Syntheaで合成データを作成してください．私たちはマサチューセッツを指定して200,000件のレコードを作成しました．

## train.py
学習，モデル保存を行うスクリプトです．学習に合成データ，検証にB.csvを指定して，F1 Scoreが最も良いモデルを保存します．

```bash
    uv run train.py \
        /path/to/synthea_data.csv \
        /path/to/B.csv \
        --model-json /path/to/json_file
```

## eval.py
検証スクリプトです．

```bash
    uv run eval.py \
        /path/to/json_file \
        --test-csv /path/to/B.csv
```