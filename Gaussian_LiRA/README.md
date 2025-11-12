予選再現：bash LiRA/shell/qualifying.sh
本戦再現：bash LiRA/shell/main_tournament.sh

# `prepare_shadows.py`
## 概要
LiRA攻撃などで用いる**シャドウモデルの学習データマスク**を生成するスクリプトです。  
各シャドウモデルがどのサンプルを訓練に使用するかを示すブール行列（`n_shadows × total_n_samples`）を `.npy` 形式で出力します。

## 機能
- 単純ランダムサンプリング  
- 参照データ（C01）に基づく**分布マッチング（層化サンプリング）**

## 主な引数
| 引数 | 説明 |
|------|------|
| `--shadow-index-file` | サンプリング可能な行インデックス（.npy） |
| `--total-n-samples` | 総サンプル数 |
| `--n-shadows` | シャドウモデル数（既定128） |
| `--per-shadow-train-frac` | 各シャドウの訓練割合 |
| `--per-shadow-train-abs` | 訓練サンプルの絶対数（優先） |
| `--data-csv-a` / `--data-csv-c` | A01, C01 のCSVパス（分布マッチング時） |
| `--match-cols` | 分布を合わせる列名（例：`AGE,GENDER`） |
| `--out` | 出力ファイルパス（既定：`masks/masks.npy`） |

## 使用例
### 単純ランダム
```bash
python prepare_shadows.py \
  --shadow-index-file indexes/shadow_idx.npy \
  --total-n-samples 100000 \
  --n-shadows 128 \
  --per-shadow-train-frac 0.5 \
  --out masks/masks.npy
```

### 分布マッチング
```bash
python prepare_shadows.py \
  --shadow-index-file indexes/shadow_idx.npy \
  --total-n-samples 100000 \
  --n-shadows 128 \
  --per-shadow-train-abs 50000 \
  --data-csv-a data/A01.csv \
  --data-csv-c data/C01.csv \
  --match-cols AGE,GENDER,RACE \
  --out masks/masks.npy
```

## 出力
`masks.npy`：shape = `(n_shadows, total_n_samples)` のブール配列（True=使用サンプル）

# `train_shadows.py`
## 概要
LiRA攻撃で使用する**シャドウモデル群**を学習するスクリプトです。  
`prepare_shadows.py` で生成したマスクを用いて、複数のXGBoostモデルを並列で学習し保存します。

## 機能
- ターゲットモデルの特徴量構成に合わせて前処理を統一  
- 各シャドウモデルが異なる訓練データを使用  
- `joblib` によるマルチプロセス並列学習  
- `--rand-params` による軽微なハイパーパラメータ乱数化  

## 主な引数
| 引数 | 説明 |
|------|------|
| `--data-csv` / `--label-col` | 入力CSVとラベル列名 |
| `--data-x` / `--data-y` | NumPy入力を使用する場合 |
| `--masks-file` | `prepare_shadows.py` の出力マスク |
| `--models-dir` | モデル保存先（既定：`models/shadows`） |
| `--target-model` | 特徴量順序を合わせる参照モデル |
| `--num-rounds` | 学習ラウンド数（既定200） |
| `--n-jobs` | 並列実行数 |
| `--rand-params` | ランダム化を有効化 |

## 使用例
```bash
python train_shadows.py \
  --data-csv data/A01.csv \
  --label-col stroke_flag \
  --masks-file masks/masks.npy \
  --models-dir models/shadows \
  --target-model models/target.json \
  --n-jobs 8 --rand-params
```

## 出力
`models/shadows/shadow_0000.json`, `shadow_0001.json`, … として学習済みXGBoostモデルを保存します。

# `attack_lira.py`
## 概要
LiRA（Likelihood-Ratio Attack）の**LLRスコア**を計算するスクリプトです。  
ターゲットモデルとシャドウモデル群の予測結果から、各サンプルが学習に含まれていた可能性を対数尤度比（LLR）として推定します。

## 機能
- ターゲットモデルの出力とシャドウモデル群の出力を比較して LLR を算出  
- in/out の分布を正規分布で近似して平均・分散をベクトル化で計算  
- 数値安定化（分散の下限・フォールバック処理）  
- LLRスコアおよびメタ情報を `.npy` と `.json` で出力

## 主な引数
| 引数 | 説明 |
|------|------|
| `--target-model` | ターゲットXGBoostモデル（特徴順序の参照） |
| `--data-csv` / `--label-col` | CSV入力を使う場合 |
| `--data-x` / `--data-y` | NumPy入力を使う場合 |
| `--targets-index-file` | スコアを計算する行インデックス（.npy） |
| `--masks-file` | シャドウごとの使用マスク（prepare_shadows.py の出力） |
| `--shadows-dir` | 学習済みシャドウモデルのディレクトリ |
| `--out-scores` | 出力スコアファイル（.npy） |
| `--sigma-floor` | 分散下限（数値安定化；既定1e-6） |

## 使用例
```bash
python attack_lira.py \
  --target-model models/target.json \
  --data-csv data/A01.csv --label-col stroke_flag \
  --targets-index-file indexes/targets.npy \
  --masks-file masks/masks.npy \
  --shadows-dir models/shadows \
  --out-scores scores/scores_targets.npy
```


## 出力
- `scores_targets.npy`：各対象の LLR スコア（float32）  
- `scores_targets_meta.json`：平均・分散・件数などのメタ情報

# `make_membership_csv.py`
## 概要
LiRA攻撃などの評価実験で使用する**メンバーシップベクトル（0/1ラベル）**を生成するスクリプトです。  
スコアの上位K件を「メンバー（1）」としてマークし、全データ長の0/1ベクトルをCSV形式で出力します。

## 機能
- スコアファイル（.npy）を読み込み、降順で上位K件を抽出  
- 対応するインデックス配列から全体長Nのベクトルを作成  
- 上位インデックスの保存（任意）  
- 生成ベクトルをCSVで出力

## 主な引数
| 引数 | 説明 |
|------|------|
| `--scores-file` | スコアファイル（例：`scores_targets.npy`） |
| `--indices-file` | スコア対応インデックス（例：`indexes/targets.npy`） |
| `--total-n-samples` | 全データ数（出力ベクトル長） |
| `--top-k` | メンバーとみなす上位件数（既定10000） |
| `--out-csv` | 出力CSVパス（例：`F_ij.csv`） |
| `--out-indices` | 上位K件のインデックスを別ファイルに保存（任意） |

## 使用例
```bash
python make_membership_csv.py \
  --scores-file scores/scores_targets.npy \
  --indices-file indexes/targets.npy \
  --total-n-samples 100000 \
  --top-k 10000 \
  --out-csv results/membership.csv \
  --out-indices results/topk_indices.txt
```

## 出力
- `membership.csv`：全長Nの0/1ベクトル（1=メンバー）  
- `topk_indices.txt`（任意）：上位K件の絶対行インデックス

# `utils.py`
## 概要
LiRA実験全体で共通して利用される**ユーティリティ関数群**をまとめたモジュールです。  
再現性の固定、XGBoost学習・推論のラッパ、CSVからの特徴量構築、特徴名の整合性維持、および単一点のGaussian LLR計算などを提供します。

## 機能
- 乱数シードの固定（NumPy・Python・環境変数）
- XGBoost 2値分類モデルの簡易学習 (`train_xgb_binary`)
- XGBoostモデルでの margin 出力予測 (`predict_margin`)
- CSVからターゲットモデルと同一構成の特徴行列を構築 (`build_X_like_xgbt_from_csv`)
- 学習済みモデルから特徴名リストを抽出 (`extract_feature_names_from_model`)
