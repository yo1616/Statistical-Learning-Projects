# Mini-Project 1 — 商品名稱價格預測（SMAPE）

本專案提供 **本地訓練 + 伺服器推論** 的完整流程。助教在評分時只會執行 `inference.sh`，不得在伺服器上進行訓練。

## 專案結構
```
r14725000/
└── miniproject1/
    ├── inference.sh
    ├── README.md
    ├── requirements.txt
    ├── configs/
    │   ├── base.yaml
    │   └── search_space.yaml
    ├── models/
    │   └── model.joblib          # 本地訓練產出的模型檔
    └── src/
        ├── preprocess.py
        ├── pipeline_utils.py
        ├── train.py
        ├── inference.py
        └── utils.py
```

## 環境安裝
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 訓練流程（僅限本地）
目前的預設設定可直接使用下列指令訓練：
```bash
python src/train.py \
  --train_path train.csv \
  --config configs/base.yaml \
  --out_model_path models/model.joblib \
  --cv 5
```

其他模型或搜尋方式：
- `--model hashing_ridge`、`--model tfidf_linear_svr` 可切換課堂示範過的組合。
- `--search random --n_iter 20` 會讀 `configs/search_space.yaml` 做隨機搜尋（請同步在報告/EXPERIMENTS.md 紀錄結果）。
- 若要減速測試可調整 `--cv 3` 或以 `train_small.csv` 做 smoke test。

## 推論流程（助教評分時執行）
模型訓練好後，確認 `models/model.joblib` 在專案根目錄。推論指令：
```bash
./inference.sh --test_path /service/res/test.csv --output_path predictions.csv
```
推論腳本會：
1. 讀入 `models/model.joblib`（內含訓練時保存的特徵轉換與模型）。
2. 對指定的 `test.csv` 預測價格，輸出單欄 `price` 的 `predictions.csv`。

> 推論必須在 **CPU 10 分鐘內完成**。本地請先跑 `./inference.sh --test_path test.csv --output_path predictions.csv` 確認正確性與耗時。

## 提交前自檢
- [ ] `models/model.joblib` 已由最新訓練流程產出。
- [ ] `./inference.sh --test_path test.csv` 可以成功執行並產生合理的價格分布（建議檢查分位數與是否存在 ≤0 值）。
- [ ] 專案目錄沒有任何 `.csv`（含 `train.csv`、`test.csv`、`predictions.csv`）；助教會在上傳前刪除所有 CSV。
- [ ] `requirements.txt` 只保留必要套件，並於乾淨環境測試過安裝與推論無誤。
- [ ] 報告中記錄訓練流程、超參數與 Kaggle 連動結果，確保能重現你上傳的成績。

## Kaggle 上傳
推論後產生的 `predictions.csv` 僅含 `price` 欄位。上傳前建議：
```bash
python - <<'PY'
import pandas as pd
p = pd.read_csv("predictions.csv")
print(p.describe(percentiles=[0.5, 0.9, 0.95, 0.99]))
print("≤0 比例：", (p["price"] <= 0).mean())
PY
```
確認預測值皆為正且分布合理，再提交到 Kaggle。

## 其他提醒
- 伺服器僅供推論使用，不可上傳 `train.csv` 或任何訓練期間產生的 CSV。
- 若在報告中採用多組模型（例如 ElasticNet、LinearSVR、Hashing Ridge），請於 `EXPERIMENTS.md` 或報告表格呈現其交叉驗證 SMAPE、訓練時間與觀察。
- 互評、報告與 Kaggle 截止時間請依課程公告為準；最終成績需同時符合程式重現與 Kaggle 指標。
