# EXPERIMENTS — CV 與超參數掃描計畫

目標：以 5 折 KFold、SMAPE 指標，建立穩健 baseline 並比較 3 條模型路線。

## A. Hashing + Ridge（快速強健）
- Vectorizer: HashingVectorizer (char 1–5gram), alternate_sign=False
- Grid（隨機抽樣）：
  - n_features ∈ {2^18, 2^19, 2^20}
  - ngram_range ∈ {(1,3), (1,4), (1,5)}
  - alpha ∈ {0.1, 0.3, 1.0, 3.0, 10.0, 30.0}
- 預期：訓練/推論極快；模型不儲存詞典，體積小，適合伺服器 10 分鐘限制。

## B. TF-IDF + ElasticNet（可稀疏化）
- Vectorizer: TfidfVectorizer (char 1–5gram, max_features 200k–400k)
- Grid（隨機抽樣）：
  - max_features ∈ {200k, 300k, 400k}
  - ngram_range ∈ {(1,4), (1,5)}
  - alpha ∈ {0.001, 0.01, 0.1, 0.3}
  - l1_ratio ∈ {0.05, 0.2, 0.5, 0.8}
- 預期：可更好地控制特徵量與稀疏度，但詞典會使模型檔案較大。

## C. TF-IDF + LinearSVR（對重尾更穩）
- Vectorizer: TfidfVectorizer (char)
- Grid（隨機抽樣）：
  - max_features ∈ {200k, 300k}
  - ngram_range ∈ {(1,4), (1,5)}
  - C ∈ {0.5, 1.0, 2.0}
  - epsilon ∈ {0.05, 0.1, 0.2}
- 預期：對高價 outliers 較穩，但推論時間可能較 Ridge 慢。

## 報告呈現建議的表格欄位
| Model | Vectorizer | Main Hyperparams | Train Time (min) | CV SMAPE | Notes |
|------:|------------|------------------|------------------:|---------:|------|
| Hashing+Ridge | char 1–5 | n_feat, alpha |  |  |  |
| TF-IDF+EN | char 1–5 | max_feat, alpha, l1 |  |  |  |
| TF-IDF+LinearSVR | char 1–5 | max_feat, C, eps |  |  |  |

> 注意：請固定 random seed、以同一清理流程對 train/test 做前處理，並避免任何手動調整輸出。
