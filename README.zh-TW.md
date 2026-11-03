# 統計學習與深度學習｜機器學習專案作品集

[English](./README.md) | [**繁體中文**](./README.zh-TW.md)

**機器學習應用 · 自然語言處理 · 預測分析 · 商業問題解決**

本作品集整理「統計學習與深度學習」課程中的兩項機器學習實作專案，分別針對**商品價格預測**與**職缺人才匹配分類**進行研究與建模。

兩項專案皆從實際商業情境出發，透過資料前處理、特徵工程、模型訓練與驗證，探索如何將非結構化與結構化資料轉換為具有預測價值的資訊。

專案重點不僅在於提升模型準確度，也包含運算效率、類別不平衡、模型泛化能力，以及不同解決方案之間的取捨。

---

## 專案總覽

| 專案 | 商業問題 | 預測任務 | 最終模型 | 主要成果 |
|---|---|---|---|---|
| **01. 商品名稱價格預測** | 利用商品名稱資訊估計價格 | 迴歸 | Hashing + Ridge | 5-Fold CV SMAPE：**39.04%** |
| **02. 職缺人才匹配分類** | 根據職缺與人才特徵預測匹配結果 | 分類 | LightGBM | Validation F1：**0.5696** |

兩項專案皆採用 Python 進行資料處理與模型實作，並透過實驗比較不同模型及特徵工程方式的效果。

上述成績來自原始課程報告，並非公開版本重新執行後的驗證結果。

---

## 專案一｜商品名稱價格預測

**Regression · NLP · Feature Engineering · Model Efficiency**

### 問題背景

在電子商務環境中，商品名稱往往包含品牌、規格、數量、容量及商品類別等資訊。

然而，商品名稱通常屬於非結構化文字資料，且可能存在全形半形混用、單位表示不一致、縮寫及符號雜訊等問題，使模型難以直接利用相關資訊。

本專案希望建立一套以商品名稱為輸入的價格預測模型，並探討不同文字特徵表示與迴歸模型在預測效果及運算成本上的差異。

### 實作方法

**資料前處理**
- Unicode NFKC 正規化
- 全形與半形字元統一
- 商品單位標準化
- 多餘符號與空白清理

**特徵工程**
- Character-level n-gram
- HashingVectorizer
- 字串長度、數字數量等數值特徵
- 數值特徵截尾與標準化

**模型訓練與驗證**
- Ridge Regression
- Logarithmic Target Transformation
- 5-Fold Cross-Validation
- SMAPE 評估指標

### 實驗成果

| 指標 | 結果 |
|---|---|
| 最終採用模型 | HashingVectorizer + Ridge Regression |
| 最佳平均 CV SMAPE | **39.04%** |
| Ridge α | 0.5 |
| 特徵維度 | 2²⁰ |
| 約略訓練時間 | 9 分鐘 |

專案曾比較 TF-IDF 搭配 Ridge、ElasticNet 與 LinearSVR 等模型。

雖然部分 TF-IDF 模型的預測誤差較低，但訓練時間及運算成本也明顯提高。

因此，最終選擇 Hashing + Ridge，作為預測表現、運算效率與推論限制之間的折衷方案。

### 核心學習

這項專案讓我理解，**機器學習模型的選擇不應只考量單一準確度指標，也必須評估計算成本、執行限制及實際使用情境。**

**[查看專案一：商品名稱價格預測](./01-product-price-prediction/)**

---

## 專案二｜職缺人才匹配分類

**Classification · NLP · Feature Engineering · Imbalanced Learning**

### 問題背景

在人才招募過程中，企業需要比較求職者的技能、經歷、職務偏好及工作條件，判斷其與職缺需求之間的匹配程度。

這類資料同時包含文字描述、類別資訊與數值條件，難以直接使用單一規則或簡單文字比對處理。

本專案透過監督式機器學習，研究如何利用文字特徵與結構化資訊預測職缺人才匹配結果。

### 實作方法

**文字與資料處理**
- TF-IDF 文字向量化
- Unigram / Bigram 特徵
- One-Hot Encoding
- 缺失值處理
- 日期與數值特徵轉換

**自訂特徵工程**
- 求職者與職缺技能重疊程度
- 工作職責匹配
- 地區偏好
- 工作時間與休假條件
- 薪資區間等結構化特徵

**模型與評估**
- Logistic Regression（基準模型）
- LightGBM（主要模型）
- Class Weight / Scale Positive Weight
- Decision Threshold Optimization
- Precision、Recall、F1 Score

此外，專案也包含以 PyTorch 與中文 RoBERTa 為基礎的深度學習訓練程式；目前主要量化結果仍以 LightGBM 與 Logistic Regression 的比較為準。

### 實驗成果

| 評估指標 | LightGBM | Logistic Regression |
|---|---:|---:|
| Precision | **0.5870** | 0.4090 |
| Recall | **0.5531** | 0.5240 |
| F1 Score | **0.5696** | 0.4592 |
| 訓練時間 | 706 秒 | 415 秒 |

LightGBM 的 F1 Score 相較 Logistic Regression **相對提升約 24.0%**。

專案報告亦記錄了特徵工程及類別權重調整過程中的 F1 改善，顯示資料處理與特徵設計對模型表現具有重要影響。

### 模型分析與決策

LightGBM 能捕捉非線性特徵交互作用，因此在本次實驗中取得較佳的 Precision、Recall 與 F1。

另一方面，訓練曲線顯示，模型在後期持續改善訓練集表現，但驗證 F1 已逐漸進入平台期。

這反映出模型複雜度與泛化效果之間的取捨，也指出 Early Stopping、正則化與額外驗證的改善方向。

### 核心學習

這項專案讓我理解，**模型表現不僅取決於演算法本身，領域特徵工程、資料分布、類別不平衡及決策門檻同樣影響預測結果。**

同時，招募相關的預測結果並不等於公平或正當的錄取決策。若要應用於真實企業環境，仍須進行偏誤檢測、隱私保護與人工審核。

**[查看專案二：職缺人才匹配分類](./02-job-talent-matching/)**

---

## 技術能力整理

| 技術領域 | 使用工具與方法 |
|---|---|
| 程式語言 | Python |
| 資料處理 | pandas、NumPy |
| 機器學習 | scikit-learn、LightGBM |
| 深度學習探索 | PyTorch、Hugging Face Transformers |
| 自然語言處理 | TF-IDF、Character n-gram、Text Normalization |
| 特徵工程 | Feature Hashing、One-Hot Encoding、自訂匹配特徵 |
| 迴歸模型 | Ridge、ElasticNet、LinearSVR |
| 分類模型 | Logistic Regression、Gradient Boosting |
| 模型評估 | K-Fold CV、SMAPE、Precision、Recall、F1 |
| 開發流程 | Hyperparameter Tuning、Threshold Optimization、Model Serialization、Inference |

---

## 專案目錄

```text
Statistical-Learning-Projects/
│
├── README.md
├── README.zh-TW.md
│
├── 01-product-price-prediction/
│   ├── README.md
│   ├── README.zh-TW.md
│   ├── EXPERIMENTS.md
│   ├── requirements.txt
│   ├── inference.sh
│   ├── configs/
│   ├── src/
│   └── models/
│
├── 02-job-talent-matching/
│   ├── README.md
│   ├── README.zh-TW.md
│   ├── requirements.txt
│   ├── inference.sh
│   ├── inference.py
│   ├── train_ml.py
│   ├── train_logreg.py
│   ├── train_dl.py
│   ├── plot_curves.py
│   ├── config/
│   ├── src/
│   ├── results/
│   └── artifacts/
│
└── .gitignore
```

兩個子專案皆附有各自的 README，提供更完整的技術實作、模型設定、實驗結果及操作說明。

部分模型產出檔可能因檔案大小或公開限制而未包含於 Repository。

---

## 跨專案學習與反思

### 1. 特徵工程是模型效能的重要基礎

商品價格預測透過文字正規化及字元特徵擷取改善輸入資料品質；職缺人才匹配則透過技能、職責與工作條件等自訂特徵，增加模型可利用的資訊。

這使我理解，除了選擇演算法，也需要思考原始資料如何轉換成具有問題意義的特徵。

### 2. 模型選擇需要考慮不同限制

兩個專案呈現不同的模型決策：

- **價格預測：** 選擇兼顧預測效果與運算效率的 Hashing + Ridge。
- **人才匹配：** 選擇在驗證集上有較佳分類表現的 LightGBM。

模型選擇並不存在適用於所有情境的單一標準，而應根據問題需求、評估指標及資源限制決定。

### 3. 必須使用正確的評估方法

迴歸與分類是不同的預測問題，需要不同的評估指標。

商品價格預測使用 SMAPE 衡量價格誤差；職缺人才匹配則以 Precision、Recall 及 F1 Score 檢視分類效果，並考量類別不平衡。

### 4. 從模型訓練到可重複執行的流程

兩個專案皆涉及資料處理、模型訓練、超參數設定、模型儲存與推論程式設計。

透過將不同步驟模組化，建立較容易維護、檢查與重現的機器學習工作流程。

### 5. 數據結果需要結合商業情境解讀

模型評估數字並不等同於真實商業成效。

在實際應用前，仍需評估資料品質、模型泛化能力、營運成本、使用情境及相關風險。

尤其涉及人才招募的模型，更需要注意公平性、個人資料保護及人工決策權限。

---

## 資料與實驗重現說明

為遵守原始課程資料的使用限制，本公開作品集不包含完整的訓練與測試資料。

Repository 保留程式碼、模型設定與實驗方法。若取得合法授權的資料集，可以依照各子專案 README 中的說明進行訓練及推論。

本頁的量化成果來自原始課程實驗報告，並非公開版本重新執行的獨立驗證結果。

---

## 關於作者

**蔡承佑 Patrick Tsai（TSAI CHENGYU）**

關注人工智慧、商業資料分析、資訊系統及企業問題解決的跨領域應用。

希望透過資訊技術、資料分析及 AI 應用，將商業問題轉換成具體的分析方法與可行解決方案。

**主要興趣領域**

- 人工智慧應用（Applied AI）
- 商業資料分析（Business Analytics）
- AI 產品開發（AI Product Development）
- 科技與數位轉型（Technology & Digital Transformation）
- 資料驅動決策（Data-Driven Decision Making）

---

## 專案聲明

本作品集為大學「統計學習與深度學習」課程之學術專案。

所有模型成績均依據原始課程實驗報告整理，尚未代表外部資料集驗證或正式商業部署。

其中職缺人才匹配專案僅作為機器學習研究與技術展示，不應直接用於真實招募流程中的自動化錄取或淘汰決策。

**作者：** 蔡承佑 Patrick Tsai  
**專案類型：** Academic Machine Learning Portfolio  
**主要技術：** Python、NLP、Machine Learning  
**核心方向：** AI × Data Analytics × Business Problem Solving
