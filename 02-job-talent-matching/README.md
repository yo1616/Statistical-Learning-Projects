# Job–Talent Matching Prediction Using Machine Learning

[English](./README.md) | [繁體中文](./README.zh-TW.md)

**Statistical Learning & Deep Learning — Mini-Project 2**

**Python · NLP · Feature Engineering · LightGBM · Logistic Regression · Classification**

An applied machine learning project that predicts job–candidate matching outcomes using textual, categorical, and engineered compatibility features.

## Project Highlights

| Metric | LightGBM | Logistic Regression |
|---|---:|---:|
| **Validation F1 Score** | **0.5696** | 0.4592 |
| Precision | 0.5870 | 0.4090 |
| Recall | 0.5531 | 0.5240 |
| Reported Training Time | 706 s | 415 s |

**Key achievements**

- Achieved a reported **0.5696 validation F1** using LightGBM, approximately **24.0% higher** than the Logistic Regression baseline.
- Combined TF-IDF text features, categorical encoding, and domain-specific compatibility features.
- Addressed class imbalance using class weighting and `scale_pos_weight`.
- Implemented decision-threshold optimization to improve the F1 score.
- Built reusable model training, serialization, and inference workflows.

*Results are taken from the original academic project report and have not been independently rerun.*

---

## 1. Problem Definition

### Business Context

Recruitment matching involves evaluating compatibility between job requirements and candidate profiles.

Both sides contain heterogeneous information:

- Job descriptions and responsibilities
- Required technical skills
- Candidate qualifications and experience
- Location and work schedule preferences
- Categorical and numerical attributes

Traditional rule-based matching may struggle to capture complex interactions among these factors.

This project investigates whether supervised machine learning can better identify patterns associated with job–candidate matching outcomes.

### Objective

The primary objective is to develop and compare classification models for predicting binary matching labels.

The project focuses on:

1. Processing structured and unstructured recruitment data.
2. Engineering features that represent job–candidate compatibility.
3. Addressing imbalanced classification labels.
4. Comparing linear and tree-based classifiers.
5. Improving predictive performance through threshold selection.
6. Building a reusable inference pipeline.

The system is an academic predictive modeling experiment, not an autonomous hiring decision system.

---

## 2. Data Preprocessing & Feature Engineering

### 2.1 Text Feature Extraction

Recruitment data includes textual fields such as job descriptions, responsibilities, professional experience, and skills.

Multiple text fields are combined after missing values are replaced with empty strings.

**TF-IDF Vectorization**

Text features are transformed using TF-IDF with unigram and bigram representations.

Current configuration:

```yaml
tfidf_max_features: 12000
tfidf_ngram_range: [1, 2]
```

These features represent individual terms and adjacent term pairs.

The original report documented experiments using 8,000 TF-IDF features, while the current configuration specifies 12,000. The exact configuration used for each reported result has not been independently reconstructed.

### 2.2 Categorical Features

Categorical attributes are processed through encoding and missing-value handling.

The original report describes:

- One-hot encoding
- Most-frequent-value imputation
- Handling previously unseen categories

Categorical information includes job-related and organizational attributes.

### 2.3 Date and Numerical Features

Date fields are transformed into structured representations.

The model also uses numerical attributes derived from recruitment information.

Examples include:

- Salary range boundaries
- Age range boundaries
- Text length features
- Matching and compatibility indicators

### 2.4 Domain-Specific Feature Engineering

A key component of the project is `src/features_match.py`, which generates features representing compatibility between job requirements and candidate information.

Examples include:

| Feature | Description |
|---|---|
| `match_skill_overlap` | Overlap between required and candidate skills |
| `match_duty_overlap` | Compatibility between job duties and candidate information |
| `match_city` | Location preference matching |
| `match_worktime` | Working-time compatibility |
| `match_vacation` | Vacation or schedule compatibility |
| `salary_min`, `salary_max`, `salary_mean` | Numerical salary-range representations |

The original project report also discusses age-range features and their role in model training.

The report describes an improvement from approximately **0.29 to 0.56 F1** during feature-engineering experimentation.

Because the report does not provide a complete controlled ablation table, this improvement should be interpreted as a reported development-stage observation rather than the isolated causal effect of one feature group.

---

## 3. Model Development

Two conventional machine learning models were used for the main comparison.

### 3.1 Logistic Regression — Baseline

Logistic Regression serves as an interpretable linear classification baseline.

The model uses transformed textual and structured features and applies regularization to manage high-dimensional feature representations.

The original experiments explored class weighting and decision-threshold tuning.

Logistic Regression provided a useful reference point for evaluating whether a more expressive nonlinear classifier offered additional predictive value.

### 3.2 LightGBM — Selected Model

LightGBM is the primary classifier selected for the final machine learning inference pipeline.

Gradient-boosted decision trees can capture nonlinear feature interactions and conditional relationships that may be difficult to model using a linear classifier.

The current configuration includes:

```yaml
ml_model_params:
  boosting_type: gbdt
  objective: binary
  metric: binary_logloss
  n_estimators: 800
  learning_rate: 0.03
  num_leaves: 127
  max_depth: -1
  subsample: 0.8
  colsample_bytree: 0.8
  min_child_samples: 40
  lambda_l1: 1.0
  lambda_l2: 1.0
  scale_pos_weight: 4.9
  n_jobs: 1
  force_col_wise: true
```

The positive class receives additional weight through `scale_pos_weight`, helping address class imbalance during training.

### 3.3 Decision Threshold Optimization

The LightGBM training pipeline generates validation probabilities and searches for an F1-maximizing decision threshold.

The implemented threshold candidates are:

```python
thresholds = np.arange(0.25, 0.56, 0.05)
```

This evaluates thresholds from approximately 0.25 to 0.55 in increments of 0.05.

The selected threshold is saved separately as:

```text
artifacts/ml/threshold.txt
```

Threshold optimization is important because the default 0.5 decision threshold does not necessarily maximize F1 for imbalanced classification tasks.

The specific winning threshold was not recorded in the supplied report.

---

## 4. Experimental Results

### 4.1 Model Comparison

The project report presents the following results:

| Model | Precision | Recall | F1 Score | Training Time |
|---|---:|---:|---:|---:|
| **LightGBM** | **0.5870** | **0.5531** | **0.5696** | 706 s |
| Logistic Regression | 0.4090 | 0.5240 | 0.4592 | 415 s |

**Interpretation**

LightGBM outperformed Logistic Regression across all three reported classification metrics.

The F1 score increased from 0.4592 to 0.5696, corresponding to an approximate **24.0% relative improvement**.

LightGBM required more training time, representing a trade-off between predictive performance and computational efficiency.

The original report attributes the stronger performance partly to LightGBM's ability to model nonlinear relationships among engineered compatibility features.

### 4.2 Training Behavior

Training and validation metrics were recorded at selected boosting rounds.

| Round | Train Logloss | Valid Logloss | Train F1 | Valid F1 |
|---:|---:|---:|---:|---:|
| 80 | 0.4385 | 0.4733 | 0.6135 | 0.5124 |
| 160 | 0.3901 | 0.4524 | 0.6849 | 0.5270 |
| 240 | 0.3443 | 0.4279 | 0.7457 | 0.5418 |
| 320 | 0.3089 | 0.4104 | 0.7932 | 0.5514 |
| 400 | 0.2798 | 0.3969 | 0.8323 | 0.5621 |
| 480 | 0.2550 | 0.3864 | 0.8634 | 0.5695 |
| 560 | 0.2335 | 0.3772 | 0.8897 | 0.5684 |
| 640 | 0.2144 | 0.3696 | 0.9114 | 0.5686 |
| 720 | 0.1975 | 0.3635 | 0.9278 | 0.5699 |
| 800 | 0.1823 | 0.3577 | 0.9422 | 0.5696 |

### 4.3 Learning Curves

![LightGBM Training Curves](./results/training_curves.png)

The recorded curves show:

- Training F1 increased continuously, reaching 0.9422 at 800 rounds.
- Validation F1 improved gradually and then approached a plateau near 0.57.
- Both training and validation logloss decreased across the recorded rounds.
- The train–validation F1 gap widened substantially in later rounds.

Although the validation loss did not show a reversal, the growing F1 gap indicates increasing training-set specialization with limited additional validation F1 gains.

This suggests that early stopping and additional regularization could be useful areas for future experimentation.

---

## 5. Class Imbalance Handling

The project report identifies class imbalance as an important modeling challenge.

Without additional treatment, a classifier may favor the majority class and under-detect positive matching examples.

The experiments explored:

**Logistic Regression**

- Positive-class weighting
- Class-balanced optimization
- Classification threshold adjustment

**LightGBM**

- `scale_pos_weight`
- F1-oriented decision threshold selection

The report describes a development-stage Logistic Regression F1 improvement from approximately 0.29 to 0.40 following class-weight adjustments.

These observations demonstrate the importance of evaluating minority-class performance rather than relying exclusively on overall classification accuracy.

---

## 6. Deep Learning Exploration

In addition to the conventional machine learning experiments, the repository includes a PyTorch-based deep learning training implementation.

### Architecture

The deep learning pipeline combines:

- A pretrained Chinese RoBERTa tokenizer and text encoder
- Structured features transformed through a tabular preprocessor
- A custom `DLJobTalentModel` implementation
- Binary classification with a weighted-loss option

The configured pretrained text model is:

`hfl/chinese-roberta-wwm-ext`

### Training Configuration

```yaml
dl_model_name: hfl/chinese-roberta-wwm-ext
max_length: 128
batch_size: 16
epochs: 3
lr: 0.00002
weight_decay: 0.01
tab_hidden_size: 256
inference_threshold: 0.5
```

The implementation uses:

- PyTorch and Hugging Face Transformers
- AdamW optimization
- Linear learning-rate scheduling with warmup
- `BCEWithLogitsLoss`
- Validation F1 monitoring
- Best-checkpoint saving

The script is structured to save model parameters, tokenizer files, preprocessing artifacts, and training history.

**Important:** The supplied course report does not include a verified validation score for the deep learning model. Accordingly, the quantitative comparison in this README is limited to LightGBM and Logistic Regression.

The default `inference.sh` uses the conventional machine learning pipeline rather than the deep learning model.

---

## 7. System Architecture

```text
Recruitment Data
       |
       v
Data Loading & Validation
       |
       v
Compatibility Feature Engineering
       |
       v
+----------------------------------+
| Preprocessing                    |
|                                  |
| TF-IDF Text Features             |
| One-Hot Categorical Features     |
| Numerical & Date Features        |
+----------------------------------+
       |
       v
Train / Validation Split
       |
       v
+----------------------+----------------------+
| Logistic Regression  | LightGBM             |
| Baseline             | Primary Classifier   |
+----------------------+----------------------+
                       |
                       v
              Validation Evaluation
                       |
                       v
              Threshold Selection
                       |
                       v
              Model Serialization
                       |
                       v
              Inference Pipeline
                       |
                       v
              Binary Predictions
```

A separate deep learning training path is provided through `train_dl.py`.

---

## 8. Repository Structure

```text
02-job-talent-matching/
├── README.md
├── README.zh-TW.md
├── requirements.txt
├── inference.sh
├── inference.py
├── train_ml.py
├── train_logreg.py
├── train_dl.py
├── plot_curves.py
├── config/
│   └── config.yaml
├── src/
│   ├── data_loader.py
│   ├── features_match.py
│   ├── preprocessing.py
│   ├── models_ml.py
│   ├── models_dl.py
│   └── utils.py
├── results/
│   ├── training_curves.png
│   └── training_curves.csv
└── artifacts/
    ├── ml/
    ├── logreg/
    └── dl/
```

`artifacts/` contains generated model outputs and may be excluded from public distribution depending on file size, licensing, and reproducibility requirements.

---

## 9. Getting Started

### Environment Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Train LightGBM

```bash
python train_ml.py \
  --train_path train.csv \
  --config config/config.yaml \
  --model_dir artifacts/ml
```

The training workflow saves:

```text
artifacts/ml/
├── preprocessor.pkl
├── lgbm_model.pkl
└── threshold.txt
```

### Train Logistic Regression

```bash
python train_logreg.py \
  --train_path train.csv \
  --config config/config.yaml \
  --model_dir artifacts/logreg
```

### Deep Learning Training

```bash
python train_dl.py \
  --train_path train.csv \
  --config config/config.yaml \
  --model_dir artifacts/dl
```

The deep learning implementation requires its configured pretrained transformer model and the corresponding Python dependencies.

### Run LightGBM Inference

```bash
./inference.sh --test_path test.csv
```

The existing inference shell script:

1. Activates the local `venv`.
2. Loads the configured machine learning preprocessing and model artifacts.
3. Performs prediction using the conventional ML pipeline.
4. Writes predictions to `predictions.csv`.

The original training and testing datasets are not distributed in this public repository. Running these commands requires appropriately authorized data and available model artifacts.

---

## 10. Limitations & Responsible Use

This project is a course-based predictive modeling exercise and has not been validated for production recruitment.

Important limitations include:

- Reported results were obtained on the original academic dataset.
- External generalization across industries and organizations has not been established.
- Dataset composition and label quality may affect predictive performance.
- The project includes potentially sensitive attributes, including age and gender-related fields.
- A model that predicts historical matching outcomes may reproduce existing hiring patterns or discrimination.
- No comprehensive fairness audit, subgroup analysis, or real-world deployment validation has been completed.

Therefore, the model should not be used to make autonomous hiring decisions or to exclude applicants based on protected or sensitive attributes.

Any future real-world application would require appropriate privacy protections, fairness evaluation, human oversight, and employment-law review.

---

## 11. Future Improvements

Potential extensions include:

1. **Threshold-independent evaluation:** Add ROC-AUC and Precision–Recall AUC.
2. **Cross-validation:** Evaluate generalization across multiple train/validation splits.
3. **Explainability:** Investigate feature importance and SHAP-based explanations.
4. **Feature ablation:** Measure the isolated contribution of textual, structured, and matching features.
5. **Early stopping:** Reduce unnecessary boosting rounds and examine generalization.
6. **Fairness evaluation:** Audit sensitive attributes and subgroup performance before any applied use.
7. **Deep learning comparison:** Evaluate the RoBERTa-based model on the same validation protocol.
8. **Deployment:** Build a reproducible inference API with automated tests and monitoring.

---

## 12. Key Learnings

This project strengthened my understanding of several practical machine learning concepts:

**Feature engineering matters.** Domain-specific compatibility features can provide useful information beyond generic text representations.

**Imbalanced labels require careful evaluation.** Class weighting and threshold selection can materially affect precision, recall, and F1.

**Model complexity involves trade-offs.** LightGBM improved reported classification performance relative to the linear baseline but required additional training time.

**Validation performance must guide model selection.** Continued training improvements do not necessarily translate into corresponding generalization gains.

**Business context matters.** Predictive outputs are not equivalent to fair or appropriate real-world hiring decisions.

---

## 13. Project Information

**Author:** Patrick Tsai (TSAI CHENGYU)

**Course:** Statistical Learning and Deep Learning

**Project:** Mini-Project 2 — Job–Talent Matching Classification

**Project Type:** Academic Machine Learning Project

**Focus Areas:** NLP, Classification, Feature Engineering, Model Evaluation, Business-oriented AI

*Quantitative results are based on the original project report. The current repository documents the implementation and experiment methodology; it does not represent a production-deployed recruitment system.*
