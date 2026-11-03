# Product Price Prediction Using Machine Learning

**Statistical Learning & Deep Learning — Mini-Project 1**

**Python | NLP | Feature Engineering | Regression | Cross-Validation**

## 1. Project Overview

This project develops a machine learning pipeline to predict product prices from product names.

Product names contain valuable but unstructured information, including brands, product categories, specifications, package sizes, and quantities. However, variations in text formatting, units, and naming conventions make it challenging to extract useful predictive features.

The objective was to build an accurate and computationally efficient regression model while considering training time, memory consumption, and inference constraints.

### Project Highlights

- **Best Reported 5-Fold CV Mean SMAPE:** 39.04%
- **Selected Model:** HashingVectorizer + Ridge Regression
- **Approximate Training Time:** 9 minutes
- **Feature Engineering:** Character n-grams + handcrafted numerical features
- **Model Selection:** Accuracy–Efficiency Trade-off

## 2. Problem Definition

### Business Problem

E-commerce platforms contain large numbers of products whose descriptions and naming conventions vary significantly.

Predicting prices directly from product names is a challenging NLP regression task because:

- Product names contain inconsistent formatting and abbreviations.
- Units and quantities can be expressed in different ways.
- Product prices can have a long-tailed distribution.
- Sparse textual features may increase training and memory costs.
- Inference must remain efficient under limited computational resources.

### Objective

Develop an end-to-end machine learning pipeline that:

1. Normalizes and processes product names.
2. Extracts textual and numerical features.
3. Predicts prices using regression models.
4. Evaluates performance using 5-fold cross-validation and SMAPE.
5. Supports model serialization and standalone inference.

## 3. Methodology

### 3.1 Data Preprocessing

Product names were normalized to reduce inconsistent text representations.

The preprocessing workflow included:

**Unicode Normalization**

Applied Unicode NFKC normalization to standardize full-width and half-width characters.

**Unit Standardization**

Converted equivalent unit expressions into consistent representations.

Examples:

| Original | Normalized |
|---|---|
| 公克、克 | g |
| 公斤、千克 | kg |
| 公升 | l |
| 毫升 | ml |
| 包 | pack |
| 入 | pcs |

**Text Cleaning**

- Lowercase conversion
- Whitespace normalization
- Symbol cleanup
- Reduction of redundant formatting

These preprocessing steps were designed to reduce feature fragmentation caused by inconsistent product naming conventions.

### 3.2 Text Feature Extraction

**Character-level n-grams**

Used `HashingVectorizer` with:

```python
analyzer="char"
ngram_range=(1, 5)
```

Character-level n-grams capture patterns within product names, including partial words, numbers, units, and product specifications.

Feature hashing provides a fixed-dimensional feature representation without storing a vocabulary dictionary.

**Handcrafted Numerical Features**

Additional features were extracted from product names, including:

- String length
- Number of digits
- Number of Latin characters
- Presence of numerical or alphabetic content

Selected numerical features were clipped to reduce the effect of extreme values.

| Feature | Clipping Upper Bound |
|---|---:|
| Digits | 15 |
| Text Length | 80 |
| Latin Characters | 40 |

Numerical features were subsequently scaled using `StandardScaler(with_mean=False)`.

### 3.3 Model Architecture

The selected pipeline combines text-based and numerical features with a regularized linear regression model.

```text
Product Names
      |
      v
Text Normalization
      |
      v
+---------------------------+
| Feature Extraction        |
|                           |
| Character N-gram Hashing  |
| Handcrafted Numeric Feats |
+---------------------------+
      |
      v
Feature Combination
      |
      v
Ridge Regression
      |
      v
Inverse Target Transformation
      |
      v
Predicted Prices
```

**Regression Model**

Ridge Regression was selected for its computational efficiency and ability to handle high-dimensional feature representations.

**Target Transformation**

The target variable was transformed using:

```python
TransformedTargetRegressor(
    regressor=...,
    func=np.log1p,
    inverse_func=np.expm1
)
```

The log transformation was introduced to address the long-tailed price distribution and reduce the influence of extreme target values.

The resulting pipeline was serialized using `joblib` for reuse during inference.

## 4. Model Experimentation

Several model configurations were explored to evaluate predictive performance and computational efficiency.

### 4.1 Compared Approaches

| Model | Configuration | Reported Training Time |
|---|---|---|
| Hashing + Ridge | 2²⁰ features, α=0.5 | ~9 min |
| TF-IDF + Ridge | max_features=300k, α=1.0 | ~30 min |
| TF-IDF + ElasticNet | max_features=400k, α=0.05, l1_ratio=0.3 | Over 60 min |
| TF-IDF + LinearSVR | max_features=300k, C=1, ε=0.1 | Over 60 min |

The original experiment report indicates that TF-IDF-based approaches could achieve lower average SMAPE, but generally required more time and computational resources.

Exact validation SMAPE scores for the alternative models were not retained in the report and are therefore not presented as numerical comparisons.

### 4.2 Hyperparameter Tuning

The selected Hashing + Ridge configuration was evaluated using 5-fold cross-validation.

The number of hashed features was fixed at 2²⁰ while adjusting Ridge regularization strength.

| Ridge α | Number of Features | Mean SMAPE |
|---|---:|---:|
| **0.5** | **2²⁰** | **39.04%** |
| 1.0 | 2²⁰ | 40.73% |
| 2.0 | 2²⁰ | 43.17% |

**Best reported configuration:**

```yaml
model: hashing_ridge
n_features: 1048576
alpha: 0.5
```

The configuration above summarizes the reported experimental parameters; the repository's actual YAML file remains the authoritative source for executable settings.

## 5. Results & Discussion

### 5.1 Final Model Performance

The selected Hashing + Ridge model achieved a reported **5-fold CV Mean SMAPE of 39.04%**, with an approximate training time of 9 minutes.

The final model was not selected solely based on the lowest validation error.

Instead, model selection considered:

- Prediction error
- Training time
- Memory efficiency
- Model complexity
- Practical inference constraints

### 5.2 Accuracy vs. Efficiency

TF-IDF approaches were explored as alternatives to feature hashing.

Although the original experiments suggested potentially better predictive performance from TF-IDF models, these approaches required substantially more computation.

Hashing + Ridge offered a practical compromise:

- Fixed-dimensional sparse feature representation
- No stored vocabulary dictionary
- Relatively fast linear model training
- Reusable serialized inference pipeline

This trade-off was especially relevant because the course evaluation environment imposed a CPU inference time constraint.

### 5.3 Error Analysis

Early model iterations produced unrealistic predictions, including:

- Excessive near-zero predicted prices
- Extremely large predicted values

To address these issues, the pipeline incorporated:

1. Log transformation of target prices.
2. Clipping of selected numerical input features.
3. Text normalization to improve feature consistency.

The original project report observed that these adjustments reduced extreme predictions and produced a more reasonable prediction distribution.

However, detailed error-distribution metrics were not preserved, so the improvement is reported qualitatively rather than as an unverified numerical gain.

## 6. Technical Implementation

### Technology Stack

| Component | Technology |
|---|---|
| Programming | Python |
| Data Processing | pandas, NumPy |
| ML Framework | scikit-learn |
| Text Features | HashingVectorizer, TfidfVectorizer |
| Regression | Ridge, ElasticNet, LinearSVR |
| Evaluation | 5-Fold KFold CV, SMAPE |
| Model Persistence | joblib |
| Configuration | YAML |
| Inference | Python, Shell Script |

### Project Structure

```text
01-product-price-prediction/
├── README.md
├── EXPERIMENTS.md
├── inference.sh
├── requirements.txt
├── configs/
│   ├── base.yaml
│   └── search_space.yaml
├── src/
│   ├── preprocess.py
│   ├── pipeline_utils.py
│   ├── train.py
│   ├── inference.py
│   └── utils.py
└── models/
    └── model.joblib
```

The model file may be excluded from the public repository depending on its size and distribution restrictions.

## 7. Reproducibility

### Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Training

With an authorized training dataset available locally:

```bash
python src/train.py \
  --train_path train.csv \
  --config configs/base.yaml \
  --out_model_path models/model.joblib \
  --cv 5
```

### Inference

Once a compatible trained model is available:

```bash
./inference.sh \
  --test_path test.csv \
  --output_path predictions.csv
```

The inference pipeline loads the serialized model and generates a CSV file containing the predicted `price` column.

### Dataset Availability

The original course datasets are not included in this public repository.

Reproducing the original validation metrics requires access to the original training data and corresponding experiment configuration.

The published code is intended to document the implementation and enable reproduction when appropriately authorized data is available.

## 8. Limitations & Future Improvements

**Limitations**

- Prediction relies primarily on product names rather than complete product attributes.
- Model performance may vary across product categories.
- Exact validation results for some alternative models were not retained.
- Evaluation results originate from the course experiments rather than an external production deployment.
- The original dataset is not distributed publicly.

**Potential Improvements**

- Category-aware price modeling
- More systematic hyperparameter optimization
- Detailed prediction-error analysis by price range
- Evaluation using additional holdout datasets
- API-based inference deployment
- Automated model validation and regression testing

## 9. Key Learnings

This project provided practical experience in:

- Designing end-to-end machine learning pipelines
- Applying text normalization and feature engineering
- Combining sparse textual and numerical features
- Evaluating regression models using cross-validation
- Comparing model quality against computational cost
- Addressing unrealistic predictions through target transformation and feature processing
- Preparing trained models for independent inference

The central lesson was that model selection requires balancing predictive performance with operational constraints, rather than optimizing a single metric in isolation.

## 10. Project Information

**Author:** Patrick Tsai (TSAI CHENGYU)

**Course:** Statistical Learning and Deep Learning

**Project:** Mini-Project 1 — Product Price Prediction

**Type:** Academic Machine Learning Project

**Focus:** NLP, Feature Engineering, Regression, Model Evaluation, Computational Efficiency

*The experimental results are based on the original course project report. The repository represents an academic implementation and should not be interpreted as a production-validated pricing system.*
