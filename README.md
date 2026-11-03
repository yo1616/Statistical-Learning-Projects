# Statistical Learning & Machine Learning Projects

[**English**](./README.md) | [繁體中文](./README.zh-TW.md)

**Applied Machine Learning · Natural Language Processing · Predictive Analytics · Business Problem Solving**

A collection of two academic machine learning projects developed as part of the **Statistical Learning and Deep Learning** course.

These projects explore how structured and unstructured data can be transformed into predictive insights through feature engineering, statistical learning, model evaluation, and systematic experimentation.

Beyond model development, the projects emphasize practical considerations such as **predictive performance, computational efficiency, class imbalance, and model selection trade-offs**.

---

## About This Repository

This repository contains two machine learning projects addressing different real-world business problems:

| Project | Business Problem | ML Task | Selected Model | Key Result |
|---|---|---|---|---|
| **01. Product Price Prediction** | Estimate product prices from unstructured product names | Regression | Hashing + Ridge | 5-Fold CV SMAPE: **39.04%** |
| **02. Job–Talent Matching** | Predict matching outcomes between job requirements and candidate profiles | Classification | LightGBM | Validation F1: **0.5696** |

Both projects were developed using Python and scikit-learn-based machine learning workflows, with additional specialized tools for text processing and classification.

The reported results originate from the original academic project reports and have not been independently rerun for this public repository.

---

## Project 01 — Product Price Prediction

**Regression · NLP · Text Feature Engineering · Model Efficiency**

### Business Problem

Product names often contain useful pricing information, including brands, specifications, quantities, and product categories.

However, inconsistent formatting, abbreviations, and unstructured text make it difficult to extract meaningful information for price estimation.

This project investigates whether machine learning can predict product prices using textual and numerical features extracted from product names.

### Technical Approach

The project implements a complete regression pipeline consisting of:

- **Text preprocessing:** Unicode normalization, lowercase conversion, unit standardization, and symbol cleanup.
- **Text feature extraction:** Character-level n-grams using `HashingVectorizer`.
- **Numerical feature engineering:** Text length, digit counts, Latin character counts, and selected feature clipping.
- **Regression modeling:** Ridge Regression with regularization.
- **Target transformation:** Logarithmic transformation using `log1p` and `expm1`.
- **Evaluation:** 5-Fold Cross-Validation using Symmetric Mean Absolute Percentage Error (SMAPE).

### Key Results

| Metric | Result |
|---|---|
| Selected Model | HashingVectorizer + Ridge Regression |
| Best Reported Mean CV SMAPE | **39.04%** |
| Ridge Regularization (α) | 0.5 |
| Hashing Feature Dimensions | 2²⁰ |
| Approximate Training Time | 9 minutes |

Additional experiments explored TF-IDF-based Ridge, ElasticNet, and LinearSVR approaches.

Although the original experiments indicated that some TF-IDF configurations could achieve lower prediction errors, these approaches required substantially greater training time and computational resources.

**Key Decision:** The final model was selected based on a balance between predictive performance, computational efficiency, and practical inference constraints rather than accuracy alone.

### Skills Demonstrated

`Regression` · `Text Normalization` · `Feature Hashing` · `Feature Engineering` · `Cross-Validation` · `Model Selection` · `Computational Efficiency`

**[Explore Project 01 — Product Price Prediction](./01-product-price-prediction/)**

---

## Project 02 — Job–Talent Matching Classification

**Classification · NLP · Feature Engineering · Imbalanced Learning**

### Business Problem

Recruitment matching involves evaluating whether candidate qualifications, skills, experience, and preferences align with job requirements.

This process combines structured information, such as job categories and location preferences, with unstructured text, such as job descriptions and professional experience.

This project explores supervised machine learning approaches for predicting job–candidate matching outcomes.

### Technical Approach

The project implements a classification pipeline featuring:

- **Text representation:** TF-IDF with unigram and bigram features.
- **Categorical processing:** One-hot encoding and missing-value handling.
- **Domain-specific feature engineering:** Skill overlap, duty compatibility, work location, working time, and other matching-related features.
- **Classification models:** LightGBM and Logistic Regression.
- **Class imbalance handling:** Class weighting and `scale_pos_weight`.
- **Decision threshold tuning:** Selection of classification thresholds based on validation F1.
- **Model evaluation:** Precision, Recall, F1 Score, and training behavior analysis.

An additional PyTorch-based deep learning training implementation using a Chinese RoBERTa backbone is included for exploration. The main quantitative comparison focuses on LightGBM and Logistic Regression.

### Key Results

| Metric | LightGBM | Logistic Regression |
|---|---:|---:|
| Precision | **0.5870** | 0.4090 |
| Recall | **0.5531** | 0.5240 |
| F1 Score | **0.5696** | 0.4592 |
| Reported Training Time | 706 s | 415 s |

**Key Achievement:** LightGBM achieved an approximately **24.0% relative improvement in validation F1** compared with the Logistic Regression baseline.

The original development report also documents improvements associated with domain-specific feature engineering and class imbalance handling.

### Model Evaluation Insights

The LightGBM training experiments showed:

- Continued improvement in training F1 across boosting rounds.
- Validation F1 approaching a plateau near 0.57.
- A growing difference between training and validation F1 at later iterations.

These observations highlight the importance of evaluating generalization rather than relying solely on training performance.

**Key Decision:** LightGBM was selected because it achieved stronger reported validation performance and better balanced precision and recall, despite requiring additional training time.

### Skills Demonstrated

`Classification` · `TF-IDF` · `Feature Engineering` · `LightGBM` · `Logistic Regression` · `Class Imbalance` · `Threshold Optimization` · `Model Evaluation`

**[Explore Project 02 — Job–Talent Matching](./02-job-talent-matching/)**

---

## Technical Skills & Tools

| Category | Technologies / Methods |
|---|---|
| Programming | Python |
| Data Processing | pandas, NumPy |
| Machine Learning | scikit-learn, LightGBM |
| Deep Learning Exploration | PyTorch, Hugging Face Transformers |
| Natural Language Processing | TF-IDF, Character n-grams, Text Normalization |
| Feature Engineering | Feature Hashing, One-Hot Encoding, Numerical Features, Compatibility Features |
| Regression | Ridge Regression, ElasticNet, LinearSVR |
| Classification | Logistic Regression, Gradient Boosting |
| Evaluation | K-Fold Cross-Validation, SMAPE, Precision, Recall, F1 Score |
| Model Development | Hyperparameter Tuning, Threshold Optimization, Model Serialization |
| Workflow | Python Scripts, YAML Configuration, Shell-based Inference |

---

## Repository Structure

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

Each project contains its own implementation, model configuration, and documentation.

Some directories containing generated model artifacts may be omitted from the public repository due to file-size or distribution considerations.

---

## Key Learnings & Takeaways

### 1. Feature Engineering Can Be as Important as Model Selection

Both projects demonstrate how transforming raw data into meaningful features affects machine learning performance.

Product price prediction required extracting useful information from unstructured product names, while job–talent matching benefited from domain-specific compatibility features.

### 2. Model Performance Must Be Evaluated in Context

A more complex model is not automatically the best choice.

In Product Price Prediction, computational efficiency and training time played an important role in model selection.

In Job–Talent Matching, the improvement in classification performance justified the additional training cost of LightGBM.

### 3. Appropriate Evaluation Metrics Matter

Different machine learning tasks require different performance measures.

- **Regression:** SMAPE was used to evaluate relative price prediction errors.
- **Classification:** Precision, Recall, and F1 were used to evaluate matching predictions, particularly under class imbalance.

### 4. Practical Machine Learning Requires Reproducible Workflows

The projects include reusable preprocessing, model training, configuration management, model serialization, and inference scripts.

These components help separate experimentation from prediction and support reproducibility when authorized datasets and compatible environments are available.

### 5. Predictive Performance Is Not the Same as Business Readiness

Although the models were developed around business-oriented problems, both projects remain academic experiments.

Before real-world deployment, additional validation would be necessary, including external testing, data governance, operational monitoring, and—particularly for recruitment—fairness and legal review.

---

## Reproducibility & Data Availability

The original course datasets are not included in this public repository.

The code, configuration files, and documented experimental methodology are provided for educational and portfolio purposes.

To reproduce the experiments, users need access to appropriately authorized datasets and the dependencies described in each project's README.

Please refer to the individual project folders for detailed setup, training, and inference instructions.

The results presented here are based on the original course project reports. They should not be interpreted as independently reproduced benchmarks or production deployment results.

---

## About the Author

**Patrick Tsai (TSAI CHENGYU)**

Undergraduate student interested in the intersection of **Artificial Intelligence, Data Analytics, Information Systems, and Business Problem Solving**.

My primary interest lies in applying technology and data-driven methods to understand business problems, design practical solutions, and support informed decision-making.

**Areas of Interest**

- Applied Artificial Intelligence
- Business Analytics
- AI Product Development
- Technology & Digital Transformation
- Data-Driven Decision Making

---

## Project Context & Disclaimer

These projects were developed as part of university coursework in Statistical Learning and Deep Learning.

Reported model performance is based on the original academic experiment reports.

The Job–Talent Matching project is intended solely as a machine learning experiment. It has not been validated for production hiring use, and its predictions should not be used for autonomous recruitment decisions.

**Author:** Patrick Tsai  
**Project Type:** Academic Machine Learning Portfolio  
**Language:** Python  
**Focus:** Applied Machine Learning × NLP × Business Analytics
