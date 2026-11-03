import argparse
import yaml
import os
import numpy as np
import time
from sklearn.metrics import precision_recall_fscore_support

# Mitigate macOS OpenMP duplicate library aborts (common when using Anaconda + lightgbm)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ["OMP_NUM_THREADS"] = "1"
from src.data_loader import load_train_df, train_valid_split
from src.features_match import add_matching_features
from src.preprocessing import TabularPreprocessor
from src.models_ml import MLJobTalentModel
from src.utils import set_seed, evaluate_f1

def train_ml_pipeline(train_path: str, config_path: str, model_dir: str):
    os.makedirs(model_dir, exist_ok=True)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    set_seed(config.get("seed", 42))

    df = load_train_df(train_path)
    df = add_matching_features(df)

    df_train, df_valid = train_valid_split(df, valid_ratio=config.get("valid_ratio", 0.2))

    target_col = config["target_col"]
    y_train = df_train[target_col].values
    y_valid = df_valid[target_col].values

    categorical_cols = config.get("categorical_cols", [])
    numeric_cols = config.get("numeric_cols", [])
    datetime_cols = config.get("datetime_cols", [])
    text_cols = config.get("text_cols_ml", [])
    tfidf_max_features = config.get("tfidf_max_features", 8000)
    tfidf_ngram_range = tuple(config.get("tfidf_ngram_range", [1, 2]))

    preprocessor = TabularPreprocessor(
        categorical_cols=categorical_cols,
        numeric_cols=numeric_cols,
        datetime_cols=datetime_cols,
        text_cols=text_cols,
        tfidf_max_features=tfidf_max_features,
        tfidf_ngram_range=tfidf_ngram_range,
    )
    preprocessor.fit(df_train)
    X_train = preprocessor.transform(df_train)
    X_valid = preprocessor.transform(df_valid)

    ml_params = config.get("ml_model_params", {})
    model = MLJobTalentModel(**ml_params)
    t0 = time.time()
    model.fit(X_train, y_train, X_valid, y_valid)
    train_time = time.time() - t0

    # 掃 threshold 找最佳 F1
    valid_probs = model.predict_proba(X_valid)
    thresholds = np.arange(0.25, 0.56, 0.05)
    best_thr, best_f1 = 0.5, -1
    best_p = best_r = 0.0
    for thr in thresholds:
        f1 = evaluate_f1(y_valid, (valid_probs >= thr).astype(int))
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
            p, r, _, _ = precision_recall_fscore_support(
                y_valid, (valid_probs >= thr).astype(int), average="binary", zero_division=0
            )
            best_p, best_r = p, r
    print(
        f"[ML] Val Precision={best_p:.4f} Recall={best_r:.4f} "
        f"F1={best_f1:.4f} @ thr={best_thr:.2f} | train_time={train_time:.2f}s"
    )

    preprocessor.save(os.path.join(model_dir, "preprocessor.pkl"))
    model.save(os.path.join(model_dir, "lgbm_model.pkl"))
    with open(os.path.join(model_dir, "threshold.txt"), "w", encoding="utf-8") as f:
        f.write(str(best_thr))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--model_dir", type=str, default="artifacts/ml")
    args = parser.parse_args()

    train_ml_pipeline(args.train_path, args.config, args.model_dir)
