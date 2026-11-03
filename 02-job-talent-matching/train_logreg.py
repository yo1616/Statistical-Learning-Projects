import argparse
import os
import joblib
import numpy as np
import yaml
import time
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support

from src.data_loader import load_train_df, train_valid_split
from src.features_match import add_matching_features
from src.preprocessing import TabularPreprocessor
from src.utils import set_seed


def find_best_threshold(y_true, y_prob, start: float = 0.1, end: float = 0.9, step: float = 0.02):
    thresholds = np.arange(start, end + 1e-8, step)
    best = {"thr": 0.5, "f1": -1, "p": 0.0, "r": 0.0}
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        if f1 > best["f1"]:
            best = {"thr": thr, "f1": f1, "p": p, "r": r}
    return best


def train_logreg_pipeline(train_path: str, config_path: str, model_dir: str):
    os.makedirs(model_dir, exist_ok=True)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    seed = config.get("seed", 42)
    set_seed(seed)

    # Load & feature engineering
    df = load_train_df(train_path)
    df = add_matching_features(df)

    df_train, df_valid = train_valid_split(
        df, valid_ratio=config.get("valid_ratio", 0.2), seed=seed
    )

    target_col = config["target_col"]
    y_train = df_train[target_col].values
    y_valid = df_valid[target_col].values

    categorical_cols = config.get("categorical_cols", [])
    numeric_cols = config.get("numeric_cols", [])
    datetime_cols = config.get("datetime_cols", [])
    text_cols = config.get("text_cols_ml", [])  # 可視需要讓 logreg 也吃 TF-IDF
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

    # Logistic Regression grid（可由 config 覆寫）
    candidate_params = config.get(
        "logreg_grid",
        [
            {"penalty": "l2", "C": 0.5, "class_weight": "balanced", "max_iter": 3000, "solver": "liblinear", "n_jobs": -1},
            {"penalty": "l2", "C": 1.0, "class_weight": "balanced", "max_iter": 3000, "solver": "liblinear", "n_jobs": -1},
            {"penalty": "l2", "C": 2.0, "class_weight": "balanced", "max_iter": 3000, "solver": "liblinear", "n_jobs": -1},
            {"penalty": "l1", "C": 1.0, "class_weight": "balanced", "max_iter": 3000, "solver": "liblinear", "n_jobs": -1},
        ],
    )

    best_global = {"f1": -1, "thr": 0.5, "p": 0.0, "r": 0.0, "params": None, "model": None, "time": 0.0}
    for idx, params in enumerate(candidate_params, 1):
        print(f"Start training Logistic Regression candidate {idx}/{len(candidate_params)}: {params}")
        clf = LogisticRegression(**params)
        t0 = time.time()
        clf.fit(X_train, y_train)
        train_time = time.time() - t0
        valid_prob = clf.predict_proba(X_valid)[:, 1]
        best = find_best_threshold(y_valid, valid_prob, start=0.05, end=0.8, step=0.01)
        print(
            f"[LogReg cand {idx}] F1={best['f1']:.4f} @ thr={best['thr']:.2f} "
            f"(P={best['p']:.4f}, R={best['r']:.4f}) time={train_time:.2f}s"
        )
        if best["f1"] > best_global["f1"]:
            best_global.update(
                {
                    "f1": best["f1"],
                    "thr": best["thr"],
                    "p": best["p"],
                    "r": best["r"],
                    "params": params,
                    "model": clf,
                    "time": train_time,
                }
            )

    print(
        f"[LogReg BEST] P={best_global['p']:.4f} R={best_global['r']:.4f} "
        f"F1={best_global['f1']:.4f} @ thr={best_global['thr']:.2f} "
        f"time={best_global['time']:.2f}s params={best_global['params']}"
    )

    preprocessor.save(os.path.join(model_dir, "preprocessor_logreg.pkl"))
    joblib.dump(best_global["model"], os.path.join(model_dir, "logreg_model.pkl"))
    with open(os.path.join(model_dir, "logreg_best_threshold.txt"), "w", encoding="utf-8") as f:
        f.write(str(best_global["thr"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--model_dir", type=str, default="artifacts/logreg")
    args = parser.parse_args()

    train_logreg_pipeline(args.train_path, args.config, args.model_dir)
