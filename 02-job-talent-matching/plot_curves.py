import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, log_loss
import yaml

from src.data_loader import load_train_df, train_valid_split
from src.features_match import add_matching_features
from src.preprocessing import TabularPreprocessor
from src.models_ml import MLJobTalentModel
from src.utils import set_seed


def staged_metrics(model, X_train, y_train, X_valid, y_valid, num_points: int = 10):
    """Compute F1/logloss on train/valid at staged num_iterations."""
    best_iter = getattr(model.model, "best_iteration_", None)
    max_iter = best_iter if best_iter and best_iter > 0 else model.model.n_estimators
    iters = np.linspace(max_iter // num_points, max_iter, num_points, dtype=int)
    iters = np.unique(np.clip(iters, 1, max_iter))

    records = []
    for n_iter in iters:
        train_prob = model.model.predict_proba(X_train, num_iteration=n_iter)[:, 1]
        valid_prob = model.model.predict_proba(X_valid, num_iteration=n_iter)[:, 1]

        train_logloss = log_loss(y_train, train_prob, labels=[0, 1])
        valid_logloss = log_loss(y_valid, valid_prob, labels=[0, 1])

        # 使用固定 threshold=0.5 觀察趨勢
        train_f1 = f1_score(y_train, (train_prob >= 0.5).astype(int))
        valid_f1 = f1_score(y_valid, (valid_prob >= 0.5).astype(int))

        records.append(
            {
                "iter": n_iter,
                "train_logloss": train_logloss,
                "valid_logloss": valid_logloss,
                "train_f1": train_f1,
                "valid_f1": valid_f1,
            }
        )
    return pd.DataFrame(records)


def main(args):
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    set_seed(config.get("seed", 42))

    df = load_train_df(args.train_path)
    df = add_matching_features(df)
    df_train, df_valid = train_valid_split(df, valid_ratio=config.get("valid_ratio", 0.2), seed=config.get("seed", 42))

    target_col = config["target_col"]
    y_train = df_train[target_col].values
    y_valid = df_valid[target_col].values

    # Load preprocessor/model
    preprocessor = TabularPreprocessor.load(os.path.join(args.model_dir, "preprocessor.pkl"))
    model = MLJobTalentModel.load(os.path.join(args.model_dir, "lgbm_model.pkl"))

    X_train = preprocessor.transform(df_train)
    X_valid = preprocessor.transform(df_valid)

    df_metrics = staged_metrics(model, X_train, y_train, X_valid, y_valid, num_points=10)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(df_metrics["iter"], df_metrics["train_logloss"], label="train")
    axes[0].plot(df_metrics["iter"], df_metrics["valid_logloss"], label="valid")
    axes[0].set_xlabel("Boosting rounds")
    axes[0].set_ylabel("Binary Logloss")
    axes[0].set_title("Logloss vs rounds")
    axes[0].legend()

    axes[1].plot(df_metrics["iter"], df_metrics["train_f1"], label="train")
    axes[1].plot(df_metrics["iter"], df_metrics["valid_f1"], label="valid")
    axes[1].set_xlabel("Boosting rounds")
    axes[1].set_ylabel("F1-score (thr=0.5)")
    axes[1].set_title("F1 vs rounds")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(args.output)
    print(f"Saved curves to {args.output}")
    df_metrics.to_csv(args.output_csv, index=False)
    print(f"Saved metrics to {args.output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="train.csv")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--model_dir", type=str, default="artifacts/ml")
    parser.add_argument("--output", type=str, default="training_curves.png")
    parser.add_argument("--output_csv", type=str, default="training_curves.csv")
    args = parser.parse_args()
    main(args)
