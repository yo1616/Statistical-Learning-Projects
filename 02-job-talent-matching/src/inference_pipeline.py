import json
import os
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from src.features_match import add_matching_features
from src.preprocessing import TabularPreprocessor
from src.models_ml import MLJobTalentModel
from src.models_dl import DLJobTalentModel


class DLInferenceDataset(Dataset):
    def __init__(self, df, tokenizer, text_cols, tabular_feats, max_length: int = 128):
        texts = df[text_cols].fillna("").astype(str).agg(" [SEP] ".join, axis=1).tolist()
        encodings = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.input_ids = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]
        if hasattr(tabular_feats, "toarray"):
            tabular_feats = tabular_feats.toarray()
        self.tabular_feats = torch.tensor(tabular_feats, dtype=torch.float32)

    def __len__(self):
        return len(self.tabular_feats)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "tabular_feats": self.tabular_feats[idx],
        }


@torch.no_grad()
def _run_dl_inference(df_test, config, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    preprocessor = TabularPreprocessor.load("artifacts/dl/preprocessor.pkl")
    with open("artifacts/dl/meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained("artifacts/dl/tokenizer")

    X_test = preprocessor.transform(df_test)
    test_dataset = DLInferenceDataset(
        df=df_test,
        tokenizer=tokenizer,
        text_cols=config.get("text_cols", ["LogTitle"]),
        tabular_feats=X_test,
        max_length=meta.get("max_length", 128),
    )
    test_loader = DataLoader(test_dataset, batch_size=config.get("infer_batch_size", 16), shuffle=False)

    model = DLJobTalentModel.load_from_checkpoint(
        ckpt_path="artifacts/dl/model.pt",
        pretrained_name=meta["pretrained_name"],
        tabular_input_dim=meta["tabular_input_dim"],
        hidden_size=meta["hidden_size"],
    ).to(device)
    model.eval()

    probs_list = []
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        tabular_feats = batch["tabular_feats"].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask, tabular_feats=tabular_feats)
        probs = torch.sigmoid(logits)
        probs_list.append(probs.cpu())

    probs = torch.cat(probs_list).numpy()
    return probs


def run_inference(test_path: str, output_path: str, model_type: str, config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    df_test = pd.read_csv(test_path)
    df_test = add_matching_features(df_test)

    if model_type == "ml":
        preprocessor = TabularPreprocessor.load("artifacts/ml/preprocessor.pkl")
        model = MLJobTalentModel.load("artifacts/ml/lgbm_model.pkl")
        X_test = preprocessor.transform(df_test)
        probs = model.predict_proba(X_test)
        thr_path = "artifacts/ml/threshold.txt"
        if os.path.exists(thr_path):
            with open(thr_path, "r", encoding="utf-8") as f:
                threshold = float(f.read().strip())
        else:
            threshold = config.get("inference_threshold", 0.5)
    elif model_type == "dl":
        probs = _run_dl_inference(df_test, config, output_path)
        meta_path = "artifacts/dl/meta.json"
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            threshold = meta.get("threshold", config.get("inference_threshold", 0.5))
        else:
            threshold = config.get("inference_threshold", 0.5)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    preds = (probs >= threshold).astype(int)

    # Kaggle requires ID and Label columns
    id_col = "ID" if "ID" in df_test.columns else None
    if id_col:
        submission = pd.DataFrame({id_col: df_test[id_col].values, "Label": preds})
    else:
        submission = pd.DataFrame({"Label": preds})
    submission.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path}")
