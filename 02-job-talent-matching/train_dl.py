import argparse
import json
import os
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import yaml
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src.data_loader import load_train_df, train_valid_split
from src.features_match import add_matching_features
from src.preprocessing import TabularPreprocessor
from src.models_dl import DLJobTalentModel
from src.utils import set_seed, evaluate_f1


class JobTalentDataset(Dataset):
    """
    將文字欄位串接後 tokenization，再搭配 tabular 特徵。
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: AutoTokenizer,
        text_cols: List[str],
        tabular_feats,
        labels=None,
        max_length: int = 128,
    ):
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

        if labels is not None:
            self.labels = torch.tensor(labels, dtype=torch.float32)
        else:
            self.labels = None

    def __len__(self):
        return len(self.tabular_feats)

    def __getitem__(self, idx):
        item = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "tabular_feats": self.tabular_feats[idx],
        }
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    device,
    pos_weight=None,
):
    model.train()
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for batch in tqdm(dataloader, desc="Train", leave=False):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        tabular_feats = batch["tabular_feats"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask, tabular_feats=tabular_feats)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * len(labels)

    return total_loss / len(dataloader.dataset)


@torch.no_grad()
def evaluate(model, dataloader, device, threshold=0.5):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    preds = []
    labels_all = []

    for batch in tqdm(dataloader, desc="Valid", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        tabular_feats = batch["tabular_feats"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask, tabular_feats=tabular_feats)
        loss = criterion(logits, labels)
        total_loss += loss.item() * len(labels)

        probs = torch.sigmoid(logits)
        preds.append((probs >= threshold).long().cpu().numpy())
        labels_all.append(labels.cpu().numpy())

    preds = np.concatenate(preds)
    labels_all = np.concatenate(labels_all)
    f1 = evaluate_f1(labels_all, preds)
    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss, f1


def train_dl_pipeline(train_path: str, config_path: str, model_dir: str):
    os.makedirs(model_dir, exist_ok=True)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    set_seed(config.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = load_train_df(train_path)
    df = add_matching_features(df)

    df_train, df_valid = train_valid_split(df, valid_ratio=config.get("valid_ratio", 0.2))
    target_col = config["target_col"]
    y_train = df_train[target_col].values
    y_valid = df_valid[target_col].values

    categorical_cols = config.get("categorical_cols", [])
    numeric_cols = config.get("numeric_cols", [])
    datetime_cols = config.get("datetime_cols", [])
    text_cols = config.get("text_cols", ["LogTitle"])

    # Tabular preprocessor
    preprocessor = TabularPreprocessor(
        categorical_cols=categorical_cols,
        numeric_cols=numeric_cols,
        datetime_cols=datetime_cols,
    )
    preprocessor.fit(df_train)
    X_train = preprocessor.transform(df_train)
    X_valid = preprocessor.transform(df_valid)
    tabular_dim = X_train.shape[1]

    # Tokenizer
    pretrained_name = config.get("dl_model_name", "hfl/chinese-roberta-wwm-ext")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_name)

    max_length = config.get("max_length", 128)
    batch_size = config.get("batch_size", 16)
    epochs = config.get("epochs", 3)
    lr = config.get("lr", 2e-5)
    weight_decay = config.get("weight_decay", 0.01)
    threshold = config.get("inference_threshold", 0.5)
    hidden_size = config.get("tab_hidden_size", 256)

    train_dataset = JobTalentDataset(
        df=df_train,
        tokenizer=tokenizer,
        text_cols=text_cols,
        tabular_feats=X_train,
        labels=y_train,
        max_length=max_length,
    )
    valid_dataset = JobTalentDataset(
        df=df_valid,
        tokenizer=tokenizer,
        text_cols=text_cols,
        tabular_feats=X_valid,
        labels=y_valid,
        max_length=max_length,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    model = DLJobTalentModel(
        pretrained_name=pretrained_name,
        tabular_input_dim=tabular_dim,
        hidden_size=hidden_size,
    ).to(device)

    total_steps = len(train_loader) * epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    pos_weight = None
    if config.get("pos_weight") is not None:
        pos_weight = torch.tensor([config["pos_weight"]], device=device)

    best_f1 = -1.0
    history = []
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            pos_weight=pos_weight,
        )
        val_loss, val_f1 = evaluate(
            model=model,
            dataloader=valid_loader,
            device=device,
            threshold=threshold,
        )
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_f1": val_f1}
        )
        print(f"[Epoch {epoch}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_f1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(model_dir, "model.pt"))

    # Save assets
    tokenizer.save_pretrained(os.path.join(model_dir, "tokenizer"))
    preprocessor.save(os.path.join(model_dir, "preprocessor.pkl"))
    meta = {
        "pretrained_name": pretrained_name,
        "tabular_input_dim": tabular_dim,
        "hidden_size": hidden_size,
        "max_length": max_length,
        "threshold": threshold,
    }
    with open(os.path.join(model_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    with open(os.path.join(model_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print(f"Best validation F1: {best_f1:.4f}")
    print(f"Artifacts saved to {model_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--model_dir", type=str, default="artifacts/dl")
    args = parser.parse_args()

    train_dl_pipeline(args.train_path, args.config, args.model_dir)
