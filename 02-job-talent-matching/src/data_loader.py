import pandas as pd
from sklearn.model_selection import train_test_split

def load_train_df(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def train_valid_split(df, valid_ratio=0.2, seed=42):
    train_df, valid_df = train_test_split(
        df,
        test_size=valid_ratio,
        random_state=seed,
        stratify=df["label"],
    )
    return train_df, valid_df
