import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
import joblib


def join_text_columns(df: pd.DataFrame) -> pd.Series:
    return df.fillna("").agg(" ".join, axis=1)


class TabularPreprocessor:
    def __init__(self, categorical_cols, numeric_cols, datetime_cols, text_cols=None, tfidf_max_features=8000, tfidf_ngram_range=(1, 2)):
        self.categorical_cols = categorical_cols or []
        self.numeric_cols = numeric_cols or []
        self.datetime_cols = datetime_cols or []
        self.text_cols = text_cols or []
        self.tfidf_max_features = tfidf_max_features
        self.tfidf_ngram_range = tfidf_ngram_range

        self.pipeline = None

    def _build_pipeline(self):
        # 處理 datetime：轉成年/月/日/時 四個數值欄位
        def process_datetime(df: pd.DataFrame) -> pd.DataFrame:
            df = df.copy()
            for col in self.datetime_cols:
                if col in df.columns:
                    dt = pd.to_datetime(df[col], errors="coerce")
                    df[f"{col}_year"] = dt.dt.year
                    df[f"{col}_month"] = dt.dt.month
                    df[f"{col}_day"] = dt.dt.day
                    df[f"{col}_hour"] = dt.dt.hour
            return df

        self._process_datetime = process_datetime

        # 更新 numeric_cols 加入 datetime 展開後的欄位
        expanded_numeric_cols = list(self.numeric_cols)
        for col in self.datetime_cols:
            expanded_numeric_cols += [
                f"{col}_year",
                f"{col}_month",
                f"{col}_day",
                f"{col}_hour",
            ]
        self.expanded_numeric_cols = expanded_numeric_cols

        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )

        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        transformers = [
            ("num", numeric_transformer, self.expanded_numeric_cols),
            ("cat", categorical_transformer, self.categorical_cols),
        ]

        if self.text_cols:
            text_transformer = Pipeline(
                steps=[
                    ("join", FunctionTransformer(join_text_columns, validate=False)),
                    (
                        "tfidf",
                        TfidfVectorizer(
                            max_features=self.tfidf_max_features,
                            ngram_range=self.tfidf_ngram_range,
                            lowercase=False,
                        ),
                    ),
                ]
            )
            transformers.append(("text", text_transformer, self.text_cols))

        preprocessor = ColumnTransformer(transformers=transformers)
        self.pipeline = preprocessor

    def fit(self, df: pd.DataFrame):
        if self.pipeline is None:
            self._build_pipeline()
        df_proc = self._process_datetime(df)
        self.pipeline.fit(df_proc)

    def transform(self, df: pd.DataFrame):
        df_proc = self._process_datetime(df)
        X = self.pipeline.transform(df_proc)
        return X

    def save(self, path: str):
        joblib.dump(
            {
                "categorical_cols": self.categorical_cols,
                "numeric_cols": self.numeric_cols,
                "datetime_cols": self.datetime_cols,
                "text_cols": self.text_cols,
                "tfidf_max_features": self.tfidf_max_features,
                "tfidf_ngram_range": self.tfidf_ngram_range,
                "pipeline": self.pipeline,
            },
            path,
        )

    @classmethod
    def load(cls, path: str):
        data = joblib.load(path)
        obj = cls(
            categorical_cols=data.get("categorical_cols"),
            numeric_cols=data.get("numeric_cols"),
            datetime_cols=data.get("datetime_cols"),
            text_cols=data.get("text_cols"),
            tfidf_max_features=data.get("tfidf_max_features", 8000),
            tfidf_ngram_range=data.get("tfidf_ngram_range", (1, 2)),
        )
        # 建立 datetime 處理與欄位展開資訊，但沿用已訓練的 pipeline
        obj._build_pipeline()
        obj.pipeline = data["pipeline"]
        return obj
