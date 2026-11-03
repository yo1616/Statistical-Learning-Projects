from __future__ import annotations

import re
from typing import Iterable, List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

_BRACKETS = [
    (r"\((.*?)\)", re.DOTALL),
    (r"（(.*?)）", re.DOTALL),
    (r"\[(.*?)\]", re.DOTALL),
]


def strip_brackets(text: str) -> str:
    cleaned = str(text)
    for pattern, flag in _BRACKETS:
        cleaned = re.sub(pattern, "", cleaned, flags=flag)
    return cleaned


def extract_brackets(text: str) -> str:
    source = str(text)
    contents: List[str] = []

    for pattern, flag in _BRACKETS:
        contents.extend(re.findall(pattern, source, flags=flag))
    return " ".join(contents)


class ParenthesesCleaner(BaseEstimator, TransformerMixin):
    def fit(self, inputs: Iterable[str], y=None):
        del inputs, y
        return self

    def transform(self, inputs: Iterable[str]) -> List[str]:
        return [strip_brackets(text) for text in inputs]


class ParenthesesContent(BaseEstimator, TransformerMixin):
    def fit(self, inputs: Iterable[str], y=None):
        del inputs, y
        return self

    def transform(self, inputs: Iterable[str]) -> List[str]:
        return [extract_brackets(text) for text in inputs]


class NumericUnitFeaturizer(BaseEstimator, TransformerMixin):
    def fit(self, inputs: Iterable[str], y=None):
        del inputs, y
        return self

    def transform(self, inputs: Iterable[str]) -> pd.DataFrame:
        series = pd.Series(inputs, dtype="string").str.lower()
        num_pattern = r"(?:\d+(?:\.\d+)?)"

        quantity = series.str.extract(
            rf"({num_pattern})\s*(kg|g|tb|t|gb|ml|l|cc|cm|mm|w)",
            expand=True,
        )
        qty_values = pd.to_numeric(quantity[0], errors="coerce").to_numpy()
        qty_units = quantity[1].fillna("").to_numpy()
        qty_values = np.nan_to_num(qty_values, nan=0.0)

        mass_g = np.where(
            qty_units == "kg",
            qty_values * 1000.0,
            np.where(qty_units == "g", qty_values, 0.0),
        )
        volume_ml = np.where(
            qty_units == "l",
            qty_values * 1000.0,
            np.where(np.isin(qty_units, ["ml", "cc"]), qty_values, 0.0),
        )
        capacity_gb = np.where(
            np.isin(qty_units, ["tb", "t"]),
            qty_values * 1024.0,
            np.where(qty_units == "gb", qty_values, 0.0),
        )
        length_cm = np.where(
            qty_units == "mm",
            qty_values / 10.0,
            np.where(qty_units == "cm", qty_values, 0.0),
        )

        watt_numbers = pd.to_numeric(
            series.str.extract(rf"({num_pattern})\s*w", expand=False),
            errors="coerce",
        ).fillna(0.0)

        count_numbers = pd.to_numeric(
            series.str.extract(
                rf"({num_pattern})\s*(入|卷|捲|張|顆|包|盒|pcs|pack)",
                expand=False,
            ),
            errors="coerce",
        ).fillna(0.0)

        feature_frame = pd.DataFrame(
            {
                "has_num": series.str.contains(r"\d", regex=True).astype(float),
                "qty_g": mass_g,
                "qty_ml": volume_ml,
                "qty_gb": capacity_gb,
                "size_cm": length_cm,
                "count": count_numbers.to_numpy(),
                "watt": watt_numbers.to_numpy(),
            }
        )
        return feature_frame.astype(float)
