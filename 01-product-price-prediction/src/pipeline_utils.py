from __future__ import annotations

import numpy as np
import pandas as pd


def wrap_to_df(values):
    if isinstance(values, pd.DataFrame):
        if 'name' in values.columns:
            out_df = values[['name']].copy()
            out_df['name'] = out_df['name'].astype(str)
            return out_df
        first_col = values.columns[0]
        out_df = values[[first_col]].copy()
        out_df.columns = ['name']
        out_df['name'] = out_df['name'].astype(str)
        return out_df

    if isinstance(values, pd.Series):
        return pd.DataFrame({'name': values.astype(str)})

    array = np.asarray(values)
    if array.ndim == 0:
        return pd.DataFrame({'name': [str(array)]})
    if array.ndim == 1:
        return pd.DataFrame({'name': array.astype(str)})
    first_col = array[:, 0]
    return pd.DataFrame({'name': first_col.astype(str)})


def simple_numeric_feats(values):
    series = pd.Series(values).astype(str)
    digits = series.str.count(r"\d").astype(float)
    length = series.str.len().astype(float)
    latin = series.str.count(r"[a-zA-Z]").astype(float)
    return pd.DataFrame({'digits': digits, 'length': length, 'latin': latin})


def clip_numeric_feats(frame: pd.DataFrame) -> pd.DataFrame:
    upper = {'digits': 15.0, 'length': 80.0, 'latin': 40.0}
    clipped = frame.copy()
    for col, cap in upper.items():
        if col in clipped.columns:
            clipped[col] = np.clip(clipped[col].to_numpy(dtype=float), 0.0, cap)
    return clipped


def numeric_to_array(frame) -> np.ndarray:
    if isinstance(frame, pd.DataFrame):
        arr = frame.to_numpy(dtype=float)
    else:
        arr = np.asarray(frame, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    return arr
