import numpy as np
import random
import os
from typing import Iterable, List

def smape(y_true, y_pred, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    return 100.0 * np.mean(np.abs(y_true - y_pred) / np.maximum(denom, eps))

def set_seed(seed: int = 42):
    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def to_list_str(x: Iterable) -> List[str]:
    return list(map(str, x))
