import numpy as np
import random
import os
from sklearn.metrics import f1_score

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

def evaluate_f1(y_true, y_pred):
    return f1_score(y_true, y_pred)
