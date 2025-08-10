from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional

def normalize_features(df: pd.DataFrame, features: List[str]) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    data = df[features].to_numpy(dtype=float)
    mean = data.mean(axis=0)
    std = data.std(axis=0) + 1e-8
    norm = (data - mean) / std
    out = df.copy()
    out[features] = norm
    return out, mean, std

def infer_input_dim(df: pd.DataFrame, features: List[str]) -> int:
    return len(features)
