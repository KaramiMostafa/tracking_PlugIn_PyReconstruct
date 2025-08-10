from __future__ import annotations
from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from .roi_adapter import ROIAdapter
from .costs import combine_costs

def build_cost_matrix(
    sec1: pd.DataFrame,
    sec2: pd.DataFrame,
    feature_cols: List[str],
    *,
    weights: Dict[str, float],
    scaling_percentile: int = 95,
    gating_max_spatial: float | None = None,
    gating_min_iou: float | None = None,
    big_m: float = 1e6
) -> np.ndarray:
    """
    Build the pairwise cost matrix C (n1 x n2), combining components with configured weights.
    Any invalid pairs get BIG_M.
    """
    parts = combine_costs(
        sec1, sec2, feature_cols,
        weights=weights, scaling_percentile=scaling_percentile,
        gating_max_spatial=gating_max_spatial, gating_min_iou=gating_min_iou
    )
    C = parts["combined"]
    # Replace NaNs/infs
    C = np.where(np.isfinite(C), C, big_m).astype(np.float32)
    return C

def _augment_for_birth_death(C: np.ndarray, birth_cost: float, death_cost: float, big_m: float = 1e6):
    """
    Augment C (n1 x n2) to a square matrix (n1+n2 x n1+n2) to allow births/deaths:
      [  C         D_death ]
      [  D_birth      0    ]
    D_death: (n1 x n1) diag = death_cost; off-diagonals = BIG_M
    D_birth: (n2 x n2) diag = birth_cost; off-diagonals = BIG_M
    """
    n1, n2 = C.shape
    top_left = C
    top_right = np.full((n1, n1), big_m, dtype=np.float32)
    np.fill_diagonal(top_right, float(death_cost))
    bottom_left = np.full((n2, n2), big_m, dtype=np.float32)
    np.fill_diagonal(bottom_left, float(birth_cost))
    bottom_right = np.zeros((n2, n1), dtype=np.float32)  # not used by interpretation

    # Block concat:
    upper = np.concatenate([top_left, top_right], axis=1)         # (n1, n2+n1)
    lower = np.concatenate([bottom_left, bottom_right], axis=1)   # (n2, n2+n1)
    C_aug = np.concatenate([upper, lower], axis=0)                # (n1+n2, n2+n1)
    return C_aug

def augmented_hungarian(C: np.ndarray, birth_cost: float, death_cost: float, big_m: float = 1e6):
    C_aug = _augment_for_birth_death(C, birth_cost, death_cost, big_m=big_m)
    rows, cols = linear_sum_assignment(C_aug)
    return rows, cols, C_aug

def interpret_assignments(
    rows: np.ndarray, cols: np.ndarray, C: np.ndarray,
    C_aug: np.ndarray, assign_threshold: float
):
    """
    Interpret the solution from the augmented matrix:
      - Real-real (i<n1, j<n2) → match if C[i,j] <= assign_threshold else unmatched
      - (i<n1, j>=n2) → death of i
      - (i>=n1, j<n2) → birth of j
    Returns:
      matches: List[Tuple[str, str]] where items look like ("1_i","2_j")
      deaths:  List[int] indices in t that ended
      births:  List[int] indices in t+1 that started
    """
    n1, n2 = C.shape
    matches, deaths, births = [], [], []

    for r, c in zip(rows, cols):
        if r < n1 and c < n2:
            if C[r, c] <= assign_threshold:
                matches.append((f"1_{r}", f"2_{c}"))
            else:
                deaths.append(r)
                births.append(c)
        elif r < n1 and c >= n2:
            deaths.append(r)
        elif r >= n1 and c < n2:
            births.append(c)
        # else: (r>=n1 and c>=n2) => dummy-dummy, ignore

    return matches, deaths, births
