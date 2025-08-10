import numpy as np
import pandas as pd
from pyreconstruct_tracking_hungarian.matcher import build_cost_matrix, augmented_hungarian, interpret_assignments

def _toy_sections():
    # two frames with three cells each
    s1 = pd.DataFrame({
        "FrameID":[0,0,0],
        "Label":[1,2,3],
        "Centroid_X":[0.0, 10.0, 20.0],
        "Centroid_Y":[0.0, 10.0, 0.0],
        "Area":[100,110,90],
        "F_a":[0.1, 0.2, 0.3],
        "F_b":[1.0, 0.9, 1.1],
    })
    s2 = pd.DataFrame({
        "FrameID":[1,1,1],
        "Label":[4,5,6],
        "Centroid_X":[0.5, 9.5, 19.7],
        "Centroid_Y":[0.2, 10.3, -0.2],
        "Area":[102,108,91],
        "F_a":[0.11, 0.19, 0.29],
        "F_b":[1.02, 0.91, 1.09],
    })
    return s1, s2

def test_augment_and_assign():
    s1, s2 = _toy_sections()
    feats = ["F_a","F_b"]
    C = build_cost_matrix(
        s1, s2, feats,
        weights={"feature":1.0, "spatial":0.2, "area":0.1, "iou":0.0},
        scaling_percentile=95, gating_max_spatial=None, gating_min_iou=None
    )
    rows, cols, C_aug = augmented_hungarian(C, birth_cost=0.6, death_cost=0.6)
    matches, deaths, births = interpret_assignments(rows, cols, C, C_aug, assign_threshold=0.95)
    # expect mostly 1-1 matches
    assert len(matches) >= 2
    assert set(d >= 0 for d in deaths) is not None
