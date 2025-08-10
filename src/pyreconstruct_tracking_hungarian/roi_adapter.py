from __future__ import annotations
from pathlib import Path
from typing import List
import json
import pandas as pd

RESERVED = ["FrameID", "Label", "Centroid_X", "Centroid_Y", "Area",
            "ROI_X1","ROI_Y1","ROI_X2","ROI_Y2"]

class ROIAdapter:
    """
    Simple ROI adapter.

    Accepts:
      - CSV with columns:
         FrameID, Label, Centroid_X, Centroid_Y, [Area], [ROI_X1, ROI_Y1, ROI_X2, ROI_Y2], and any feature columns.
      - JSON (list of dicts) with same keys per row.
        Example row:
          {
            "FrameID": 0, "Label": 12,
            "Centroid_X": 105.2, "Centroid_Y": 78.0,
            "Area": 123.0,
            "ROI_X1": 90, "ROI_Y1": 60, "ROI_X2": 120, "ROI_Y2": 95,
            "F_texture": 0.31, "F_roundness": 0.82
          }

    Output: pandas DataFrame sorted by (FrameID, Label).
    """
    def __init__(self, path: str):
        self.path = Path(path)

    def dataframe(self) -> pd.DataFrame:
        if self.path.suffix.lower() == ".csv":
            df = pd.read_csv(self.path)
        elif self.path.suffix.lower() == ".json":
            with open(self.path, "r") as f:
                data = json.load(f)
            if isinstance(data, dict) and "rows" in data:
                data = data["rows"]
            df = pd.DataFrame(data)
        else:
            raise ValueError("ROIAdapter expects .csv or .json")

        # sanity checks
        required = ["FrameID", "Label", "Centroid_X", "Centroid_Y"]
        for c in required:
            if c not in df.columns:
                raise ValueError(f"Missing required column: {c}")

        df = df.sort_values(["FrameID", "Label"]).reset_index(drop=True)
        return df

    @staticmethod
    def feature_columns(df: pd.DataFrame, explicit: List[str] | None = None) -> List[str]:
        if explicit:
            for c in explicit:
                if c not in df.columns:
                    raise ValueError(f"Configured feature column not found: {c}")
            return explicit
        return [c for c in df.columns if c not in RESERVED]
