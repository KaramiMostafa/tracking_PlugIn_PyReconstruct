from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
import numpy as np
import pandas as pd

REQUIRED_COLS = ["FrameID", "Label", "Centroid_X", "Centroid_Y"]

@dataclass
class BaseAdapter:
    """
    Base adapter: produce a DataFrame with rows per object per frame.

    Required columns:
      - FrameID: int
      - Label:   int (object id within frame)
      - Centroid_X, Centroid_Y: float
    You may add: 'Area', 'Perimeter', other shape descriptors.

    Implementations should guarantee rows are sorted by (FrameID, Label).
    """
    def dataframe(self) -> pd.DataFrame:
        raise NotImplementedError

    @staticmethod
    def remove_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
        nonconst = []
        for c in df.columns:
            if c in ["FrameID", "Label"]:
                nonconst.append(c); continue
            s = df[c].dropna()
            if len(s.unique()) > 1:
                nonconst.append(c)
        return df[nonconst]

    @staticmethod
    def features_to_use(df: pd.DataFrame) -> List[str]:
        return [c for c in df.columns if c not in ["FrameID", "Label"]]

class CSVAdapter(BaseAdapter):
    def __init__(self, path: str):
        self.path = Path(path)

    def dataframe(self) -> pd.DataFrame:
        df = pd.read_csv(self.path)
        # Basic cleanup
        df = df.dropna(axis=1, how="all")
        df = self.remove_constant_columns(df)
        # Checks
        for col in REQUIRED_COLS:
            if col not in df.columns:
                raise ValueError(f"CSV missing required column: {col}")
        # Sort index
        df = df.sort_values(["FrameID", "Label"]).reset_index(drop=True)
        return df

class JSERAdapter(BaseAdapter):
    """
    Minimal .jser reader that:
      - iterates sections (frames) in order
      - extracts per-trace polygons (closed -> area, centroid; open -> centroid only)
      - applies 2D affine (a,b,c,d,e,f) if found
    Assumptions:
      series_json["sections"] -> list of sections
      section.get("traces") or section.get("Traces") -> list of traces
      trace points under keys like "points" or "polyline" as [[x,y], ...]
      trace has a "name" or "label" used for Label; otherwise use index
    Adjust field names for your actual schema if needed.
    """
    def __init__(self, jser_path: str):
        self.path = Path(jser_path)

    def dataframe(self) -> pd.DataFrame:
        with open(self.path, "r") as f:
            series = json.load(f)

        sections = series.get("sections") or series.get("Sections")
        if sections is None:
            raise ValueError("Could not find 'sections' in .jser")

        rows = []
        for t, sec in enumerate(sections):
            traces = sec.get("traces") or sec.get("Traces") or []
            # Try to get affine: [a,b,c,d,e,f]
            affine = _section_affine(sec)
            for idx, tr in enumerate(traces):
                pts = tr.get("points") or tr.get("polyline") or tr.get("Pts") or []
                pts = np.asarray(pts, dtype=float)
                if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 2:
                    continue
                if affine is not None:
                    pts = apply_affine(pts, affine)

                # Compute centroid & area (shoelace); open polylines => area=0
                is_closed = _is_closed(pts)
                cx, cy = polygon_centroid(pts)
                area = polygon_area(pts) if is_closed else 0.0
                label = tr.get("label") or tr.get("name") or idx
                rows.append({
                    "FrameID": int(t),
                    "Label": int(label),
                    "Centroid_X": float(cx),
                    "Centroid_Y": float(cy),
                    "Area": float(area),
                    "Perimeter": float(perimeter(pts))
                })

        df = pd.DataFrame(rows)
        if df.empty:
            raise ValueError("No traces found in .jser")
        df = self.remove_constant_columns(df)
        for col in REQUIRED_COLS:
            if col not in df.columns:
                raise ValueError(f".jser-derived DataFrame missing column: {col}")
        df = df.sort_values(["FrameID", "Label"]).reset_index(drop=True)
        return df

def _section_affine(sec: Dict[str, Any]) -> Optional[np.ndarray]:
    # Look for common keys; adjust to your schema as needed.
    aff = sec.get("affine") or sec.get("alignment") or sec.get("transform")
    if isinstance(aff, dict):
        # e.g. {"a":..,"b":..,"c":..,"d":..,"e":..,"f":..}
        keys = ["a","b","c","d","e","f"]
        if all(k in aff for k in keys):
            return np.array([aff["a"], aff["b"], aff["c"], aff["d"], aff["e"], aff["f"]], dtype=float)
    if isinstance(aff, list) and len(aff) == 6:
        return np.array(aff, dtype=float)
    return None

def apply_affine(pts: np.ndarray, affine: np.ndarray) -> np.ndarray:
    a,b,c,d,e,f = affine.tolist()
    xy1 = np.c_[pts, np.ones((len(pts),1))]
    M = np.array([[a,b,c],[d,e,f],[0,0,1]], dtype=float)
    tr = (xy1 @ M.T)[:, :2]
    return tr

def _is_closed(pts: np.ndarray, tol: float = 1e-6) -> bool:
    return np.linalg.norm(pts[0] - pts[-1]) < tol

def polygon_area(pts: np.ndarray) -> float:
    """Signed area via shoelace (absolute value)."""
    x, y = pts[:,0], pts[:,1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def perimeter(pts: np.ndarray) -> float:
    diffs = np.diff(pts, axis=0)
    segs = np.linalg.norm(diffs, axis=1)
    if not _is_closed(pts):
        return float(segs.sum())
    return float(segs.sum() + np.linalg.norm(pts[-1] - pts[0]))

def polygon_centroid(pts: np.ndarray) -> Tuple[float,float]:
    """Centroid for polygon (open polyline uses mean of points)."""
    if _is_closed(pts) and len(pts) >= 3:
        A = polygon_area(pts)
        if A == 0.0:
            return float(pts[:,0].mean()), float(pts[:,1].mean())
        x, y = pts[:,0], pts[:,1]
        c = x*np.roll(y,-1) - y*np.roll(x,-1)
        cx = (1/(6*A)) * np.sum((x + np.roll(x,-1)) * c)
        cy = (1/(6*A)) * np.sum((y + np.roll(y,-1)) * c)
        return float(cx), float(cy)
    # fallback: mean
    return float(pts[:,0].mean()), float(pts[:,1].mean())
