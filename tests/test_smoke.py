import numpy as np
import pandas as pd
from pyreconstruct_tracking.dataset_adapter import CSVAdapter, BaseAdapter

def test_constant_column_removal(tmp_path):
    p = tmp_path / "toy.csv"
    df = pd.DataFrame({
        "FrameID": [0,0,1,1],
        "Label": [1,2,1,2],
        "Centroid_X": [10,20,11,21],
        "Centroid_Y": [5,6,5.5,7.0],
        "Area": [100,100,100,100]  # constant, should be removed
    })
    df.to_csv(p, index=False)
    ad = CSVAdapter(str(p))
    out = ad.dataframe()
    assert "Area" not in out.columns
    assert set(["FrameID","Label","Centroid_X","Centroid_Y"]).issubset(out.columns)
