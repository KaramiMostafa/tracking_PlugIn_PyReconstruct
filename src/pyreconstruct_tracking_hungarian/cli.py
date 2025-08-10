from __future__ import annotations
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd

from pyreconstruct_tracking.config import load_config, ensure_outdir   # reuse shared config helpers
from pyreconstruct_tracking.visualize import visualize_matching_matplotlib, visualize_matching_plotly
from pyreconstruct_tracking.lineage import (
    update_tracking_with_divisions, export_lineage_to_mantrack, generate_output_csv_with_divisions, attach_results_to_jser
)

from .roi_adapter import ROIAdapter
from .matcher import build_cost_matrix, augmented_hungarian, interpret_assignments

def _load_hcfg(cfg_dict):
    # pull the "hungarian" section from a full config (and supply defaults)
    h = cfg_dict.get("hungarian", {})
    features = h.get("features", None)
    weights = h.get("weights", {"feature": 1.0, "spatial": 0.2, "area": 0.1, "iou": 0.5})
    scaling = h.get("scaling", {"percentile": 95})
    gating = h.get("gating", {"max_spatial": 200.0, "min_iou": 0.0})
    costs  = h.get("costs",  {"birth": 0.6, "death": 0.6, "max_cost": 1e6, "assign_threshold": 0.95})
    return features, weights, scaling, gating, costs

def cmd_predict(args):
    cfg = load_config(args.config)
    outdir = ensure_outdir(args.out or cfg.paths.output_dir)

    # Load ROI table
    adapter = ROIAdapter(args.roi)
    df = adapter.dataframe().sort_values(["FrameID","Label"]).reset_index(drop=True)

    feat_cfg, weights, scaling, gating, costs = _load_hcfg(json.loads(json.dumps(cfg.__dict__)))  # convert dataclass to dict-ish
    feature_cols = ROIAdapter.feature_columns(df, feat_cfg)

    num_frames = int(df["FrameID"].max()) + 1
    tracking, lineage_info = {}, {}
    links_json, births_json = [], []

    for t in range(args.frames[0], args.frames[1]):
        sec1 = df[df["FrameID"] == t]
        sec2 = df[df["FrameID"] == t + 1]
        if sec1.empty or sec2.empty:
            continue

        sec04_names = [f"{cid}_{t}" for cid in sec1["Label"].tolist()]
        sec05_names = [f"{cid}_{t+1}" for cid in sec2["Label"].tolist()]

        # Build pairwise costs
        C = build_cost_matrix(
            sec1, sec2, feature_cols,
            weights=weights,
            scaling_percentile=int(scaling.get("percentile", 95)),
            gating_max_spatial=gating.get("max_spatial", None),
            gating_min_iou=gating.get("min_iou", None),
            big_m=float(costs.get("max_cost", 1e6))
        )

        # Augment for births/deaths and solve
        rows, cols, C_aug = augmented_hungarian(
            C,
            birth_cost=float(costs.get("birth", 0.6)),
            death_cost=float(costs.get("death", 0.6)),
            big_m=float(costs.get("max_cost", 1e6))
        )
        matches, deaths, births = interpret_assignments(
            rows, cols, C, C_aug, assign_threshold=float(costs.get("assign_threshold", 0.95))
        )

        # record JSON links for optional .jser attach (one-to-one only here)
        for (n1, n2) in matches:
            i = int(n1.split('_')[1]); j = int(n2.split('_')[1])
            id_t  = int(sec04_names[i].split('_')[0])
            id_t1 = int(sec05_names[j].split('_')[0])
            links_json.append({"t": t, "id_t": id_t, "t1": t+1, "id_t1": id_t1})

        # Update tracks (uses the same lineage helper; divisions not used here)
        tracking = update_tracking_with_divisions(
            tracking, matches, t, num_frames, sec04_names, sec05_names, lineage_info, sec1, sec2
        )

        # Viz (reusing shared helpers)
        if args.static_every and (t - args.frames[0]) % args.static_every == 0:
            visualize_matching_matplotlib(
                sec1[feature_cols].to_numpy(dtype=np.float32),
                sec2[feature_cols].to_numpy(dtype=np.float32),
                matches, sec04_names, sec05_names, sec1, sec2, t, outdir / "figs"
            )
        if args.interactive_every and (t - args.frames[0]) % args.interactive_every == 0:
            visualize_matching_plotly(
                sec1[feature_cols].to_numpy(dtype=np.float32),
                sec2[feature_cols].to_numpy(dtype=np.float32),
                matches, sec04_names, sec05_names, sec1, sec2, t, outdir / "html"
            )

        print(f"[Hungarian {t}->{t+1}] matches={len(matches)} deaths={len(deaths)} births={len(births)}")

    # Exports
    export_lineage_to_mantrack(lineage_info, outdir / "man_track.txt")
    generate_output_csv_with_divisions(tracking, num_frames, outdir / "tracking_results.csv")

    # Optional: attach to .jser
    if args.attach_jser:
        if not args.jser:
            raise ValueError("--attach-jser requires --jser (the source series).")
        attach_results_to_jser(Path(args.jser), Path(args.attach_jser), links_json, divisions=[], version=1)

def main():
    ap = argparse.ArgumentParser("pyreconstruct-hungarian")
    ap.add_argument("--roi", type=str, required=True, help="ROI table (.csv or .json)")
    ap.add_argument("--frames", type=int, nargs=2, required=True, help="start end (end exclusive)")
    ap.add_argument("--config", type=str, required=True, help="hungarian_config.yaml")
    ap.add_argument("--out", type=str, default=None, help="Output dir (overrides config.paths.output_dir)")
    ap.add_argument("--static-every", type=int, default=1, help="save matplotlib every N frames (0=off)")
    ap.add_argument("--interactive-every", type=int, default=0, help="save plotly every N frames (0=off)")
    ap.add_argument("--jser", type=str, default=None, help="Source .jser (only needed if attaching results)")
    ap.add_argument("--attach-jser", type=str, default=None, help="Write an updated .jser with links")
    args = ap.parse_args()
    cmd_predict(args)
