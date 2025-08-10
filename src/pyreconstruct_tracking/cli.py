from __future__ import annotations
import argparse
from pathlib import Path
import json
import numpy as np
import torch
import pandas as pd

from .config import load_config, ensure_outdir
from .dataset_adapter import CSVAdapter, JSERAdapter, BaseAdapter
from .features import normalize_features, infer_input_dim
from .model.tracker import BayesianTransformerForCellTracking, train_bnn
from .matching import higher_order_graph_matching_with_divisions
from .lineage import (
    update_tracking_with_divisions, export_lineage_to_mantrack,
    generate_output_csv_with_divisions, attach_results_to_jser
)
from .visualize import visualize_matching_matplotlib, visualize_matching_plotly

def _select_device(pref: str) -> str:
    if pref == "cpu":
        return "cpu"
    if pref == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    # auto
    return "cuda" if torch.cuda.is_available() else "cpu"

def _adapter_from_args(args) -> BaseAdapter:
    if args.csv:
        return CSVAdapter(args.csv)
    if args.jser:
        return JSERAdapter(args.jser)
    raise ValueError("Provide --csv or --jser")

def cmd_train(args):
    cfg = load_config(args.config)
    outdir = ensure_outdir(cfg.paths.output_dir)
    adapter = _adapter_from_args(args)
    df = adapter.dataframe()
    df = df.sort_values(["FrameID", "Label"]).reset_index(drop=True)
    feats = [c for c in df.columns if c not in ["FrameID", "Label"]]
    df_norm, mean, std = normalize_features(df, feats)

    input_dim = cfg.model.input_dim or infer_input_dim(df_norm, feats)
    model = BayesianTransformerForCellTracking(
        input_dim=input_dim, embed_dim=cfg.model.embed_dim, num_heads=cfg.model.num_heads,
        ff_hidden_dim=cfg.model.ff_hidden_dim, num_layers=cfg.model.num_layers,
        output_dim=cfg.model.output_dim, prior_mu=cfg.model.prior_mu, prior_sigma=cfg.model.prior_sigma,
        dropout=cfg.model.dropout, use_layernorm=cfg.model.use_layernorm
    )

    frame_pairs = [(i, i+1) for i in range(args.frames[0], args.frames[1])]
    device = _select_device(cfg.train.device)
    model = train_bnn(
        bnn=model, frame_pairs=frame_pairs, df=df_norm, features_to_use=feats,
        num_epochs=cfg.train.num_epochs, lr=cfg.train.lr, margin=cfg.train.margin,
        weight_decay=cfg.train.weight_decay, batch_size=cfg.train.batch_size,
        early_stopping_patience=cfg.train.early_stopping_patience,
        reduce_lr_patience=cfg.train.reduce_lr_patience, device=device,
        kl_beta=cfg.model.kl_beta
    )

    # Save model + normalization stats
    out_model = Path(args.model_out)
    out_model.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "mean": mean, "std": std, "features": feats,
                "model_cfg": cfg.model.__dict__}, out_model)
    print(f"[INFO] Saved model: {out_model}")

def _load_model(model_path: Path):
    ckpt = torch.load(model_path, map_location="cpu")
    mc = ckpt.get("model_cfg", {})
    model = BayesianTransformerForCellTracking(
        input_dim=mc.get("input_dim", len(ckpt["features"])),
        embed_dim=mc.get("embed_dim", 64),
        num_heads=mc.get("num_heads", 2),
        ff_hidden_dim=mc.get("ff_hidden_dim", 256),
        num_layers=mc.get("num_layers", 2),
        output_dim=mc.get("output_dim", 2),
        prior_mu=mc.get("prior_mu", 0.0),
        prior_sigma=mc.get("prior_sigma", 0.1),
        dropout=mc.get("dropout", 0.1),
        use_layernorm=mc.get("use_layernorm", True)
    )
    model.load_state_dict(ckpt["state_dict"])
    mean, std, features = ckpt["mean"], ckpt["std"], ckpt["features"]
    return model, np.asarray(mean), np.asarray(std), features

def _embed_section(model, section_df, features, mean, std, device):
    X = section_df[features].to_numpy(dtype=np.float32)
    X = (X - mean) / (std + 1e-8)
    x_tensor = torch.from_numpy(X).float().unsqueeze(1).to(device)
    with torch.no_grad():
        mu, logv, _ = model(x_tensor, sample=False)
    return mu.cpu(), logv.cpu()

def cmd_predict(args):
    cfg = load_config(args.config)
    outdir = ensure_outdir(args.out or cfg.paths.output_dir)
    adapter = _adapter_from_args(args)
    df = adapter.dataframe().sort_values(["FrameID","Label"]).reset_index(drop=True)

    model, mean, std, features = _load_model(Path(args.model))
    device = _select_device(cfg.train.device)
    model.to(device)
    model.eval()

    num_frames = int(df["FrameID"].max()) + 1
    tracking = {}
    lineage_info = {}
    links_json = []
    divisions_json = []

    for frame_idx in range(args.frames[0], args.frames[1]):
        sec1 = df[df["FrameID"] == frame_idx]
        sec2 = df[df["FrameID"] == frame_idx + 1]
        if sec1.empty or sec2.empty:
            continue

        sec04_names = [f"{cid}_{frame_idx}" for cid in sec1["Label"].tolist()]
        sec05_names = [f"{cid}_{frame_idx+1}" for cid in sec2["Label"].tolist()]

        mu1, logv1 = _embed_section(model, sec1, features, mean, std, device)
        mu2, logv2 = _embed_section(model, sec2, features, mean, std, device)

        # Use raw features matrix for graph distances (same columns you trained on)
        s1 = sec1[features].to_numpy(dtype=np.float32)
        s2 = sec2[features].to_numpy(dtype=np.float32)

        matches, C, d1, d2, thresh = higher_order_graph_matching_with_divisions(
            s1, s2, mu1.squeeze(1), logv1.squeeze(1), mu2.squeeze(1), logv2.squeeze(1),
            max_single_neighbors=cfg.match.max_single_neighbors,
            max_triplet_neighbors=cfg.match.max_triplet_neighbors,
            triplet_dist_thresh=cfg.match.triplet_dist_thresh,
            belief_max_iter=cfg.match.belief_max_iter,
            belief_damping=cfg.match.belief_damping,
            base_percentile=cfg.match.base_percentile,
            motion_threshold=cfg.match.motion_threshold,
            area_variation=cfg.match.area_variation
        )

        # Build JSON link records (object ids, not local indices)
        for m in matches:
            parent_node = m[0]
            parent_idx = int(parent_node.split('_')[1])
            parent_id = int(sec04_names[parent_idx].split('_')[0])
            children = m[1] if isinstance(m[1], list) else [m[1]]
            child_ids = []
            for ch in children:
                ci = int(ch.split('_')[1])
                child_ids.append(int(sec05_names[ci].split('_')[0]))
            if len(child_ids) == 1:
                links_json.append({"t": frame_idx, "id_t": parent_id, "t1": frame_idx+1, "id_t1": child_ids[0],
                                   "score": float(1.0 - np.min(C))})
            else:
                divisions_json.append({"parent": parent_id, "children": child_ids, "t": frame_idx+1,
                                       "score": float(1.0 - np.min(C))})

        # Update in-memory lineage for export
        tracking = update_tracking_with_divisions(
            tracking, matches, frame_idx, num_frames, sec04_names, sec05_names, lineage_info, sec1, sec2
        )

        # Visualization
        if cfg.viz.static_every and (cfg.viz.static_every > 0) and ((frame_idx - args.frames[0]) % cfg.viz.static_every == 0):
            visualize_matching_matplotlib(
                s1, s2, matches, sec04_names, sec05_names, sec1, sec2, frame_idx, outdir / "figs"
            )
        if cfg.viz.interactive_every and (cfg.viz.interactive_every > 0) and ((frame_idx - args.frames[0]) % cfg.viz.interactive_every == 0):
            visualize_matching_plotly(
                s1, s2, matches, sec04_names, sec05_names, sec1, sec2, frame_idx, outdir / "html"
            )

        print(f"[Frame {frame_idx}->{frame_idx+1}] matched={len(matches)}  dummy_t={len(d1)}  dummy_t1={len(d2)}")

    export_lineage_to_mantrack(lineage_info, outdir / "man_track.txt")
    generate_output_csv_with_divisions(tracking, num_frames, outdir / "tracking_results.csv")

    if args.attach_jser:
        attach_results_to_jser(Path(args.jser), Path(args.attach_jser), links_json, divisions_json, version=1)

def cmd_visualize(args):
    # In case you only want to re-render visualization from CSV features
    cfg = load_config(args.config)
    outdir = ensure_outdir(args.out or cfg.paths.output_dir)
    adapter = CSVAdapter(args.csv)
    df = adapter.dataframe().sort_values(["FrameID","Label"]).reset_index(drop=True)
    print(df.head())

def main():
    p = argparse.ArgumentParser("pyreconstruct-tracking")
    sub = p.add_subparsers(dest="cmd", required=True)

    # train
    t = sub.add_parser("train", help="Train model from CSV or JSER")
    t.add_argument("--csv", type=str, default=None, help="Feature CSV")
    t.add_argument("--jser", type=str, default=None, help="PyReconstruct series .jser")
    t.add_argument("--frames", type=int, nargs=2, required=True, help="start end (end exclusive)")
    t.add_argument("--config", type=str, required=True)
    t.add_argument("--model-out", type=str, required=True)
    t.set_defaults(func=cmd_train)

    # predict
    pr = sub.add_parser("predict", help="Run matching and lineage export")
    pr.add_argument("--csv", type=str, default=None)
    pr.add_argument("--jser", type=str, default=None)
    pr.add_argument("--frames", type=int, nargs=2, required=True)
    pr.add_argument("--config", type=str, required=True)
    pr.add_argument("--model", type=str, required=True)
    pr.add_argument("--out", type=str, default=None, help="Output directory")
    pr.add_argument("--attach-jser", type=str, default=None, help="Write results into this .jser")
    pr.set_defaults(func=cmd_predict)

    # visualize (optional helper)
    vz = sub.add_parser("visualize", help="Utility to preview a CSV")
    vz.add_argument("--csv", type=str, required=True)
    vz.add_argument("--config", type=str, required=True)
    vz.add_argument("--out", type=str, default=None)
    vz.set_defaults(func=cmd_visualize)

    args = p.parse_args()
    args.func(args)
