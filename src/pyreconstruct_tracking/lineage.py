from __future__ import annotations
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import json
from pathlib import Path

def update_tracking_with_divisions(
    tracking: Dict[int, list],
    matches,
    frame_idx: int,
    num_frames: int,
    sec04_names: List[str],
    sec05_names: List[str],
    lineage_info: Dict[int, Dict[str, int]],
    sec04_cells,  # DataFrame (frame t)
    sec05_cells   # DataFrame (frame t+1)
):
    name_to_index_sec04 = {name: idx for idx, name in enumerate(sec04_names)}
    name_to_index_sec05 = {name: idx for idx, name in enumerate(sec05_names)}
    matched_t = set()
    matched_t1 = set()

    for match in matches:
        parent_node = match[0]
        child_nodes = match[1] if isinstance(match[1], list) else [match[1]]
        parent_idx = int(parent_node.split('_')[1])
        parent_cell_id = int(sec04_names[parent_idx].split('_')[0])
        parent_centroid = np.array([
            sec04_cells.iloc[parent_idx]["Centroid_X"],
            sec04_cells.iloc[parent_idx]["Centroid_Y"]
        ], dtype=np.float32)
        child_ids, child_centroids = [], []
        for ch in child_nodes:
            cidx = int(ch.split('_')[1])
            cid = int(sec05_names[cidx].split('_')[0])
            child_ids.append(cid)
            child_centroids.append(np.array([
                sec05_cells.iloc[cidx]["Centroid_X"],
                sec05_cells.iloc[cidx]["Centroid_Y"]
            ], dtype=np.float32))
        matched_t.add(parent_cell_id)
        matched_t1.update(child_ids)

        if len(child_nodes) == 1:
            # continuation
            child_id = child_ids[0]; child_cent = child_centroids[0]
            track_id = None
            for tid, arr in tracking.items():
                if isinstance(arr[frame_idx], dict) and arr[frame_idx]['cell_id'] == parent_cell_id:
                    track_id = tid; break
            if track_id is None:
                track_id = len(tracking)
                tracking[track_id] = [None] * num_frames
                lineage_info[track_id] = {'start': frame_idx, 'end': frame_idx, 'parent': 0}
            tracking[track_id][frame_idx] = {'cell_id': parent_cell_id, 'centroid': parent_centroid}
            tracking[track_id][frame_idx + 1] = {'cell_id': child_id, 'centroid': child_cent}
            lineage_info[track_id]['end'] = frame_idx + 1
        else:
            # division
            track_id = None
            for tid, arr in tracking.items():
                if isinstance(arr[frame_idx], dict) and arr[frame_idx]['cell_id'] == parent_cell_id:
                    track_id = tid; break
            if track_id is None:
                track_id = len(tracking)
                tracking[track_id] = [None] * num_frames
                tracking[track_id][frame_idx] = {'cell_id': parent_cell_id, 'centroid': parent_centroid}
                lineage_info[track_id] = {'start': frame_idx, 'end': frame_idx, 'parent': 0}
            for f in range(frame_idx + 1, num_frames):
                tracking[track_id][f] = None
            lineage_info[track_id]['end'] = frame_idx
            for (cid, ccent) in zip(child_ids, child_centroids):
                new_tid = len(tracking)
                tracking[new_tid] = [None] * num_frames
                tracking[new_tid][frame_idx + 1] = {'cell_id': cid, 'centroid': ccent}
                lineage_info[new_tid] = {'start': frame_idx + 1, 'end': frame_idx + 1, 'parent': track_id}

    # Unmatched in t+1 => new tracks
    all_t1 = set(int(n.split('_')[0]) for n in sec05_names)
    unmatched_t1 = all_t1 - matched_t1
    for cid in unmatched_t1:
        tid = len(tracking)
        tracking[tid] = [None] * num_frames
        tracking[tid][frame_idx + 1] = {'cell_id': cid, 'centroid': np.array([0,0], dtype=np.float32)}
        lineage_info[tid] = {'start': frame_idx + 1, 'end': frame_idx + 1, 'parent': 0}
    return tracking

def generate_output_csv_with_divisions(tracking: Dict[int, list], num_frames: int, out_csv: Path):
    data = []
    for track_id, track in tracking.items():
        row = [track_id] + track[:num_frames]
        data.append(row)
    cols = ["Tracking_ID"] + [f"Frame_{i}" for i in range(num_frames)]
    df = pd.DataFrame(data, columns=cols)
    df.to_csv(out_csv, index=False)
    print(f"[INFO] Tracking results saved: {out_csv}")

def export_lineage_to_mantrack(lineage_info: Dict[int, Dict[str,int]], output_path: Path):
    with open(output_path, 'w') as f:
        for track_id in sorted(lineage_info.keys()):
            L = track_id + 1
            B = lineage_info[track_id]['start']
            E = lineage_info[track_id]['end']
            parent_track_id = lineage_info[track_id]['parent']
            P = 0 if parent_track_id == 0 else (parent_track_id + 1)
            f.write(f"{L} {B} {E} {P}\n")
    print(f"[INFO] man_track.txt written: {output_path}")

def attach_results_to_jser(
    input_jser: Path, output_jser: Path,
    links: list, divisions: list, version: int = 1
):
    with open(input_jser, "r") as f:
        series = json.load(f)
    if "extensions" not in series or not isinstance(series["extensions"], dict):
        series["extensions"] = {}
    series["extensions"]["pyreconstruct_tracking"] = {
        "version": version,
        "links": links,
        "divisions": divisions
    }
    with open(output_jser, "w") as f:
        json.dump(series, f, indent=2)
    print(f"[INFO] Attached tracking results to: {output_jser}")
