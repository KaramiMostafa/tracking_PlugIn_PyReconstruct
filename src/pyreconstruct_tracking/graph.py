from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple

def construct_higher_order_graph(
    section_1: np.ndarray,  # (n1, d')
    section_2: np.ndarray,  # (n2, d')
    max_single_neighbors=8,
    max_triplet_neighbors=10,
    triplet_dist_thresh=50.0
) -> Dict[str, List]:
    """
    Third-order (triplet) graph:
      - edges_single: nearest neighbors by L2 in feature space
      - edges_triplet: connect triangles (i1,i2,i3) -> (j1,j2,j3) whose geometry is similar
    """
    n1, n2 = section_1.shape[0], section_2.shape[0]
    G = {'nodes_t': [f"t_{i}" for i in range(n1)],
         'nodes_t+1': [f"t+1_{j}" for j in range(n2)],
         'edges_single': [], 'edges_triplet': []}

    # First-order edges
    for i in range(n1):
        dist_vec = np.linalg.norm(section_2 - section_1[i], axis=1)
        nn_sorted = np.argsort(dist_vec)[:max_single_neighbors]
        for j in nn_sorted:
            G['edges_single'].append({'i': f"t_{i}", 'j': f"t+1_{j}", 'cost': float(dist_vec[j])})

    # Triplets in t
    triplets_t = []
    for i1 in range(n1):
        for i2 in range(i1+1, n1):
            for i3 in range(i2+1, n1):
                coords = [section_1[i1], section_1[i2], section_1[i3]]
                if _all_within_thresh(coords, triplet_dist_thresh):
                    triplets_t.append((i1, i2, i3, _triplet_geometry(coords)))
    # Triplets in t+1
    triplets_tp1 = []
    for j1 in range(n2):
        for j2 in range(j1+1, n2):
            for j3 in range(j2+1, n2):
                coords = [section_2[j1], section_2[j2], section_2[j3]]
                if _all_within_thresh(coords, triplet_dist_thresh):
                    triplets_tp1.append((j1, j2, j3, _triplet_geometry(coords)))

    # Cross-connect by geometry cost
    for (i1, i2, i3, tri_t) in triplets_t:
        cands = []
        for (j1, j2, j3, tri_tp1) in triplets_tp1:
            cands.append((j1, j2, j3, _triplet_cost(tri_t, tri_tp1)))
        cands.sort(key=lambda x: x[3])
        for (j1, j2, j3, c) in cands[:max_triplet_neighbors]:
            G['edges_triplet'].append({
                'i_i2_i3': (f"t_{i1}", f"t_{i2}", f"t_{i3}"),
                'j1_j2_j3': (f"t+1_{j1}", f"t+1_{j2}", f"t+1_{j3}"),
                'cost': float(c)
            })
    return G

def _all_within_thresh(coords, dist_thresh):
    for a in range(3):
        for b in range(a+1, 3):
            if np.linalg.norm(coords[a] - coords[b]) > dist_thresh:
                return False
    return True

def _triplet_geometry(coords):
    s1 = np.linalg.norm(coords[0] - coords[1])
    s2 = np.linalg.norm(coords[1] - coords[2])
    s3 = np.linalg.norm(coords[0] - coords[2])
    return [s1, s2, s3]

def _triplet_cost(triA, triB):
    triA = sorted(triA); triB = sorted(triB)
    return float(np.sum(np.abs(np.array(triA) - np.array(triB))))
