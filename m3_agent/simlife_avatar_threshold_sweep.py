"""Diagnose what cosine similarity an in-the-wild face cluster actually gets
against the avatar references.

For one unit's already-precomputed face JSONs, this script:
  1. Re-runs HDBSCAN over the union (same parameters as
     ``recluster_unit_faces``), with avatar anchoring TURNED OFF.
  2. For each raw cluster, computes its centroid embedding and the
     cosine similarity to each of the four avatar embeddings.
  3. Prints a similarity matrix sorted by cluster size, plus a
     per-threshold sweep showing which avatar slots get populated at
     each candidate ``avatar_match_threshold``.

Usage:
    python -m m3_agent.simlife_avatar_threshold_sweep \
        --intermediate_dir data/intermediate/video_001285
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np

from mmagent.face_processing import filter_score_based, processing_config
from mmagent.simlife_avatars import load_avatar_references
from mmagent.src.face_clustering import cluster_faces


def _load_pool(intermediate_dir):
    """Return the same pool of (cluster_idx_in_pool, face_dict) pairs that
    ``recluster_unit_faces`` would use — score-filtered, flattened across
    every clip in the unit.
    """
    paths = sorted(
        glob.glob(os.path.join(intermediate_dir, "clip_*_faces.json")),
        key=lambda p: int(os.path.basename(p).split("_")[1]),
    )
    pool = []
    for p in paths:
        for f in json.load(open(p)):
            if filter_score_based(f):
                pool.append(f)
    return pool


def _centroid(embeddings):
    arr = np.asarray(embeddings, dtype=np.float32)
    c = arr.mean(axis=0)
    n = float(np.linalg.norm(c))
    return c / n if n > 0 else c


def _similarity_matrix(pool, raw_labels, references):
    """Returns
        clusters: [(raw_label, n_faces, centroid_emb), ...] sorted by n desc
        sims: dict raw_label -> [sim_to_father, sim_to_mother, sim_to_son, sim_to_daughter]
    """
    by_raw = defaultdict(list)
    for f, lbl in zip(pool, raw_labels):
        by_raw[int(lbl)].append(f)

    ref_embs = []
    ref_names = []
    for ref in references:
        emb = ref.get("embedding")
        if emb is None:
            ref_embs.append(np.zeros(512, dtype=np.float32))
            ref_names.append(ref["name"] + " (NO AVATAR)")
        else:
            ref_embs.append(np.asarray(emb, dtype=np.float32))
            ref_names.append(ref["name"])

    clusters = []
    sims = {}
    for raw, faces in by_raw.items():
        if raw == -1:
            continue
        embs = [f["face_emb"] for f in faces]
        centroid = _centroid(embs)
        clusters.append((raw, len(faces), centroid))
        sims[raw] = [float(np.dot(centroid, e)) for e in ref_embs]
    clusters.sort(key=lambda x: -x[1])
    return clusters, sims, ref_names


def _assign(sims, threshold):
    """Mirror of ``_avatar_remap_table``: each cluster is anchored to
    its best-matching avatar if the cosine similarity is above the
    threshold. **Multiple clusters can share the same avatar slot.**
    Returns:
        slot_to_raws: {fixed_id: [raw_label, ...]}
        raw_to_slot:  {raw_label: fixed_id}
    """
    raw_to_slot = {}
    slot_to_raws = {}
    for raw, row in sims.items():
        best_idx = int(np.argmax(row))
        best_sim = float(row[best_idx])
        if best_sim >= threshold:
            raw_to_slot[raw] = best_idx
            slot_to_raws.setdefault(best_idx, []).append(raw)
    return slot_to_raws, raw_to_slot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--intermediate_dir", required=True,
                        help="data/intermediate/<unit_id>/")
    parser.add_argument("--min_cluster_size", type=int,
                        default=processing_config.get("face_min_cluster_size", 3))
    parser.add_argument("--distance_threshold", type=float, default=0.5,
                        help="HDBSCAN eps = 1 - this; matches recluster_unit_faces default")
    parser.add_argument("--top_n", type=int, default=12,
                        help="Print this many largest clusters")
    parser.add_argument("--thresholds", default="0.10,0.15,0.20,0.25,0.30,0.35,0.40",
                        help="Comma-separated list of avatar_match_thresholds to sweep")
    args = parser.parse_args()

    config_thr = processing_config.get("avatar_match_threshold", 0.3)

    pool = _load_pool(args.intermediate_dir)
    if not pool:
        print(f"no faces pass filter under {args.intermediate_dir}; nothing to diagnose")
        return

    relabeled = cluster_faces(
        pool,
        min_cluster_size=args.min_cluster_size,
        distance_threshold=args.distance_threshold,
    )
    raw_labels = [int(rf["cluster_id"]) for rf in relabeled]
    n_noise = sum(1 for x in raw_labels if x == -1)

    references = load_avatar_references()
    clusters, sims, ref_names = _similarity_matrix(pool, raw_labels, references)

    print(f"unit: {args.intermediate_dir}")
    print(f"pool size (filtered): {len(pool)}  (noise: {n_noise})")
    print(f"raw clusters formed: {len(clusters)}")
    print(f"min_cluster_size={args.min_cluster_size}  distance_threshold={args.distance_threshold}")
    print(f"current configs/processing_config.json:avatar_match_threshold = {config_thr}")
    print()

    print(f"top {args.top_n} clusters: cluster_id_raw / size / cosine to each avatar")
    print(f"{'raw':>5} {'size':>5}  " + "  ".join(f"{n[:14]:>14s}" for n in ref_names))
    for raw, n, _ in clusters[: args.top_n]:
        row = sims[raw]
        best_idx = int(np.argmax(row))
        best_sim = row[best_idx]
        cells = []
        for i, s in enumerate(row):
            tag = "*" if i == best_idx else " "
            cells.append(f"{s:>+13.3f}{tag}")
        print(f"{raw:>5d} {n:>5d}  " + "  ".join(cells)
              + f"   <- best={ref_names[best_idx]} ({best_sim:+.3f})")
    print()

    raw_size = {raw: n for raw, n, _ in clusters}

    print("threshold sweep — total faces folded into each avatar slot at each cutoff:")
    for thr_str in args.thresholds.split(","):
        thr = float(thr_str.strip())
        slot_to_raws, raw_to_slot = _assign(sims, thr)
        slots = []
        for fid in range(len(references)):
            name = ref_names[fid]
            raws = slot_to_raws.get(fid, [])
            if not raws:
                slots.append(f"{fid}={name[:10]}(EMPTY)")
                continue
            n_total = sum(raw_size[r] for r in raws)
            slots.append(f"{fid}={name[:10]}(raws={sorted(raws)}, n={n_total})")
        print(f"  thr={thr:.2f}: {' | '.join(slots)}")

    # Suggest a threshold: the highest one that still fills all 4 slots
    # without anchoring "everything" to one avatar (a common failure mode
    # when the threshold is too low and noisy clusters all pull toward
    # whichever avatar is most ambiguous).
    print()
    cands = sorted(set(round(t, 2) for t in [0.1, 0.12, 0.15, 0.18, 0.2,
                                              0.22, 0.25, 0.28, 0.3, 0.35, 0.4, 0.45]))
    rec = None
    for thr in cands:
        slot_to_raws, _ = _assign(sims, thr)
        if len(slot_to_raws) == 4:
            rec = thr
    if rec is not None:
        print(f"recommendation: avatar_match_threshold ~ {rec:.2f}  "
              f"(highest threshold that still fills all 4 avatar slots)")
    else:
        best_n, best_t = -1, None
        for thr in cands:
            slot_to_raws, _ = _assign(sims, thr)
            if len(slot_to_raws) > best_n:
                best_n, best_t = len(slot_to_raws), thr
        print(f"no threshold fills all 4; best is {best_t:.2f} -> "
              f"{best_n}/4 slots filled. Consider lowering further or "
              f"checking if some characters are simply absent from this unit.")


if __name__ == "__main__":
    main()
