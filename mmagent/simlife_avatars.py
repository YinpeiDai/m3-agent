"""SimLife avatar reference embeddings.

The benchmark questions all centre on four characters living in the same
household: Father Sim, Mother Sim, Son Sim, Daughter Sim. Their portraits
sit in a 2x2 grid at ``SimLife-Data-HF/avatars.png``:

    +-------------+-------------+
    | Father Sim  | Mother Sim  |
    +-------------+-------------+
    | Son Sim     | Daughter Sim|
    +-------------+-------------+

This module:
  - extracts an InsightFace embedding for each portrait once per snapshot
    of avatars.png (cached under ``data/intermediate/_avatars/``);
  - exposes ``load_avatar_references()`` for the per-unit reclusterer and
    the Stage B graph seeder so the four mains end up with **fixed**
    integer ids across every unit and every chain.

Layout choices:
  - Fixed cluster_id / graph-node-id mapping
        0 = Father Sim
        1 = Mother Sim
        2 = Son Sim
        3 = Daughter Sim
        >=4 = anyone else (assigned in order of appearance per unit / chain)
  - ``CHARACTER_NAMES`` is the source of truth; if you ever extend the
    cast keep it in lockstep with the avatars.png layout.
"""
import base64
import json
import logging
import os
import sys

import cv2
import numpy as np

logger = logging.getLogger(__name__)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_AVATAR_PATH = os.path.join(REPO_ROOT, "SimLife-Data-HF", "avatars.png")
DEFAULT_CACHE_DIR = os.path.join(REPO_ROOT, "data", "intermediate", "_avatars")

# Order matches both the avatars.png 2x2 grid (row-major: top-left,
# top-right, bottom-left, bottom-right) AND the fixed cluster_id assignment.
CHARACTER_NAMES = ["Father Sim", "Mother Sim", "Son Sim", "Daughter Sim"]
NAME_TO_FIXED_ID = {name: i for i, name in enumerate(CHARACTER_NAMES)}
FIXED_ID_TO_NAME = {i: name for i, name in enumerate(CHARACTER_NAMES)}
N_FIXED = len(CHARACTER_NAMES)


def _quadrant_for(cx, cy, w, h):
    """Map (cx, cy) on a 2x2 image to a fixed id matching CHARACTER_NAMES order."""
    col = 1 if cx >= w / 2 else 0   # 0 = left,  1 = right
    row = 1 if cy >= h / 2 else 0   # 0 = top,   1 = bottom
    return row * 2 + col            # 0 TL, 1 TR, 2 BL, 3 BR


def _detect_avatar_faces(image_path):
    """Run InsightFace on the avatar grid and return the highest-scoring
    detection per quadrant, paired with the character name."""
    from .face_processing import face_app

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    h, w = img.shape[:2]
    faces = face_app.get(img)
    if not faces:
        raise RuntimeError(f"InsightFace found no faces in {image_path}")

    # Bucket by quadrant; in case a quadrant has multiple detections (e.g.,
    # a stray detection on shoulders) keep the one with the highest score.
    by_quadrant = {}
    for f in faces:
        bbox = [int(x) for x in f.bbox.astype(int).tolist()]
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        q = _quadrant_for(cx, cy, w, h)
        if 0 <= q < N_FIXED:
            cur = by_quadrant.get(q)
            if cur is None or float(f.det_score) > float(cur.det_score):
                by_quadrant[q] = f

    out = []
    for fixed_id in range(N_FIXED):
        face = by_quadrant.get(fixed_id)
        name = FIXED_ID_TO_NAME[fixed_id]
        if face is None:
            logger.warning("No avatar face detected for %s (quadrant %d)", name, fixed_id)
            out.append((fixed_id, name, None, None, None))
            continue
        bbox = [int(x) for x in face.bbox.astype(int).tolist()]
        crop = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        emb = np.asarray(face.normed_embedding, dtype=np.float32)
        norm = float(np.linalg.norm(emb))
        if norm > 0:
            emb = emb / norm
        out.append((fixed_id, name, emb.tolist(), bbox, crop))
    return out, img.shape[:2]


def extract_avatar_references(avatar_path=DEFAULT_AVATAR_PATH,
                              cache_dir=DEFAULT_CACHE_DIR,
                              overwrite=False):
    """Extract reference embeddings + crops, cache to ``cache_dir``.

    Idempotent: if ``manifest.json`` already exists and the avatar source's
    mtime hasn't moved past it, returns the cached manifest unchanged.
    """
    manifest_path = os.path.join(cache_dir, "manifest.json")
    if (not overwrite and os.path.exists(manifest_path)
            and os.path.exists(avatar_path)
            and os.path.getmtime(manifest_path) >= os.path.getmtime(avatar_path)):
        return json.load(open(manifest_path))

    os.makedirs(cache_dir, exist_ok=True)
    rows, _hw = _detect_avatar_faces(avatar_path)

    refs = []
    for fixed_id, name, emb, bbox, crop in rows:
        if emb is None:
            refs.append({"fixed_id": fixed_id, "name": name, "embedding": None})
            continue
        crop_path = os.path.join(cache_dir, f"avatar_{fixed_id:02d}.jpg")
        cv2.imwrite(crop_path, crop)
        refs.append({
            "fixed_id": fixed_id,
            "name": name,
            "embedding": emb,
            "bounding_box": bbox,
            "crop_file": os.path.relpath(crop_path, REPO_ROOT),
        })

    manifest = {
        "source": os.path.relpath(avatar_path, REPO_ROOT),
        "characters": refs,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def load_avatar_references(cache_dir=DEFAULT_CACHE_DIR,
                           avatar_path=DEFAULT_AVATAR_PATH,
                           auto_extract=True):
    """Return the embedding manifest, lazily extracting if missing.

    Returns a list of ``{"fixed_id", "name", "embedding"}`` dicts in
    fixed-id order. ``embedding`` is None for characters whose avatar
    InsightFace couldn't detect (so callers can decide to skip / warn).
    """
    manifest_path = os.path.join(cache_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        if not auto_extract:
            return []
        return extract_avatar_references(avatar_path, cache_dir)["characters"]
    return json.load(open(manifest_path))["characters"]


def avatar_face_info():
    """Return the avatar references in the {"embeddings", "contents"} shape
    expected by ``VideoGraph.add_img_node``. Useful for Stage B seeding.

    Each character with a valid embedding becomes one entry; ``contents``
    is the base64-encoded crop. Characters whose face InsightFace couldn't
    detect are SKIPPED — Stage B then doesn't seed them, and the first
    actual video face matching that character will create their node
    (without the fixed-id guarantee).
    """
    refs = load_avatar_references()
    out = []  # one (fixed_id, face_info) pair per detected avatar
    for ref in refs:
        emb = ref.get("embedding")
        if emb is None:
            continue
        crop_path = os.path.join(REPO_ROOT, ref.get("crop_file", ""))
        if os.path.exists(crop_path):
            with open(crop_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
        else:
            b64 = ""
        out.append((int(ref["fixed_id"]), ref["name"], {
            "embeddings": [emb],
            "contents": [b64],
        }))
    return out


def match_clusters_to_avatars(cluster_centroid_emb, references, threshold=0.3):
    """Pick the best-matching avatar for one cluster's centroid embedding.

    Args:
        cluster_centroid_emb: 1-D numpy array (already normalized).
        references: list from ``load_avatar_references``.
        threshold: minimum cosine similarity to accept a match.

    Returns:
        ``(fixed_id, name, similarity)`` of the best match, or
        ``(None, None, similarity_of_best_anyway)`` if no avatar passes
        the threshold.
    """
    best_id, best_name, best_sim = None, None, -1.0
    for ref in references:
        emb = ref.get("embedding")
        if emb is None:
            continue
        sim = float(np.dot(cluster_centroid_emb, np.asarray(emb, dtype=np.float32)))
        if sim > best_sim:
            best_id, best_name, best_sim = ref["fixed_id"], ref["name"], sim
    if best_sim < threshold:
        return None, None, best_sim
    return best_id, best_name, best_sim


def main():
    """CLI entry point: extract & cache references, print a summary."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--avatar_path", default=DEFAULT_AVATAR_PATH)
    parser.add_argument("--cache_dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest = extract_avatar_references(args.avatar_path, args.cache_dir,
                                         overwrite=args.overwrite)
    for ch in manifest["characters"]:
        emb_marker = "ok" if ch.get("embedding") else "MISSING"
        print(f"  {ch['fixed_id']}: {ch['name']:<14}  embedding={emb_marker}  "
              f"crop={ch.get('crop_file', '-')}")
    print(f"manifest at {os.path.join(args.cache_dir, 'manifest.json')}")


if __name__ == "__main__":
    main()
