# Copyright (2025) Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import json
import os
import logging
from insightface.app import FaceAnalysis
from mmagent.src.face_extraction import extract_faces
from mmagent.src.face_clustering import cluster_faces
from mmagent.utils.video_processing import process_video_clip

processing_config = json.load(open("configs/processing_config.json"))
face_app = FaceAnalysis(name="buffalo_l")  # RetinaFace + ArcFace
face_app.prepare(ctx_id=-1)
cluster_size = processing_config["cluster_size"]
logger = logging.getLogger(__name__)

class Face:
    def __init__(self, frame_id, bounding_box, face_emb, cluster_id, extra_data):
        self.frame_id = frame_id
        self.bounding_box = bounding_box
        self.face_emb = face_emb
        self.cluster_id = cluster_id
        self.extra_data = extra_data

def get_face(frames):
    extracted_faces = extract_faces(face_app, frames)
    faces = [Face(frame_id=f['frame_id'], bounding_box=f['bounding_box'], face_emb=f['face_emb'], cluster_id=f['cluster_id'], extra_data=f['extra_data']) for f in extracted_faces]
    return faces


def filter_score_based(face):
    """Keep faces above the configured detection/quality thresholds."""
    dthresh = processing_config["face_detection_score_threshold"]
    qthresh = processing_config["face_quality_score_threshold"]
    return (
        float(face["extra_data"]["face_detection_score"]) > dthresh
        and float(face["extra_data"]["face_quality_score"]) > qthresh
    )


def _avatar_remap_table(pool_faces, raw_labels, references,
                        avatar_match_threshold=0.3):
    """Decide how to renumber HDBSCAN's raw labels so the four SimLife
    main characters (Father/Mother/Son/Daughter) take fixed cluster_ids
    0..3, anyone else takes 4..N.

    Strategy:
      1. For each non-noise raw cluster, take the centroid of its face
         embeddings (already normalized by InsightFace).
      2. Match each centroid against the four avatar embeddings; if the
         best-matching avatar's similarity is above
         ``avatar_match_threshold``, anchor the cluster to that fixed id.
      3. **Multiple raw clusters may anchor to the same avatar.** This is
         deliberate: HDBSCAN often splits one identity across several
         clusters (different poses, different sessions), and the greedy
         "one cluster per avatar" rule was discarding the bigger, more
         representative cluster in favour of one that happens to look
         more like the head-shot avatar. Letting them all merge means the
         fixed-id slots actually capture the main characters' total
         screen-time.
      4. Anything below threshold gets the next free id starting at
         len(references).

    Returns:
        ``{raw_label: new_label}`` (always includes ``-1: -1``).
    """
    import numpy as np
    from collections import defaultdict

    raw_to_indices = defaultdict(list)
    for i, lbl in enumerate(raw_labels):
        raw_to_indices[int(lbl)].append(i)

    raw_to_fixed = {}
    raw_to_best_sim = {}
    for raw, idxs in raw_to_indices.items():
        if raw == -1:
            continue
        embs = np.asarray([pool_faces[i]["face_emb"] for i in idxs], dtype=np.float32)
        # Re-normalize the centroid; InsightFace's normed_embedding is unit-norm
        # but the average of unit vectors generally is not.
        centroid = embs.mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm > 0:
            centroid = centroid / norm
        best_id, best_sim = None, -1.0
        for ref in references:
            emb = ref.get("embedding")
            if emb is None:
                continue
            sim = float(np.dot(centroid, np.asarray(emb, dtype=np.float32)))
            if sim > best_sim:
                best_id, best_sim = int(ref["fixed_id"]), sim
        if best_id is not None and best_sim >= avatar_match_threshold:
            raw_to_fixed[raw] = best_id
            raw_to_best_sim[raw] = best_sim

    n_fixed = len(references)
    next_free = n_fixed
    remap = {-1: -1}
    for raw in raw_to_indices:
        if raw == -1:
            continue
        if raw in raw_to_fixed:
            remap[raw] = raw_to_fixed[raw]
        else:
            remap[raw] = next_free
            next_free += 1
    return remap


def recluster_unit_faces(intermediate_dir, min_cluster_size=None,
                         distance_threshold=0.5,
                         use_avatar_anchors=True,
                         avatar_match_threshold=None):
    """Re-cluster face detections across all clips of one unit, optionally
    pinning the four SimLife main characters to fixed cluster_ids 0..3.

    Per-clip HDBSCAN is data-starved: 30 s clips often hold fewer than
    ``face_min_cluster_size`` "good" faces, so most detections end up
    labeled -1 (noise) and never make it into the graph. Running a single
    HDBSCAN over the union of every ``clip_*_faces.json`` in this unit
    gives clustering enough density to find characters, then we write the
    unit-wide ``cluster_id`` back into each per-clip JSON.

    With ``use_avatar_anchors=True`` (default) the resulting cluster_ids
    are also remapped against the avatar references at
    ``data/intermediate/_avatars/``, so the four mains end up on the same
    integers in *every* unit:

        0 = Father Sim, 1 = Mother Sim, 2 = Son Sim, 3 = Daughter Sim
        >= 4 = anyone else (assigned in arbitrary HDBSCAN order)

    The original per-clip cluster_id is preserved as ``cluster_id_per_clip``
    for debugging; the original raw HDBSCAN id (pre-remap) is preserved as
    ``cluster_id_raw``.

    Args:
        intermediate_dir: ``data/intermediate/<unit_id>/``.
        min_cluster_size: HDBSCAN ``min_cluster_size``. Defaults to
            ``processing_config["face_min_cluster_size"]``.
        distance_threshold: cosine-similarity threshold; HDBSCAN ``eps``
            becomes ``1 - distance_threshold``.
        use_avatar_anchors: when True, remap clusters that match the
            avatar references onto fixed ids 0..3. Set to False for
            diagnostics or non-SimLife data.
        avatar_match_threshold: minimum cosine similarity between a
            cluster centroid and an avatar embedding to count as a match.
            ``None`` (default) reads
            ``processing_config["avatar_match_threshold"]`` (default 0.3).

    Returns:
        ``{cluster_id: count}`` dict — total clusters formed and the
        number of faces per cluster (key ``-1`` is noise).
    """
    import glob
    from collections import Counter

    if min_cluster_size is None:
        min_cluster_size = processing_config.get("face_min_cluster_size", 3)
    if avatar_match_threshold is None:
        avatar_match_threshold = processing_config.get("avatar_match_threshold", 0.3)

    paths = sorted(
        glob.glob(os.path.join(intermediate_dir, "clip_*_faces.json")),
        key=lambda p: int(os.path.basename(p).split("_")[1]),
    )

    # Flatten with provenance so we can split back per-clip after clustering.
    pool = []   # list of (clip_idx_in_paths, face_idx_in_clip, face_dict)
    for ci, p in enumerate(paths):
        for fi, f in enumerate(json.load(open(p))):
            if filter_score_based(f):
                pool.append((ci, fi, f))

    if not pool:
        logger.warning("[%s] no faces pass filter; nothing to recluster",
                       intermediate_dir)
        return {}

    pool_faces = [t[2] for t in pool]
    relabeled = cluster_faces(
        pool_faces,
        min_cluster_size=min_cluster_size,
        distance_threshold=distance_threshold,
    )
    raw_labels = [int(rf["cluster_id"]) for rf in relabeled]
    raw_label_by_pos = {(t[0], t[1]): raw_labels[i]
                        for i, t in enumerate(pool)}

    # Build the raw -> remapped table. With avatar anchors, the four mains
    # collapse to fixed ids 0..3; otherwise the raw HDBSCAN ids carry through.
    if use_avatar_anchors:
        from .simlife_avatars import load_avatar_references
        try:
            references = load_avatar_references()
        except Exception as e:
            logger.warning("[%s] avatar references unavailable (%s); "
                           "skipping avatar anchoring", intermediate_dir, e)
            references = []
        if references:
            remap = _avatar_remap_table(
                pool_faces, raw_labels, references,
                avatar_match_threshold=avatar_match_threshold,
            )
        else:
            remap = {raw: raw for raw in set(raw_labels)}
    else:
        remap = {raw: raw for raw in set(raw_labels)}
    remap.setdefault(-1, -1)

    # Rewrite each per-clip JSON. Faces in the pool get the unit-level id;
    # faces that didn't pass the score filter (and so weren't reclustered)
    # are forced to -1 so they're treated as noise everywhere.
    summary = Counter()
    for ci, p in enumerate(paths):
        faces = json.load(open(p))
        for fi, f in enumerate(faces):
            f["cluster_id_per_clip"] = f.get("cluster_id", -1)
            raw = raw_label_by_pos.get((ci, fi), -1)
            f["cluster_id_raw"] = int(raw)
            new = remap.get(raw, -1)
            f["cluster_id"] = int(new)
            summary[int(new)] += 1
        with open(p, "w") as out:
            json.dump(faces, out)

    return dict(summary)


def establish_mapping(faces, key="cluster_id", filter=None):
    """Group face dicts by a key (default cluster_id), keeping the top-K best per group.

    K is configs/processing_config.json:max_faces_per_character. Used both at
    detection time (graph-aware update) and at stage-B graph assembly to recover
    cluster_id -> graph node id mappings without re-running detection.
    """
    mapping = {}
    for face in faces:
        if key not in face.keys():
            raise ValueError(f"key {key} not found in faces")
        if filter and not filter(face):
            continue
        id = face[key]
        if id not in mapping:
            mapping[id] = []
        mapping[id].append(face)
    max_faces = processing_config["max_faces_per_character"]
    for id in mapping:
        mapping[id] = sorted(
            mapping[id],
            key=lambda x: (
                float(x["extra_data"]["face_detection_score"]),
                float(x["extra_data"]["face_quality_score"]),
            ),
            reverse=True,
        )[:max_faces]
    return mapping


def add_face_clusters_to_graph(video_graph, tempid2faces):
    """Match each per-clip face cluster against existing img nodes and either
    extend (update_node) or create a new node. Returns:

    - id2faces:        {graph_node_id: [face, ...]} (for downstream Qwen context)
    - cluster_to_node: {cluster_id: graph_node_id} (the local->global label map
                        Stage B uses to rewrite memory text — captured BEFORE
                        the per-node top-K cap that ``id2faces`` applies, so no
                        cluster_id is lost when several clusters merge).
    """
    id2faces = {}
    cluster_to_node = {}
    for tempid, faces in tempid2faces.items():
        if tempid == -1 or not faces:
            continue
        face_info = {
            "embeddings": [face["face_emb"] for face in faces],
            "contents": [face["extra_data"]["face_base64"] for face in faces],
        }
        matched_nodes = video_graph.search_img_nodes(face_info)
        if matched_nodes:
            matched_node = matched_nodes[0][0]
            video_graph.update_node(matched_node, face_info)
        else:
            matched_node = video_graph.add_img_node(face_info)
        for face in faces:
            face["matched_node"] = matched_node
        cluster_to_node[int(tempid)] = int(matched_node)
        id2faces.setdefault(matched_node, []).extend(faces)

    max_faces = processing_config["max_faces_per_character"]
    for nid, faces in id2faces.items():
        id2faces[nid] = sorted(
            faces,
            key=lambda x: (
                float(x["extra_data"]["face_detection_score"]),
                float(x["extra_data"]["face_quality_score"]),
            ),
            reverse=True,
        )[:max_faces]
    return id2faces, cluster_to_node

def cluster_face(faces):
    faces_json = [{'frame_id': f.frame_id, 'bounding_box': f.bounding_box, 'face_emb': f.face_emb, 'cluster_id': f.cluster_id, 'extra_data': f.extra_data} for f in faces]
    min_cluster_size = processing_config.get("face_min_cluster_size", 20)
    clustered_faces = cluster_faces(faces_json, min_cluster_size, 0.5)
    faces = [Face(frame_id=f['frame_id'], bounding_box=f['bounding_box'], face_emb=f['face_emb'], cluster_id=f['cluster_id'], extra_data=f['extra_data']) for f in clustered_faces]
    return faces

def process_faces(video_graph, base64_frames, save_path, preprocessing=[]):
    """
    Process video frames to detect, cluster and track faces.

    Args:
        video_graph: Graph object to store face embeddings and relationships
        base64_frames (list): List of base64 encoded video frames to process

    Returns:
        dict: Mapping of face IDs to lists of face detections, where each face detection contains:
            - frame_id (int): Frame number where face was detected
            - bounding_box (list): Face bounding box coordinates [x1,y1,x2,y2]
            - face_emb (list): Face embedding vector
            - cluster_id (int): ID of face cluster from initial clustering
            - extra_data (dict): Additional face detection metadata
            - matched_node (int): ID of matched face node in video graph

    The function:
    1. Splits frames into batches and processes them in parallel to detect faces
    2. Clusters detected faces to group similar faces together
    3. Converts face detections to JSON format
    4. Updates video graph with face embeddings and relationships
    5. Returns mapping of face IDs to face detections
    """
    batch_size = max(len(base64_frames) // cluster_size, 4)
    
    def _process_batch(params):
        """
        Process a batch of video frames to detect faces.

        Args:
            params (tuple): A tuple containing:
                - frames (list): List of video frames to process
                - offset (int): Frame offset to add to detected face frame IDs

        Returns:
            list: List of detected faces with adjusted frame IDs

        The function:
        1. Extracts frames and offset from input params
        2. Creates face detection request for the batch
        3. Gets face detection response from service
        4. Adjusts frame IDs of detected faces by adding offset
        5. Returns list of detected faces
        """
        frames = params[0]
        offset = params[1]
        faces = get_face(frames)
        for face in faces:
            face.frame_id += offset
        return faces

    def get_embeddings(base64_frames, batch_size):
        num_batches = (len(base64_frames) + batch_size - 1) // batch_size
        batched_frames = [
            (base64_frames[i * batch_size : (i + 1) * batch_size], i * batch_size)
            for i in range(num_batches)
        ]

        faces = []

        # parallel process the batches
        with ThreadPoolExecutor(max_workers=num_batches) as executor:
            for batch_faces in tqdm(
                executor.map(_process_batch, batched_frames), total=num_batches
            ):
                faces.extend(batch_faces)

        faces = cluster_face(faces)
        return faces

    # Check if intermediate results exist
    try:
        with open(save_path, "r") as f:
            faces_json = json.load(f)
    except Exception as e:
        faces = get_embeddings(base64_frames, batch_size)

        faces_json = [
            {
                "frame_id": face.frame_id,
                "bounding_box": face.bounding_box,
                "face_emb": face.face_emb,
                "cluster_id": int(face.cluster_id),
                "extra_data": face.extra_data,
            }
            for face in faces
        ]

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
        with open(save_path, "w") as f:
            json.dump(faces_json, f)
            
    if "face" in preprocessing:
        return

    if len(faces_json) == 0:
        return {}

    tempid2faces = establish_mapping(faces_json, key="cluster_id", filter=filter_score_based)
    if len(tempid2faces) == 0:
        return {}

    id2faces, _ = add_face_clusters_to_graph(video_graph, tempid2faces)

    return id2faces

def main():
    _, frames, _ = process_video_clip(
        "/mnt/hdfs/foundation/longlin.kylin/mmagent/data/video_clips/CZ_2/-OCrS_r5GHc/11.mp4"
    )
    process_faces(None, frames, "data/temp/face_detection_results.json", preprocessing=["face"])

if __name__ == "__main__":
    main()