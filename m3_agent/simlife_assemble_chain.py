"""Stage B: assemble a per-video-chain VideoGraph from precomputed unit JSONs.

A "chain" is one ``video_chain_id`` (vc). Every SimLife task JSON sharing
the same vc has identical ``video_ids``; only ``stop_day_position`` and the
attached questions differ. Building one graph per chain (rather than per
task) saves ~62% Stage B work and storage. Per-task differences live on
the QA side as ``before_clip`` inside ``data/annotations/simlife.json``.

For each video unit referenced in chain["video_ids"] (in order), iterate
its clips; replay cached faces/voices/memories through the existing
``add_*_node`` / ``update_node`` machinery so node IDs are minted globally
within this chain's graph; then pickle the result.

We do NOT re-run face detection, speaker embedding, or Qwen captioning here —
all expensive work happened in stage A.
"""
import argparse
import collections
import glob
import json
import logging
import os
import pickle
import re
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from mmagent.face_processing import (
    add_face_clusters_to_graph,
    establish_mapping,
    filter_score_based,
)
from mmagent.simlife_voice_processing import update_videograph_from_cache
from mmagent.memory_processing_qwen import process_memories
from mmagent.videograph import VideoGraph

logger = logging.getLogger(__name__)

CLIPS_PER_UNIT_OFFSET = 100  # must match m3_agent/simlife_data_prep.py

# Matches the same id-bearing tokens parse_video_caption looks for, but only
# for the entity types we know how to remap. <character_*> tokens, if any,
# are left untouched (refresh_equivalences handles those after assembly).
_FACE_RE = re.compile(r"<face_(\d+)>")
_VOICE_RE = re.compile(r"<voice_(\d+)>")


def _count_clips(unit_inter_dir):
    return len(glob.glob(os.path.join(unit_inter_dir, "clip_*_faces.json")))


def _process_clip_faces(graph, face_path):
    """Stage-B per-clip face replay. Returns the lossless cluster_id ->
    graph_node_id map (process_faces alone can't surface this — its return
    value applies a top-K cap that may discard cluster_ids when several
    clusters merge into one node).
    """
    if not os.path.exists(face_path):
        return {}
    with open(face_path) as f:
        faces_json = json.load(f)
    if not faces_json:
        return {}
    tempid2faces = establish_mapping(faces_json, key="cluster_id",
                                     filter=filter_score_based)
    if not tempid2faces:
        return {}
    _, cluster_to_node = add_face_clusters_to_graph(graph, tempid2faces)
    return cluster_to_node


def _build_voice_local_to_global(voice_entries, speaker_order):
    """speaker_order is the per-clip order written at precompute time (first
    appearance). voice_entries are the in-memory list AFTER
    update_videograph_from_cache has populated 'matched_node' on each entry.
    """
    by_speaker = collections.defaultdict(list)
    for entry in voice_entries:
        by_speaker[entry.get("speaker") or "unknown"].append(entry)
    out = {}
    for idx, spk in enumerate(speaker_order):
        entries = by_speaker.get(spk, [])
        if entries and "matched_node" in entries[0]:
            out[idx] = int(entries[0]["matched_node"])
    return out


def _rewrite_ids(text, face_map, voice_map):
    def face_sub(m):
        local = int(m.group(1))
        if local in face_map:
            return f"<face_{face_map[local]}>"
        return ""  # drop unresolved placeholders so parse_video_caption ignores them

    def voice_sub(m):
        local = int(m.group(1))
        if local in voice_map:
            return f"<voice_{voice_map[local]}>"
        return ""

    text = _FACE_RE.sub(face_sub, text)
    text = _VOICE_RE.sub(voice_sub, text)
    return text


def _resolve_clip_paths(unit_dir, override_dir, k):
    """Return (face_path, voice_path, mem_path) for clip K.

    Faces always come from the per-unit default (override only changes
    audio/text). Voices and memories are an *atomic pair*: we use the
    per-chain override only when BOTH files exist for that clip — mixing an
    override voice JSON with the default memory JSON would break the
    local->global voice id rewrite (speaker_order in the memory JSON
    wouldn't match the override's actual speakers).
    """
    face_path = os.path.join(unit_dir, f"clip_{k}_faces.json")
    default_voice = os.path.join(unit_dir, f"clip_{k}_voices.json")
    default_mem = os.path.join(unit_dir, f"clip_{k}_memory.json")

    if override_dir:
        ov_voice = os.path.join(override_dir, f"clip_{k}_voices.json")
        ov_mem = os.path.join(override_dir, f"clip_{k}_memory.json")
        if os.path.exists(ov_voice) and os.path.exists(ov_mem):
            return face_path, ov_voice, ov_mem
    return face_path, default_voice, default_mem


def assemble(chain_row, memory_config, inter_root="data/intermediate"):
    graph = VideoGraph(**memory_config)
    chain_id = chain_row.get("chain_id")

    for unit_idx, unit_id in enumerate(chain_row["video_ids"]):
        unit_dir = os.path.join(inter_root, unit_id)
        if not os.path.isdir(unit_dir):
            logger.warning("Missing %s; skipping unit", unit_dir)
            continue
        n_clips = _count_clips(unit_dir)
        if n_clips == 0:
            logger.warning("No face JSONs in %s; skipping", unit_dir)
            continue

        override_dir = None
        if chain_id:
            candidate = os.path.join(inter_root, "per_chain", chain_id, unit_id)
            if os.path.isdir(candidate):
                override_dir = candidate

        for k in range(n_clips):
            face_path, voice_path, mem_path = _resolve_clip_paths(unit_dir, override_dir, k)

            # Faces — replay cached cluster_ids through the graph and capture
            # the local->global label map before any top-K cap clobbers it.
            face_map = _process_clip_faces(graph, face_path)

            # Voices — load JSON ourselves so we can read matched_node back per entry.
            speaker_order = []
            voice_map = {}
            if os.path.exists(voice_path):
                with open(voice_path) as f:
                    voice_entries = json.load(f)
                if voice_entries:
                    update_videograph_from_cache(graph, voice_entries)
                    # speaker_order is also written into the memory JSON; load that
                    # canonical copy if present, else reconstruct.
                    if os.path.exists(mem_path):
                        try:
                            speaker_order = json.load(open(mem_path)).get(
                                "voice_speaker_order", []) or []
                        except Exception:
                            speaker_order = []
                    if not speaker_order:
                        seen = []
                        for entry in voice_entries:
                            spk = entry.get("speaker") or "unknown"
                            if spk not in seen:
                                seen.append(spk)
                        speaker_order = seen
                    voice_map = _build_voice_local_to_global(voice_entries, speaker_order)

            # Memory — rewrite ids, then process via existing pipeline.
            if not os.path.exists(mem_path):
                continue
            mem = json.load(open(mem_path))
            episodic_raw = mem.get("episodic", []) or []
            semantic_raw = mem.get("semantic", []) or []
            episodic = [_rewrite_ids(t, face_map, voice_map) for t in episodic_raw if t]
            semantic = [_rewrite_ids(t, face_map, voice_map) for t in semantic_raw if t]

            clip_id = unit_idx * CLIPS_PER_UNIT_OFFSET + k
            if episodic:
                process_memories(graph, episodic, clip_id, type="episodic")
            if semantic:
                process_memories(graph, semantic, clip_id, type="semantic")

    graph.refresh_equivalences()
    return graph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chain", help="chain_id basename, e.g. vc_000001")
    parser.add_argument("--data_chains", default="data/simlife/data_chains.jsonl")
    parser.add_argument("--inter_root", default="data/intermediate")
    parser.add_argument("--memory_config", default="configs/memory_config.json")
    parser.add_argument("--log_level", default="INFO")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    memory_config = json.load(open(args.memory_config))

    chain_row = None
    with open(args.data_chains) as f:
        for line in f:
            row = json.loads(line)
            if args.chain is None or row["chain_id"] == args.chain:
                chain_row = row
                break
    if chain_row is None:
        raise SystemExit(f"Chain {args.chain!r} not found in {args.data_chains}")

    out_path = chain_row["mem_path"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path) and not args.overwrite:
        logger.info("Already exists: %s (use --overwrite to rebuild)", out_path)
        return

    graph = assemble(chain_row, memory_config, inter_root=args.inter_root)

    with open(out_path, "wb") as f:
        pickle.dump(graph, f)
    logger.info("wrote %s (nodes=%d edges=%d characters=%d)",
                out_path, len(graph.nodes), len(graph.edges),
                len(getattr(graph, "character_mappings", {})))


if __name__ == "__main__":
    main()
