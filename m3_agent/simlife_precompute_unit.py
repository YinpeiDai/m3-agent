"""Stage A: precompute face/voice/memory JSONs for one SimLife video unit.

Pipeline (idempotent):
  A1. Cut video.mp4 into 30s silent clips under data/clips/video_XXXXXX/
  A2. For each clip K: face JSON via mmagent.face_processing.process_faces
  A3. Once per unit: voice JSONs via mmagent.simlife_voice_processing
  A4. For each clip K: memory JSON via mmagent.memory_processing_qwen.generate_memories
      (uses clip-local face cluster_ids and per-clip speaker indices as IDs).

The graph is NOT built here — that happens in stage B per task.
"""
import argparse
import json
import logging
import os
import sys

# Repo-relative paths and absolute path normalization.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from m3_agent.simlife_clip_video import cut_clips
from mmagent.face_processing import (
    process_faces,
    establish_mapping,
    filter_score_based,
    recluster_unit_faces,
)
from mmagent.simlife_voice_processing import build_unit_voice_jsons
from mmagent.utils.video_processing import process_video_clip

logger = logging.getLogger(__name__)
CLIP_INTERVAL_SEC = 30


def _voice_speaker_grouping(voice_entries):
    """Return ``id2voices`` keyed by per-clip speaker index (deterministic order
    of first appearance), and a parallel ``speaker_order`` list. The same
    grouping is recomputed verbatim at stage B; consistency is what makes
    local IDs translatable.
    """
    speaker_order = []
    seen = {}
    id2voices = {}
    for entry in voice_entries:
        spk = entry.get("speaker") or "unknown"
        if spk not in seen:
            seen[spk] = len(speaker_order)
            speaker_order.append(spk)
        idx = seen[spk]
        id2voices.setdefault(idx, []).append(entry)
    return id2voices, speaker_order


def _build_local_id2faces(faces_json):
    """Reconstruct the same per-clip cluster->faces grouping the original
    pipeline uses inside process_faces (post score-filter, top-K per cluster).
    """
    id2faces = establish_mapping(faces_json, key="cluster_id", filter=filter_score_based)
    id2faces.pop(-1, None)  # outliers
    return {k: v for k, v in id2faces.items() if v}


def _generate_memory_for_clip(clip_path, faces_json, voice_entries):
    """Run Qwen2.5-Omni on a single clip. Lazy import keeps the heavy model out
    of the import path for non-Qwen stages (e.g., the prep / voice-only runs).
    """
    from mmagent.memory_processing_qwen import generate_memories

    base64_video, base64_frames, _ = process_video_clip(clip_path)
    if not base64_frames:
        return [], []
    id2faces = _build_local_id2faces(faces_json)
    id2voices, _ = _voice_speaker_grouping(voice_entries)

    if not id2faces and not id2voices:
        # Still ask Qwen for a description; just won't include face/voice IDs.
        return generate_memories(base64_frames, {}, {}, clip_path)
    return generate_memories(base64_frames, id2faces, id2voices, clip_path)


def _list_existing_clips(clip_dir):
    """Return ``data/clips/<unit>/{0,1,2,...}.mp4`` sorted by integer name."""
    if not os.path.isdir(clip_dir):
        return []
    paths = []
    for f in os.listdir(clip_dir):
        if not f.endswith(".mp4"):
            continue
        stem = f[:-4]
        if stem.isdigit():
            paths.append(os.path.join(clip_dir, f))
    return sorted(paths, key=lambda p: int(os.path.basename(p)[:-4]))


def precompute_unit(unit_id, src_root="SimLife-Data-HF/video_units",
                    clips_root="data/clips", inter_root="data/intermediate",
                    skip_memory=False, skip_voice=False, skip_face=False,
                    skip_clip=False):
    src_dir = os.path.join(src_root, unit_id)
    clip_dir = os.path.join(clips_root, unit_id)
    inter_dir = os.path.join(inter_root, unit_id)
    os.makedirs(inter_dir, exist_ok=True)

    video_path = os.path.join(src_dir, "video.mp4")
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)

    # A1 — cut clips (or trust the existing set when --skip_clip).
    if skip_clip:
        clips = _list_existing_clips(clip_dir)
        if not clips:
            raise FileNotFoundError(
                f"--skip_clip set but no <K>.mp4 clips under {clip_dir}; "
                f"run without --skip_clip first to generate them."
            )
        logger.info("[%s] skip_clip: reusing %d existing clips at %s",
                    unit_id, len(clips), clip_dir)
    else:
        clips = cut_clips(video_path, clip_dir)
    n_clips = len(clips)
    logger.info("[%s] %d clips at %s", unit_id, n_clips, clip_dir)

    # A2 — face JSON per clip (cache-only via preprocessing=["face"])
    if not skip_face:
        any_new = False
        for k in range(n_clips):
            face_path = os.path.join(inter_dir, f"clip_{k}_faces.json")
            if os.path.exists(face_path):
                continue
            clip_path = os.path.join(clip_dir, f"{k}.mp4")
            _, base64_frames, _ = process_video_clip(clip_path)
            if not base64_frames:
                with open(face_path, "w") as f:
                    json.dump([], f)
                continue
            process_faces(None, base64_frames, save_path=face_path, preprocessing=["face"])
            logger.info("[%s] wrote %s", unit_id, face_path)
            any_new = True

        # A2.5 — re-cluster across the whole unit so cluster_id is a stable
        # unit-level identity instead of per-clip noise. Only re-run when we
        # just wrote new face JSONs (presence of "cluster_id_per_clip" in any
        # face dict means recluster has already been done).
        sentinel_path = os.path.join(inter_dir, "clip_0_faces.json")
        already = False
        if os.path.exists(sentinel_path):
            try:
                first = json.load(open(sentinel_path))
                already = bool(first) and "cluster_id_per_clip" in first[0]
            except Exception:
                already = False
        if any_new or not already:
            summary = recluster_unit_faces(inter_dir)
            n_clusters = sum(1 for k in summary if k != -1)
            n_noise = summary.get(-1, 0)
            n_total = sum(summary.values())
            logger.info("[%s] unit-level recluster: %d clusters across %d faces (%d noise)",
                        unit_id, n_clusters, n_total, n_noise)

    # A3 — voice JSONs (one pass over log.jsonl)
    if not skip_voice:
        build_unit_voice_jsons(src_dir, inter_dir, n_clips)
        logger.info("[%s] wrote voice JSONs", unit_id)

    # A4 — memory JSON per clip via Qwen2.5-Omni
    if not skip_memory:
        for k in range(n_clips):
            mem_path = os.path.join(inter_dir, f"clip_{k}_memory.json")
            if os.path.exists(mem_path):
                continue
            face_path = os.path.join(inter_dir, f"clip_{k}_faces.json")
            voice_path = os.path.join(inter_dir, f"clip_{k}_voices.json")
            if not (os.path.exists(face_path) and os.path.exists(voice_path)):
                logger.warning("[%s] missing face/voice for clip %d, skipping memory", unit_id, k)
                continue
            faces_json = json.load(open(face_path))
            voice_entries = json.load(open(voice_path))
            clip_path = os.path.join(clip_dir, f"{k}.mp4")
            try:
                episodic, semantic = _generate_memory_for_clip(clip_path, faces_json, voice_entries)
            except Exception as e:
                logger.exception("[%s] memory generation failed for clip %d: %s", unit_id, k, e)
                episodic, semantic = [], []
            _, speaker_order = _voice_speaker_grouping(voice_entries)
            with open(mem_path, "w") as f:
                json.dump({
                    "episodic": episodic,
                    "semantic": semantic,
                    "voice_speaker_order": speaker_order,
                }, f)
            logger.info("[%s] wrote %s (epi=%d, sem=%d)",
                        unit_id, mem_path, len(episodic), len(semantic))


def main():
    import time
    tstart = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit", required=True, help="e.g. video_001285")
    parser.add_argument("--src_root", default="SimLife-Data-HF/video_units")
    parser.add_argument("--clips_root", default="data/clips")
    parser.add_argument("--inter_root", default="data/intermediate")
    parser.add_argument("--skip_clip", action="store_true",
                        help="Reuse existing <out>/<K>.mp4 clips instead of "
                             "re-cutting (the source video.mp4 is not even read).")
    parser.add_argument("--skip_face", action="store_true")
    parser.add_argument("--skip_voice", action="store_true")
    parser.add_argument("--skip_memory", action="store_true")
    parser.add_argument("--log_level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    precompute_unit(
        args.unit,
        src_root=args.src_root,
        clips_root=args.clips_root,
        inter_root=args.inter_root,
        skip_clip=args.skip_clip,
        skip_face=args.skip_face,
        skip_voice=args.skip_voice,
        skip_memory=args.skip_memory,
    )
    
    tend = time.time()
    print(f"Precomputation for unit {args.unit} completed in {tend - tstart:.2f} seconds.")


if __name__ == "__main__":
    main()
