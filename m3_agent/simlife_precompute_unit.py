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
    """Return ``id2voices`` keyed by clip-local **per-utterance** voice id.

    Each utterance gets its own id (0..N-1 in ``voice_entries`` order):

        {0: [utt0], 1: [utt1], 2: [utt2], ...}

    Two utterances by the same speaker therefore land on different prompt
    ids — Qwen sees one ``<voice_X>`` per piece of speech. Stage B's
    ``update_videograph_from_cache`` still merges same-speaker utterances
    into one **graph voice node** via embedding similarity, so the
    different ``<voice_X>`` refs typically *rewrite to the same graph node
    id* in the final memory text. The prompt-side count of voice ids
    matches the number of utterances; the graph-side count matches the
    number of distinct speakers.
    """
    return {i: [entry] for i, entry in enumerate(voice_entries)}


def _build_local_id2faces(faces_json):
    """Reconstruct the same per-clip cluster->faces grouping the original
    pipeline uses inside process_faces (post score-filter, top-K per cluster).
    """
    id2faces = establish_mapping(faces_json, key="cluster_id", filter=filter_score_based)
    id2faces.pop(-1, None)  # outliers
    return {k: v for k, v in id2faces.items() if v}


def _embed_memory_texts(texts):
    """Embed a list of memory strings via text-embedding-3-large.

    Returns ``[]`` for an empty input. Done at precompute time so Stage B
    doesn't have to re-fetch ~14k embeddings per chain on every assemble
    run. Embeddings are cached alongside the memory JSON; Stage B reads
    them and bypasses the API call.
    """
    if not texts:
        return []
    from mmagent.utils.chat_api import parallel_get_embedding
    return parallel_get_embedding("text-embedding-3-large", texts)[0]


def _force_correct_equivalences(semantic, voice_entries):
    """Inject the correct ``Equivalence: <face_X>, <voice_Y>`` lines at
    the top of the semantic memory list.

    Why we can do this: per-utterance voice ids are exactly the indices
    in ``voice_entries`` (see ``_voice_speaker_grouping``), and
    ``recluster_unit_faces`` anchored the four mains' face cluster_ids
    to ``NAME_TO_FIXED_ID``. So whenever an utterance's speaker is one
    of the four mains we can derive::

        "Equivalence: <face_{NAME_TO_FIXED_ID[speaker]}>, <voice_{i}>"

    deterministically from the data — Qwen doesn't need to guess.

    We strip any ``Equivalence:`` lines Qwen produced (they're often
    missing or partially wrong) and prepend our derived list. Speakers
    not in NAME_TO_FIXED_ID (e.g., Servo Bot, visitors) are skipped:
    we don't know which face cluster id they took.

    Stage B relies on the ``<face_0..3>`` references being resolvable
    even when the speaker's face doesn't visually appear in this clip;
    that's handled in ``simlife_assemble_chain._process_clip_faces``,
    which seeds the cluster→node map with the avatar identity mapping.
    """
    from mmagent.simlife_avatars import NAME_TO_FIXED_ID

    cleaned = [s for s in semantic
               if not str(s).strip().lower().startswith("equivalence:")]
    derived = []
    for i, entry in enumerate(voice_entries):
        spk = entry.get("speaker")
        if spk in NAME_TO_FIXED_ID:
            derived.append(
                f"Equivalence: <face_{NAME_TO_FIXED_ID[spk]}>, <voice_{i}>"
            )
    return derived + cleaned


def _generate_memory_for_clip(clip_path, faces_json, voice_entries,
                              use_audio_in_video=True):
    """Run Qwen2.5-Omni on a single clip in either audio or vision-only mode.

    Lazy import keeps the heavy model out of the import path for non-Qwen
    stages (e.g., the prep / voice-only runs).

    When ``use_audio_in_video=False`` we also pass an empty
    ``voices_list={}`` so Qwen sees neither the dialogue audio (model-side
    audio modality off) nor the voice JSON in its prompt — that matches
    SimLife's "vision only" task setting where the answer must come from
    visual cues alone.
    """
    from mmagent.memory_processing_qwen import generate_memories

    base64_video, base64_frames, _ = process_video_clip(clip_path)
    if not base64_frames:
        return [], []
    id2faces = _build_local_id2faces(faces_json)
    id2voices = _voice_speaker_grouping(voice_entries) if use_audio_in_video else {}

    return generate_memories(
        base64_frames, id2faces, id2voices, clip_path,
        use_audio_in_video=use_audio_in_video,
    )


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

    # A4 — memory JSON per clip via Qwen2.5-Omni, in two variants:
    #   audio   -> use_audio_in_video=True  + full voice JSON in prompt
    #   noaudio -> use_audio_in_video=False + voices stripped from prompt
    # Files: clip_K_memory_audio.json, clip_K_memory_noaudio.json
    if not skip_memory:
        variants = [
            ("audio", True),
            ("noaudio", False),
        ]
        for k in range(n_clips):
            face_path = os.path.join(inter_dir, f"clip_{k}_faces.json")
            voice_path = os.path.join(inter_dir, f"clip_{k}_voices.json")
            if not (os.path.exists(face_path) and os.path.exists(voice_path)):
                logger.warning("[%s] missing face/voice for clip %d, skipping memory", unit_id, k)
                continue
            faces_json = None       # lazy-loaded once per clip
            voice_entries = None
            for variant, use_audio in variants:
                mem_path = os.path.join(inter_dir, f"clip_{k}_memory_{variant}.json")
                if os.path.exists(mem_path):
                    continue
                if faces_json is None:
                    faces_json = json.load(open(face_path))
                    voice_entries = json.load(open(voice_path))
                clip_path = os.path.join(clip_dir, f"{k}.mp4")
                try:
                    episodic, semantic = _generate_memory_for_clip(
                        clip_path, faces_json, voice_entries,
                        use_audio_in_video=use_audio,
                    )
                except Exception as e:
                    logger.exception("[%s] memory generation (%s) failed for clip %d: %s",
                                     unit_id, variant, k, e)
                    episodic, semantic = [], []

                # Force correct Equivalence lines for the audio variant —
                # we know each utterance's speaker name from the voice
                # JSON and which face cluster id is anchored to each main,
                # so deriving them deterministically beats relying on
                # Qwen to remember to emit them. The vision variant has
                # no voice ids in its prompt, so any stray Equivalence
                # lines from a hallucinating model are also stripped (the
                # function is a no-op when voice_entries doesn't carry
                # main-speaker entries that match the 4 anchors).
                semantic = _force_correct_equivalences(semantic, voice_entries
                                                       if variant == "audio" else [])

                # Cache text-embedding-3-large embeddings here so Stage B
                # never has to re-fetch them — at SimLife scale that's
                # ~14k API calls per chain rebuild otherwise. Failure to
                # embed (network blip, missing creds) is caught and just
                # leaves the embeddings out; Stage B falls back to the
                # online path automatically.
                try:
                    epi_embeddings = _embed_memory_texts(episodic)
                    sem_embeddings = _embed_memory_texts(semantic)
                except Exception as e:
                    logger.warning("[%s] embedding failed for clip %d (%s): %s; "
                                   "Stage B will re-fetch on demand",
                                   unit_id, k, variant, e)
                    epi_embeddings, sem_embeddings = [], []

                payload = {
                    "episodic": episodic,
                    "semantic": semantic,
                }
                if epi_embeddings and len(epi_embeddings) == len(episodic):
                    payload["episodic_embeddings"] = epi_embeddings
                if sem_embeddings and len(sem_embeddings) == len(semantic):
                    payload["semantic_embeddings"] = sem_embeddings

                with open(mem_path, "w") as f:
                    json.dump(payload, f)
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
