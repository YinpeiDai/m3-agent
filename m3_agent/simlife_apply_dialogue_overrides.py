"""Stage A.5: apply per-chain dialogue overrides on top of the default per-unit
intermediate JSONs.

Some video units appear in multiple chains, and a chain can swap out specific
dialogue *sessions* for that chain only. In the current SimLife layout the
authoritative override file is::

    SimLife-Data-HF/video_chains/<chain>/overlay/<unit>/dialogue/sessions.json

That file lists EVERY session of the unit (overridden and not), each with:

  - ``is_overridden``: bool
  - ``audio_path``: relative path string (points into the overlay dir for
    overridden sessions, into ``video_units/<unit>/dialogue/`` otherwise)
  - ``transcript``: per-utterance text + start_offset_sec + duration_sec

We do NOT need to merge against the unit's default ``sessions.json``: the
overlay file is self-contained. We only diff against the default to find
which session_ids changed, so we can scope work to the affected clips
(other clips would be byte-identical to the default per-unit voice/memory
JSON).

Outputs go to a per-(chain, unit) directory so Stage B can pick them up
without disturbing the per-unit defaults::

    data/intermediate/per_chain/<chain>/<unit>/clip_K_voices.json
    data/intermediate/per_chain/<chain>/<unit>/clip_K_memory_audio.json   (if --regenerate_memories)

**Atomic pairing rule.** Stage B uses the per-chain pair *only* when both
``voices`` and ``memory_audio`` files exist for that clip — otherwise it
falls back to the default pair. So if you skip ``--regenerate_memories``,
the override voice JSONs sit unused; the override only takes effect once
memories are regenerated for the same clips.

Only the ``audio`` memory variant is regenerated per chain. The ``noaudio``
variant doesn't depend on the audio modality at all and is shared across
every chain that uses the same unit.
"""
import argparse
import glob
import json
import logging
import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from mmagent.simlife_voice_processing import (
    build_voice_jsons,
    iter_utterances_from_sessions,
    load_sessions,
    sessions_to_legacy_events,
    CLIP_INTERVAL_SEC,
)
from mmagent.simlife_audio_mixing import (
    build_full_audio_track,
    cut_single_clip_with_audio,
)

logger = logging.getLogger(__name__)

SIMLIFE_ROOT = "SimLife-Data-HF"
VIDEO_CHAINS = os.path.join(SIMLIFE_ROOT, "video_chains")
VIDEO_UNITS = os.path.join(SIMLIFE_ROOT, "video_units")
DEFAULT_INTER_ROOT = "data/intermediate"


def _overlay_dir(chain_id):
    return os.path.join(VIDEO_CHAINS, chain_id, "overlay")


def _list_chain_units(chain_id):
    """Units that have an overlay dir under the given chain."""
    base = _overlay_dir(chain_id)
    if not os.path.isdir(base):
        return []
    return sorted(d for d in os.listdir(base)
                  if d.startswith("video_") and os.path.isdir(os.path.join(base, d)))


def _count_unit_clips(inter_root, unit_id):
    """Stage A writes one clip_K_faces.json per actual clip; use that as the
    canonical n_clips so we never write a voice JSON for a clip that never
    had faces processed."""
    paths = glob.glob(os.path.join(inter_root, unit_id, "clip_*_faces.json"))
    return len(paths)


def _resolve_audio_abspath(rel_path):
    """Sessions.json ``audio_path`` strings are relative to the SimLife
    dataset root (``video_units/...`` or ``video_chains/.../overlay/...``).
    Resolve against ``SIMLIFE_ROOT`` to get an absolute file path."""
    if not rel_path:
        return None
    return os.path.join(SIMLIFE_ROOT, rel_path)


def process_chain_unit(chain_id, unit_id, *,
                       inter_root=DEFAULT_INTER_ROOT,
                       regenerate_memories=False,
                       overwrite=False,
                       min_duration_sec=2):
    """Apply (chain, unit) overrides. Returns a dict of stats for logging."""
    overlay_sessions_path = os.path.join(_overlay_dir(chain_id), unit_id,
                                         "dialogue", "sessions.json")
    if not os.path.exists(overlay_sessions_path):
        return {"skipped": "no overlay sessions.json"}

    overlay_sessions = load_sessions(overlay_sessions_path)
    if not overlay_sessions:
        return {"skipped": "empty overlay sessions list"}

    overridden_ids = {s["session_id"] for s in overlay_sessions
                      if s.get("is_overridden")}
    if not overridden_ids:
        # Overlay file exists but nothing actually overridden — Stage B's
        # default pair already produces the right answer.
        return {"skipped": "no overridden sessions"}

    unit_src = os.path.join(VIDEO_UNITS, unit_id)
    out_dir = os.path.join(inter_root, "per_chain", chain_id, unit_id)

    n_clips = _count_unit_clips(inter_root, unit_id)
    if n_clips == 0:
        logger.warning("[%s/%s] no faces JSONs in %s; run Stage A first",
                       chain_id, unit_id, os.path.join(inter_root, unit_id))
        return {"skipped": "stage A not run"}

    # Determine which clips need re-processing. A clip is "affected" if
    # any of its utterances came from an overridden session — checked
    # against BOTH the overlay and the default sessions.json so we also
    # cover the case where an override REMOVED utterances (default has
    # them, overlay doesn't, the corresponding clip's default memory text
    # would still reference removed speech).
    default_sessions_path = os.path.join(unit_src, "dialogue", "sessions.json")
    default_sessions = load_sessions(default_sessions_path)

    affected_clips = set()
    for utt in iter_utterances_from_sessions(overlay_sessions):
        if utt["session_id"] in overridden_ids:
            k = int(utt["abs_start_sec"] // CLIP_INTERVAL_SEC)
            if 0 <= k < n_clips:
                affected_clips.add(k)
    for utt in iter_utterances_from_sessions(default_sessions):
        if utt["session_id"] in overridden_ids:
            k = int(utt["abs_start_sec"] // CLIP_INTERVAL_SEC)
            if 0 <= k < n_clips:
                affected_clips.add(k)
    affected_clips = sorted(affected_clips)

    if not affected_clips:
        return {"skipped": "no affected clips after diff"}

    def audio_resolver(utt):
        return _resolve_audio_abspath(utt.get("audio_rel_path"))

    os.makedirs(out_dir, exist_ok=True)
    build_voice_jsons(
        None, audio_resolver, out_dir, n_clips,
        utterances=iter_utterances_from_sessions(overlay_sessions),
        min_duration_sec=min_duration_sec,
        overwrite=overwrite,
        log_label=f"{chain_id}/{unit_id}",
        only_clips=affected_clips,
    )

    stats = {
        "overridden_sessions": sorted(overridden_ids),
        "n_clips": n_clips,
        "affected_clips": affected_clips,
    }

    if regenerate_memories:
        # Build a full-length audio track that has the override session WAVs
        # mixed in at their unit-local timestamps, then cut just the affected
        # clips (Qwen with audio modality on needs the actual override audio,
        # not just the override ASR text).
        chain_clip_dir = os.path.join("data", "clips", "per_chain", chain_id, unit_id)
        os.makedirs(chain_clip_dir, exist_ok=True)
        ovr_audio_path = os.path.join(chain_clip_dir, "_full_audio.wav")

        # Per-session audio resolver for the audio mixer. The mixer wants
        # ``session_id -> wav_path``; we satisfy that by indexing the
        # overlay sessions list (which already encodes the right audio_path
        # for both overridden and default sessions).
        sid_to_path = {s["session_id"]: _resolve_audio_abspath(s.get("audio_path"))
                       for s in overlay_sessions}

        def mixer_audio_resolver(session_id):
            return sid_to_path.get(session_id)

        legacy_events = sessions_to_legacy_events(overlay_sessions)
        build_full_audio_track(
            unit_src, ovr_audio_path,
            dialogue_events=legacy_events,
            audio_resolver=mixer_audio_resolver,
            overwrite=overwrite,
        )
        unit_video = os.path.join(unit_src, "video.mp4")
        for k in affected_clips:
            ovr_clip_path = os.path.join(chain_clip_dir, f"{k}.mp4")
            if os.path.exists(ovr_clip_path) and not overwrite:
                continue
            cut_single_clip_with_audio(
                unit_video, ovr_audio_path, ovr_clip_path, clip_index=k,
            )
        stats["chain_clip_dir"] = chain_clip_dir

        stats["memories_regenerated"] = _regenerate_memories(
            chain_id, unit_id, affected_clips, out_dir, inter_root,
            chain_clip_dir=chain_clip_dir,
            overwrite=overwrite,
        )
    return stats


def _regenerate_memories(chain_id, unit_id, affected_clips, out_dir,
                         inter_root, chain_clip_dir=None, overwrite=False):
    """Re-run Qwen for the **audio** memory variant of clips whose override
    voice JSON we just wrote.

    Only the audio variant changes when an override swaps in different
    dialogue audio + ASR; the noaudio variant doesn't depend on the audio
    modality and is safely shared with the per-unit defaults. We
    therefore regen exactly one file per affected clip:
    ``<out_dir>/clip_K_memory_audio.json``.

    ``chain_clip_dir`` holds the per-(chain, unit) override clip MP4s
    (audio = override) so Qwen sees the override audio. When None we fall
    back to the unit's default clip — but that defeats the point of the
    override.

    Imports the heavy precompute helpers lazily so the override script can
    be run in voice-only mode without the Qwen-Omni model load cost.
    """
    from m3_agent.simlife_precompute_unit import (
        _generate_memory_for_clip, _force_correct_equivalences,
        _embed_memory_texts,
    )

    unit_inter = os.path.join(inter_root, unit_id)
    default_clip_root = os.path.join("data", "clips", unit_id)
    regenerated = []
    for k in affected_clips:
        out_path = os.path.join(out_dir, f"clip_{k}_memory_audio.json")
        if os.path.exists(out_path) and not overwrite:
            continue
        voice_path = os.path.join(out_dir, f"clip_{k}_voices.json")
        face_path = os.path.join(unit_inter, f"clip_{k}_faces.json")
        clip_path = None
        if chain_clip_dir:
            candidate = os.path.join(chain_clip_dir, f"{k}.mp4")
            if os.path.exists(candidate):
                clip_path = candidate
        if clip_path is None:
            clip_path = os.path.join(default_clip_root, f"{k}.mp4")
        if not (os.path.exists(voice_path) and os.path.exists(face_path)
                and os.path.exists(clip_path)):
            continue
        voices = json.load(open(voice_path))
        faces = json.load(open(face_path))
        try:
            epi, sem = _generate_memory_for_clip(
                clip_path, faces, voices, use_audio_in_video=True,
            )
        except Exception as e:
            logger.exception("[%s/%s] audio-memory regen failed for clip %d: %s",
                             chain_id, unit_id, k, e)
            epi, sem = [], []
        sem = _force_correct_equivalences(sem, voices)

        try:
            epi_emb = _embed_memory_texts(epi)
            sem_emb = _embed_memory_texts(sem)
        except Exception as e:
            logger.warning("[%s/%s] embedding failed for clip %d: %s",
                           chain_id, unit_id, k, e)
            epi_emb, sem_emb = [], []

        payload = {"episodic": epi, "semantic": sem}
        if epi_emb and len(epi_emb) == len(epi):
            payload["episodic_embeddings"] = epi_emb
        if sem_emb and len(sem_emb) == len(sem):
            payload["semantic_embeddings"] = sem_emb
        with open(out_path, "w") as f:
            json.dump(payload, f)
        regenerated.append(k)
    return regenerated


def process_chain(chain_id, **kwargs):
    units = _list_chain_units(chain_id)
    if not units:
        logger.warning("No overlay units under %s", _overlay_dir(chain_id))
        return {}
    out = {}
    for u in units:
        stats = process_chain_unit(chain_id, u, **kwargs)
        out[u] = stats
        logger.info("[%s/%s] %s", chain_id, u, stats)
    return out


def list_all_chains():
    """Every chain that has at least one unit under
    ``video_chains/<chain>/overlay/``. Chains without overlay dirs (i.e.,
    no per-chain dialogue tweaks) don't need Stage A.5 at all."""
    if not os.path.isdir(VIDEO_CHAINS):
        return []
    out = []
    for d in sorted(os.listdir(VIDEO_CHAINS)):
        if not d.startswith("vc_"):
            continue
        if os.path.isdir(_overlay_dir(d)):
            out.append(d)
    return out


def main():
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--chain", help="Process a single chain, e.g. vc_000001")
    g.add_argument("--all", action="store_true",
                   help="Process every chain that has an overlay/ dir")
    parser.add_argument("--inter_root", default=DEFAULT_INTER_ROOT)
    parser.add_argument("--regenerate_memories", action="store_true",
                        help="Re-run Qwen2.5-Omni for affected clips' audio "
                             "memory. Required for the override to take effect "
                             "at Stage B (atomic pairing).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-emit per-clip JSONs even if present.")
    parser.add_argument("--log_level", default="INFO")
    args = parser.parse_args()

    # mmagent/__init__.py installs its own root handlers (or sets the level
    # to CRITICAL in training mode), so basicConfig() is a no-op. Replace
    # root handlers with one StreamHandler at the requested level so this
    # script's progress is always visible without doubled output.
    level = getattr(logging, args.log_level.upper())
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root.setLevel(level)
    root.addHandler(handler)

    chains = [args.chain] if args.chain else list_all_chains()
    if not chains:
        raise SystemExit(f"No chains with overlay/ dirs under {VIDEO_CHAINS}")

    for chain_id in chains:
        process_chain(
            chain_id,
            inter_root=args.inter_root,
            regenerate_memories=args.regenerate_memories,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
