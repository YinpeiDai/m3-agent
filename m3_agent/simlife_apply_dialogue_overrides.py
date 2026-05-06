"""Stage A.5: apply per-chain dialogue overrides on top of the default per-unit
intermediate JSONs.

Some video units appear in multiple chains, and a chain can swap out specific
dialogue *sessions* (typically one per unit) for that chain only. Each
override lives at::

    SimLife-Data-HF/task_dialogue_audio/<chain>/<unit>/
        session_NNN.wav        # replacement audio for that session
        asr.jsonl              # replacement dialogue events (same schema as
                                 video_units/<unit>/log.jsonl rows where
                                 type==dialogue) — assumed present alongside
                                 the WAV(s)

Other sessions of the same unit are unchanged. We therefore *merge* the
default ``log.jsonl`` events with the override ``asr.jsonl`` (one event per
``session_id``: override wins) and route audio reads through a resolver that
prefers the override WAV when present.

Outputs go to a per-(chain, unit) directory so Stage B can pick them up
without disturbing the per-unit defaults::

    data/intermediate/per_chain/<chain>/<unit>/clip_K_voices.json
    data/intermediate/per_chain/<chain>/<unit>/clip_K_memory.json   (if --regenerate_memories)

**Atomic pairing rule.** Voice ids in Qwen's prompt are clip-local and
**per-utterance** (the i-th entry in ``clip_K_voices.json`` is
``<voice_i>``); same-speaker utterances are merged at the graph level by
embedding cosine, but the prompt-side ids match the number of speech
pieces. So when an override:

  - swaps **only the audio + ASR text** of an existing session, the
    clip-local voice-id count stays the same (one per utterance) — but
    the dialogue *content* the captions reference doesn't, so captions
    need a Qwen regeneration to stay truthful.
  - adds or removes utterances, the per-utterance ids 0..N shift and
    captions definitely need regen.

Either way Stage B uses the per-chain pair *only* when both ``voices``
and ``memory_audio`` files exist for that clip — otherwise it falls
back to the default pair. So if you skip ``--regenerate_memories``,
the override voice JSONs sit unused; the override only takes effect
once memories are regenerated for the same clips.
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
    iter_utterances,
    load_dialogue_events,
    CLIP_INTERVAL_SEC,
)
from mmagent.simlife_audio_mixing import (
    build_full_audio_track,
    cut_single_clip_with_audio,
)

logger = logging.getLogger(__name__)

SIMLIFE_ROOT = "SimLife-Data-HF"
TASK_DIALOGUE_AUDIO = os.path.join(SIMLIFE_ROOT, "task_dialogue_audio")
VIDEO_UNITS = os.path.join(SIMLIFE_ROOT, "video_units")
DEFAULT_INTER_ROOT = "data/intermediate"


def _list_chain_units(chain_dir):
    if not os.path.isdir(chain_dir):
        return []
    return sorted(d for d in os.listdir(chain_dir)
                  if d.startswith("video_") and os.path.isdir(os.path.join(chain_dir, d)))


def _override_session_ids(override_dir):
    """Set of session_ids whose WAV is present under the (chain, unit) folder."""
    return {
        f[: -len(".wav")]
        for f in os.listdir(override_dir)
        if f.endswith(".wav") and f.startswith("session_")
    }


def _merge_events(default_events, override_events):
    """Replace any default event whose session_id is in override_events.

    Order is preserved (default order). If the override has events for
    sessions not in the default, they are appended at the end so any
    out-of-band session still contributes utterances.
    """
    if not override_events:
        return list(default_events)
    by_session = {e.get("session_id"): e for e in override_events}
    merged = []
    consumed = set()
    for ev in default_events:
        sid = ev.get("session_id")
        if sid in by_session:
            merged.append(by_session[sid])
            consumed.add(sid)
        else:
            merged.append(ev)
    for sid, ev in by_session.items():
        if sid not in consumed:
            merged.append(ev)
    return merged


def _count_unit_clips(inter_root, unit_id):
    """Stage A writes one clip_K_faces.json per actual clip; use that as the
    canonical n_clips so we never write a voice JSON for a clip that never
    had faces processed."""
    paths = glob.glob(os.path.join(inter_root, unit_id, "clip_*_faces.json"))
    return len(paths)


def process_chain_unit(chain_id, unit_id, *,
                       inter_root=DEFAULT_INTER_ROOT,
                       regenerate_memories=False,
                       overwrite=False,
                       min_duration_sec=2):
    """Apply (chain, unit) overrides. Returns a dict of stats for logging."""
    override_dir = os.path.join(TASK_DIALOGUE_AUDIO, chain_id, unit_id)
    unit_src = os.path.join(VIDEO_UNITS, unit_id)
    out_dir = os.path.join(inter_root, "per_chain", chain_id, unit_id)

    if not os.path.isdir(override_dir):
        return {"skipped": "no override dir"}

    override_sessions = _override_session_ids(override_dir)
    if not override_sessions:
        return {"skipped": "no override sessions"}

    asr_path = os.path.join(override_dir, "asr.jsonl")
    if not os.path.exists(asr_path):
        # The user said to assume asr.jsonl is present. If it isn't, we'd be
        # mixing override audio with default ASR text — refuse rather than
        # silently produce inconsistent data.
        logger.warning("[%s/%s] missing asr.jsonl alongside override WAVs; skipping",
                       chain_id, unit_id)
        return {"skipped": "no asr.jsonl"}

    default_events = load_dialogue_events(os.path.join(unit_src, "log.jsonl"))
    override_events = load_dialogue_events(asr_path)
    merged = _merge_events(default_events, override_events)

    n_clips = _count_unit_clips(inter_root, unit_id)
    if n_clips == 0:
        logger.warning("[%s/%s] no faces JSONs in %s; run Stage A first",
                       chain_id, unit_id, os.path.join(inter_root, unit_id))
        return {"skipped": "stage A not run"}

    def audio_resolver(session_id):
        override_wav = os.path.join(override_dir, f"{session_id}.wav")
        if os.path.exists(override_wav):
            return override_wav
        return os.path.join(unit_src, "dialogue_audio", f"{session_id}.wav")

    # Only write override JSONs for clips that actually contain at least one
    # utterance from an overridden session. Other clips' contents would be
    # bit-identical to the default per-unit voice JSON, so re-emitting them
    # just wastes ERes2NetV2 inference. Stage B's atomic pairing rule plus
    # the fallback to the default pair makes those clips work transparently.
    #
    # We union the default and override utterance positions so we still
    # cover clips where the override REMOVED utterances (e.g., a session
    # became silent in the override) — those clips need memory regen too,
    # because the default memory text still references speech that no
    # longer happens.
    affected_clips = set()
    for source_events in (default_events, override_events):
        for utt in iter_utterances(source_events):
            if utt["session_id"] not in override_sessions:
                continue
            k = int(utt["abs_start_sec"] // CLIP_INTERVAL_SEC)
            if 0 <= k < n_clips:
                affected_clips.add(k)
    affected_clips = sorted(affected_clips)

    os.makedirs(out_dir, exist_ok=True)
    build_voice_jsons(
        merged, audio_resolver, out_dir, n_clips,
        min_duration_sec=min_duration_sec,
        overwrite=overwrite,
        log_label=f"{chain_id}/{unit_id}",
        only_clips=affected_clips,
    )

    stats = {
        "override_sessions": sorted(override_sessions),
        "n_clips": n_clips,
        "affected_clips": affected_clips,
    }

    if regenerate_memories:
        # Per-chain clip MP4s with override audio muxed in. Stage A.4 (Qwen)
        # needs the audio modality, so feeding it the default-audio default
        # clip would defeat the override. We build a full-length override
        # audio track once and cut just the affected clips from the unit's
        # source video.
        chain_clip_dir = os.path.join("data", "clips", "per_chain", chain_id, unit_id)
        os.makedirs(chain_clip_dir, exist_ok=True)
        ovr_audio_path = os.path.join(chain_clip_dir, "_full_audio.wav")
        build_full_audio_track(
            unit_src, ovr_audio_path,
            dialogue_events=merged,
            audio_resolver=audio_resolver,
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
    dialogue audio + ASR; the vision-only variant of the memory
    (``clip_K_memory_vision.json``) doesn't depend on the audio modality
    at all and is safely shared with the per-unit defaults. We therefore
    regen exactly one file per affected clip:
    ``<out_dir>/clip_K_memory_audio.json``. Stage B's atomic-pairing
    rule for the audio variant requires both
    ``clip_K_voices.json`` (override) and ``clip_K_memory_audio.json``
    (override) to be present; otherwise the default audio pair is used.
    The vision variant always reads from the per-unit default.

    ``affected_clips`` is the set the caller already computed from
    ``override_sessions``: every clip with at least one overridden
    utterance.

    ``chain_clip_dir`` is where per-(chain, unit) override clip MP4s
    live; Qwen reads those (audio = override) instead of the default
    per-unit clips (audio = default). When None we fall back to defaults
    — but that defeats the override for the audio modality.

    Imported lazily so callers that don't pass --regenerate_memories can run
    without paying the Qwen-Omni model load cost.
    """
    from m3_agent.simlife_precompute_unit import (
        _generate_memory_for_clip, _force_correct_equivalences,
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
        # Prefer the per-chain override clip (override audio muxed in);
        # fall back to the default clip if it's missing.
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
        # Force correct Equivalence lines using the override speaker labels
        # in this clip's voice JSON (voice_speaker_order isn't needed —
        # voice ids are per-utterance now, indices into ``voices``).
        sem = _force_correct_equivalences(sem, voices)
        with open(out_path, "w") as f:
            json.dump({
                "episodic": epi,
                "semantic": sem,
            }, f)
        regenerated.append(k)
    return regenerated


def process_chain(chain_id, **kwargs):
    chain_dir = os.path.join(TASK_DIALOGUE_AUDIO, chain_id)
    units = _list_chain_units(chain_dir)
    if not units:
        logger.warning("No units under %s", chain_dir)
        return {}
    out = {}
    for u in units:
        stats = process_chain_unit(chain_id, u, **kwargs)
        out[u] = stats
        logger.info("[%s/%s] %s", chain_id, u, stats)
    return out


def list_all_chains():
    if not os.path.isdir(TASK_DIALOGUE_AUDIO):
        return []
    return sorted(d for d in os.listdir(TASK_DIALOGUE_AUDIO)
                  if d.startswith("vc_") and os.path.isdir(os.path.join(TASK_DIALOGUE_AUDIO, d)))


def main():
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--chain", help="Process a single chain, e.g. vc_000001")
    g.add_argument("--all", action="store_true", help="Process every chain under task_dialogue_audio/")
    parser.add_argument("--inter_root", default=DEFAULT_INTER_ROOT)
    parser.add_argument("--regenerate_memories", action="store_true",
                        help="Re-run Qwen2.5-Omni for clips whose override voices changed. "
                             "Required for the override to take effect at Stage B (atomic pairing).")
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
        raise SystemExit(f"No chains under {TASK_DIALOGUE_AUDIO}")

    for chain_id in chains:
        process_chain(
            chain_id,
            inter_root=args.inter_root,
            regenerate_memories=args.regenerate_memories,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
