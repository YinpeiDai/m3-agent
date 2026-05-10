"""SimLife voice processing.

Replaces the Gemini-based diarization in mmagent/voice_processing.py with a
``dialogue/sessions.json``-driven flow. Output JSON is bit-compatible with
the existing pipeline so stage-B graph assembly can reuse the same
update_videograph logic.

Two source formats are supported transparently:

  - ``video_units/<u>/dialogue/sessions.json``   — per-unit defaults
  - ``video_chains/<vc>/overlay/<u>/dialogue/sessions.json`` — per-chain
    overrides (every session listed, with ``is_overridden`` + ``audio_path``
    per session pointing into either the overlay dir or the unit dir).

Each session entry's ``audio_path`` is the authoritative WAV pointer (an
overridden session's path lives under the overlay dir, otherwise it points
back at ``video_units/<u>/dialogue/<sid>.wav``). The legacy log.jsonl-based
path is retained only for backward compat with already-generated caches.

Also exposes ``process_voices_from_cache`` for stage-B: it bypasses the
``if not base64_audio: return {}`` short-circuit at the top of
``mmagent/voice_processing.py:process_voices`` (which would otherwise discard
a perfectly fine cached JSON when we don't pass an audio blob).
"""
import base64
import io
import json
import logging
import math
import os

import numpy as np
from pydub import AudioSegment

from .voice_processing import embedding_model, feature_extractor
import torch
import torchaudio

logger = logging.getLogger(__name__)

CLIP_INTERVAL_SEC = 30


def _format_mmss(seconds):
    seconds = max(0, int(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def _wav_bytes_to_embedding(wav_bytes):
    """Compute a normalized 192-d ERes2NetV2 speaker embedding from WAV bytes.

    Uses torchaudio.functional.resample (kernel-based) instead of sox_effects;
    the latter pulls libsox.so as a runtime dep and is not always present.
    """
    buf = io.BytesIO(wav_bytes)
    wav, fs = torchaudio.load(buf)
    if fs != 16000:
        wav = torchaudio.functional.resample(wav, fs, 16000)
    if wav.shape[0] > 1:
        wav = wav[0, :].unsqueeze(0)
    feat = feature_extractor(wav).unsqueeze(0).to(torch.device("cuda"))
    with torch.no_grad():
        emb = embedding_model(feat).detach().squeeze(0).cpu().numpy()
    norm = float(np.linalg.norm(emb))
    if norm <= 0:
        return emb.tolist()
    return (emb / norm).tolist()


def _slice_to_wav_bytes(audio_segment, start_ms, end_ms):
    """Slice a pydub AudioSegment and return WAV bytes."""
    seg = audio_segment[start_ms:end_ms]
    out = io.BytesIO()
    seg.export(out, format="wav")
    return out.getvalue()


def load_dialogue_events(log_path):
    """Legacy reader: every ``type==dialogue`` row from a log.jsonl-shaped file.

    Kept for backward compatibility with caches generated before the switch
    to ``dialogue/sessions.json``. New code paths should call
    :func:`load_sessions` instead. Returns a list so callers can splice.
    """
    events = []
    if not os.path.exists(log_path):
        return events
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("type") != "dialogue":
                continue
            events.append(row)
    return events


def load_sessions(sessions_json_path):
    """Load a ``dialogue/sessions.json`` file (per-unit default OR per-chain
    overlay) and return its ``sessions`` list.

    Returns an empty list when the file is absent — the caller treats that
    as "no dialogue for this unit" rather than an error so missing dialogue
    folders don't break the precompute pipeline for action-only days.
    """
    if not os.path.exists(sessions_json_path):
        return []
    with open(sessions_json_path) as f:
        return json.load(f).get("sessions") or []


def iter_utterances_from_sessions(sessions):
    """Flatten ``dialogue/sessions.json`` sessions into per-utterance dicts.

    The output dict shape matches :func:`iter_utterances` so downstream
    consumers (``build_voice_jsons``) don't need to know which source
    format was loaded.

    Sessions JSON differs from log.jsonl rows in:
      - ``start_time``       -> ``start_time_sec``
      - ``utterances``       -> ``transcript``
      - ``audio_path`` is now per-session (relative path string)

    Per-session ``audio_path`` is propagated to every yielded utterance so
    the audio resolver in ``build_voice_jsons`` can route to the right WAV
    directly (no separate session->path map needed). Overridden sessions
    in an overlay file naturally point into the overlay dir; non-overridden
    ones point back at the unit's default WAV.
    """
    for sess in sessions:
        session_id = sess.get("session_id")
        # tolerate legacy log.jsonl-style start_time too
        sess_start = float(sess.get("start_time_sec", sess.get("start_time", 0.0)))
        audio_rel = sess.get("audio_path")
        for utt in sess.get("transcript", sess.get("utterances", [])):
            offset = float(utt.get("start_offset_sec", 0.0))
            duration = float(utt.get("duration_sec", 0.0))
            if duration <= 0:
                continue
            abs_start = sess_start + offset
            abs_end = abs_start + duration
            yield {
                "session_id": session_id,
                "session_offset_sec": offset,
                "duration_sec": duration,
                "abs_start_sec": abs_start,
                "abs_end_sec": abs_end,
                "speaker": utt.get("speaker"),
                "text": utt.get("text", ""),
                # Path is relative to the SimLife-Data-HF root (e.g.
                # "video_units/video_001128/dialogue/session_001.wav" or
                # "video_chains/vc_000001/overlay/.../session_005.wav").
                # Caller resolves to an absolute path.
                "audio_rel_path": audio_rel,
            }


def sessions_to_legacy_events(sessions):
    """Convert a ``dialogue/sessions.json`` session list into the
    log.jsonl-row shape the audio mixer (and any other legacy consumer)
    expects.

    The resulting rows are NOT written to disk — they're a thin in-memory
    adapter so we don't have to teach every consumer two formats.
    """
    out = []
    for s in sessions:
        out.append({
            "type": "dialogue",
            "session_id": s.get("session_id"),
            "start_time": float(s.get("start_time_sec", s.get("start_time", 0.0))),
            "end_time": float(s.get("end_time_sec", s.get("end_time", 0.0))),
            "participants": list(s.get("participants") or []),
            "utterances": list(s.get("transcript", s.get("utterances", []))),
            # carry through so per-row resolvers can use it if they want
            "audio_path": s.get("audio_path"),
        })
    return out


def iter_utterances(events):
    """Legacy: flatten log.jsonl-shaped dialogue events to per-utterance dicts.

    Kept for backward compat with the override script's old code paths.
    New code should use :func:`iter_utterances_from_sessions`.
    """
    for row in events:
        session_id = row.get("session_id")
        dialogue_start = float(row["start_time"])
        for utt in row.get("utterances", []):
            offset = float(utt.get("start_offset_sec", 0.0))
            duration = float(utt.get("duration_sec", 0.0))
            if duration <= 0:
                continue
            abs_start = dialogue_start + offset
            abs_end = abs_start + duration
            yield {
                "session_id": session_id,
                "session_offset_sec": offset,
                "duration_sec": duration,
                "abs_start_sec": abs_start,
                "abs_end_sec": abs_end,
                "speaker": utt.get("speaker"),
                "text": utt.get("text", ""),
            }


def build_voice_jsons(utterances_or_events, audio_resolver, out_dir, n_clips,
                      clip_interval_sec=CLIP_INTERVAL_SEC, min_duration_sec=2,
                      overwrite=False, log_label="", only_clips=None,
                      utterances=None):
    """Generic per-clip voice-JSON writer.

    Args:
        utterances_or_events: legacy positional — when ``utterances`` is
            None this is treated as a list of log.jsonl-shaped dialogue
            events and flattened via :func:`iter_utterances`. New callers
            should pass ``utterances=`` instead and leave this empty.
        utterances: explicit pre-flattened utterance iterable (each dict
            shaped like :func:`iter_utterances_from_sessions` output).
            Recommended path for new sessions.json-driven callers.
        audio_resolver: callable ``utt_dict -> path-to-wav`` (or ``None``).
            Receives the full utterance dict so resolvers can use the
            per-session ``audio_rel_path`` for sessions.json sources, or
            fall back to ``session_id``-based lookup for log.jsonl
            sources. For backward compat, a callable that accepts a
            single ``session_id`` string is also detected and adapted.
        out_dir: writes ``clip_K_voices.json`` for K in [0, n_clips).
        n_clips: total number of 30s clips for this unit.
        overwrite: if False, leave existing per-clip JSONs untouched.
        log_label: free-form label included in log lines for debugability.
        only_clips: optional iterable of clip indices to write. When set,
            other clips are left untouched (used by the per-chain override
            flow to avoid recomputing JSONs for clips not affected by any
            overridden session).

    Empty clips (in scope) still get an empty list written so downstream
    stages can detect "processed" cleanly.
    """
    os.makedirs(out_dir, exist_ok=True)

    if utterances is None:
        utterances = iter_utterances(utterances_or_events)

    # Detect legacy session_id-only resolvers and adapt them so the rest of
    # the function can pass full utterance dicts unconditionally.
    resolver = audio_resolver
    try:
        import inspect
        sig = inspect.signature(audio_resolver)
        if len(sig.parameters) == 1:
            first = next(iter(sig.parameters))
            if first in ("session_id", "sid"):
                resolver = lambda utt, _orig=audio_resolver: _orig(utt["session_id"])
    except (TypeError, ValueError):
        pass

    write_set = None if only_clips is None else set(only_clips)

    buckets = {k: [] for k in range(n_clips)}
    skipped_short = 0
    for utt in utterances:
        if utt["duration_sec"] < min_duration_sec:
            skipped_short += 1
            continue
        clip_idx = int(utt["abs_start_sec"] // clip_interval_sec)
        if 0 <= clip_idx < n_clips:
            buckets[clip_idx].append(utt)

    if skipped_short:
        logger.debug("[%s] skipped %d utterances < %ds", log_label, skipped_short, min_duration_sec)

    # Cache loaded WAVs by resolved absolute path so two utterances from
    # the same session decode the WAV exactly once even if the resolver
    # routes them through different code paths.
    seg_cache = {}

    def _get_segment(utt):
        wav_path = resolver(utt)
        if not wav_path:
            return None
        if wav_path in seg_cache:
            return seg_cache[wav_path]
        if not os.path.exists(wav_path):
            seg_cache[wav_path] = None
            return None
        seg = AudioSegment.from_wav(wav_path)
        seg_cache[wav_path] = seg
        return seg

    for k in range(n_clips):
        if write_set is not None and k not in write_set:
            continue

        out_path = os.path.join(out_dir, f"clip_{k}_voices.json")
        if os.path.exists(out_path) and not overwrite:
            continue

        clip_start_abs = k * clip_interval_sec
        entries = []
        for utt in buckets[k]:
            seg = _get_segment(utt)
            if seg is None:
                logger.warning("[%s] missing wav for session %s", log_label, utt["session_id"])
                continue
            start_ms = int(utt["session_offset_sec"] * 1000)
            end_ms = min(start_ms + int(utt["duration_sec"] * 1000), len(seg))
            if end_ms <= start_ms:
                continue
            try:
                wav_bytes = _slice_to_wav_bytes(seg, start_ms, end_ms)
                emb = _wav_bytes_to_embedding(wav_bytes)
            except Exception as e:
                logger.warning("[%s] embedding failed for session %s: %s",
                               log_label, utt["session_id"], e)
                continue

            clip_local_start = utt["abs_start_sec"] - clip_start_abs
            clip_local_end = utt["abs_end_sec"] - clip_start_abs
            entries.append({
                "start_time": _format_mmss(clip_local_start),
                "end_time": _format_mmss(clip_local_end),
                "asr": utt["text"],
                "duration": int(math.ceil(utt["duration_sec"])),
                "speaker": utt["speaker"],
                "session_id": utt["session_id"],
                "audio_segment": base64.b64encode(wav_bytes).decode("utf-8"),
                "embedding": emb,
            })

        with open(out_path, "w") as f:
            json.dump(entries, f)


def build_unit_voice_jsons(unit_dir, out_dir, n_clips, clip_interval_sec=CLIP_INTERVAL_SEC,
                          min_duration_sec=2, overwrite=False,
                          simlife_root="SimLife-Data-HF"):
    """Default per-unit pipeline.

    Source preference:
      1. ``<unit_dir>/dialogue/sessions.json`` — current SimLife layout.
         Each session lists its own ``audio_path`` (relative to
         ``simlife_root``), so the resolver simply joins that.
      2. ``<unit_dir>/log.jsonl`` (legacy) — falls back to the old
         ``dialogue_audio/<sid>.wav`` convention.

    Empty units (no dialogue source) still get empty per-clip JSONs so
    the absence is distinguishable from "not yet processed."
    """
    label = os.path.basename(unit_dir.rstrip("/"))
    sessions_path = os.path.join(unit_dir, "dialogue", "sessions.json")
    log_path = os.path.join(unit_dir, "log.jsonl")

    if os.path.exists(sessions_path):
        sessions = load_sessions(sessions_path)
        utts = iter_utterances_from_sessions(sessions)

        # ``audio_path`` in sessions.json is relative to the SimLife dataset
        # root. Defensive default: when missing, fall back to the canonical
        # per-unit dialogue dir convention.
        unit_id = label
        default_dialogue_dir = os.path.join(unit_dir, "dialogue")

        def audio_resolver(utt):
            rel = utt.get("audio_rel_path")
            if rel:
                return os.path.join(simlife_root, rel)
            sid = utt.get("session_id")
            return os.path.join(default_dialogue_dir, f"{sid}.wav") if sid else None

        build_voice_jsons(
            None, audio_resolver, out_dir, n_clips,
            clip_interval_sec=clip_interval_sec,
            min_duration_sec=min_duration_sec,
            overwrite=overwrite, log_label=label,
            utterances=utts,
        )
        return

    if not os.path.exists(log_path):
        logger.warning("No dialogue/sessions.json or log.jsonl at %s; "
                       "writing empty voice JSONs.", unit_dir)
        os.makedirs(out_dir, exist_ok=True)
        for k in range(n_clips):
            with open(os.path.join(out_dir, f"clip_{k}_voices.json"), "w") as f:
                json.dump([], f)
        return

    # Legacy fallback path.
    events = load_dialogue_events(log_path)
    audio_dir = os.path.join(unit_dir, "dialogue_audio")

    def audio_resolver(session_id):
        return os.path.join(audio_dir, f"{session_id}.wav")

    build_voice_jsons(events, audio_resolver, out_dir, n_clips,
                      clip_interval_sec=clip_interval_sec,
                      min_duration_sec=min_duration_sec,
                      overwrite=overwrite, log_label=label)


def update_videograph_from_cache(video_graph, audios):
    """Fold one clip's cached voice JSON into ``video_graph``.

    Per-utterance: every entry in ``audios`` (one utterance) is matched
    against existing voice nodes via cosine similarity (≥
    ``audio_matching_threshold``); same-speaker utterances therefore
    typically *merge into the same graph voice node* without us having
    to know speaker names. New utterances that don't match any existing
    node create a fresh voice node.

    The clip-local prompt id (``<voice_X>``) is NOT decided here — we
    keep one id per utterance (assigned in voice-JSON order by
    ``_voice_speaker_grouping`` in
    ``m3_agent.simlife_precompute_unit``). Stage B's
    ``_build_voice_local_to_global`` then maps each utterance index to
    the matched_node we wrote here, so two utterances that the graph
    merged into one node both rewrite to the same ``<voice_node_id>`` in
    the final memory text.
    """
    id2audios = {}
    for audio in audios:
        audio_info = {
            "embeddings": [audio["embedding"]],
            "contents": [audio["asr"]],
        }
        matched_nodes = video_graph.search_voice_nodes(audio_info)
        if matched_nodes:
            matched_node = matched_nodes[0][0]
            video_graph.update_node(matched_node, audio_info)
        else:
            matched_node = video_graph.add_voice_node(audio_info)
        audio["matched_node"] = matched_node
        id2audios.setdefault(matched_node, []).append(audio)
    return id2audios


def process_voices_from_cache(video_graph, save_path):
    """Stage-B entry point: load a precomputed clip_K_voices.json and fold its
    voices into the given graph. Returns ``{matched_node_id: [audio, ...]}``.
    """
    if not os.path.exists(save_path):
        return {}
    with open(save_path) as f:
        audios = json.load(f)
    if not audios:
        return {}
    return update_videograph_from_cache(video_graph, audios)
