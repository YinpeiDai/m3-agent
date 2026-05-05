"""SimLife voice processing.

Replaces the Gemini-based diarization in mmagent/voice_processing.py with a
log.jsonl + dialogue_audio/*.wav driven flow. Output JSON is bit-compatible
with the existing pipeline so stage-B graph assembly can reuse the same
update_videograph logic.

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
    """Read every ``type==dialogue`` row from a log.jsonl-shaped file.

    Used by both the per-unit pipeline (``video_units/<u>/log.jsonl``) and
    the per-chain override script
    (``task_dialogue_audio/<vc>/<u>/asr.jsonl``), which share this schema.
    Returns a list (not a generator) so callers can splice/replace events.
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


def iter_utterances(events):
    """Flatten dialogue events to per-utterance dicts in absolute (unit-local) time."""
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


def build_voice_jsons(events, audio_resolver, out_dir, n_clips,
                      clip_interval_sec=CLIP_INTERVAL_SEC, min_duration_sec=2,
                      overwrite=False, log_label="", only_clips=None):
    """Generic per-clip voice-JSON writer.

    Used by the default per-unit precompute (with log.jsonl + dialogue_audio)
    and by the per-chain override flow (with possibly merged events and
    override session WAVs). All slicing/embedding/format choices live here so
    both paths emit byte-compatible JSON for stage-B consumers.

    Args:
        events: list of dialogue-event dicts (same shape as log.jsonl rows).
        audio_resolver: callable ``session_id -> path-to-wav`` (or ``None``).
            Lets callers prefer override WAVs and fall back to defaults.
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

    write_set = None if only_clips is None else set(only_clips)

    buckets = {k: [] for k in range(n_clips)}
    skipped_short = 0
    for utt in iter_utterances(events):
        if utt["duration_sec"] < min_duration_sec:
            skipped_short += 1
            continue
        clip_idx = int(utt["abs_start_sec"] // clip_interval_sec)
        if 0 <= clip_idx < n_clips:
            buckets[clip_idx].append(utt)

    if skipped_short:
        logger.debug("[%s] skipped %d utterances < %ds", log_label, skipped_short, min_duration_sec)

    # Cache loaded session WAVs (one resolver call per session, decoded once).
    session_cache = {}

    def _get_segment(session_id):
        if session_id in session_cache:
            return session_cache[session_id]
        wav_path = audio_resolver(session_id)
        if not wav_path or not os.path.exists(wav_path):
            session_cache[session_id] = None
            return None
        seg = AudioSegment.from_wav(wav_path)
        session_cache[session_id] = seg
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
            seg = _get_segment(utt["session_id"])
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
                          min_duration_sec=2, overwrite=False):
    """Default per-unit pipeline: read log.jsonl + dialogue_audio/<sid>.wav.

    Empty units (no log.jsonl) still get empty per-clip JSONs so the absence
    is distinguishable from "not yet processed."
    """
    log_path = os.path.join(unit_dir, "log.jsonl")
    if not os.path.exists(log_path):
        logger.warning("No log.jsonl at %s; writing empty voice JSONs.", log_path)
        os.makedirs(out_dir, exist_ok=True)
        for k in range(n_clips):
            with open(os.path.join(out_dir, f"clip_{k}_voices.json"), "w") as f:
                json.dump([], f)
        return

    events = load_dialogue_events(log_path)

    audio_dir = os.path.join(unit_dir, "dialogue_audio")

    def audio_resolver(session_id):
        return os.path.join(audio_dir, f"{session_id}.wav")

    label = os.path.basename(unit_dir.rstrip("/"))
    build_voice_jsons(events, audio_resolver, out_dir, n_clips,
                      clip_interval_sec=clip_interval_sec,
                      min_duration_sec=min_duration_sec,
                      overwrite=overwrite, log_label=label)


def update_videograph_from_cache(video_graph, audios):
    """Mirror of mmagent/voice_processing.py:update_videograph (reimplemented to
    avoid that closure being unreachable from outside its parent function).
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
