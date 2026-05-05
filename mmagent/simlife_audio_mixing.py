"""Audio mixing helpers for SimLife clip generation.

SimLife video_units ship video.mp4 *without* an audio track. Real audio
lives in two side files:

  ambient_audio.wav         — ~24 min, 16 kHz mono; full-length non-speech
  dialogue_audio/session_NNN.wav  — short, 24 kHz mono; speech only,
                                    timestamped via log.jsonl rows where
                                    type == 'dialogue'.

Qwen2.5-Omni's memory pipeline runs with USE_AUDIO_IN_VIDEO=True, so clips
need an embedded audio track. We do this in two steps:

1. ``build_full_audio_track`` overlays every dialogue utterance onto a copy
   of ambient_audio.wav at its absolute unit-local timestamp
   (``dialogue.start_time + utterance.start_offset_sec``), producing one
   full-length WAV.
2. ``cut_clips_with_audio_segmented`` (or ``cut_single_clip_with_audio``)
   muxes that WAV with video.mp4 in a single ffmpeg pass.

The override flow reuses (1) with a custom ``audio_resolver`` that returns
override WAVs from ``task_dialogue_audio/<chain>/<unit>/<session>.wav``
when present, so the same code path produces per-chain clip MP4s with
the swapped dialogue.
"""
import logging
import os
import subprocess

import imageio_ffmpeg
from pydub import AudioSegment

from .simlife_voice_processing import load_dialogue_events

CLIP_INTERVAL_SEC = 30
logger = logging.getLogger(__name__)


def _default_audio_resolver(unit_dir):
    def resolver(session_id):
        return os.path.join(unit_dir, "dialogue_audio", f"{session_id}.wav")
    return resolver


def build_full_audio_track(unit_dir, out_path, *,
                           dialogue_events=None,
                           audio_resolver=None,
                           overwrite=False):
    """Mix ``ambient_audio.wav`` with dialogue utterances overlaid at their
    absolute unit-local timestamps.

    Args:
        unit_dir: SimLife ``video_units/<unit>/`` folder.
        out_path: Output WAV path. Idempotent unless ``overwrite=True``.
        dialogue_events: Optional pre-loaded events list. Defaults to the
            ``type==dialogue`` rows of ``<unit_dir>/log.jsonl``.
        audio_resolver: Optional ``session_id -> wav_path`` callable. Lets
            the override flow swap in chain-specific session WAVs while
            falling back to the unit's default ``dialogue_audio/`` for
            unaffected sessions.
        overwrite: re-mix even if ``out_path`` already exists.

    Returns:
        ``out_path`` on success.
    """
    if os.path.exists(out_path) and not overwrite:
        return out_path

    ambient_path = os.path.join(unit_dir, "ambient_audio.wav")
    if not os.path.exists(ambient_path):
        raise FileNotFoundError(ambient_path)

    track = AudioSegment.from_wav(ambient_path)

    if dialogue_events is None:
        dialogue_events = load_dialogue_events(os.path.join(unit_dir, "log.jsonl"))
    if audio_resolver is None:
        audio_resolver = _default_audio_resolver(unit_dir)

    session_cache = {}

    for ev in dialogue_events:
        sid = ev.get("session_id")
        if sid not in session_cache:
            wav_path = audio_resolver(sid)
            if not wav_path or not os.path.exists(wav_path):
                logger.warning("Missing session WAV for %s under %s", sid, unit_dir)
                session_cache[sid] = None
                continue
            sess = AudioSegment.from_wav(wav_path)
            if sess.frame_rate != track.frame_rate:
                sess = sess.set_frame_rate(track.frame_rate)
            if sess.channels != track.channels:
                sess = sess.set_channels(track.channels)
            session_cache[sid] = sess
        sess = session_cache[sid]
        if sess is None:
            continue

        dialogue_start = float(ev["start_time"])
        for utt in ev.get("utterances", []):
            offset = float(utt.get("start_offset_sec", 0.0))
            duration = float(utt.get("duration_sec", 0.0))
            if duration <= 0:
                continue
            abs_start_ms = int((dialogue_start + offset) * 1000)
            seg_start_ms = int(offset * 1000)
            seg_end_ms = min(seg_start_ms + int(duration * 1000), len(sess))
            if seg_end_ms <= seg_start_ms:
                continue
            seg = sess[seg_start_ms:seg_end_ms]
            track = track.overlay(seg, position=abs_start_ms)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    track.export(out_path, format="wav")
    return out_path


def cut_silent_clips_segmented(video_path, out_dir, interval=CLIP_INTERVAL_SEC):
    """Original (silent) segment muxer pass. We use this and then remux
    audio per-clip below — combining audio with the segment muxer in one
    pass triggers a known ffmpeg quirk where the first segment absorbs an
    extra keyframe interval (clip 0 ends up 2x length).
    """
    os.makedirs(out_dir, exist_ok=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    out_pattern = os.path.join(out_dir, "%d.mp4")
    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-i", video_path,
        "-an",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
        "-force_key_frames", f"expr:gte(t,n_forced*{interval})",
        "-reset_timestamps", "1",
        "-f", "segment",
        "-segment_time", str(interval),
        "-segment_format", "mp4",
        out_pattern,
    ]
    subprocess.run(cmd, check=True)


def remux_audio_into_clip(silent_clip, audio_path, out_clip,
                          audio_start_sec, audio_duration_sec):
    """Mux a slice of ``audio_path`` (``[audio_start_sec, +duration)``) onto
    ``silent_clip`` (video copied without re-encode), writing ``out_clip``.

    Uses ``-c:v copy -c:a aac`` so this is fast and only the audio is
    re-encoded.
    """
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    os.makedirs(os.path.dirname(out_clip), exist_ok=True)
    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-i", silent_clip,
        "-ss", str(audio_start_sec),
        "-t", str(audio_duration_sec),
        "-i", audio_path,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "128k",
        "-shortest",
        out_clip,
    ]
    subprocess.run(cmd, check=True)


def cut_clips_with_audio_segmented(video_path, audio_path, out_dir,
                                   interval=CLIP_INTERVAL_SEC):
    """Cut clips with audio, in two passes for reliability:

    1. ffmpeg segment muxer cuts the silent video into 30 s pieces. The
       segment muxer is well-behaved without an audio stream attached.
    2. For each clip, remux the corresponding ``audio_path`` slice with
       ``-c:v copy``. No video re-encode in pass 2; only the audio gets
       re-encoded as AAC.

    The intermediate silent clips are overwritten in place by the audio'd
    versions (so the final on-disk layout is identical to the existing
    one, just with an audio stream now present).
    """
    os.makedirs(out_dir, exist_ok=True)

    cut_silent_clips_segmented(video_path, out_dir, interval=interval)

    silent_paths = sorted(
        (os.path.join(out_dir, f) for f in os.listdir(out_dir)
         if f.endswith(".mp4") and f[:-4].isdigit()),
        key=lambda p: int(os.path.basename(p)[:-4]),
    )
    for k, sp in enumerate(silent_paths):
        # Remux into a sibling temp file, then atomically replace the silent
        # clip — keeps the final filename ``<K>.mp4``.
        tmp = sp + ".audio.mp4"
        try:
            remux_audio_into_clip(sp, audio_path, tmp,
                                  audio_start_sec=k * interval,
                                  audio_duration_sec=interval)
            os.replace(tmp, sp)
        except Exception:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            raise


def cut_single_clip_with_audio(video_path, audio_path, out_path, clip_index,
                               interval=CLIP_INTERVAL_SEC):
    """Cut clip K (covering ``[K*interval, (K+1)*interval)``) with audio.

    Used by the override flow: we typically only need a handful of clips
    per (chain, unit), so paying the segment-muxer's full re-encode cost
    is wasteful. Per-clip cuts seek to the start time and re-encode just
    that 30 s window.
    """
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    start = clip_index * interval
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-ss", str(start), "-i", video_path,
        "-ss", str(start), "-i", audio_path,
        "-t", str(interval),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
        "-c:a", "aac", "-b:a", "128k",
        "-shortest",
        out_path,
    ]
    subprocess.run(cmd, check=True)


def has_audio_stream(path):
    """Quick check: does this mp4 file carry any audio stream?

    We use ``ffmpeg -i`` (no output specified) and look for the
    ``Audio:`` substring in stderr, since imageio_ffmpeg ships ffmpeg
    only — no separate ffprobe binary.
    """
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    try:
        result = subprocess.run(
            [ffmpeg, "-i", path],
            capture_output=True, text=True, timeout=15,
        )
    except subprocess.TimeoutExpired:
        return False
    return "Audio:" in (result.stderr or "")
