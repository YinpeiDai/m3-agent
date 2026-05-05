"""Cut a SimLife video unit into 30-second MP4 clips with audio.

video.mp4 has no audio track in SimLife. We synthesize one by overlaying
every dialogue utterance (timestamped from log.jsonl) onto a copy of
ambient_audio.wav, then mux the result into the cut clips so Qwen-Omni's
``USE_AUDIO_IN_VIDEO=True`` path actually has speech to listen to.

Re-encodes video with libx264 ultrafast (forced keyframes at every
interval boundary so segments are frame-accurate) and audio with AAC.

Output: <out_dir>/0.mp4, 1.mp4, ... matching the int(filename) parsing in
m3_agent/memorization_intermediate_outputs.py:72.
"""
import argparse
import logging
import math
import os
import subprocess
import sys

import imageio_ffmpeg

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from mmagent.simlife_audio_mixing import (
    build_full_audio_track,
    cut_clips_with_audio_segmented,
    has_audio_stream,
)

CLIP_INTERVAL_SEC = 30
logger = logging.getLogger(__name__)


def _ffprobe_duration(ffmpeg, path):
    """Return float seconds, or None if the file can't be opened.

    We avoid ffprobe-as-separate-binary by using ffmpeg's null muxer; a
    well-formed mp4 prints "Duration: HH:MM:SS.ms" to stderr, a truncated
    one (no moov atom) errors out and returns None.
    """
    try:
        result = subprocess.run(
            [ffmpeg, "-v", "error", "-i", path, "-f", "null", "-"],
            capture_output=True, text=True, timeout=30,
        )
    except subprocess.TimeoutExpired:
        return None
    if result.returncode != 0:
        return None
    return 0.0  # readable; exact duration not needed for our checks


def _existing_clip_paths(out_dir):
    return sorted(
        (os.path.join(out_dir, f) for f in os.listdir(out_dir)
         if f.endswith(".mp4") and f[:-4].isdigit()),
        key=lambda p: int(os.path.basename(p)[:-4]),
    )


def _expected_clip_count(ffmpeg, video_path, interval):
    """Round-up of source duration / interval. Returns None on probe failure."""
    result = subprocess.run(
        [ffmpeg, "-v", "error", "-i", video_path, "-f", "null", "-"],
        capture_output=True, text=True,
    )
    # Both stderr and stdout may contain the Duration line.
    text = (result.stderr or "") + (result.stdout or "")
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("Duration:"):
            try:
                hms = line.split("Duration:", 1)[1].split(",", 1)[0].strip()
                h, m, s = hms.split(":")
                seconds = int(h) * 3600 + int(m) * 60 + float(s)
                return max(1, math.ceil(seconds / interval))
            except Exception:
                return None
    return None


def _cached_clips_are_valid(out_dir, video_path, interval, ffmpeg):
    """Quickly decide whether the existing clip set was finalized cleanly.

    Catches three failure modes:
      * partial cut (previous ffmpeg run killed mid-write — last segment
        lacks a moov atom and is unopenable),
      * count mismatch vs ``ceil(source_duration / interval)``,
      * legacy silent clips from before audio muxing landed.
    """
    existing = _existing_clip_paths(out_dir)
    if not existing:
        return False

    # All clips must be newer than the source video.
    src_mtime = os.path.getmtime(video_path)
    if any(os.path.getmtime(p) < src_mtime for p in existing):
        return False

    expected = _expected_clip_count(ffmpeg, video_path, interval)
    if expected is None or len(existing) != expected:
        return False

    # The last clip is the one most likely to be truncated.
    if _ffprobe_duration(ffmpeg, existing[-1]) is None:
        return False

    # Legacy silent clips — cheap to recut, do it now so Qwen-Omni's
    # USE_AUDIO_IN_VIDEO path actually has audio to consume.
    if not has_audio_stream(existing[0]):
        logger.info("[%s] existing clips have no audio track; will re-cut",
                    os.path.basename(out_dir.rstrip("/")))
        return False

    return True


def cut_clips(video_path, out_dir, interval=CLIP_INTERVAL_SEC, force=False,
              src_unit_dir=None, audio_track_path=None):
    """Cut ``video_path`` into 30 s clips, muxing in a synthesized audio track.

    Args:
        video_path: SimLife ``video_units/<unit>/video.mp4``.
        out_dir: Where to write ``0.mp4, 1.mp4, ...``.
        interval: Segment length in seconds.
        force: Re-cut even if the cache check passes.
        src_unit_dir: ``video_units/<unit>/`` directory holding
            ``ambient_audio.wav``, ``log.jsonl`` and ``dialogue_audio/``.
            Defaults to ``dirname(video_path)``.
        audio_track_path: Optional pre-built full-length audio WAV. If
            absent, we mix one from the unit's ambient + dialogue files
            into ``<out_dir>/_full_audio.wav``.
    """
    os.makedirs(out_dir, exist_ok=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    if not force and _cached_clips_are_valid(out_dir, video_path, interval, ffmpeg):
        return _existing_clip_paths(out_dir)

    # Wipe any stale/truncated clips so the segment muxer doesn't trip on
    # them and so a partial previous run doesn't leave residue behind.
    for p in _existing_clip_paths(out_dir):
        try:
            os.remove(p)
        except OSError:
            pass

    if src_unit_dir is None:
        src_unit_dir = os.path.dirname(os.path.abspath(video_path))

    if audio_track_path is None:
        audio_track_path = os.path.join(out_dir, "_full_audio.wav")
        build_full_audio_track(src_unit_dir, audio_track_path, overwrite=force)

    cut_clips_with_audio_segmented(video_path, audio_track_path, out_dir, interval=interval)

    out = _existing_clip_paths(out_dir)
    if out and _ffprobe_duration(ffmpeg, out[-1]) is None:
        raise RuntimeError(
            f"Last clip {out[-1]} appears truncated after ffmpeg run; check disk space"
        )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to source video.mp4")
    parser.add_argument("--out_dir", required=True, help="Directory for 0.mp4, 1.mp4, ...")
    parser.add_argument("--interval", type=int, default=CLIP_INTERVAL_SEC)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    clips = cut_clips(args.video, args.out_dir, interval=args.interval, force=args.force)
    print(f"wrote {len(clips)} clips to {args.out_dir}")


if __name__ == "__main__":
    main()
