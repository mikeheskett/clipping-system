"""Cut clips out of the recorded chunks with ffmpeg."""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

from .recorder import Chunk

log = logging.getLogger(__name__)

VERTICAL_FILTER = (
    "[0:v]split[a][b];"
    "[a]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=20:5[bg];"
    "[b]scale=1080:-2[fg];"
    "[bg][fg]overlay=(W-w)/2:(H-h)/2,format=yuv420p[v]"
)


def chunks_for_window(chunks: list[Chunk], start: float, end: float) -> list[Chunk]:
    """Contiguous chunks overlapping [start, end), in time order."""
    return [c for c in sorted(chunks, key=lambda c: c.start) if c.end > start and c.start < end]


def probe_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        capture_output=True, check=True, text=True).stdout
    return float(json.loads(out)["format"]["duration"])


def cut_clip(chunks: list[Chunk], start: float, end: float, out_path: Path, fmt: str = "vertical") -> Path:
    """Write the [start, end) wall-clock window to an mp4, joining chunks if the window crosses a boundary.

    Chunks from one recording keep continuous MPEG-TS timestamps, so they can be joined with ffmpeg's
    concat protocol and seeked as one stream."""
    parts = chunks_for_window(chunks, start, end)
    if not parts:
        raise ValueError("no recorded chunks cover this clip window")
    start = max(start, parts[0].start)
    end = min(end, parts[-1].end)
    offset = start - parts[0].start
    duration = end - start
    if duration <= 1:
        raise ValueError("clip window is too short")

    src = parts[0].path if len(parts) == 1 else "concat:" + "|".join(str(c.path) for c in parts)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # MPEG-TS input seeking is byte-based and can land just past a keyframe, which drops all video
    # until the next one. So seek coarsely to 10 s early, then decode forward to the exact frame.
    coarse = max(0.0, offset - 10)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y"]
    if coarse:
        cmd += ["-ss", f"{coarse:.3f}"]
    cmd += ["-i", str(src), "-ss", f"{offset - coarse:.3f}", "-t", f"{duration:.3f}"]
    if fmt == "vertical":
        cmd += ["-filter_complex", VERTICAL_FILTER, "-map", "[v]", "-map", "0:a:0?"]
    else:
        cmd += ["-map", "0:v:0", "-map", "0:a:0?", "-vf", "format=yuv420p"]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-movflags", "+faststart", str(out_path)]
    subprocess.run(cmd, check=True, capture_output=True)
    return out_path


def make_preview(clip: Path, out_path: Path, max_mb: float) -> Path | None:
    """Return a file small enough to attach to Discord: the clip itself if it fits, else a re-encode."""
    limit = max_mb * 1024 * 1024
    if clip.stat().st_size <= limit:
        return clip
    duration = probe_duration(clip)
    audio_kbps = 96
    video_kbps = int(limit * 8 / 1000 / duration * 0.92) - audio_kbps
    if video_kbps < 150:
        log.warning("clip %s too long for a %s MB preview", clip.name, max_mb)
        return None
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i", str(clip),
         "-vf", "scale=-2:'min(960,ih)'", "-r", "30",
         "-c:v", "libx264", "-preset", "veryfast", "-b:v", f"{video_kbps}k",
         "-maxrate", f"{video_kbps}k", "-bufsize", f"{video_kbps * 2}k",
         "-c:a", "aac", "-b:a", f"{audio_kbps}k", "-movflags", "+faststart", str(out_path)],
        check=True, capture_output=True)
    return out_path if out_path.stat().st_size <= limit else None
