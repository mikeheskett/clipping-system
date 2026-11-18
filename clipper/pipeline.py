"""Turn a finished chunk (+ chat) into clips. Shared by the live system and the `analyze` command."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from .analysis import Moment, pick_moments, score_chunk
from .audio import get_detector, load_audio
from .chat import ChatMessage
from .clips import EditStyle, can_edit_text, cut_clip, make_preview, retitle
from .config import Config
from .recorder import Chunk

log = logging.getLogger(__name__)


@dataclass
class ClipFile:
    moment: Moment
    path: Path
    preview: Path | None
    video_text: str = ""   # text drawn on the video (editable from Discord when a text-free copy exists)


def find_moments(chunk: Chunk, messages: list[ChatMessage] | None, cfg: Config,
                 max_moments: int | None = None) -> list[Moment]:
    """Score every second of a chunk and return the best moments (blocking: run in a thread)."""
    seconds = int(chunk.duration)
    if seconds <= 0:
        return []
    audio_laughter = None
    detector = get_detector(cfg.detection.laughter_model)
    if detector is not None:
        audio_laughter = detector.per_second(load_audio(chunk.path))
    combined, signals, raw = score_chunk(chunk.start, seconds, cfg.detection, messages, audio_laughter)
    moments = pick_moments(chunk.start, combined, signals, raw, cfg.detection, max_moments)
    log.info("%s: %d moment(s) above %.2f (best second scored %.2f)", chunk.path.name, len(moments),
             cfg.detection.threshold, float(combined.max()) if len(combined) else 0.0)
    return moments


def clip_window(m: Moment, cfg: Config) -> tuple[float, float]:
    return m.ts - cfg.clips.preroll_seconds, m.ts + cfg.clips.postroll_seconds


def edit_style(streamer: str, cfg: Config) -> EditStyle | None:
    c = cfg.clips
    if not c.edit:
        return None
    title = c.title.format(streamer=streamer).strip() if streamer and c.title else ""
    return EditStyle(title=title, punch_zoom=c.punch_zoom, progress_bar=c.progress_bar,
                     color_grade=c.color_grade, vary=c.vary_each_clip, font_file=c.font_file)


def render_clip(m: Moment, chunks: list[Chunk], out_dir: Path, stem: str, cfg: Config,
                streamer: str = "") -> ClipFile:
    """Cut the clip and its Discord preview (blocking)."""
    start, end = clip_window(m, cfg)
    style = edit_style(streamer, cfg)
    path = cut_clip(chunks, start, end, out_dir / f"{stem}.mp4", cfg.clips.format, style=style, peak=m.ts)
    preview = make_preview(path, out_dir / f"{stem}_preview.mp4", cfg.discord.preview_max_mb)
    text = style.title if style and can_edit_text(path) else ""
    return ClipFile(m, path, preview, text)


def rerender_text(clip_path: Path, text: str, streamer: str, cfg: Config) -> Path | None:
    """Redraw the text on a clip and remake its Discord preview (blocking). Returns the new preview."""
    style = edit_style(streamer, cfg) or EditStyle(vary=cfg.clips.vary_each_clip, font_file=cfg.clips.font_file)
    retitle(clip_path, text, style)
    return make_preview(clip_path, clip_path.with_name(f"{clip_path.stem}_preview.mp4"),
                        cfg.discord.preview_max_mb)


def caption_for(streamer: str, cfg: Config) -> tuple[str, str]:
    tag = "".join(ch for ch in streamer if ch.isalnum()).lower()
    fmt = {"streamer": streamer, "streamer_tag": tag}
    return cfg.postforme.caption_template.format(**fmt), cfg.postforme.title_template.format(**fmt)
