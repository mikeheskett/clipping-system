"""Cut clips out of the recorded chunks with ffmpeg."""

from __future__ import annotations

import json
import logging
import os
import platform
import random
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .recorder import Chunk

log = logging.getLogger(__name__)

VERTICAL_FILTER = (
    "[0:v]split[a][b];"
    "[a]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=20:5[bg];"
    "[b]scale=1080:-2[fg];"
    "[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1,format=yuv420p[v]"
)


# Accent colours for the title card and progress bar; each clip picks one.
ACCENTS = ["0xFFD400", "0x00C8FF", "0xFF3D7F", "0x5BE33A", "0xFF8A00", "0x9B6BFF"]

FONT_CANDIDATES = [
    "C:/Windows/Fonts/impact.ttf", "C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/segoeuib.ttf",
    "/System/Library/Fonts/Supplementary/Impact.ttf", "/Library/Fonts/Impact.ttf",
    "/System/Library/Fonts/Supplementary/Arial Bold.ttf", "/Library/Fonts/Arial Bold.ttf",
    "/usr/share/fonts/truetype/msttcorefonts/Impact.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
]


@dataclass
class EditStyle:
    """Edits that make each clip look like an edit rather than a raw re-upload."""
    title: str = ""             # text in the top bar of a vertical clip ("" = none)
    punch_zoom: bool = True     # zoom in on the funny moment for a few seconds
    progress_bar: bool = True   # thin bar along the bottom that fills up as the clip plays
    color_grade: bool = True    # punchier contrast and colour
    vary: bool = True           # small random differences per clip (zoom, grade, colours, layout)
    font_file: str = ""         # font for the title ("" = find a bold system font)
    # Where the streamer's webcam is in the stream, as fractions of the frame (x, y, width, height).
    # With it, vertical clips are split: webcam zoomed in on top, gameplay below. Without it, the
    # gameplay sits on a blurred copy of itself.
    facecam: tuple | None = None
    title_seconds: float = 7.0  # how long the text stays on screen from the start (0 = whole clip)


@lru_cache(maxsize=1)
def has_drawtext() -> bool:
    try:
        out = subprocess.run(["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True).stdout
    except OSError:
        return False
    return " drawtext " in out


def find_font(preferred: str = "") -> Path | None:
    if preferred:
        return Path(preferred) if Path(preferred).is_file() else None
    windir = os.environ.get("WINDIR")
    candidates = list(FONT_CANDIDATES)
    if windir:
        candidates[:0] = [f"{windir}/Fonts/impact.ttf", f"{windir}/Fonts/arialbd.ttf"]
    for c in candidates:
        if Path(c).is_file():
            return Path(c)
    if platform.system() != "Windows" and shutil.which("fc-match"):
        out = subprocess.run(["fc-match", "-f", "%{file}", "sans:bold"], capture_output=True, text=True).stdout
        if out and Path(out).is_file():
            return Path(out)
    return None


@dataclass
class Look:
    """The randomised look of one clip. Seeded by the clip's file name, so it can be rebuilt later
    (to redraw the text on the video) and comes out exactly the same."""
    zoom: float
    contrast: float
    saturation: float
    gamma: float
    blur: int
    hue: float
    punch: float
    shift: int
    accent: str
    cam_h: int      # height of the webcam panel in the split layout (gameplay gets the rest)

    @classmethod
    def for_clip(cls, seed: str, vary: bool) -> "Look":
        rng = random.Random(seed)

        def pick(lo: float, hi: float, default: float) -> float:
            return round(rng.uniform(lo, hi), 3) if vary else default

        return cls(
            zoom=pick(1.0, 1.06, 1.03),            # base zoom on the gameplay
            contrast=pick(1.04, 1.12, 1.08),
            saturation=pick(1.10, 1.30, 1.2),
            gamma=pick(0.96, 1.03, 1.0),
            blur=int(pick(14, 28, 20)),
            hue=pick(-25, 25, 0),                  # tint of the blurred background
            punch=pick(1.18, 1.32, 1.25),
            shift=int(pick(-90, 90, 0)) // 2 * 2,  # move the gameplay up/down a little
            accent=rng.choice(ACCENTS) if vary else ACCENTS[0],
            cam_h=int(pick(740, 860, 800)) // 2 * 2,
        )


@dataclass
class TextSpot:
    """Where and how the text sits on a clip. Saved next to the text-free copy so the text can be
    redrawn identically even if the layout settings change later."""
    y: str            # drawtext y expression
    accent: str
    seconds: float    # 0 = whole clip

    def save(self, path: Path):
        path.write_text(json.dumps(self.__dict__), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "TextSpot | None":
        try:
            return cls(**json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError):
            return None


def text_free_path(clip: Path) -> Path:
    """Copy of an edited vertical clip without the text on it, kept so the text can be changed later."""
    return clip.with_name(clip.stem + "_notext.mp4")


def text_spot_path(clip: Path) -> Path:
    return clip.with_name(clip.stem + "_notext.json")


def can_edit_text(clip: Path) -> bool:
    return text_free_path(clip).exists()


def _text_font(style: EditStyle) -> Path | None:
    font = find_font(style.font_file)
    if font is None or not has_drawtext():
        return None
    return font


def _text_spot(look: Look, split: bool, seconds: float) -> TextSpot:
    if split:   # centred on the line between the webcam and the gameplay
        return TextSpot(y=f"{look.cam_h}-text_h/2", accent=look.accent, seconds=seconds)
    top_of_game = (1920 - int(1080 * look.zoom * 9 / 16)) // 2 + look.shift   # Twitch is 16:9
    return TextSpot(y=f"{top_of_game}-text_h-90", accent=look.accent, seconds=seconds)


def _drawtext(spot: TextSpot, text: str, font: Path, workdir: Path, t0: float = 0.0) -> str:
    """drawtext filter for the text on the video, shown from t0 for `spot.seconds`. Font and text are
    copied into `workdir`, which ffmpeg runs in, so no paths have to be escaped in the filter string."""
    font_name = "font" + font.suffix.lower()
    shutil.copyfile(font, workdir / font_name)
    (workdir / "title.txt").write_text(text, encoding="utf-8")
    size = max(48, min(96, int(1500 / max(len(text), 1))))
    timing = f":enable='lt(t,{t0 + spot.seconds:.3f})'" if spot.seconds > 0 else ""
    return (f"drawtext=fontfile={font_name}:textfile=title.txt:"
            f"fontsize={size}:fontcolor=white:borderw=5:bordercolor=black:"
            f"box=1:boxcolor={spot.accent}@0.9:boxborderw=26:x=(w-text_w)/2:y={spot.y}{timing}")


def _edited_filter(style: EditStyle, fmt: str, t0: float, duration: float, peak: float | None,
                   seed: str, workdir: Path, text_free: bool = False) -> str:
    """Build the filtergraph for an edited clip; its output is [v].

    Frames reach the filters with timestamps starting at the coarse seek point, so the clip itself runs
    from t0 to t0 + duration. With `text_free`, a second output [vnotext] has everything but the title."""
    look = Look.for_clip(seed, style.vary)
    vertical = fmt == "vertical"
    split = vertical and style.facecam is not None
    width = 1080 if vertical else 1920
    grade = (f",eq=contrast={look.contrast}:saturation={look.saturation}:gamma={look.gamma}"
             if style.color_grade else "")
    src = "[src_fg]" if vertical else "[0:v]"
    parts = ["[0:v]split[src_bg][src_fg]"] if vertical else []
    if split:
        # Gameplay fills the bottom panel edge to edge: scale to its height, keep the middle.
        game_h = 1920 - look.cam_h
        tall = int(round(game_h * look.zoom / 2)) * 2
        parts.append(f"{src}scale=-2:{tall},crop={width}:{game_h},setsar=1{grade}[g]")
    else:
        scaled = int(round(width * look.zoom / 2)) * 2
        crop = f"{width}:ih" if vertical else f"{width}:1080"
        parts.append(f"{src}scale={scaled}:-2,crop={crop},setsar=1{grade}[g]")
    fg = "[g]"
    if style.punch_zoom and peak is not None:
        # Hard punch-in: cut to a tighter crop from just before the moment until a few seconds after.
        a, b = t0 + max(0.0, peak - 0.4), t0 + min(duration, peak + 2.6)
        parts.append(f"{fg}split[g1][g2]")
        parts.append(f"[g2]crop=w=iw/{look.punch}:h=ih/{look.punch}:x=(iw-ow)/2:y=(ih-oh)*0.45,"
                     f"scale={width}:-2[gz]")
        parts.append(f"[g1][gz]overlay=(W-w)/2:(H-h)/2:enable='between(t,{a:.3f},{b:.3f})'[gp]")
        fg = "[gp]"

    out = fg
    if split:
        x, y, w, h = style.facecam
        parts.append(f"[src_bg]crop=w=iw*{w}:h=ih*{h}:x=iw*{x}:y=ih*{y},"
                     f"scale=1080:{look.cam_h}:force_original_aspect_ratio=increase,"
                     f"crop=1080:{look.cam_h},setsar=1{grade}[cam]")
        parts.append(f"[cam]{fg}vstack,drawbox=x=0:y={look.cam_h - 4}:w=iw:h=8:color={look.accent}:t=fill[v0]")
        out = "[v0]"
    elif vertical:
        bg_grade = f",hue=h={look.hue}" if style.vary else ""
        parts.append(f"[src_bg]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,"
                     f"boxblur={look.blur}:5,eq=brightness=-0.12:saturation=1.3{bg_grade}[bg]")
        parts.append(f"[bg]{fg}overlay=(W-w)/2:(H-h)/2+{look.shift},setsar=1[v0]")
        out = "[v0]"

    if style.progress_bar:
        parts.append(f"color=c={look.accent}:s={width}x14:r=30[bar]")
        parts.append(f"{out}[bar]overlay=x='-w+w*(t-{t0:.3f})/{duration:.3f}':y=H-h:shortest=1[v2]")
        out = "[v2]"

    text = ""
    font = None
    if vertical and (style.title or text_free):
        font = _text_font(style)
        if font is None:
            log.warning("no usable font or ffmpeg lacks drawtext: clips are made without text on them")
            text_free = False
        else:
            text = style.title

    spot = _text_spot(look, split, style.title_seconds)
    if text_free:
        parts.append(f"{out}split[vt][vn]")
        parts.append("[vn]format=yuv420p[vnotext]")
        spot.save(workdir / "spot.json")
        out = "[vt]"
    if text:
        parts.append(f"{out}{_drawtext(spot, text, font, workdir, t0)}[v1]")
        out = "[v1]"
    parts.append(f"{out}format=yuv420p[v]")
    return ";".join(parts)


def retitle(clip: Path, text: str, style: EditStyle) -> Path:
    """Redraw the text on an edited clip from its text-free copy, replacing the clip file."""
    base = text_free_path(clip)
    if not base.exists():
        raise FileNotFoundError(f"no text-free copy of {clip.name}, so its text can't be changed")
    text = text.strip()
    tmp_out = clip.with_name(clip.stem + "_retitle.mp4")
    if not text:
        shutil.copyfile(base, tmp_out)
    else:
        font = _text_font(style)
        if font is None:
            raise RuntimeError("no usable font, or ffmpeg was built without drawtext")
        with tempfile.TemporaryDirectory(prefix="clipedit_") as tmp:
            # Clips made before the spot was saved used the blurred layout with the text on all the time.
            spot = (TextSpot.load(text_spot_path(clip))
                    or _text_spot(Look.for_clip(clip.stem, style.vary), split=False, seconds=0))
            vf = _drawtext(spot, text, font, Path(tmp)) + ",format=yuv420p"
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                            "-i", str(base.resolve()), "-map", "0:v:0", "-map", "0:a:0?", "-vf", vf,
                            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-c:a", "copy",
                            "-movflags", "+faststart", str(tmp_out.resolve())],
                           check=True, capture_output=True, cwd=tmp)
    os.replace(tmp_out, clip)
    return clip


def chunks_for_window(chunks: list[Chunk], start: float, end: float) -> list[Chunk]:
    """Contiguous chunks overlapping [start, end), in time order."""
    return [c for c in sorted(chunks, key=lambda c: c.start) if c.end > start and c.start < end]


def probe_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        capture_output=True, check=True, text=True).stdout
    return float(json.loads(out)["format"]["duration"])


def cut_clip(chunks: list[Chunk], start: float, end: float, out_path: Path, fmt: str = "vertical",
             style: EditStyle | None = None, peak: float | None = None) -> Path:
    """Write the [start, end) wall-clock window to an mp4, joining chunks if the window crosses a boundary.

    Chunks from one recording keep continuous MPEG-TS timestamps, so they can be joined with ffmpeg's
    concat protocol and seeked as one stream.

    With a `style`, the clip is edited (title, punch-in at `peak`, colour grade, progress bar), and a
    vertical clip also gets a text-free copy (see `text_free_path`). If the edited render fails for any
    reason, the plain clip is made instead so no moment is lost."""
    parts = chunks_for_window(chunks, start, end)
    if not parts:
        raise ValueError("no recorded chunks cover this clip window")
    start = max(start, parts[0].start)
    end = min(end, parts[-1].end)
    offset = start - parts[0].start
    duration = end - start
    if duration <= 1:
        raise ValueError("clip window is too short")

    paths = [c.path.resolve() for c in parts]
    src = str(paths[0]) if len(paths) == 1 else "concat:" + "|".join(str(p) for p in paths)
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # MPEG-TS input seeking is byte-based and can land just past a keyframe, which drops all video
    # until the next one. So seek coarsely to 10 s early, then decode forward to the exact frame.
    coarse = max(0.0, offset - 10)
    fine = offset - coarse
    head = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y"]
    if coarse:
        head += ["-ss", f"{coarse:.3f}"]
    head += ["-i", src]

    def output(label: str, path: Path, crf: int) -> list[str]:
        # -ss/-t after -i trim each output separately, so they're repeated for every output file.
        return ["-ss", f"{fine:.3f}", "-t", f"{duration:.3f}", "-map", label, "-map", "0:a:0?",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
                "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-movflags", "+faststart", str(path)]

    base = text_free_path(out_path)
    base.unlink(missing_ok=True)
    text_spot_path(out_path).unlink(missing_ok=True)
    if style is not None:
        # Vertical clips also get a text-free copy so the text can be changed from Discord later.
        keep_base = fmt == "vertical"
        with tempfile.TemporaryDirectory(prefix="clipedit_") as tmp:
            try:
                graph = _edited_filter(style, fmt, fine, duration, None if peak is None else peak - start,
                                       out_path.stem, Path(tmp), text_free=keep_base)
                cmd = head + ["-filter_complex", graph] + output("[v]", out_path, 20)
                if "[vnotext]" in graph:
                    cmd += output("[vnotext]", base, 18)
                subprocess.run(cmd, check=True, capture_output=True, cwd=tmp)
                if base.exists() and (Path(tmp) / "spot.json").exists():
                    shutil.copyfile(Path(tmp) / "spot.json", text_spot_path(out_path))
                return out_path
            except subprocess.CalledProcessError as e:
                base.unlink(missing_ok=True)
                log.warning("edited render failed, making a plain clip instead: %s",
                            (e.stderr or b"").decode(errors="replace").strip()[-500:])

    if fmt == "vertical":
        cmd = head + ["-filter_complex", VERTICAL_FILTER] + output("[v]", out_path, 20)
    else:
        cmd = head + ["-vf", "format=yuv420p"] + output("0:v:0", out_path, 20)
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
