"""Uses real ffmpeg: a synthetic MPEG-TS stream stands in for Streamlink."""

import asyncio
import shutil
import time

import pytest

from clipper.clips import (EditStyle, _edited_filter, chunks_for_window, cut_clip, make_preview, probe_duration,
                          retitle, text_free_path)
from clipper.recorder import Recorder

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")


def fake_stream_cmd(seconds: int):
    return ["ffmpeg", "-hide_banner", "-loglevel", "error", "-re",
            "-f", "lavfi", "-i", f"testsrc2=size=640x360:rate=30:duration={seconds}",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
            "-c:v", "libx264", "-preset", "ultrafast", "-g", "30", "-c:a", "aac", "-f", "mpegts", "pipe:1"]


@pytest.fixture(scope="module")
def recorded(tmp_path_factory):
    chunk_dir = tmp_path_factory.mktemp("chunks")

    async def go():
        rec = Recorder("tester", chunk_dir, "best", chunk_seconds=4, source_cmd=fake_stream_cmd(11))
        await rec.start()
        await asyncio.wait_for(rec.wait(), 60)
        finished = time.time()
        chunks = []
        while not rec.queue.empty():
            chunks.append(rec.queue.get_nowait())
        return chunks, finished

    return asyncio.run(go())


def test_recorder_writes_contiguous_timed_chunks(recorded):
    chunks, finished = recorded
    assert len(chunks) == 3
    assert len({c.path for c in chunks}) == 3 and all(c.path.exists() for c in chunks)
    assert 3.5 <= chunks[0].duration <= 4.5
    assert abs(sum(c.duration for c in chunks) - 11) < 0.6
    for a, b in zip(chunks, chunks[1:]):
        assert abs(b.start - a.end) < 0.1          # back to back on the wall clock
    assert abs(chunks[-1].end - finished) < 2.5     # anchored to real time


def test_clip_across_chunk_boundary(recorded, tmp_path):
    chunks, _ = recorded
    t0 = chunks[0].start
    start, end = t0 + 2.0, t0 + 9.0
    assert len(chunks_for_window(chunks, start, end)) == 3
    out = cut_clip(chunks, start, end, tmp_path / "clip.mp4", fmt="vertical")
    assert abs(probe_duration(out) - 7.0) < 0.3
    import json, subprocess
    info = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(out)],
                                     capture_output=True, text=True).stdout)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    assert (v["width"], v["height"]) == (1080, 1920)
    assert v.get("sample_aspect_ratio", "1:1") == "1:1"   # square pixels: YouTube Shorts needs a true 9:16
    assert any(s["codec_type"] == "audio" for s in info["streams"])


def test_edited_clip_is_vertical_with_audio(recorded, tmp_path):
    chunks, _ = recorded
    t0 = chunks[0].start
    out = cut_clip(chunks, t0 + 1.0, t0 + 6.0, tmp_path / "edited.mp4", fmt="vertical",
                   style=EditStyle(title="TheBurntPeanut"), peak=t0 + 4.0)
    assert abs(probe_duration(out) - 5.0) < 0.3
    import json, subprocess
    info = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(out)],
                                     capture_output=True, text=True).stdout)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    assert (v["width"], v["height"]) == (1080, 1920)
    assert any(s["codec_type"] == "audio" for s in info["streams"])


def title_area_hash(path):
    """Hash of the top part of the 1 s frame, where the text on the video sits."""
    import hashlib, subprocess
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", "1", "-i", str(path), "-frames:v", "1",
                          "-vf", "crop=1080:700:0:0,scale=108:70", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                         capture_output=True, check=True).stdout
    return hashlib.md5(bytes(b // 32 for b in raw)).hexdigest()


def test_text_on_video_can_be_changed(recorded, tmp_path):
    chunks, _ = recorded
    t0 = chunks[0].start
    clip = cut_clip(chunks, t0 + 1.0, t0 + 4.0, tmp_path / "xqc_1.mp4", fmt="vertical",
                    style=EditStyle(title="xQc"), peak=t0 + 2.0)
    base = text_free_path(clip)
    assert base.exists()
    original = title_area_hash(clip)
    assert original != title_area_hash(base)                     # the copy really has no text

    retitle(clip, "BEST MOMENT EVER", EditStyle())
    assert title_area_hash(clip) not in (original, title_area_hash(base))
    assert abs(probe_duration(clip) - 3.0) < 0.3

    retitle(clip, "", EditStyle())                               # empty text = no text at all
    assert title_area_hash(clip) == title_area_hash(base)


def test_plain_and_landscape_clips_have_no_text_free_copy(recorded, tmp_path):
    chunks, _ = recorded
    t0 = chunks[0].start
    plain = cut_clip(chunks, t0, t0 + 2, tmp_path / "plain.mp4", fmt="vertical")
    wide = cut_clip(chunks, t0, t0 + 2, tmp_path / "wide.mp4", fmt="original", style=EditStyle(title="x"))
    assert not text_free_path(plain).exists() and not text_free_path(wide).exists()


def test_each_clip_gets_its_own_look(tmp_path):
    a = _edited_filter(EditStyle(), "vertical", 0.0, 60.0, 40.0, "xqc_20261118-200000", tmp_path)
    b = _edited_filter(EditStyle(), "vertical", 0.0, 60.0, 40.0, "xqc_20261118-201500", tmp_path)
    again = _edited_filter(EditStyle(), "vertical", 0.0, 60.0, 40.0, "xqc_20261118-200000", tmp_path)
    assert a != b and a == again


def test_edited_render_falls_back_to_plain_clip(recorded, tmp_path, monkeypatch):
    import clipper.clips as clips
    monkeypatch.setattr(clips, "_edited_filter", lambda *a, **k: "[0:v]nosuchfilter[v]")
    chunks, _ = recorded
    out = cut_clip(chunks, chunks[0].start, chunks[0].start + 3, tmp_path / "f.mp4", style=EditStyle())
    assert 2.5 < probe_duration(out) < 3.5


def test_clip_window_is_clamped_to_recording(recorded, tmp_path):
    chunks, _ = recorded
    out = cut_clip(chunks, chunks[0].start - 40, chunks[0].start + 3, tmp_path / "early.mp4", fmt="original")
    assert 2.5 < probe_duration(out) < 3.5


def test_preview_fits_size_limit(recorded, tmp_path):
    chunks, _ = recorded
    clip = cut_clip(chunks, chunks[0].start, chunks[0].start + 8, tmp_path / "c.mp4")
    assert make_preview(clip, tmp_path / "p.mp4", max_mb=50) == clip   # already small enough
    small = make_preview(clip, tmp_path / "p.mp4", max_mb=0.4)
    assert small is not None and small.stat().st_size <= 0.4 * 1024 * 1024


def video_frames(path):
    import subprocess
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                          "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True).stdout.strip()
    return int(out or 0)


@pytest.mark.parametrize("gop,offset", [(300, 0.5), (60, 20.0), (60, 25.3)])
def test_cut_always_contains_video(tmp_path, gop, offset):
    """Regression: input seeking in MPEG-TS once produced audio-only clips."""
    import subprocess
    from clipper.recorder import Chunk
    src = tmp_path / "src.ts"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30:duration=30",
                    "-f", "lavfi", "-i", "sine=duration=30", "-c:v", "libx264", "-preset", "ultrafast",
                    "-g", str(gop), "-c:a", "aac", "-f", "mpegts", str(src)], check=True)
    out = cut_clip([Chunk(src, 1000.0, 30.0)], 1000 + offset, 1000 + offset + 3, tmp_path / "c.mp4")
    assert video_frames(out) >= 85  # ~3 s at 30 fps


def test_streamlink_records_hls_when_pycryptodome_is_blocked(tmp_path):
    """Windows Smart App Control can block pycryptodome's .pyd files; Streamlink must still record HLS."""
    import functools
    import http.server
    import os
    import subprocess
    import sys
    import threading

    hls = tmp_path / "hls"
    hls.mkdir()
    subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30:duration=6",
                    "-c:v", "libx264", "-preset", "ultrafast", "-g", "60", "-f", "hls", "-hls_time", "2",
                    "-hls_playlist_type", "vod", str(hls / "index.m3u8")], check=True)
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(hls))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        out = subprocess.run(
            [sys.executable, "-m", "clipper.streamlink_runner", "--stdout",
             f"hls://http://127.0.0.1:{server.server_port}/index.m3u8", "best"],
            capture_output=True, timeout=60, env={**os.environ, "CLIPPER_FORCE_CRYPTO_PLACEHOLDER": "1"})
    finally:
        server.shutdown()
    assert out.returncode == 0, out.stderr.decode()[-1000:]
    assert b"continuing without it" in out.stderr
    (tmp_path / "out.ts").write_bytes(out.stdout)
    assert abs(probe_duration(tmp_path / "out.ts") - 6) < 0.5
