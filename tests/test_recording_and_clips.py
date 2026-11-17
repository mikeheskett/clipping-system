"""Uses real ffmpeg: a synthetic MPEG-TS stream stands in for Streamlink."""

import asyncio
import shutil
import time

import pytest

from clipper.clips import chunks_for_window, cut_clip, make_preview, probe_duration
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
    assert any(s["codec_type"] == "audio" for s in info["streams"])


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
