"""Step 2a: record a live stream with Streamlink into fixed-length chunks (default 10 minutes).

streamlink --stdout  ──pipe──▶  ffmpeg -f segment  ──▶  chunk_<run>_0000.ts, chunk_<run>_0001.ts, …

ffmpeg appends "file,start,end" (stream timestamps) to the run's segment list each time a chunk is
finished. The recorder checks that list every second and turns each new row into a Chunk with
wall-clock times, so chunks can be lined up with chat.

Wall-clock mapping: the first time a row appears, its end timestamp is "now" (the stream arrives in
real time once Streamlink is past its start-up burst). Every chunk in the run is placed relative to that
anchor, so chunks from one run are exactly contiguous.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class Chunk:
    path: Path
    start: float  # unix seconds (wall clock) of the chunk's first frame
    duration: float

    @property
    def end(self) -> float:
        return self.start + self.duration


def read_segment_rows(list_path: Path) -> list[tuple[str, float, float]]:
    """Complete rows of ffmpeg's CSV segment list as (filename, start, end)."""
    try:
        text = list_path.read_text()
    except FileNotFoundError:
        return []
    text = text[:text.rfind("\n") + 1]  # ignore a row ffmpeg is still writing
    rows = []
    for line in text.splitlines():
        parts = line.rsplit(",", 2)
        if len(parts) == 3:
            try:
                rows.append((parts[0].strip('"'), float(parts[1]), float(parts[2])))
            except ValueError:
                log.warning("skipping unreadable segment list row %r", line)
    return rows


def _interrupt(proc: asyncio.subprocess.Process):
    """Ask a process to stop. Windows can't send SIGINT to a child, so it is terminated instead;
    either way ffmpeg sees the pipe close and finishes the last chunk."""
    if sys.platform == "win32":
        proc.terminate()
    else:
        proc.send_signal(signal.SIGINT)


class Recorder:
    def __init__(self, login: str, chunk_dir: Path, quality: str, chunk_seconds: int,
                 queue: asyncio.Queue | None = None, source_cmd: list[str] | None = None):
        self.login = login
        self.chunk_dir = chunk_dir
        self.chunk_seconds = chunk_seconds
        self.queue: asyncio.Queue[Chunk] = queue if queue is not None else asyncio.Queue()
        # Any command that writes an MPEG-TS stream to stdout; Streamlink unless overridden (tests).
        self.source_cmd = source_cmd or [
            # Run Streamlink through Python rather than streamlink.exe: Windows app-control policies
            # (Smart App Control, AppLocker) often block the unsigned launcher but allow python.exe.
            sys.executable, "-m", "clipper.streamlink_runner", "--stdout", "--twitch-disable-ads", "--retry-open", "3", "--loglevel", "warning",
            f"https://www.twitch.tv/{login}", quality,
        ]
        self.list_path: Path | None = None
        self._streamlink: asyncio.subprocess.Process | None = None
        self._ffmpeg: asyncio.subprocess.Process | None = None
        self._watch_task: asyncio.Task | None = None
        self._rows_seen = 0
        self._anchor: tuple[float, float] | None = None  # (stream timestamp, wall clock)

    async def start(self):
        self.chunk_dir.mkdir(parents=True, exist_ok=True)
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        self.list_path = self.chunk_dir / f"segments_{run_id}.csv"

        read_fd, write_fd = os.pipe()
        try:
            self._streamlink = await asyncio.create_subprocess_exec(
                *self.source_cmd, stdout=write_fd, stderr=asyncio.subprocess.PIPE)
            self._ffmpeg = await asyncio.create_subprocess_exec(
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
                "-i", "pipe:0",
                "-map", "0:v:0?", "-map", "0:a:0?", "-c", "copy",
                "-f", "segment", "-segment_time", str(self.chunk_seconds), "-segment_format", "mpegts",
                "-segment_list", str(self.list_path), "-segment_list_type", "csv",
                str(self.chunk_dir / f"chunk_{run_id}_%04d.ts"),
                stdin=read_fd, stderr=asyncio.subprocess.PIPE)
        finally:
            os.close(read_fd)
            os.close(write_fd)
        asyncio.create_task(self._log_stderr(self._streamlink, "streamlink"))
        asyncio.create_task(self._log_stderr(self._ffmpeg, "ffmpeg"))
        self._watch_task = asyncio.create_task(self._watch_list())
        log.info("recording %s into %s", self.login, self.chunk_dir)

    def _scan(self):
        rows = read_segment_rows(self.list_path)
        now = time.time()
        for name, start, end in rows[self._rows_seen:]:
            if self._anchor is None:
                self._anchor = (end, now)
            ref_ts, ref_wall = self._anchor
            self.queue.put_nowait(Chunk(self.chunk_dir / name, ref_wall + (start - ref_ts), end - start))
        self._rows_seen = len(rows)

    async def _watch_list(self):
        while self._ffmpeg.returncode is None:
            self._scan()
            await asyncio.sleep(1)
        self._scan()  # the last chunk is listed when ffmpeg exits

    async def _log_stderr(self, proc: asyncio.subprocess.Process, name: str):
        async for line in proc.stderr:
            text = line.decode(errors="replace").strip()
            if text:
                log.info("%s[%s]: %s", name, self.login, text)

    async def wait(self) -> int:
        """Wait until the recording ends (stream ended, network failure, or stop()) and every
        finished chunk has been queued."""
        code = await self._ffmpeg.wait()
        await self._watch_task
        return code

    async def stop(self):
        """Stop streamlink; ffmpeg then sees end-of-input and finalises the last chunk."""
        if self._streamlink and self._streamlink.returncode is None:
            _interrupt(self._streamlink)
            try:
                await asyncio.wait_for(self._streamlink.wait(), 20)
            except asyncio.TimeoutError:
                self._streamlink.kill()
        if self._ffmpeg and self._ffmpeg.returncode is None:
            try:
                await asyncio.wait_for(self._ffmpeg.wait(), 30)
            except asyncio.TimeoutError:
                _interrupt(self._ffmpeg)
                await self._ffmpeg.wait()
        if self._watch_task:
            await self._watch_task
