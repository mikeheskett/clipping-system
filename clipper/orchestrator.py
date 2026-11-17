"""Runs everything: poll Twitch → record + read chat while live → analyse each 10-minute chunk →
cut clips → Discord review → Post for Me on approval."""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from .chat import ChatRecorder
from .config import Config
from .discord_bot import DiscordBot, LogOnlyNotifier
from .pipeline import caption_for, clip_window, find_moments, render_clip
from .postforme import PostForMe
from .recorder import Chunk, Recorder
from .store import FAILED, POSTED, Store
from .twitch import LiveStream, TwitchClient

log = logging.getLogger(__name__)

# Analysis is CPU-heavy; one worker keeps it from starving the recorders when several streams run.
ANALYSIS_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="analysis")
RENDER_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="render")


class StreamerSession:
    """One live stream: recorder + chat logger + a worker that analyses each finished chunk."""

    def __init__(self, app: "App", stream: LiveStream):
        self.app = app
        self.cfg = app.cfg
        self.stream = stream
        self.name = stream.display_name
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        self.dir = self.cfg.data_dir / "streams" / stream.login / stamp
        self.chunk_dir = self.dir / "chunks"
        self.clip_dir = self.cfg.data_dir / "clips" / stream.login
        self.chunk_dir.mkdir(parents=True, exist_ok=True)
        self.chat = ChatRecorder(stream.login, self.dir / "chat.jsonl",
                                 keep_seconds=self.cfg.detection.chat_baseline_seconds + 3600)
        self.queue: asyncio.Queue = asyncio.Queue()   # finished chunks from every recorder run
        self.recorder: Recorder | None = None
        self.source_cmd: list[str] | None = None      # tests swap Streamlink for a fake stream
        self.chunks: list[Chunk] = []                 # chunks of the current recorder run, in order
        self.pending: list = []                       # moments waiting for their post-roll to be recorded
        self.chunk_count = 0
        self.clip_count = 0
        self._stopping = False
        self._supervisor: asyncio.Task | None = None
        self._worker: asyncio.Task | None = None

    async def start(self):
        self.chat.start()
        await self._start_recorder()
        self._supervisor = asyncio.create_task(self._supervise(), name=f"supervise-{self.stream.login}")
        self._worker = asyncio.create_task(self._work(), name=f"chunks-{self.stream.login}")

    async def _start_recorder(self):
        self.recorder = Recorder(self.stream.login, self.chunk_dir, self.cfg.recording.quality,
                                 self.cfg.recording.chunk_minutes * 60, self.queue, self.source_cmd)
        self.queue.put_nowait("new-run")  # before start(), so it is queued ahead of the run's chunks
        await self.recorder.start()

    async def _supervise(self):
        """Restart the recording if it drops while Twitch still says the channel is live."""
        while True:
            code = await self.recorder.wait()
            if self._stopping:
                return
            log.warning("recording of %s exited (code %s); restarting in 15s", self.name, code)
            await asyncio.sleep(15)
            if self._stopping:
                return
            await self._start_recorder()

    async def _work(self):
        while True:
            item = await self.queue.get()
            if item is None:            # stop() has finished recording: cut whatever is left
                await self._render_ready(force=True)
                return
            if item == "new-run":       # timestamps restart with each run; clips can't span runs
                await self._render_ready(force=True)
                self.chunks = []
                continue
            await self._process_chunk(item)

    async def _process_chunk(self, chunk: Chunk):
        self.chunks.append(chunk)
        self.chunk_count += 1
        log.info("%s: chunk ready %s (%.0fs)", self.name, chunk.path.name, chunk.duration)
        # Chat reacts after the moment: give it a few more seconds before scoring the chunk's tail.
        wait = chunk.end + self.cfg.detection.chat_window_seconds - time.time()
        if wait > 0 and not self._stopping:
            await asyncio.sleep(wait)
        msgs = self.chat.between(chunk.start - self.cfg.detection.chat_baseline_seconds,
                                 chunk.end + self.cfg.detection.chat_window_seconds)
        loop = asyncio.get_running_loop()
        try:
            moments = await loop.run_in_executor(ANALYSIS_POOL, find_moments, chunk, msgs, self.cfg)
        except Exception:
            log.exception("%s: analysis failed for %s", self.name, chunk.path.name)
            moments = []
        self.pending.extend(moments)
        await self._render_ready()

    async def _render_ready(self, force: bool = False):
        """Cut every pending moment whose post-roll has been recorded (or all of them if forced)."""
        recorded_until = self.chunks[-1].end if self.chunks else 0
        still = []
        for m in self.pending:
            _, end = clip_window(m, self.cfg)
            if end <= recorded_until or force:
                await self._make_clip(m)
            else:
                still.append(m)
        self.pending = still

    async def _make_clip(self, m):
        stem = f"{self.stream.login}_{datetime.fromtimestamp(m.ts, timezone.utc):%Y%m%d-%H%M%S}"
        loop = asyncio.get_running_loop()
        try:
            cf = await loop.run_in_executor(RENDER_POOL, render_clip, m, list(self.chunks),
                                            self.clip_dir, stem, self.cfg)
        except Exception:
            log.exception("%s: failed to cut clip at %s", self.name, m.ts)
            return
        self.clip_count += 1
        await self.app.new_clip(self.name, cf)

    async def stop(self):
        self._stopping = True
        if self.recorder:
            await self.recorder.stop()      # queues the final chunk
        if self._supervisor:
            await asyncio.gather(self._supervisor, return_exceptions=True)
        self.queue.put_nowait(None)
        if self._worker:
            await asyncio.gather(self._worker, return_exceptions=True)
        await self.chat.stop()


class App:
    def __init__(self, cfg: Config, twitch=None, notifier=None, pfm=None):
        self.cfg = cfg
        self.store = Store(cfg.data_dir / "clips.db")
        self.sessions: dict[str, StreamerSession] = {}
        self._ending: set[asyncio.Task] = set()
        self.twitch = twitch or TwitchClient(cfg.secrets.twitch_client_id, cfg.secrets.twitch_client_secret)
        self.pfm = pfm or (PostForMe(cfg.secrets.post_for_me_api_key) if cfg.secrets.post_for_me_api_key else None)
        if notifier is not None:
            self.notifier = notifier
        elif cfg.secrets.discord_bot_token and cfg.discord.channel_id:
            self.notifier = DiscordBot(cfg.secrets.discord_bot_token, cfg.discord.channel_id,
                                       cfg.discord.approver_user_ids, self.store, self.publish)
        else:
            log.warning("DISCORD_BOT_TOKEN or discord.channel_id not set: notifications go to the log only")
            self.notifier = LogOnlyNotifier()

    async def run(self):
        await self.notifier.start()
        names = ", ".join(self.cfg.streamers)
        await self._notify(f"👀 Clipping system online. Watching: {names}")
        asyncio.create_task(self._cleanup_loop())
        try:
            while True:
                try:
                    await self.poll()
                except Exception:
                    log.exception("live check failed")
                await asyncio.sleep(self.cfg.poll_interval_seconds)
        finally:
            for s in list(self.sessions.values()):
                await s.stop()
            await asyncio.gather(*self._ending, return_exceptions=True)
            await self.notifier.close()

    async def poll(self):
        live = await self.twitch.get_live(self.cfg.streamers)
        for login, stream in live.items():
            if login not in self.sessions:
                session = StreamerSession(self, stream)
                self.sessions[login] = session
                await session.start()
                await self._notify(
                    f"🔴 **{stream.display_name}** just went live — *{stream.title}* ({stream.game}, "
                    f"{stream.viewers:,} viewers). Beginning to record the stream.\n"
                    f"https://twitch.tv/{login}")
        for login in [l for l in self.sessions if l not in live]:
            # Finishing the last chunk takes a minute or two; don't hold up live checks for the others.
            self._ending.add(asyncio.create_task(self._end_session(self.sessions.pop(login))))

    async def _end_session(self, session: StreamerSession):
        try:
            await session.stop()
        finally:
            self._ending.discard(asyncio.current_task())
        await self._notify(f"⚫ **{session.name}** went offline. {session.clip_count} clip(s) "
                           f"from {session.chunk_count} recorded chunk(s).")

    async def _notify(self, text: str):
        try:
            await self.notifier.send(text)
        except Exception:
            log.exception("could not send notification: %s", text)

    async def new_clip(self, streamer: str, cf):
        caption, title = caption_for(streamer, self.cfg)
        clip = self.store.add(streamer, cf.path, cf.preview, cf.moment.ts, cf.moment.score,
                              cf.moment.signals, cf.moment.details, caption, title)
        try:
            msg_id = await self.notifier.send_clip(clip)
            self.store.update(clip.id, discord_message_id=msg_id or None)
        except Exception:
            log.exception("could not send clip %s to Discord", clip.id)

    async def publish(self, clip_id: int):
        """Called when a clip is approved in Discord: upload to Post for Me and post everywhere."""
        clip = self.store.get(clip_id)
        try:
            if not self.pfm:
                raise RuntimeError("POST_FOR_ME_API_KEY is not set")
            accounts = self.cfg.postforme.social_account_ids
            if not accounts:
                accounts = [a["id"] for a in await self.pfm.list_accounts()]
            if not accounts:
                raise RuntimeError("no connected social accounts in Post for Me")
            # On a retry, only post to the accounts that failed last time (no duplicate posts).
            succeeded = [r for r in clip.post_results if r.get("success")]
            done_ids = {r.get("social_account_id") for r in succeeded}
            accounts = [a for a in accounts if a not in done_ids]
            if not accounts:
                self.store.update(clip_id, status=POSTED)
                await self.notifier.refresh_clip(clip_id)
                return
            media_url = await self.pfm.upload(Path(clip.path))
            post = await self.pfm.create_post(clip.caption, accounts, media_url, clip.title,
                                              external_id=f"clip-{clip.id}")
            self.store.update(clip_id, post_id=post.get("id"))
            await self.notifier.refresh_clip(clip_id)
            results = await self.pfm.wait_for_results(post["id"], expected=len(accounts))
            # Post for Me reports results as each platform finishes; none after the wait means still processing.
            ok = all(r.get("success") for r in results)
            self.store.update(clip_id, post_results=succeeded + results, status=POSTED if ok else FAILED)
        except Exception as e:
            log.exception("posting clip %s failed", clip_id)
            kept = [r for r in clip.post_results if r.get("success")]
            self.store.update(clip_id, status=FAILED,
                              post_results=kept + [{"success": False, "error": str(e)}])
        await self.notifier.refresh_clip(clip_id)

    async def _cleanup_loop(self):
        """Delete recorded chunks older than recording.keep_chunks_hours (clips are kept)."""
        keep = self.cfg.recording.keep_chunks_hours * 3600
        while True:
            cutoff = time.time() - keep
            for p in (self.cfg.data_dir / "streams").glob("*/*/chunks/chunk_*.ts"):
                try:
                    if p.stat().st_mtime < cutoff:
                        p.unlink()
                except FileNotFoundError:
                    pass
            await asyncio.sleep(600)
