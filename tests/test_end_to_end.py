"""Whole flow offline: Twitch says live → record (fake stream) + chat → chunk analysis → clip →
Discord stand-in → approve → Post for Me (mocked) → offline."""

import asyncio
import shutil
import time

import pytest

from clipper.chat import ChatMessage
from clipper.config import Config, DetectionConfig, ClipConfig, RecordingConfig
from clipper.orchestrator import App
from clipper.store import POSTED
from clipper.twitch import LiveStream

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")


class FakeTwitch:
    def __init__(self):
        self.live = True

    async def get_live(self, logins):
        if not self.live:
            return {}
        return {"xqc": LiveStream("xqc", "xQc", "GTA 6 DAY 1", "Grand Theft Auto VI", 90000, "")}


class FakeNotifier:
    def __init__(self):
        self.texts, self.clips, self.refreshed = [], [], []

    async def start(self): ...
    async def close(self): ...

    async def send(self, text):
        self.texts.append(text)

    async def send_clip(self, clip):
        self.clips.append(clip)
        return 555

    async def refresh_clip(self, clip_id, new_video=False):
        self.refreshed.append((clip_id, new_video) if new_video else clip_id)


class FakePostForMe:
    def __init__(self):
        self.posts = []

    async def list_accounts(self):
        return [{"id": "spc_tiktok"}, {"id": "spc_yt"}]

    async def upload(self, path):
        assert path.exists()
        return "https://media.example/clip.mp4"

    async def create_post(self, caption, accounts, media_url, title="", external_id=None):
        self.posts.append((caption, accounts, media_url, title))
        return {"id": "sp_1"}

    async def wait_for_results(self, post_id, expected):
        return [{"success": True, "social_account_id": a, "platform_data": {"url": f"https://x/{a}"}}
                for a in ("spc_tiktok", "spc_yt")]


def fake_stream_cmd(seconds):
    return ["ffmpeg", "-hide_banner", "-loglevel", "error", "-re",
            "-f", "lavfi", "-i", f"testsrc2=size=640x360:rate=30:duration={seconds}",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
            "-c:v", "libx264", "-preset", "ultrafast", "-g", "30", "-c:a", "aac", "-f", "mpegts", "pipe:1"]


def test_live_to_posted(tmp_path, monkeypatch):
    cfg = Config(
        streamers=["xQc"], data_dir=tmp_path,
        recording=RecordingConfig(chunk_minutes=1),
        detection=DetectionConfig(laughter_model="off", chat_window_seconds=3, chat_baseline_seconds=20,
                                  min_seconds_between_clips=5),
        clips=ClipConfig(preroll_seconds=4, postroll_seconds=2, format="original"),
    )
    # 6-second chunks so the test runs quickly (chunk_minutes is whole minutes in config)
    monkeypatch.setattr("clipper.orchestrator.ChatRecorder.start", lambda self: None)  # chat injected below
    from clipper import orchestrator
    real_start = orchestrator.StreamerSession._start_recorder

    async def short_chunks(self):
        self.cfg.recording.chunk_minutes = 0.1
        self.source_cmd = fake_stream_cmd(16)
        await real_start(self)

    monkeypatch.setattr(orchestrator.StreamerSession, "_start_recorder", short_chunks)

    twitch, notifier, pfm = FakeTwitch(), FakeNotifier(), FakePostForMe()
    app = App(cfg, twitch=twitch, notifier=notifier, pfm=pfm)

    async def go():
        await app.poll()                       # goes live → session starts recording + chat
        session = app.sessions["xqc"]
        t0 = time.time()
        # steady chat, then a laughing burst ~9s in
        while time.time() - t0 < 15:
            el = time.time() - t0
            burst = 8.5 < el < 11
            for _ in range(8 if burst else 1):
                session.chat.messages.append(ChatMessage(time.time(), "v", "KEKW LMAO" if burst else "hi"))
            await asyncio.sleep(0.25)
        await asyncio.sleep(3)
        twitch.live = False
        await app.poll()                       # goes offline → final chunk processed in the background
        await asyncio.gather(*list(app._ending))
        return session

    session = asyncio.run(asyncio.wait_for(go(), 120))
    assert session.chunk_count >= 2
    assert "just went live" in notifier.texts[0] and "went offline" in notifier.texts[-1]
    assert len(notifier.clips) >= 1, notifier.texts
    clip = notifier.clips[0]
    assert clip.streamer == "xQc" and clip.details["chat_activity"].endswith("s)")
    assert app.store.get(clip.id).discord_message_id == 555

    # Approve in Discord → posted to every connected account
    assert app.store.claim_for_posting(clip.id)
    asyncio.run(app.publish(clip.id))
    done = app.store.get(clip.id)
    assert done.status == POSTED and len(done.post_results) == 2
    caption, accounts, media, title = pfm.posts[0]
    assert accounts == ["spc_tiktok", "spc_yt"] and "#xqc" in caption and title.startswith("xQc")


def test_retry_only_reposts_failed_accounts(tmp_path):
    class FlakyPFM(FakePostForMe):
        def __init__(self):
            super().__init__()
            self.calls = 0

        async def wait_for_results(self, post_id, expected):
            self.calls += 1
            accounts = self.posts[-1][1]
            return [{"success": not (a == "spc_yt" and self.calls == 1), "social_account_id": a,
                     "error": None if a != "spc_yt" else "YouTube quota", "platform_data": {}}
                    for a in accounts]

    clip_file = tmp_path / "c.mp4"
    clip_file.write_bytes(b"x")
    pfm = FlakyPFM()
    app = App(Config(streamers=["xQc"], data_dir=tmp_path), twitch=FakeTwitch(), notifier=FakeNotifier(), pfm=pfm)
    clip = app.store.add("xQc", clip_file, None, 1.0, 0.9, {}, {}, "cap", "title")

    app.store.claim_for_posting(clip.id)
    asyncio.run(app.publish(clip.id))
    assert app.store.get(clip.id).status == "failed"

    assert app.store.claim_for_posting(clip.id)   # Approve pressed again
    asyncio.run(app.publish(clip.id))
    final = app.store.get(clip.id)
    assert [p[1] for p in pfm.posts] == [["spc_tiktok", "spc_yt"], ["spc_yt"]]   # TikTok not posted twice
    assert final.status == "posted"
    assert sorted(r["social_account_id"] for r in final.post_results) == ["spc_tiktok", "spc_yt"]


def test_changing_text_on_video_rerenders_clip_and_preview(tmp_path):
    import subprocess
    from clipper.clips import EditStyle, cut_clip, text_free_path
    from clipper.recorder import Chunk
    from clipper.store import PENDING, RENDERING
    src = tmp_path / "src.ts"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30:duration=4",
                    "-f", "lavfi", "-i", "sine=duration=4", "-c:v", "libx264", "-preset", "ultrafast",
                    "-c:a", "aac", "-f", "mpegts", str(src)], check=True)
    clip_file = cut_clip([Chunk(src, 1000.0, 4.0)], 1000.5, 1003.5, tmp_path / "xqc_1.mp4",
                         style=EditStyle(title="xQc"), peak=1002.0)
    before = clip_file.read_bytes()
    notifier = FakeNotifier()
    cfg = Config(streamers=["xQc"], data_dir=tmp_path)
    app = App(cfg, twitch=FakeTwitch(), notifier=notifier, pfm=FakePostForMe())
    clip = app.store.add("xQc", clip_file, None, 1002.0, 0.9, {}, {}, "cap", "title", "xQc")
    app.store.update(clip.id, status=RENDERING)

    asyncio.run(app.change_video_text(clip.id, "NO WAY", PENDING))
    done = app.store.get(clip.id)
    assert done.video_text == "NO WAY" and done.status == PENDING
    assert clip_file.read_bytes() != before and text_free_path(clip_file).exists()
    assert done.preview_path == str(clip_file)   # small enough to attach as-is
    assert notifier.refreshed == [(clip.id, True)]
