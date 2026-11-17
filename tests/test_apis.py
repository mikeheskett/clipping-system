import asyncio
import json

import httpx

from clipper.discord_bot import clip_embed, clip_view, modal_values
from clipper.postforme import PostForMe
from clipper.store import FAILED, PENDING, POSTED, Store
from clipper.twitch import TwitchClient


def run(coro):
    return asyncio.run(coro)


def test_twitch_live_and_token_refresh():
    calls = {"token": 0, "streams": 0}

    def handler(req: httpx.Request):
        if req.url.host == "id.twitch.tv":
            calls["token"] += 1
            return httpx.Response(200, json={"access_token": f"tok{calls['token']}", "expires_in": 3600})
        calls["streams"] += 1
        assert req.headers["Client-Id"] == "cid"
        if req.headers["Authorization"] == "Bearer tok1":
            return httpx.Response(401)
        assert req.url.params.get_list("user_login") == ["timthetatman", "xqc"]
        return httpx.Response(200, json={"data": [{
            "user_login": "xqc", "user_name": "xQc", "type": "live", "title": "GTA 6 day 1",
            "game_name": "Grand Theft Auto VI", "viewer_count": 81234, "started_at": "2026-11-17T18:00:00Z"}]})

    tw = TwitchClient("cid", "secret", http=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    live = run(tw.get_live(["TimTheTatman", "xQc"]))
    assert list(live) == ["xqc"] and live["xqc"].display_name == "xQc" and live["xqc"].viewers == 81234
    assert calls == {"token": 2, "streams": 2}


def test_postforme_upload_and_post(tmp_path):
    clip = tmp_path / "c.mp4"
    clip.write_bytes(b"\x00" * 1000)
    seen = []

    def handler(req: httpx.Request):
        seen.append((req.method, str(req.url)))
        if req.url.path == "/v1/media/create-upload-url":
            assert req.headers["Authorization"] == "Bearer key"
            return httpx.Response(200, json={"upload_url": "https://storage.example/up?sig=1",
                                             "media_url": "https://media.example/c.mp4"})
        if req.url.host == "storage.example":
            assert req.method == "PUT" and req.headers["Content-Type"] == "video/mp4"
            assert len(req.content) == 1000 and "Authorization" not in req.headers
            return httpx.Response(200)
        if req.url.path == "/v1/social-posts":
            body = json.loads(req.content)
            assert body["social_accounts"] == ["spc_1", "spc_2"]
            assert body["media"] == [{"url": "https://media.example/c.mp4"}]
            assert body["platform_configurations"]["youtube"]["title"] == "xQc moment"
            return httpx.Response(200, json={"id": "sp_123", "status": "processing"})
        if req.url.path == "/v1/social-post-results":
            assert req.url.params["post_id"] == "sp_123"
            return httpx.Response(200, json={"data": [
                {"success": True, "social_account_id": "spc_1", "platform_data": {"url": "https://tiktok.com/x"}},
                {"success": True, "social_account_id": "spc_2", "platform_data": {}}], "meta": {}})
        return httpx.Response(404)

    pfm = PostForMe("key", http=httpx.AsyncClient(transport=httpx.MockTransport(handler)))

    async def go():
        url = await pfm.upload(clip)
        post = await pfm.create_post("caption", ["spc_1", "spc_2"], url, "xQc moment")
        return post, await pfm.wait_for_results(post["id"], expected=2, interval=0)

    post, results = run(go())
    assert post["id"] == "sp_123" and len(results) == 2


def test_store_claim_prevents_double_post(tmp_path):
    s = Store(tmp_path / "db.sqlite")
    c = s.add("xQc", tmp_path / "c.mp4", None, 1.0, 0.8, {"chat_spike": 1}, {}, "cap", "title")
    assert c.status == PENDING
    assert s.claim_for_posting(c.id) is True
    assert s.claim_for_posting(c.id) is False
    s.update(c.id, status=FAILED)
    assert s.claim_for_posting(c.id) is True   # failed posts can be retried
    s.update(c.id, status=POSTED, post_results=[{"success": True}])
    assert s.get(c.id).post_results == [{"success": True}]


def test_discord_embed_and_buttons(tmp_path):
    s = Store(tmp_path / "db.sqlite")
    c = s.add("xQc", tmp_path / "c.mp4", None, 1_800_000_000, 0.82, {}, {"chat_activity": "4.1x normal"},
              "xQc moment 😂", "xQc moment")
    e = clip_embed(c)
    assert "clip #1" in e.title and any(f.value == "4.1x normal" for f in e.fields)
    ids = [b.custom_id for b in clip_view(c).children]
    assert ids == ["clip:approve:1", "clip:caption:1", "clip:reject:1"]
    s.update(c.id, status=POSTED)
    assert clip_view(s.get(c.id)) is None


def test_modal_values_both_layouts():
    rows = [{"type": 1, "components": [{"type": 4, "custom_id": "caption", "value": "new cap"}]},
            {"type": 18, "component": {"type": 4, "custom_id": "title", "value": "new title"}}]
    assert modal_values(rows) == {"caption": "new cap", "title": "new title"}


def test_discord_setup_errors_are_readable(tmp_path):
    import discord
    import pytest
    from clipper.discord_bot import DiscordBot, DiscordSetupError

    class Resp:
        status, reason = 403, "Forbidden"

    def make(start_behaviour, fetch_exc=None):
        bot = DiscordBot("tok", 42, [], Store(tmp_path / "d.sqlite"), on_approve=None)

        async def fake_start(token):
            if start_behaviour == "bad-token":
                raise discord.LoginFailure("Improper token has been passed.")
            await bot.on_ready()
            await asyncio.sleep(3600)

        async def fake_fetch(cid):
            raise fetch_exc

        bot.client.start = fake_start
        bot.client.get_channel = lambda cid: None
        bot.client.fetch_channel = fake_fetch
        bot.client.close = lambda: asyncio.sleep(0)
        return bot

    with pytest.raises(DiscordSetupError, match="rejected the bot token"):
        asyncio.run(make("bad-token").start())
    with pytest.raises(DiscordSetupError, match="can't see channel 42"):
        asyncio.run(make("ok", discord.Forbidden(Resp(), {"code": 50001, "message": "Missing Access"})).start())
    with pytest.raises(DiscordSetupError, match="No channel with ID 42"):
        asyncio.run(make("ok", discord.NotFound(Resp(), "Unknown Channel")).start())
