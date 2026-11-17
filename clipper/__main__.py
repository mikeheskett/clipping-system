"""python -m clipper {run,check,accounts,analyze}"""

from __future__ import annotations

import argparse
import asyncio
import logging
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from .config import load_config


def cmd_run(cfg, args):
    from .orchestrator import App
    if not cfg.streamers:
        sys.exit("No streamers in config.yaml (copy config.example.yaml to config.yaml)")
    if not (cfg.secrets.twitch_client_id and cfg.secrets.twitch_client_secret):
        sys.exit("Set TWITCH_CLIENT_ID and TWITCH_CLIENT_SECRET in .env (see README)")
    try:
        asyncio.run(App(cfg).run())
    except KeyboardInterrupt:
        pass


async def _check(cfg):
    ok = True
    for tool in ("ffmpeg", "ffprobe", "streamlink"):
        found = shutil.which(tool)
        print(f"{'✓' if found else '✗'} {tool}: {found or 'not found on PATH'}")
        ok &= bool(found)

    from .twitch import TwitchClient
    try:
        tw = TwitchClient(cfg.secrets.twitch_client_id, cfg.secrets.twitch_client_secret)
        live = await tw.get_live(cfg.streamers)
        await tw.aclose()
        print("✓ Twitch API")
        for s in cfg.streamers:
            st = live.get(s.lower())
            print(f"    {s}: " + (f"LIVE — {st.title} ({st.game}, {st.viewers:,} viewers)" if st else "offline"))
    except Exception as e:
        ok = False
        print(f"✗ Twitch API: {e}")

    if cfg.secrets.discord_bot_token and cfg.discord.channel_id:
        print("• Discord: token and channel set (connection is tested by `run`)")
    else:
        print("✗ Discord: set DISCORD_BOT_TOKEN and discord.channel_id")
        ok = False

    if cfg.secrets.post_for_me_api_key:
        from .postforme import PostForMe
        try:
            pfm = PostForMe(cfg.secrets.post_for_me_api_key)
            accts = await pfm.list_accounts()
            await pfm.aclose()
            print(f"✓ Post for Me: {len(accts)} connected account(s)")
        except Exception as e:
            ok = False
            print(f"✗ Post for Me: {e}")
    else:
        print("✗ Post for Me: set POST_FOR_ME_API_KEY")
        ok = False
    return ok


def cmd_check(cfg, args):
    sys.exit(0 if asyncio.run(_check(cfg)) else 1)


async def _accounts(cfg):
    from .postforme import PostForMe
    pfm = PostForMe(cfg.secrets.post_for_me_api_key)
    for a in await pfm.list_accounts(connected_only=False):
        print(f"{a['id']}  {a.get('platform', ''):<10} {a.get('username') or '':<25} {a.get('status')}")
    await pfm.aclose()


def cmd_accounts(cfg, args):
    asyncio.run(_accounts(cfg))


def cmd_analyze(cfg, args):
    """Run detection + clipping on a local recording, to test and tune thresholds."""
    from .chat import load_chat_log
    from .clips import probe_duration
    from .pipeline import find_moments, render_clip
    from .recorder import Chunk

    video = Path(args.video)
    start = (datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc).timestamp()
             if args.start else 0.0)
    messages = load_chat_log(args.chat) if args.chat else None
    if messages and not args.start:
        start = messages[0].ts  # assume the chat log began when the recording did
    if args.threshold is not None:
        cfg.detection.threshold = args.threshold
    if args.laughter:
        cfg.detection.laughter_model = args.laughter
    chunk = Chunk(video, start, probe_duration(video))
    moments = find_moments(chunk, messages, cfg, max_moments=args.max)
    if not moments:
        print("No moments above the threshold. Try --threshold lower.")
    out = Path(args.out)
    for m in moments:
        t = m.ts - start
        line = f"{int(t // 60):02d}:{int(t % 60):02d}  score {m.score:.2f}  {m.signals}  {m.details}"
        if args.no_clips:
            print(line)
            continue
        cf = render_clip(m, [chunk], out, f"{video.stem}_{int(t)}s", cfg)
        print(f"{line}\n    → {cf.path}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="clipper", description="Twitch live clipping system")
    p.add_argument("-c", "--config", default="config.yaml")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run", help="watch streamers, record, clip, send to Discord")
    sub.add_parser("check", help="verify tools and API keys, show who is live")
    sub.add_parser("accounts", help="list Post for Me social accounts and their IDs")
    a = sub.add_parser("analyze", help="find and cut clips from a local video file")
    a.add_argument("video")
    a.add_argument("--chat", help="chat log (JSONL with ts/user/text, as written by `run`)")
    a.add_argument("--start", help="UTC time the video starts, e.g. 2026-11-17T20:00:00 (to line up chat)")
    a.add_argument("--threshold", type=float)
    a.add_argument("--laughter", choices=["ast", "loudness", "off"])
    a.add_argument("--max", type=int, default=10)
    a.add_argument("--out", default="data/clips/manual")
    a.add_argument("--no-clips", action="store_true", help="only list moments")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpx2", "huggingface_hub", "discord.gateway", "discord.client", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    cfg = load_config(args.config)
    {"run": cmd_run, "check": cmd_check, "accounts": cmd_accounts, "analyze": cmd_analyze}[args.cmd](cfg, args)


if __name__ == "__main__":
    main()
