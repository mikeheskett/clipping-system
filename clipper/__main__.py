"""python -m clipper {run,check,accounts,analyze,facecam}"""

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
    from .discord_bot import DiscordSetupError
    try:
        asyncio.run(App(cfg).run())
    except KeyboardInterrupt:
        pass
    except DiscordSetupError as e:
        sys.exit(f"Discord setup problem: {e}")


async def _check(cfg):
    ok = True
    for tool in ("ffmpeg", "ffprobe"):
        found = shutil.which(tool)
        print(f"{'✓' if found else '✗'} {tool}: {found or 'not found on PATH'}")
        ok &= bool(found)
    import subprocess
    try:
        ver = subprocess.run([sys.executable, "-m", "clipper.streamlink_runner", "--version"],
                             capture_output=True, text=True, timeout=60).stdout.strip()
        print(f"{'✓' if ver else '✗'} streamlink: {ver or 'installed but would not start'}")
        ok &= bool(ver)
    except Exception as e:
        print(f"✗ streamlink: {e} (run: python -m pip install -r requirements.txt)")
        ok = False

    if cfg.detection.laughter_model == "ast":
        try:
            import onnxruntime  # noqa: F401
            import huggingface_hub  # noqa: F401
            print("✓ laughter detector: AI model (downloads ~90 MB on first run)")
        except ImportError:
            print("• laughter detector: AI model not installed, using loudness instead "
                  "(optional: python -m pip install -r requirements-laughter.txt)")
    else:
        print(f"• laughter detector: {cfg.detection.laughter_model}")

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
        cf = render_clip(m, [chunk], out, f"{video.stem}_{int(t)}s", cfg, args.streamer or "")
        print(f"{line}\n    → {cf.path}")


def _grab_live(login: str, out: Path, seconds: int = 4) -> Path:
    """Record a few seconds of a live stream with Streamlink."""
    import subprocess
    sl = subprocess.Popen([sys.executable, "-m", "clipper.streamlink_runner", "--stdout", "--loglevel", "error",
                           "--twitch-disable-ads", f"twitch.tv/{login}", "720p60,720p,best"],
                          stdout=subprocess.PIPE)
    try:
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", "pipe:0", "-t", str(seconds),
                        "-c", "copy", "-f", "mpegts", str(out)], stdin=sl.stdout, check=True, timeout=90)
    finally:
        sl.kill()
    if not out.exists() or out.stat().st_size == 0:
        sys.exit(f"Couldn't get video from twitch.tv/{login}. Are they live?")
    return out


def cmd_facecam(cfg, args):
    """Show a stream frame with a percent grid, and optionally test a facecam box in the split layout."""
    import subprocess
    import tempfile
    from .clips import EditStyle, cut_clip, find_font, has_drawtext, probe_duration
    from .recorder import Chunk

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    source = Path(args.source)
    name = source.stem
    if not source.is_file():
        name = args.source.lower()
        print(f"Grabbing a few seconds of twitch.tv/{name}…")
        source = _grab_live(name, out_dir / f"{name}_sample.ts")
    at = min(args.at, max(0.0, probe_duration(source) - 2.5))

    grid_png = out_dir / f"{name}_grid.png"
    vf = "scale=1280:720,drawgrid=w=iw/10:h=ih/10:t=2:c=white@0.6"
    font = find_font(cfg.clips.font_file) if has_drawtext() else None
    with tempfile.TemporaryDirectory() as tmp:
        if font:
            import shutil as sh
            sh.copyfile(font, Path(tmp) / ("font" + font.suffix.lower()))
            f = "font" + font.suffix.lower()
            for i in range(1, 10):
                lab = f"fontfile={f}:fontsize=22:fontcolor=yellow:borderw=3:bordercolor=black:text={i * 10}"
                vf += f",drawtext={lab}:x={i * 128 + 4}:y=4,drawtext={lab}:x=4:y={i * 72 + 4}"
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{at:.2f}",
                        "-i", str(source.resolve()), "-frames:v", "1", "-vf", vf, str(grid_png.resolve())],
                       check=True, cwd=tmp)
    print(f"Grid (lines every 10%): {grid_png}")

    if args.box:
        x, y, w, h = args.box
        style = EditStyle(title=name, facecam=(x / 100, y / 100, w / 100, h / 100), vary=False)
        clip = cut_clip([Chunk(source, 0.0, probe_duration(source))], at, at + 2, out_dir / f"{name}_test.mp4",
                        style=style, peak=None)
        layout_png = out_dir / f"{name}_layout.png"
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", "1", "-i", str(clip),
                        "-frames:v", "1", "-vf", "scale=540:-2", str(layout_png)], check=True)
        for p in (clip, clip.with_name(clip.stem + "_notext.mp4"), clip.with_name(clip.stem + "_notext.json")):
            p.unlink(missing_ok=True)
        print(f"Split layout with that box: {layout_png}")
        print("If it looks right, add this under clips: in config.yaml:\n"
              f"  facecams:\n    {args.source if not Path(args.source).is_file() else 'StreamerName'}: "
              f"[{x:g}, {y:g}, {w:g}, {h:g}]")
    else:
        print("Find the webcam on the grid, then run again with --box X Y WIDTH HEIGHT (percent), "
              "e.g. --box 0 0 25 30 for a webcam in the top-left quarter.")


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
    a.add_argument("--streamer", help="name shown as the clip title")
    f = sub.add_parser("facecam", help="find where a streamer's webcam is, for the split layout")
    f.add_argument("source", help="a streamer who is live now (e.g. xQc), or a video file / recorded chunk")
    f.add_argument("--box", nargs=4, type=float, metavar=("X", "Y", "W", "H"),
                   help="webcam box to test, in percent of the frame")
    f.add_argument("--at", type=float, default=1.0, help="seconds into the video to take the frame from")
    f.add_argument("--out", default="data/facecam")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("discord.client").setLevel(logging.ERROR)  # hides "PyNaCl/davey not installed" voice notes
    for noisy in ("httpx", "httpx2", "huggingface_hub", "discord.gateway", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    cfg = load_config(args.config)
    {"run": cmd_run, "check": cmd_check, "accounts": cmd_accounts, "analyze": cmd_analyze,
     "facecam": cmd_facecam}[args.cmd](cfg, args)


if __name__ == "__main__":
    main()
