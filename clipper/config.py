"""Configuration: non-secret settings from config.yaml, secrets from the environment (.env)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_LAUGH_TERMS = [
    "LUL", "LULW", "KEKW", "KEKL", "OMEGALUL", "ICANT", "pepeLaugh", "EleGiggle",
    "LOL", "LMAO", "LMFAO", "HAHA", "XD", "💀", "😂", "🤣",
]


@dataclass
class RecordingConfig:
    quality: str = "720p60,720p,best"
    chunk_minutes: int = 10
    keep_chunks_hours: float = 6.0


@dataclass
class DetectionConfig:
    # Audio laughter detector: "ast" (AudioSet model, recommended), "loudness" (cheap fallback), "off".
    laughter_model: str = "ast"
    # Chat is counted in a window starting at each second, because chat reacts after the moment.
    chat_window_seconds: int = 12
    # How far back to look when working out what "normal" chat speed is.
    chat_baseline_seconds: int = 300
    # Chat running this many times faster than normal scores a full 1.0 for "chat spike".
    chat_spike_full_ratio: float = 3.0
    # This share of messages being laugh reactions scores a full 1.0 for "chat laughter".
    chat_laughter_full_share: float = 0.35
    # This AudioSet laughter probability scores a full 1.0 for "audio laughter".
    audio_laughter_full_prob: float = 0.3
    weights: dict = field(default_factory=lambda: {
        "chat_spike": 0.4, "chat_laughter": 0.3, "audio_laughter": 0.3,
    })
    threshold: float = 0.6
    min_seconds_between_clips: int = 90
    max_clips_per_chunk: int = 3
    laugh_terms: list = field(default_factory=lambda: list(DEFAULT_LAUGH_TERMS))


@dataclass
class ClipConfig:
    preroll_seconds: int = 40
    postroll_seconds: int = 20
    # "vertical" = 9:16 with blurred background (TikTok / Shorts / Reels), "original" = keep 16:9.
    format: str = "vertical"
    # Editing so clips look like edits, not raw re-uploads (platforms push down duplicate/unoriginal videos).
    edit: bool = True
    title: str = "{streamer}"   # text above the gameplay in vertical clips ("" = none; emoji won't render)
    punch_zoom: bool = True     # zoom in on the funny moment for a few seconds
    progress_bar: bool = True   # bar along the bottom that fills as the clip plays
    color_grade: bool = True    # punchier contrast and colour
    vary_each_clip: bool = True # randomise zoom, grade, colours and layout a little per clip
    font_file: str = ""         # font for the title; empty = find a bold system font (Impact/Arial Bold)
    title_seconds: float = 7    # how long the text stays on screen from the start (0 = whole clip)
    # "split" = streamer's webcam zoomed in on top, gameplay below (needs that streamer in `facecams`);
    # "blur" = gameplay on a blurred copy of itself. Streamers without a facecam entry always use blur.
    layout: str = "split"
    # Where each streamer's webcam is in their stream: [x, y, width, height] in percent of the frame.
    # Find it with `python -m clipper facecam <streamer or video>`.
    facecams: dict = field(default_factory=dict)


@dataclass
class DiscordConfig:
    channel_id: int = 0
    # Discord user IDs allowed to press Approve/Reject. Empty = anyone who can see the channel.
    approver_user_ids: list = field(default_factory=list)
    # Bots without a boosted server can upload 10 MB; previews are squeezed under this.
    preview_max_mb: float = 9.5


@dataclass
class PostForMeConfig:
    # spc_... IDs from `python -m clipper accounts`. Empty = every connected account.
    social_account_ids: list = field(default_factory=list)
    caption_template: str = "{streamer} moment 😂 #{streamer_tag} #twitch #gta6 #clips"
    # Title used where a platform needs one (YouTube, TikTok).
    title_template: str = "{streamer} moment 😂"


@dataclass
class Secrets:
    twitch_client_id: str = ""
    twitch_client_secret: str = ""
    discord_bot_token: str = ""
    post_for_me_api_key: str = ""


@dataclass
class Config:
    streamers: list = field(default_factory=list)
    poll_interval_seconds: int = 120
    data_dir: Path = Path("data")
    recording: RecordingConfig = field(default_factory=RecordingConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    clips: ClipConfig = field(default_factory=ClipConfig)
    discord: DiscordConfig = field(default_factory=DiscordConfig)
    postforme: PostForMeConfig = field(default_factory=PostForMeConfig)
    secrets: Secrets = field(default_factory=Secrets)


def _build(cls, data: dict | None):
    data = data or {}
    known = {f for f in cls.__dataclass_fields__}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys in config: {sorted(unknown)}")
    return cls(**data)


def load_config(path: str | Path = "config.yaml") -> Config:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    path = Path(path)
    raw = yaml.safe_load(path.read_text()) if path.exists() else {}
    raw = raw or {}

    cfg = Config(
        streamers=[s.strip() for s in raw.get("streamers", []) if str(s).strip()],
        poll_interval_seconds=int(raw.get("poll_interval_seconds", 120)),
        data_dir=Path(raw.get("data_dir", "data")),
        recording=_build(RecordingConfig, raw.get("recording")),
        detection=_build(DetectionConfig, raw.get("detection")),
        clips=_build(ClipConfig, raw.get("clips")),
        discord=_build(DiscordConfig, raw.get("discord")),
        postforme=_build(PostForMeConfig, raw.get("postforme")),
        secrets=Secrets(
            twitch_client_id=os.environ.get("TWITCH_CLIENT_ID", ""),
            twitch_client_secret=os.environ.get("TWITCH_CLIENT_SECRET", ""),
            discord_bot_token=os.environ.get("DISCORD_BOT_TOKEN", ""),
            post_for_me_api_key=os.environ.get("POST_FOR_ME_API_KEY", ""),
        ),
    )
    cfg.discord.channel_id = int(cfg.discord.channel_id or 0)
    cfg.discord.approver_user_ids = [int(u) for u in cfg.discord.approver_user_ids]
    return cfg
