# Clipping system

Watches Twitch streamers, records them while they're live, finds the funniest moments and sends them
to Discord for approval. When you approve a clip, it is posted to your socials through
[Post for Me](https://postforme.dev).

```
 Twitch API ──(every 2 min)──▶ who's live?
                                  │ went live → 🔴 Discord: "xQc just went live — beginning to record"
                                  ▼
 Streamlink ──▶ ffmpeg ──▶ 10-minute chunks          Twitch chat (anonymous) ──▶ timestamped log
                                  │                                   │
                                  └──────────── each finished chunk ──┘
                                                      ▼
                        score every second:  chat spike (chat suddenly faster than normal)
                                           + chat laughter (KEKW / LUL / 😂 / lmao share)
                                           + audio laughter (AudioSet AI model)
                                                      ▼
                        moment found → clip from 40 s before to 20 s after, vertical 9:16
                                                      ▼
                 Discord: preview video + [Approve & post] [Edit title / caption] [Reject]
                                                      ▼  Approve
                 Post for Me: upload → post to every connected account → links back in Discord
```

## Setup

You need a computer or server that stays on while the streamers are live, with **Python 3.10+** and
**ffmpeg** installed (`brew install ffmpeg`, `sudo apt install ffmpeg`, or ffmpeg.org on Windows).

```bash
git clone <this repo> && cd clipping-system
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt                       # includes streamlink
python -m pip install -r requirements-laughter.txt              # optional: AI laughter detector
cp config.example.yaml config.yaml
cp .env.example .env
```

Then fill in `.env` and `config.yaml`.

### 1. Twitch (live checks)
1. Go to <https://dev.twitch.tv/console/apps> → **Register Your Application**.
2. OAuth Redirect URL: `http://localhost`, category: anything, client type: **Confidential**.
3. Copy the **Client ID** and click **New Secret** → put both in `.env`.

The streamers are already listed in `config.yaml`: TimTheTatman, TheBurntPeanut, xQc, Jynxzi.

### 2. Discord
1. <https://discord.com/developers/applications> → **New Application** → **Bot** → **Reset Token** → copy it into
   `DISCORD_BOT_TOKEN` in `.env`. No privileged intents are needed.
2. **OAuth2 → URL Generator**: scope `bot`; permissions **View Channels, Send Messages, Embed Links, Attach Files,
   Read Message History**. Open the URL and add the bot to your server.
3. In Discord, turn on **Settings → Advanced → Developer Mode**, right-click the channel for clips →
   **Copy Channel ID** → `discord.channel_id` in `config.yaml`.
4. Optional: put your own user ID (right-click yourself → Copy User ID) in `discord.approver_user_ids` so only you
   can approve.

### 3. Post for Me
1. Sign up at <https://postforme.dev>, connect your TikTok / YouTube / Instagram etc. accounts, and create an API key
   → `POST_FOR_ME_API_KEY` in `.env`.
2. `python -m clipper accounts` lists the connected accounts. Leave `postforme.social_account_ids` empty to post to
   all of them, or list the `spc_…` IDs you want.

### 4. Check and run

```bash
python -m clipper check   # verifies ffmpeg/streamlink, Twitch, Post for Me; shows who is live now
python -m clipper run     # runs until you stop it (Ctrl+C finishes the current chunks cleanly)
```

On first run the laughter model (~90 MB) downloads from Hugging Face.

If `requirements-laughter.txt` won't install ("No matching distribution found for onnxruntime"), your Python
version is newer than onnxruntime supports yet, or isn't 64-bit. Everything else still works, using a
loudness-based detector instead. To get the AI detector, install 64-bit Python 3.12 from python.org and
create the venv with it: `py -3.12 -m venv .venv`.

## Tuning

Detection settings are in `config.yaml` under `detection`. Each second gets a score from 0 to 1. A clip is
made when the score passes `threshold`, which is 0.6 by default. At most 3 clips are made per 10-minute chunk,
and clips are at least 90 seconds apart.

| Too many boring clips | Missing good moments |
|---|---|
| raise `threshold` (e.g. 0.7) | lower `threshold` (e.g. 0.5) |
| raise `chat_spike_full_ratio` | raise the weight of the signal that catches them |
| lower `max_clips_per_chunk` | lower `min_seconds_between_clips` |

Test settings on a recording you already have:

```bash
python -m clipper analyze some_vod.mp4 --no-clips          # list moments with their scores
python -m clipper analyze data/streams/xqc/<session>/chunks/chunk_..._0003.ts \
    --chat data/streams/xqc/<session>/chat.jsonl --start 2026-11-19T20:30:00
```

Each Discord clip card shows why the clip was picked, for example "chat activity 4.2x normal", "chat laughing 61%"
or "audio laughter 0.43".

## Clip editing

TikTok, Shorts and Reels push down videos that look like straight re-uploads, so each clip is edited (settings
under `clips` in `config.yaml`):

- **Title**: the streamer's name in a coloured box above the gameplay (`title`, vertical clips only).
- **Punch-in**: the video cuts to a closer zoom just before the funny moment and holds for about 3 seconds (`punch_zoom`).
- **Colour grade**: more contrast and saturation (`color_grade`).
- **Progress bar**: a thin bar along the bottom that fills as the clip plays (`progress_bar`).
- **Different every clip**: the zoom, grade, background tint and blur, accent colour and gameplay position change a
  little each time (`vary_each_clip`). The same clip always comes out the same if it is cut again.

**Changing it in Discord:** **Edit title / caption** on a clip card opens a form with the post title, the
caption and the **text on video**. Changing the text re-makes the clip (about as long as cutting it took) and
swaps in the new preview. Leave it empty for no text. To make this possible, each vertical clip keeps a copy
without text (`…_notext.mp4`), so clips take about twice the disk space.

`edit: false` turns all of this off. If an edited render fails (for example ffmpeg has no `drawtext`), a plain clip
is made instead and a warning is logged. If no title appears, set `font_file`, for example
`C:/Windows/Fonts/impact.ttf`.

## Files and disk use

- `data/clips/<streamer>/…mp4` is the finished clips, plus a `_notext` copy of each for changing the text on
  the video. These are kept.
- `data/streams/<streamer>/<session>/chunks/` is the raw 10-minute recordings. They are deleted after
  `recording.keep_chunks_hours`, which is 6 hours by default. At 720p60 a stream uses about 3 GB per hour.
- `data/streams/<streamer>/<session>/chat.jsonl` is the full chat log.
- `data/clips.db` records every clip and its approval/posting status.

## Notes

- Laughter detection uses the Audio Spectrogram Transformer fine-tuned on Google's AudioSet. It runs on the CPU
  through onnxruntime and takes about 1.5 minutes per 10-minute chunk. Chunks are analysed one at a time, so four
  streams at once is fine on a normal desktop. If that is too slow, set `laughter_model: loudness` or `off`.
- Game audio is mixed into the stream, so laughter that happens in the game also counts.
- Chat is read anonymously, so no Twitch account is needed for chat.
- Recording restarts on its own if Streamlink drops while the streamer is still live.

## Tests

```bash
python -m pip install pytest && python -m pytest
```

The tests need ffmpeg. They feed a synthetic stream through the real recorder and clip cutter, and run the whole
live → clip → approve → post flow against mocked Twitch, Discord and Post for Me.
