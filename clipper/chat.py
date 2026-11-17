"""Read a channel's Twitch chat anonymously (no account needed) and keep a timestamped log."""

from __future__ import annotations

import asyncio
import json
import logging
import random
import ssl
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

IRC_HOST = "irc.chat.twitch.tv"
IRC_PORT = 6697


@dataclass
class ChatMessage:
    ts: float  # unix seconds
    user: str
    text: str


def parse_privmsg(line: str) -> tuple[str, str] | None:
    """Return (user, text) for a PRIVMSG line, else None. Handles lines with or without IRCv3 tags."""
    if line.startswith("@"):
        _, _, line = line.partition(" ")
    if not line.startswith(":") or " PRIVMSG " not in line:
        return None
    prefix, _, rest = line[1:].partition(" PRIVMSG ")
    user = prefix.split("!", 1)[0]
    _, _, text = rest.partition(" :")
    return user, text


def load_chat_log(path: str | Path) -> list[ChatMessage]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                out.append(ChatMessage(float(d["ts"]), d.get("user", ""), d.get("text", "")))
    return out


class ChatRecorder:
    """Keeps the last `keep_seconds` of chat in memory and appends every message to a JSONL file."""

    def __init__(self, channel: str, log_path: Path, keep_seconds: int = 3600):
        self.channel = channel.lower()
        self.log_path = log_path
        self.keep_seconds = keep_seconds
        self.messages: deque[ChatMessage] = deque()
        self._task: asyncio.Task | None = None

    def start(self):
        self._task = asyncio.create_task(self._run(), name=f"chat-{self.channel}")

    async def stop(self):
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def between(self, start: float, end: float) -> list[ChatMessage]:
        return [m for m in self.messages if start <= m.ts < end]

    def _add(self, user: str, text: str):
        m = ChatMessage(time.time(), user, text)
        self.messages.append(m)
        cutoff = m.ts - self.keep_seconds
        while self.messages and self.messages[0].ts < cutoff:
            self.messages.popleft()
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": m.ts, "user": user, "text": text}, ensure_ascii=False) + "\n")

    async def _run(self):
        backoff = 2
        while True:
            try:
                await self._session()
                backoff = 2
            except asyncio.CancelledError:
                raise
            except Exception as e:  # network blips: reconnect
                log.warning("chat %s disconnected (%s); reconnecting in %ss", self.channel, e, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)

    async def _session(self):
        reader, writer = await asyncio.open_connection(IRC_HOST, IRC_PORT, ssl=ssl.create_default_context())
        try:
            nick = f"justinfan{random.randint(10000, 99999)}"  # Twitch's anonymous read-only login
            writer.write(f"PASS SCHMOOPIIE\r\nNICK {nick}\r\nJOIN #{self.channel}\r\n".encode())
            await writer.drain()
            log.info("chat %s connected", self.channel)
            while True:
                raw = await asyncio.wait_for(reader.readline(), timeout=360)
                if not raw:
                    raise ConnectionError("server closed connection")
                line = raw.decode("utf-8", "replace").rstrip("\r\n")
                if line.startswith("PING"):
                    writer.write(("PONG" + line[4:] + "\r\n").encode())
                    await writer.drain()
                    continue
                if " RECONNECT" in line:
                    raise ConnectionError("server asked to reconnect")
                parsed = parse_privmsg(line)
                if parsed:
                    self._add(*parsed)
        finally:
            writer.close()
