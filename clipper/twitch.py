"""Step 1: ask the Twitch API which streamers are live."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import httpx

log = logging.getLogger(__name__)

TOKEN_URL = "https://id.twitch.tv/oauth2/token"
STREAMS_URL = "https://api.twitch.tv/helix/streams"


@dataclass
class LiveStream:
    login: str
    display_name: str
    title: str
    game: str
    viewers: int
    started_at: str


class TwitchClient:
    def __init__(self, client_id: str, client_secret: str, http: httpx.AsyncClient | None = None):
        if not client_id or not client_secret:
            raise ValueError("TWITCH_CLIENT_ID and TWITCH_CLIENT_SECRET must be set")
        self.client_id = client_id
        self.client_secret = client_secret
        self.http = http or httpx.AsyncClient(timeout=20)
        self._token: str | None = None
        self._token_expires = 0.0

    async def _get_token(self, force: bool = False) -> str:
        if force or not self._token or time.time() > self._token_expires - 300:
            r = await self.http.post(TOKEN_URL, params={
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "grant_type": "client_credentials",
            })
            r.raise_for_status()
            body = r.json()
            self._token = body["access_token"]
            self._token_expires = time.time() + body.get("expires_in", 3600)
        return self._token

    async def get_live(self, logins: list[str]) -> dict[str, LiveStream]:
        """Return {lowercase login: LiveStream} for the logins that are live right now."""
        live: dict[str, LiveStream] = {}
        for i in range(0, len(logins), 100):  # Helix accepts up to 100 logins per call
            batch = logins[i:i + 100]
            params = [("user_login", l.lower()) for l in batch] + [("first", "100")]
            for attempt in range(2):
                token = await self._get_token(force=attempt > 0)
                r = await self.http.get(STREAMS_URL, params=params, headers={
                    "Client-Id": self.client_id,
                    "Authorization": f"Bearer {token}",
                })
                if r.status_code == 401 and attempt == 0:
                    continue  # token expired or revoked: get a new one and retry once
                r.raise_for_status()
                break
            for s in r.json().get("data", []):
                if s.get("type") != "live":
                    continue
                live[s["user_login"].lower()] = LiveStream(
                    login=s["user_login"].lower(),
                    display_name=s.get("user_name") or s["user_login"],
                    title=s.get("title", ""),
                    game=s.get("game_name", ""),
                    viewers=int(s.get("viewer_count", 0)),
                    started_at=s.get("started_at", ""),
                )
        return live

    async def aclose(self):
        await self.http.aclose()
