"""Step 4: publish an approved clip to every connected social account through Post for Me.

API: https://api.postforme.dev (Bearer API key)
  POST /v1/media/create-upload-url → {upload_url, media_url}; PUT the file to upload_url
  POST /v1/social-posts {caption, social_accounts, media: [{url}], platform_configurations}
  GET  /v1/social-post-results?post_id=… → per-account success / error / platform_data.url
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import httpx

log = logging.getLogger(__name__)

BASE_URL = "https://api.postforme.dev"


class PostForMeError(RuntimeError):
    pass


class PostForMe:
    def __init__(self, api_key: str, http: httpx.AsyncClient | None = None, base_url: str = BASE_URL):
        if not api_key:
            raise ValueError("POST_FOR_ME_API_KEY must be set")
        self.http = http or httpx.AsyncClient(timeout=httpx.Timeout(60, write=600))
        self.base = base_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {api_key}"}

    async def _req(self, method: str, path: str, **kw) -> dict:
        r = await self.http.request(method, self.base + path, headers=self.headers, **kw)
        if r.status_code >= 400:
            raise PostForMeError(f"{method} {path} failed: {r.status_code} {r.text[:500]}")
        return r.json() if r.content else {}

    async def list_accounts(self, connected_only: bool = True) -> list[dict]:
        params = [("limit", "100")]
        if connected_only:
            params.append(("status", "connected"))
        return (await self._req("GET", "/v1/social-accounts", params=params)).get("data", [])

    async def upload(self, path: Path, content_type: str = "video/mp4") -> str:
        """Upload a local file and return the media_url to attach to a post."""
        urls = await self._req("POST", "/v1/media/create-upload-url")
        with open(path, "rb") as f:
            data = f.read()
        r = await self.http.put(urls["upload_url"], content=data, headers={"Content-Type": content_type})
        if r.status_code >= 400:
            raise PostForMeError(f"media upload failed: {r.status_code} {r.text[:300]}")
        return urls["media_url"]

    async def create_post(self, caption: str, account_ids: list[str], media_url: str,
                          title: str = "", external_id: str | None = None) -> dict:
        body: dict = {
            "caption": caption,
            "social_accounts": account_ids,
            "media": [{"url": media_url}],
        }
        if title:
            body["platform_configurations"] = {
                "youtube": {"title": title[:100]},
                "tiktok": {"title": title[:90]},
            }
        if external_id:
            body["external_id"] = external_id
        return await self._req("POST", "/v1/social-posts", json=body)

    async def results(self, post_id: str) -> list[dict]:
        return (await self._req("GET", "/v1/social-post-results",
                                params=[("post_id", post_id), ("limit", "100")])).get("data", [])

    async def wait_for_results(self, post_id: str, expected: int, timeout: float = 600,
                               interval: float = 15) -> list[dict]:
        """Poll until every account has a result (or timeout); returns whatever results exist."""
        deadline = asyncio.get_running_loop().time() + timeout
        res: list[dict] = []
        while asyncio.get_running_loop().time() < deadline:
            res = await self.results(post_id)
            if len(res) >= expected:
                break
            await asyncio.sleep(interval)
        return res

    async def aclose(self):
        await self.http.aclose()
