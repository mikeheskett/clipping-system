"""Step 3: Discord — live/recording notifications, and clip review with Approve / Edit caption / Reject.

Buttons carry the clip ID in their custom_id and are handled in on_interaction, so they keep working
after the bot restarts.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Awaitable, Callable

import discord

from .store import FAILED, PENDING, POSTED, POSTING, REJECTED, Clip, Store

log = logging.getLogger(__name__)

STATUS_LABEL = {
    PENDING: "⏳ Waiting for approval",
    POSTING: "📤 Posting…",
    POSTED: "✅ Posted",
    REJECTED: "🗑️ Rejected",
    FAILED: "⚠️ Posting failed — press Approve to retry",
}
STATUS_COLOR = {PENDING: 0xF1C40F, POSTING: 0x3498DB, POSTED: 0x2ECC71, REJECTED: 0x7F8C8D, FAILED: 0xE74C3C}


def clip_embed(clip: Clip) -> discord.Embed:
    e = discord.Embed(
        title=f"😂 {clip.streamer} — clip #{clip.id}",
        description=clip.caption,
        color=STATUS_COLOR.get(clip.status, 0x95A5A6),
    )
    e.add_field(name="Status", value=STATUS_LABEL.get(clip.status, clip.status), inline=False)
    e.add_field(name="Moment", value=f"<t:{int(clip.moment_ts)}:T> (<t:{int(clip.moment_ts)}:R>)")
    e.add_field(name="Score", value=f"{clip.score:.2f}")
    for k, v in clip.details.items():
        e.add_field(name=k.replace("_", " ").capitalize(), value=v)
    if clip.post_results:
        lines = []
        for r in clip.post_results:
            name = r.get("platform") or r.get("social_account_id", "account")
            if r.get("success"):
                url = (r.get("platform_data") or {}).get("url")
                lines.append(f"✅ {name}" + (f": {url}" if url else ""))
            else:
                lines.append(f"❌ {name}: {str(r.get('error'))[:150]}")
        e.add_field(name="Results", value="\n".join(lines)[:1024], inline=False)
    elif clip.post_id:
        e.add_field(name="Post for Me", value=f"post `{clip.post_id}`", inline=False)
    e.set_footer(text=Path(clip.path).name)
    return e


def clip_view(clip: Clip) -> discord.ui.View | None:
    if clip.status not in (PENDING, FAILED):
        return None
    v = discord.ui.View(timeout=None)
    v.add_item(discord.ui.Button(label="Approve & post", style=discord.ButtonStyle.success,
                                 custom_id=f"clip:approve:{clip.id}"))
    v.add_item(discord.ui.Button(label="Edit caption", style=discord.ButtonStyle.secondary,
                                 custom_id=f"clip:caption:{clip.id}"))
    v.add_item(discord.ui.Button(label="Reject", style=discord.ButtonStyle.danger,
                                 custom_id=f"clip:reject:{clip.id}"))
    return v


class CaptionModal(discord.ui.Modal):
    def __init__(self, clip: Clip):
        super().__init__(title=f"Caption for clip #{clip.id}", custom_id=f"clip:captionmodal:{clip.id}")
        self.caption = discord.ui.TextInput(label="Caption", style=discord.TextStyle.paragraph,
                                            default=clip.caption, max_length=2000, custom_id="caption")
        self.title_input = discord.ui.TextInput(label="Title (YouTube / TikTok)", default=clip.title,
                                                max_length=100, required=False, custom_id="title")
        self.add_item(self.caption)
        self.add_item(self.title_input)


def modal_values(components: list) -> dict[str, str]:
    """Collect {custom_id: value} from a modal submit, whether inputs sit in action rows or labels."""
    out: dict[str, str] = {}
    stack = list(components)
    while stack:
        c = stack.pop()
        if "value" in c and "custom_id" in c:
            out[c["custom_id"]] = c["value"]
        stack.extend(c.get("components", []))
        if isinstance(c.get("component"), dict):
            stack.append(c["component"])
    return out


class DiscordBot:
    def __init__(self, token: str, channel_id: int, approver_ids: list[int], store: Store,
                 on_approve: Callable[[int], Awaitable[None]]):
        self.token = token
        self.channel_id = channel_id
        self.approver_ids = set(approver_ids)
        self.store = store
        self.on_approve = on_approve
        self.client = discord.Client(intents=discord.Intents.default())
        self._ready = asyncio.Event()
        self._channel: discord.abc.Messageable | None = None
        self.client.event(self.on_ready)
        self.client.event(self.on_interaction)

    async def start(self):
        asyncio.create_task(self.client.start(self.token), name="discord")
        await asyncio.wait_for(self._ready.wait(), timeout=60)

    async def close(self):
        await self.client.close()

    async def on_ready(self):
        self._channel = self.client.get_channel(self.channel_id) or await self.client.fetch_channel(self.channel_id)
        log.info("discord connected as %s, posting to #%s", self.client.user, getattr(self._channel, "name", "?"))
        self._ready.set()

    async def send(self, text: str):
        await self._channel.send(text)

    async def send_clip(self, clip: Clip) -> int:
        kwargs = {"embed": clip_embed(clip)}
        view = clip_view(clip)
        if view:
            kwargs["view"] = view
        if clip.preview_path and Path(clip.preview_path).exists():
            kwargs["file"] = discord.File(clip.preview_path, filename=f"clip_{clip.id}.mp4")
        else:
            kwargs["content"] = "(preview too large to attach — the full clip is saved on the server)"
        msg = await self._channel.send(**kwargs)
        return msg.id

    async def refresh_clip(self, clip_id: int):
        clip = self.store.get(clip_id)
        if not clip or not clip.discord_message_id:
            return
        try:
            msg = await self._channel.fetch_message(clip.discord_message_id)
            await msg.edit(embed=clip_embed(clip), view=clip_view(clip))
        except discord.HTTPException as e:
            log.warning("could not update Discord message for clip %s: %s", clip_id, e)

    def _allowed(self, user: discord.abc.User) -> bool:
        return not self.approver_ids or user.id in self.approver_ids

    async def on_interaction(self, interaction: discord.Interaction):
        data = interaction.data or {}
        custom_id = data.get("custom_id", "")
        if not custom_id.startswith("clip:"):
            return
        try:
            _, action, raw_id = custom_id.split(":")
            clip_id = int(raw_id)
        except ValueError:
            return
        clip = self.store.get(clip_id)
        if not clip:
            await interaction.response.send_message("That clip no longer exists.", ephemeral=True)
            return
        if not self._allowed(interaction.user):
            await interaction.response.send_message("You're not on the approvers list.", ephemeral=True)
            return

        if action == "approve":
            if not self.store.claim_for_posting(clip_id):
                await interaction.response.send_message(
                    f"Clip #{clip_id} is already {clip.status}.", ephemeral=True)
                return
            clip = self.store.get(clip_id)
            await interaction.response.edit_message(embed=clip_embed(clip), view=None)
            log.info("clip %s approved by %s", clip_id, interaction.user)
            asyncio.create_task(self.on_approve(clip_id))

        elif action == "reject":
            if clip.status not in (PENDING, FAILED):
                await interaction.response.send_message(f"Clip #{clip_id} is already {clip.status}.", ephemeral=True)
                return
            self.store.update(clip_id, status=REJECTED)
            await interaction.response.edit_message(embed=clip_embed(self.store.get(clip_id)), view=None)

        elif action == "caption":
            if clip.status not in (PENDING, FAILED):
                await interaction.response.send_message("Captions can only change before posting.", ephemeral=True)
                return
            await interaction.response.send_modal(CaptionModal(clip))

        elif action == "captionmodal":
            values = modal_values(data.get("components", []))
            caption = values.get("caption") or clip.caption
            title = values.get("title", clip.title)
            self.store.update(clip_id, caption=caption, title=title)
            clip = self.store.get(clip_id)
            await interaction.response.edit_message(embed=clip_embed(clip), view=clip_view(clip))


class LogOnlyNotifier:
    """Used when no Discord token is configured: notifications go to the log, clips wait as 'pending'."""

    async def start(self): ...
    async def close(self): ...

    async def send(self, text: str):
        log.info("[notify] %s", text)

    async def send_clip(self, clip: Clip) -> int:
        log.info("[notify] new clip #%s for %s (score %.2f): %s", clip.id, clip.streamer, clip.score, clip.path)
        return 0

    async def refresh_clip(self, clip_id: int): ...
