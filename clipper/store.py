"""SQLite record of every clip and where it is in the approve → post flow."""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

PENDING, POSTING, POSTED, REJECTED, FAILED = "pending", "posting", "posted", "rejected", "failed"
RENDERING = "rendering"  # text on the video is being redrawn; goes back to pending/failed after


@dataclass
class Clip:
    id: int
    streamer: str
    path: str
    preview_path: str | None
    moment_ts: float
    score: float
    signals: dict
    details: dict
    caption: str
    title: str
    status: str
    discord_message_id: int | None
    post_id: str | None
    post_results: list
    created_at: float
    video_text: str = ""


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS clips (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                streamer TEXT NOT NULL,
                path TEXT NOT NULL,
                preview_path TEXT,
                moment_ts REAL NOT NULL,
                score REAL NOT NULL,
                signals TEXT NOT NULL DEFAULT '{}',
                details TEXT NOT NULL DEFAULT '{}',
                caption TEXT NOT NULL,
                title TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'pending',
                discord_message_id INTEGER,
                post_id TEXT,
                post_results TEXT NOT NULL DEFAULT '[]',
                created_at REAL NOT NULL
            )""")
        cols = {r["name"] for r in self.db.execute("PRAGMA table_info(clips)")}
        if "video_text" not in cols:  # databases made before the text on the video could be edited
            self.db.execute("ALTER TABLE clips ADD COLUMN video_text TEXT NOT NULL DEFAULT ''")
        # A redraw interrupted by a restart never finished: let the clip be reviewed again.
        self.db.execute("UPDATE clips SET status = ? WHERE status = ?", (PENDING, RENDERING))
        self.db.commit()

    def add(self, streamer: str, path: Path, preview: Path | None, moment_ts: float, score: float,
            signals: dict, details: dict, caption: str, title: str, video_text: str = "") -> Clip:
        cur = self.db.execute(
            "INSERT INTO clips (streamer, path, preview_path, moment_ts, score, signals, details, caption,"
            " title, created_at, video_text) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (streamer, str(path), str(preview) if preview else None, moment_ts, score,
             json.dumps(signals), json.dumps(details), caption, title, time.time(), video_text))
        self.db.commit()
        return self.get(cur.lastrowid)

    def get(self, clip_id: int) -> Clip | None:
        row = self.db.execute("SELECT * FROM clips WHERE id = ?", (clip_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        for k in ("signals", "details", "post_results"):
            d[k] = json.loads(d[k])
        return Clip(**d)

    def update(self, clip_id: int, **fields):
        if "post_results" in fields:
            fields["post_results"] = json.dumps(fields["post_results"])
        cols = ", ".join(f"{k} = ?" for k in fields)
        self.db.execute(f"UPDATE clips SET {cols} WHERE id = ?", (*fields.values(), clip_id))
        self.db.commit()

    def claim_for_posting(self, clip_id: int) -> bool:
        """Atomically move a pending (or failed) clip to 'posting' so a double-click can't post twice."""
        cur = self.db.execute(
            "UPDATE clips SET status = ? WHERE id = ? AND status IN (?, ?)",
            (POSTING, clip_id, PENDING, FAILED))
        self.db.commit()
        return cur.rowcount == 1
