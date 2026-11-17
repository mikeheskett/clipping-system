"""Step 2c: combine chat activity, chat laughter and audio laughter into a per-second "clip-worthiness"
score, then pick the best moments in a chunk."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from .chat import ChatMessage
from .config import DetectionConfig


@dataclass
class Moment:
    ts: float          # unix seconds of the peak
    score: float
    signals: dict = field(default_factory=dict)  # each signal's 0..1 value at the peak
    details: dict = field(default_factory=dict)  # human-readable numbers for Discord


def laugh_matcher(terms: list[str]):
    """Match laugh terms as whole words (case-insensitive), so 'LUL' doesn't match 'LULLABY'.
    Emoji are matched anywhere. Also catches 'hahaha', 'lmaooo', 'lolol'."""
    words = [t for t in terms if t.isascii()]
    emoji = [t for t in terms if not t.isascii()]
    parts = [r"(?<![A-Za-z0-9])(?:" + "|".join(re.escape(w) for w in words) + r")(?![A-Za-z0-9])"] if words else []
    parts.append(r"(?<![A-Za-z])(?:a?(?:ha){2,}h?|lmao+|lo+l(?:ol)*|kek)(?![A-Za-z])")
    parts += [re.escape(e) for e in emoji]
    rx = re.compile("|".join(parts), re.IGNORECASE)
    return lambda text: bool(rx.search(text))


def chat_signals(messages: list[ChatMessage], start: float, seconds: int, cfg: DetectionConfig):
    """Per-second chat spike and chat laughter scores for [start, start+seconds).

    `messages` should include chat from `cfg.chat_baseline_seconds` before `start` (for the baseline)
    and `cfg.chat_window_seconds` after the end (chat reacts after the moment)."""
    w = cfg.chat_window_seconds
    lookback = cfg.chat_baseline_seconds
    origin = start - lookback
    total = lookback + seconds + w
    counts = np.zeros(total)
    laughs = np.zeros(total)
    is_laugh = laugh_matcher(cfg.laugh_terms)
    for m in messages:
        i = int(m.ts - origin)
        if 0 <= i < total:
            counts[i] += 1
            laughs[i] += is_laugh(m.text)

    csum = np.concatenate([[0], np.cumsum(counts)])
    lsum = np.concatenate([[0], np.cumsum(laughs)])
    idx = np.arange(total - w + 1)
    rate = csum[idx + w] - csum[idx]   # messages in the w seconds starting at each second
    lrate = lsum[idx + w] - lsum[idx]

    spike = np.zeros(seconds)
    laugh_share = np.zeros(seconds)
    ratio = np.zeros(seconds)
    full = max(cfg.chat_spike_full_ratio - 1, 1e-6)
    for s in range(seconds):
        i = lookback + s
        hist = rate[max(0, i - lookback):max(1, i - w)]  # history that doesn't overlap this window
        baseline = max(float(np.median(hist)) if len(hist) else 0.0, 1.0)
        ratio[s] = rate[i] / baseline
        spike[s] = np.clip((ratio[s] - 1) / full, 0, 1)
        if rate[i] >= 3:  # need a few messages before a laugh share means anything
            laugh_share[s] = lrate[i] / rate[i]
    chat_laughter = np.clip(laugh_share / cfg.chat_laughter_full_share, 0, 1)
    return spike, chat_laughter, ratio, laugh_share, rate[lookback:lookback + seconds]


def score_chunk(start: float, seconds: int, cfg: DetectionConfig,
                messages: list[ChatMessage] | None, audio_laughter: np.ndarray | None):
    """Return (combined per-second score, per-signal arrays, raw detail arrays)."""
    signals: dict[str, np.ndarray] = {}
    raw: dict[str, np.ndarray] = {}
    if messages is not None and len(messages) > 0:
        spike, chat_laugh, ratio, share, rate = chat_signals(messages, start, seconds, cfg)
        signals["chat_spike"], signals["chat_laughter"] = spike, chat_laugh
        raw.update(chat_ratio=ratio, chat_laugh_share=share, chat_msgs=rate)
    if audio_laughter is not None:
        a = np.zeros(seconds)
        n = min(seconds, len(audio_laughter))
        a[:n] = audio_laughter[:n]
        raw["audio_laugh_prob"] = a
        signals["audio_laughter"] = np.clip(a / cfg.audio_laughter_full_prob, 0, 1)

    weights = {k: float(cfg.weights.get(k, 0)) for k in signals}
    total_w = sum(weights.values())
    combined = np.zeros(seconds)
    if total_w > 0:
        for k, arr in signals.items():
            combined += arr * weights[k] / total_w  # renormalise when a signal is missing
    return combined, signals, raw


def pick_moments(start: float, combined: np.ndarray, signals: dict, raw: dict,
                 cfg: DetectionConfig, max_moments: int | None = None) -> list[Moment]:
    """Greedy peak picking: highest second above threshold, then suppress its neighbourhood."""
    max_moments = cfg.max_clips_per_chunk if max_moments is None else max_moments
    score = combined.copy()
    gap = cfg.min_seconds_between_clips
    out: list[Moment] = []
    while len(out) < max_moments and len(score):
        i = int(np.argmax(score))
        if score[i] < cfg.threshold:
            break
        details = {}
        if "chat_ratio" in raw:
            details["chat_activity"] = f"{raw['chat_ratio'][i]:.1f}x normal ({int(raw['chat_msgs'][i])} msgs / {cfg.chat_window_seconds}s)"
            details["chat_laughing"] = f"{raw['chat_laugh_share'][i]:.0%} of messages"
        if "audio_laugh_prob" in raw:
            details["audio_laughter"] = f"{raw['audio_laugh_prob'][i]:.2f}"
        out.append(Moment(
            ts=start + i,
            score=float(combined[i]),
            signals={k: round(float(v[i]), 3) for k, v in signals.items()},
            details=details,
        ))
        score[max(0, i - gap):i + gap + 1] = -1
    return sorted(out, key=lambda m: m.ts)
