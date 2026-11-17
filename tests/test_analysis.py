import numpy as np

from clipper.analysis import laugh_matcher, pick_moments, score_chunk
from clipper.chat import ChatMessage, parse_privmsg
from clipper.config import DetectionConfig

T0 = 1_800_000_000.0


def steady_chat(start, end, per_second=2, text="nice"):
    return [ChatMessage(t + k / per_second, "u", text) for t in range(int(start), int(end))
            for k in range(per_second)]


def test_laugh_matcher():
    m = laugh_matcher(DetectionConfig().laugh_terms)
    for yes in ["KEKW", "omegalul", "LMAOOO", "hahaha", "that was so funny 😂", "ICANT", "lol", "lolol", "💀"]:
        assert m(yes), yes
    for no in ["LULLABY", "hello", "Pog", "lollipop", "pharaoh"]:
        assert not m(no), no


def test_parse_privmsg():
    assert parse_privmsg(":bob!bob@bob.tmi.twitch.tv PRIVMSG #xqc :KEKW what") == ("bob", "KEKW what")
    tagged = "@badge-info=;color=#FF0000 :amy!amy@amy.tmi.twitch.tv PRIVMSG #xqc :hi :)"
    assert parse_privmsg(tagged) == ("amy", "hi :)")
    assert parse_privmsg("PING :tmi.twitch.tv") is None


def test_chat_spike_with_laughter_is_picked():
    cfg = DetectionConfig()
    start, seconds = T0, 600
    msgs = steady_chat(start - 300, start + 612)
    # At 300s into the chunk chat explodes with laughter for ~10 seconds.
    msgs += [ChatMessage(start + 300 + k * 0.1, "u", "KEKW") for k in range(100)]
    combined, signals, raw = score_chunk(start, seconds, cfg, msgs, audio_laughter=None)
    moments = pick_moments(start, combined, signals, raw, cfg)
    assert len(moments) == 1
    m = moments[0]
    assert 285 <= m.ts - start <= 302  # chat window starts at the moment, so the peak is just before the burst
    assert m.signals["chat_spike"] == 1.0 and m.signals["chat_laughter"] > 0.5
    assert "x normal" in m.details["chat_activity"]


def test_quiet_chunk_gives_nothing():
    cfg = DetectionConfig()
    msgs = steady_chat(T0 - 300, T0 + 612)
    combined, signals, raw = score_chunk(T0, 600, cfg, msgs, audio_laughter=None)
    assert pick_moments(T0, combined, signals, raw, cfg) == []


def test_audio_laughter_combines_with_chat_and_respects_gap():
    cfg = DetectionConfig(threshold=0.5, min_seconds_between_clips=90)
    msgs = steady_chat(T0 - 300, T0 + 612)
    msgs += [ChatMessage(T0 + 100 + k * 0.1, "u", "LUL") for k in range(80)]
    msgs += [ChatMessage(T0 + 140 + k * 0.1, "u", "LUL") for k in range(80)]   # too close to the first
    msgs += [ChatMessage(T0 + 400 + k * 0.1, "u", "LUL") for k in range(80)]
    audio = np.zeros(600)
    audio[95:110] = 0.9
    combined, signals, raw = score_chunk(T0, 600, cfg, msgs, audio)
    moments = pick_moments(T0, combined, signals, raw, cfg)
    offsets = [m.ts - T0 for m in moments]
    assert len(moments) == 2
    assert 90 <= offsets[0] <= 110 and 385 <= offsets[1] <= 402
    assert moments[0].score > moments[1].score  # the one with audio laughter scores higher


def test_weights_renormalise_without_chat():
    cfg = DetectionConfig(threshold=0.5)
    audio = np.zeros(600)
    audio[200:205] = 0.8
    combined, signals, raw = score_chunk(T0, 600, cfg, None, audio)
    assert set(signals) == {"audio_laughter"}
    moments = pick_moments(T0, combined, signals, raw, cfg)
    assert [int(m.ts - T0) for m in moments] == [200]
