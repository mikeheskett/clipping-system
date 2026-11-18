"""Step 2b (audio): how much laughter is in each second of a chunk.

The "ast" detector runs the Audio Spectrogram Transformer fine-tuned on Google's AudioSet (527 sound
classes, several of them laughter) as a quantized ONNX model, so it needs only onnxruntime + numpy,
not PyTorch. The model (~90 MB) downloads from Hugging Face on first use.
"""

from __future__ import annotations

import json
import logging
import subprocess
import threading
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
AST_REPO = "onnx-community/ast-finetuned-audioset-10-10-0.4593-ONNX"
AST_FILE = "onnx/model_quantized.onnx"
LAUGH_LABELS = {"Laughter", "Baby laughter", "Giggle", "Snicker", "Belly laugh", "Chuckle, chortle"}

# AST front-end constants (Kaldi-compatible log-mel filterbank, as in the original AST recipe).
_FRAME_LEN, _HOP, _NFFT, _NMELS, _MAX_FRAMES = 400, 160, 512, 128, 1024
_AST_MEAN, _AST_STD = -4.2677393, 4.5689974


def load_audio(path: str | Path, start: float = 0.0, duration: float | None = None) -> np.ndarray:
    """Decode any media file's audio to 16 kHz mono float32 via ffmpeg."""
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin"]
    if start:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(path)]
    if duration:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "pipe:1"]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, np.int16).astype(np.float32) / 32768.0


def _kaldi_mel_filters() -> np.ndarray:
    """(257, 128) triangular filters laid out in Kaldi's mel space, 20 Hz to Nyquist."""
    mel = lambda f: 1127.0 * np.log(1.0 + np.asarray(f) / 700.0)
    fft_freqs = np.linspace(0, SAMPLE_RATE / 2, _NFFT // 2 + 1)
    fft_mels = mel(fft_freqs)
    edges = np.linspace(mel(20.0), mel(SAMPLE_RATE / 2), _NMELS + 2)
    lo, center, hi = edges[:-2], edges[1:-1], edges[2:]
    up = (fft_mels[:, None] - lo) / (center - lo)
    down = (hi - fft_mels[:, None]) / (hi - center)
    return np.maximum(0.0, np.minimum(up, down)).astype(np.float32)


_MEL = _kaldi_mel_filters()
_WINDOW = np.hanning(_FRAME_LEN).astype(np.float32)  # symmetric Hann, as Kaldi's "hanning"


def ast_features(wave: np.ndarray) -> np.ndarray:
    """One waveform (≤10.24 s at 16 kHz) → normalised (1024, 128) log-mel input for AST."""
    if len(wave) < _FRAME_LEN:
        wave = np.pad(wave, (0, _FRAME_LEN - len(wave)))
    n = 1 + (len(wave) - _FRAME_LEN) // _HOP
    idx = np.arange(_FRAME_LEN)[None, :] + _HOP * np.arange(n)[:, None]
    frames = wave[idx].astype(np.float32)
    frames = frames - frames.mean(axis=1, keepdims=True)                  # remove DC offset
    frames = np.concatenate([frames[:, :1] * (1 - 0.97),                   # pre-emphasis
                             frames[:, 1:] - 0.97 * frames[:, :-1]], axis=1)
    power = np.abs(np.fft.rfft(frames * _WINDOW, n=_NFFT)) ** 2
    fbank = np.log(np.maximum(power @ _MEL, np.finfo(np.float32).eps))
    if len(fbank) < _MAX_FRAMES:
        fbank = np.pad(fbank, ((0, _MAX_FRAMES - len(fbank)), (0, 0)))
    fbank = fbank[:_MAX_FRAMES]
    return ((fbank - _AST_MEAN) / (_AST_STD * 2)).astype(np.float32)


def _model_file(name: str) -> str:
    """Path to a model file: the copy downloaded on an earlier run if there is one, so starting up needs
    no network, else download it. Downloads are anonymous; the model is public, so no HF_TOKEN is needed."""
    import warnings
    from huggingface_hub import hf_hub_download
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)  # hides the "set a HF_TOKEN" warning
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*HF_TOKEN.*")
        try:
            return hf_hub_download(AST_REPO, name, local_files_only=True)
        except Exception:
            log.info("downloading laughter model file %s (first run only)", name)
            return hf_hub_download(AST_REPO, name)


class AstLaughterDetector:
    window_seconds = 10.0
    hop_seconds = 5.0

    def __init__(self):
        import onnxruntime as ort

        model_path = _model_file(AST_FILE)
        labels = json.loads(Path(_model_file("config.json")).read_text())["id2label"]
        self.laugh_ids = [int(i) for i, name in labels.items() if name in LAUGH_LABELS]
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self._lock = threading.Lock()

    def per_second(self, wave: np.ndarray) -> np.ndarray:
        """Laughter probability for every whole second of `wave`."""
        seconds = int(np.ceil(len(wave) / SAMPLE_RATE))
        win, hop = int(self.window_seconds * SAMPLE_RATE), int(self.hop_seconds * SAMPLE_RATE)
        starts = list(range(0, max(len(wave) - win, 0) + 1, hop)) or [0]
        if starts[-1] + win < len(wave):
            starts.append(max(len(wave) - win, 0))
        scores = np.zeros(seconds, np.float32)
        for b in range(0, len(starts), 8):
            batch_starts = starts[b:b + 8]
            x = np.stack([ast_features(wave[s:s + win]) for s in batch_starts])
            with self._lock:
                logits = self.session.run(None, {self.input_name: x})[0]
            probs = 1.0 / (1.0 + np.exp(-logits[:, self.laugh_ids]))
            for s, p in zip(batch_starts, probs.max(axis=1)):
                a, z = s // SAMPLE_RATE, min(seconds, int(np.ceil((s + win) / SAMPLE_RATE)))
                scores[a:z] = np.maximum(scores[a:z], p)
        return scores


class LoudnessDetector:
    """Cheap stand-in when the AST model can't be used: flags seconds that are much louder than usual
    (shouting and laughing are loud). Returns a 0..1 score, not a probability."""

    def per_second(self, wave: np.ndarray) -> np.ndarray:
        seconds = int(np.ceil(len(wave) / SAMPLE_RATE))
        padded = np.pad(wave, (0, seconds * SAMPLE_RATE - len(wave)))
        rms = np.sqrt((padded.reshape(seconds, SAMPLE_RATE) ** 2).mean(axis=1) + 1e-12)
        db = 20 * np.log10(rms)
        med = np.median(db)
        mad = np.median(np.abs(db - med)) * 1.4826 + 1e-6
        return np.clip((db - med) / (4 * mad), 0, 1).astype(np.float32)


_detector_cache: dict[str, object] = {}


def get_detector(kind: str):
    """Return a laughter detector, or None when audio analysis is off."""
    if kind == "off":
        return None
    if kind not in _detector_cache:
        if kind == "ast":
            try:
                _detector_cache[kind] = AstLaughterDetector()
            except Exception as e:
                log.warning("AST laughter model unavailable (%s); falling back to loudness", e)
                _detector_cache[kind] = LoudnessDetector()
        elif kind == "loudness":
            _detector_cache[kind] = LoudnessDetector()
        else:
            raise ValueError(f"unknown laughter_model {kind!r} (use ast, loudness or off)")
    return _detector_cache[kind]
