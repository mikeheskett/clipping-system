import numpy as np

from clipper.audio import SAMPLE_RATE, LoudnessDetector, ast_features


def test_ast_features_shape_and_padding():
    f = ast_features(np.zeros(SAMPLE_RATE * 3, np.float32))
    assert f.shape == (1024, 128) and f.dtype == np.float32
    long = ast_features(np.random.default_rng(0).standard_normal(SAMPLE_RATE * 20).astype(np.float32))
    assert long.shape == (1024, 128)  # truncated to the model's 10.24 s


def test_loudness_detector_flags_loud_seconds():
    rng = np.random.default_rng(1)
    wave = rng.standard_normal(SAMPLE_RATE * 30).astype(np.float32) * 0.01
    wave[SAMPLE_RATE * 12:SAMPLE_RATE * 14] *= 30
    s = LoudnessDetector().per_second(wave)
    assert len(s) == 30 and s[12] == 1.0 and s[13] == 1.0 and s[:10].max() < 0.5
