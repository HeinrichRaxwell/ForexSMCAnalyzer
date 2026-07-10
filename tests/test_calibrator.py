import os
import tempfile

import numpy as np
import pytest

from src.calibrator import fit_calibrator, apply_calibrator, save_calibrator, load_calibrator


def test_calibrator_monotonic_nondecreasing():
    rng = np.random.default_rng(0)
    raw = rng.uniform(0, 1, 500)
    labels = (rng.uniform(0, 1, 500) < raw).astype(int)

    calibrator = fit_calibrator(raw, labels)
    mapped = apply_calibrator(calibrator, np.linspace(0, 1, 50))

    assert (np.diff(mapped) >= -1e-9).all()


def test_calibrator_maps_into_unit_interval():
    rng = np.random.default_rng(1)
    raw = rng.uniform(0, 1, 300)
    labels = (rng.uniform(0, 1, 300) < raw).astype(int)

    calibrator = fit_calibrator(raw, labels)
    out = apply_calibrator(calibrator, np.array([0.0, 0.5, 1.0]))

    assert (out >= 0.0).all()
    assert (out <= 1.0).all()


def test_fit_calibrator_insufficient_data_returns_none():
    calibrator = fit_calibrator(np.array([0.5, 0.6]), np.array([1, 0]), min_samples=50)

    assert calibrator is None


def test_apply_none_calibrator_is_identity():
    out = apply_calibrator(None, np.array([0.2, 0.8]))

    assert out[0] == pytest.approx(0.2)
    assert out[1] == pytest.approx(0.8)


def test_save_and_load_roundtrip():
    rng = np.random.default_rng(2)
    raw = rng.uniform(0, 1, 200)
    labels = (rng.uniform(0, 1, 200) < raw).astype(int)
    calibrator = fit_calibrator(raw, labels)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "cal.joblib")
        save_calibrator(calibrator, path)
        loaded = load_calibrator(path)

    original = apply_calibrator(calibrator, np.array([0.3, 0.7]))
    restored = apply_calibrator(loaded, np.array([0.3, 0.7]))
    assert np.allclose(original, restored)
