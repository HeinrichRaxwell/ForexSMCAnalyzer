"""Isotonic confidence calibration for ML setup probabilities."""

import os

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression


def fit_calibrator(raw_probs, labels, min_samples: int = 50):
    """Fit raw probability -> empirical winrate mapping from OOF rows."""
    raw = np.asarray(raw_probs, dtype=float)
    y = np.asarray(labels, dtype=float)
    mask = np.isfinite(raw) & np.isfinite(y)
    raw = raw[mask]
    y = y[mask]
    if len(raw) < int(min_samples):
        return None

    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calibrator.fit(raw, y)
    return calibrator


def apply_calibrator(calibrator, raw_probs):
    """Apply calibrator to raw probabilities, or return identity if unavailable."""
    raw = np.asarray(raw_probs, dtype=float)
    if calibrator is None:
        return raw
    return np.clip(calibrator.predict(raw), 0.0, 1.0)


def save_calibrator(calibrator, path: str):
    """Persist a fitted calibrator. No-op when there is not enough data to fit one."""
    if calibrator is None:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    joblib.dump(calibrator, path)


def load_calibrator(path: str):
    """Load a calibrator, returning None for missing or unreadable files."""
    if not path or not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None
