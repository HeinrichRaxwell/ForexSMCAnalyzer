import numpy as np
import pandas as pd

from src.calibration_report import fit_calibrator_from_scored


def test_fit_calibrator_from_scored_df():
    rng = np.random.default_rng(3)
    n = 300
    raw = rng.uniform(0, 1, n)
    labels = (rng.uniform(0, 1, n) < raw).astype(int)
    df = pd.DataFrame({
        "confidence": raw,
        "label": labels,
        "confidence_source": ["walk_forward_oof"] * n,
    })

    calibrator = fit_calibrator_from_scored(df)
    mapped = calibrator.predict(np.linspace(0, 1, 20))

    assert calibrator is not None
    assert (np.diff(mapped) >= -1e-9).all()


def test_fit_calibrator_ignores_unscored_rows():
    df = pd.DataFrame({
        "confidence": [np.nan, np.nan, 0.5],
        "label": [1, 0, 1],
        "confidence_source": ["unscored", "unscored", "walk_forward_oof"],
    })

    calibrator = fit_calibrator_from_scored(df, min_samples=50)

    assert calibrator is None
