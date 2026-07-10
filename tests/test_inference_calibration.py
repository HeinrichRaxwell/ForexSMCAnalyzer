import numpy as np

from src.inference import apply_confidence_calibration, _build_inference_matrix


def test_apply_calibration_uses_calibrator(tmp_path):
    from src.calibrator import fit_calibrator, save_calibrator

    raw = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    calibrator = fit_calibrator(raw, labels, min_samples=4)
    path = tmp_path / "confidence_calibrator.joblib"
    save_calibrator(calibrator, str(path))

    out = apply_confidence_calibration(0.9, calibrator_path=str(path))

    assert 0.0 <= out <= 1.0


def test_apply_calibration_fallback_identity_when_missing():
    out = apply_confidence_calibration(0.73, calibrator_path="does/not/exist.joblib")

    assert out == 0.73


def test_build_inference_matrix_fills_missing_model_features_with_zero():
    rows = [{"hour": 8, "direction": 1}]
    expected_features = ["hour", "direction", "rr_ratio", "reaction_strength"]

    matrix = _build_inference_matrix(rows, expected_features)

    assert matrix.columns.tolist() == expected_features
    assert matrix.iloc[0]["rr_ratio"] == 0.0
    assert matrix.iloc[0]["reaction_strength"] == 0.0
