import os
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.calibration_report import (
    assign_confidence_bucket,
    build_calibration_report,
    recommend_threshold,
    score_outcome_dataset,
)


class ConfidenceByHourModel:
    feature_names_in_ = np.array(["hour", "timeframe", "direction"])

    def predict_proba(self, X):
        probs = X["hour"].astype(float).to_numpy() / 100.0
        probs = np.clip(probs, 0.0, 1.0)
        return np.column_stack([1.0 - probs, probs])


def test_assign_confidence_bucket_uses_expected_ranges():
    assert assign_confidence_bucket(0.0) == "0.00-0.30"
    assert assign_confidence_bucket(0.299) == "0.00-0.30"
    assert assign_confidence_bucket(0.30) == "0.30-0.40"
    assert assign_confidence_bucket(0.50) == "0.50-0.60"
    assert assign_confidence_bucket(0.999) == "0.90-1.00"
    assert assign_confidence_bucket(1.0) == "0.90-1.00"


def test_score_outcome_dataset_uses_active_model_probabilities(tmp_path):
    data_path = tmp_path / "labeled_setups.csv"
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    pd.DataFrame({
        "time": ["2026-06-01 08:00:00", "2026-06-01 09:00:00"],
        "hour": [35, 75],
        "timeframe": [15, 60],
        "direction": [1, -1],
        "pnl_relative": [-1.0, 2.0],
        "label": [0, 1],
    }).to_csv(data_path, index=False)
    joblib.dump(ConfidenceByHourModel(), model_dir / "smc_xgb_classifier.joblib")

    scored = score_outcome_dataset(str(data_path), model_dir=str(model_dir))

    assert scored["confidence"].round(2).tolist() == [0.35, 0.75]
    assert scored["confidence_bucket"].tolist() == ["0.30-0.40", "0.70-0.80"]
    assert scored["sample_source"].tolist() == ["real", "real"]


def test_score_outcome_dataset_falls_back_to_stored_shadow_confidence_when_no_model(tmp_path):
    data_path = tmp_path / "labeled_setups.csv"
    shadow_path = tmp_path / "shadow_labeled_setups.csv"
    pd.DataFrame({
        "time": ["2026-06-01 08:00:00"],
        "hour": [8],
        "timeframe": [15],
        "direction": [1],
        "pnl_relative": [2.0],
        "label": [1],
    }).to_csv(data_path, index=False)
    pd.DataFrame({
        "signal_id": ["shadow-a"],
        "sample_source": ["shadow"],
        "time": ["2026-06-01 09:00:00"],
        "hour": [9],
        "timeframe": [60],
        "direction": [-1],
        "confidence": [0.44],
        "pnl_relative": [-1.0],
        "label": [0],
    }).to_csv(shadow_path, index=False)

    scored = score_outcome_dataset(
        str(data_path),
        shadow_labeled_data_path=str(shadow_path),
        model_dir=str(tmp_path / "missing_models"),
    )

    assert scored.loc[scored["sample_source"] == "real", "confidence"].isna().all()
    assert scored.loc[scored["sample_source"] == "shadow", "confidence"].tolist() == [0.44]
    assert scored.loc[scored["sample_source"] == "shadow", "confidence_bucket"].tolist() == ["0.40-0.50"]


def test_build_calibration_report_groups_by_threshold_bucket_timeframe_and_source(tmp_path):
    scored = pd.DataFrame({
        "confidence": [0.35, 0.45, 0.55, 0.65, 0.85],
        "confidence_bucket": ["0.30-0.40", "0.40-0.50", "0.50-0.60", "0.60-0.70", "0.80-0.90"],
        "timeframe": [15, 15, 60, 60, 60],
        "hour": [8, 8, 9, 9, 10],
        "killzone": [1, 1, 2, 2, 0],
        "setup_type": [0, 1, 0, 1, 1],
        "strategy": ["FVG", "BPR", "FVG", "Pivot", "Pivot"],
        "direction": [1, 1, -1, -1, -1],
        "sample_source": ["shadow", "shadow", "real", "real", "real"],
        "pnl_relative": [-1.0, 2.0, -1.0, 2.0, 2.0],
        "label": [0, 1, 0, 1, 1],
    })
    output_path = tmp_path / "calibration_report.json"

    report = build_calibration_report(scored, output_path=str(output_path))

    assert report["overall"]["sample_count"] == 5
    assert report["overall"]["winrate_pct"] == 60.0
    assert report["overall"]["expectancy_r"] == 0.8
    assert report["overall"]["max_drawdown_r"] == 1.0
    assert report["thresholds"]["0.50"]["sample_count"] == 3
    assert report["thresholds"]["0.50"]["winrate_pct"] == 66.67
    assert report["thresholds"]["0.50"]["max_drawdown_r"] == 1.0
    assert report["thresholds"]["0.50"]["profit_factor"] == 4.0
    assert report["thresholds"]["0.50"]["max_consecutive_losses"] == 1
    assert report["thresholds"]["0.70"]["sample_count"] == 1
    assert report["buckets"]["0.40-0.50"]["winrate_pct"] == 100.0
    assert report["timeframes"]["60"]["sample_count"] == 3
    assert report["hours"]["8"]["sample_count"] == 2
    assert report["killzones"]["2"]["sample_count"] == 2
    assert report["setup_types"]["1"]["sample_count"] == 3
    assert report["strategies"]["FVG"]["sample_count"] == 2
    assert report["strategies"]["Pivot"]["winrate_pct"] == 100.0
    assert report["directions"]["-1"]["winrate_pct"] == 66.67
    assert report["sources"]["shadow"]["expectancy_r"] == 0.5
    assert output_path.exists()


def test_build_calibration_report_uses_setup_type_strategy_fallback_when_strategy_missing():
    scored = pd.DataFrame({
        "confidence": [0.55, 0.65, 0.75],
        "confidence_bucket": ["0.50-0.60", "0.60-0.70", "0.70-0.80"],
        "setup_type": [0, 1, 2],
        "sample_source": ["real", "real", "shadow"],
        "pnl_relative": [1.0, -1.0, 2.0],
        "label": [1, 0, 1],
    })

    report = build_calibration_report(scored, output_path=None)

    assert report["strategies"]["FVG_OR_BPR"]["sample_count"] == 1
    assert report["strategies"]["OB_OR_SWAPZONE_IC_SND"]["loss_count"] == 1
    assert report["strategies"]["PIVOT_REJECTION"]["winrate_pct"] == 100.0


def test_build_calibration_report_calculates_drawdown_in_time_order():
    scored = pd.DataFrame({
        "time": [
            "2026-06-01 08:15:00",
            "2026-06-01 08:00:00",
            "2026-06-01 08:30:00",
            "2026-06-01 08:45:00",
        ],
        "confidence": [0.80, 0.80, 0.80, 0.80],
        "confidence_bucket": ["0.80-0.90"] * 4,
        "pnl_relative": [-1.0, 2.0, -1.0, -1.0],
        "label": [0, 1, 0, 0],
    })

    report = build_calibration_report(scored, output_path=None)

    assert report["overall"]["max_drawdown_r"] == 3.0


def test_recommend_threshold_selects_lowest_threshold_that_meets_quality_rules():
    report = {
        "thresholds": {
            "0.30": {"sample_count": 100, "expectancy_r": -0.1, "max_drawdown_r": 12.0},
            "0.40": {"sample_count": 90, "expectancy_r": 0.2, "max_drawdown_r": 8.0},
            "0.50": {"sample_count": 70, "expectancy_r": 0.8, "max_drawdown_r": 3.0},
            "0.60": {"sample_count": 20, "expectancy_r": 1.2, "max_drawdown_r": 1.0},
        }
    }

    recommendation = recommend_threshold(
        report,
        min_samples=50,
        min_expectancy_r=0.5,
        max_drawdown_r=5.0,
    )

    assert recommendation == {
        "threshold": "0.50",
        "sample_count": 70,
        "expectancy_r": 0.8,
        "max_drawdown_r": 3.0,
        "reason": "lowest_threshold_meeting_rules",
    }


def test_recommend_threshold_falls_back_to_most_conservative_when_none_pass():
    report = {
        "thresholds": {
            "0.30": {"sample_count": 100, "expectancy_r": -0.1, "max_drawdown_r": 12.0},
            "0.40": {"sample_count": 90, "expectancy_r": 0.2, "max_drawdown_r": 8.0},
        }
    }

    recommendation = recommend_threshold(
        report,
        min_samples=50,
        min_expectancy_r=0.5,
        max_drawdown_r=5.0,
    )

    assert recommendation["threshold"] == "0.40"
    assert recommendation["reason"] == "no_threshold_met_all_rules"
