import json

from src.rollout_status import (
    evaluate_rollout_status,
    load_env_values,
    normalize_threshold_key,
)


def test_evaluate_rollout_status_ready_for_recommended_threshold():
    report = {
        "thresholds": {
            "0.50": {
                "sample_count": 326,
                "expectancy_r": 1.53,
                "max_drawdown_r": 2.0,
                "winrate_pct": 95.09,
                "profit_factor": 36.24,
                "max_consecutive_losses": 2,
            }
        },
        "recommendation": {
            "threshold": "0.50",
            "reason": "lowest_threshold_meeting_rules",
        },
    }

    status = evaluate_rollout_status(
        report,
        env_values={
            "MT5_EXECUTE_TRADES": "True",
            "MT5_MAX_CONCURRENT_TRADES": "1",
            "MT5_ALLOWED_TIMEFRAMES": "H4,D1",
            "MT5_ENFORCE_ENTRY_GATE": "False",
        },
    )

    assert status["status"] == "READY"
    assert status["threshold"] == "0.50"
    assert status["live_execution"] is True
    assert status["metrics"]["expectancy_r"] == 1.53
    assert {check["name"]: check["status"] for check in status["checks"]} == {
        "sample_count": "PASS",
        "expectancy_r": "PASS",
        "max_drawdown_r": "PASS",
        "live_execution": "WARN",
        "max_concurrent_trades": "PASS",
        "allowed_timeframes": "PASS",
        "entry_gate_not_enforced": "PASS",
    }


def test_evaluate_rollout_status_blocks_risky_live_configuration():
    report = {
        "thresholds": {
            "0.50": {
                "sample_count": 326,
                "expectancy_r": 1.53,
                "max_drawdown_r": 2.0,
            }
        },
        "recommendation": {"threshold": "0.50"},
    }

    status = evaluate_rollout_status(
        report,
        env_values={
            "MT5_EXECUTE_TRADES": "True",
            "MT5_MAX_CONCURRENT_TRADES": "0",
            "MT5_ALLOWED_TIMEFRAMES": "",
            "MT5_ENFORCE_ENTRY_GATE": "True",
        },
    )

    checks = {check["name"]: check["status"] for check in status["checks"]}
    assert status["status"] == "BLOCKED"
    assert checks["max_concurrent_trades"] == "FAIL"
    assert checks["allowed_timeframes"] == "FAIL"
    assert checks["entry_gate_not_enforced"] == "FAIL"


def test_evaluate_rollout_status_real_money_requires_artifacts_allowlist_and_governor(tmp_path):
    report = {
        "thresholds": {
            "0.50": {
                "sample_count": 326,
                "expectancy_r": 1.53,
                "max_drawdown_r": 2.0,
                "profit_factor": 2.4,
                "max_consecutive_losses": 2,
            }
        },
        "live_policy": {
            "allowed_timeframes": ["M30", "H1"],
            "blocked_strategies": ["Pivot", "PIVOT_REJECTION", "SND", "Swapzone"],
            "allowed_strategies": [],
            "thresholds": {
                "0.50": {
                    "sample_count": 326,
                    "expectancy_r": 1.53,
                    "max_drawdown_r": 2.0,
                    "profit_factor": 2.4,
                    "max_consecutive_losses": 2,
                }
            },
            "sources": {
                "real": {
                    "sample_count": 200,
                    "expectancy_r": 0.5,
                    "profit_factor": 1.4,
                }
            },
        },
    }

    status = evaluate_rollout_status(
        report,
        env_values={
            "MT5_EXECUTE_TRADES": "True",
            "MT5_MAX_CONCURRENT_TRADES": "1",
            "MT5_ALLOWED_TIMEFRAMES": "M30,H1",
            "MT5_LIVE_STRATEGY_BLOCKLIST": "Pivot,SND,Swapzone",
            "MT5_ENFORCE_ENTRY_GATE": "False",
            "MT5_DAILY_GOVERNOR_ENABLED": "False",
            "ML_LIVE_MIN_THRESHOLD": "0.50",
        },
        requested_threshold=0.50,
        min_samples=100,
        min_expectancy_r=0.25,
        max_drawdown_r=5.0,
        min_profit_factor=1.25,
        max_consecutive_losses=5,
        profile="real-money",
        required_artifacts=[str(tmp_path / "missing.joblib")],
    )

    checks = {check["name"]: check["status"] for check in status["checks"]}
    assert status["status"] == "BLOCKED"
    assert checks["strategy_allowlist"] == "FAIL"
    assert checks["daily_governor_enabled"] == "FAIL"
    assert checks["required_artifacts"] == "FAIL"


def test_evaluate_rollout_status_real_money_passes_with_explicit_policy_and_artifacts(tmp_path):
    artifacts = []
    for filename in ["xgb.joblib", "lgb.joblib", "calibrator.joblib"]:
        path = tmp_path / filename
        path.write_text("ok", encoding="utf-8")
        artifacts.append(str(path))

    report = {
        "live_policy": {
            "allowed_timeframes": ["M30", "H1"],
            "blocked_strategies": ["Pivot", "PIVOT_REJECTION", "SND", "Swapzone"],
            "allowed_strategies": ["FVG_OR_BPR"],
            "overall": {"expectancy_r": 0.4},
            "thresholds": {
                "0.50": {
                    "sample_count": 326,
                    "expectancy_r": 0.53,
                    "max_drawdown_r": 2.0,
                    "profit_factor": 1.6,
                    "max_consecutive_losses": 2,
                }
            },
            "sources": {
                "real": {
                    "sample_count": 200,
                    "expectancy_r": 0.5,
                    "profit_factor": 1.4,
                }
            },
        },
    }

    status = evaluate_rollout_status(
        report,
        env_values={
            "MT5_EXECUTE_TRADES": "True",
            "MT5_MAX_CONCURRENT_TRADES": "1",
            "MT5_ALLOWED_TIMEFRAMES": "M30,H1",
            "MT5_LIVE_STRATEGY_ALLOWLIST": "FVG_OR_BPR",
            "MT5_LIVE_STRATEGY_BLOCKLIST": "Pivot,SND,Swapzone",
            "MT5_ENFORCE_ENTRY_GATE": "False",
            "MT5_DAILY_GOVERNOR_ENABLED": "True",
            "ML_LIVE_MIN_THRESHOLD": "0.50",
        },
        requested_threshold=0.50,
        min_samples=100,
        min_expectancy_r=0.25,
        max_drawdown_r=5.0,
        min_profit_factor=1.25,
        max_consecutive_losses=5,
        profile="real-money",
        required_artifacts=artifacts,
    )

    assert status["status"] == "READY"
    assert {check["status"] for check in status["checks"]} <= {"PASS", "WARN"}


def test_evaluate_rollout_status_blocks_bad_drawdown_threshold():
    report = {
        "thresholds": {
            "0.40": {
                "sample_count": 396,
                "expectancy_r": 1.27,
                "max_drawdown_r": 5.0,
                "winrate_pct": 83.59,
            }
        }
    }

    status = evaluate_rollout_status(
        report,
        env_values={"MT5_EXECUTE_TRADES": "False"},
        requested_threshold=0.40,
    )

    assert status["status"] == "BLOCKED"
    assert status["threshold"] == "0.40"
    assert status["live_execution"] is False
    assert status["checks"][2] == {
        "name": "max_drawdown_r",
        "status": "FAIL",
        "message": "5.0R > max 3.0R",
    }


def test_evaluate_rollout_status_prefers_live_policy_metrics_when_present():
    report = {
        "thresholds": {
            "0.40": {
                "sample_count": 200,
                "expectancy_r": 1.5,
                "max_drawdown_r": 2.0,
                "winrate_pct": 80.0,
            }
        },
        "live_policy": {
            "thresholds": {
                "0.40": {
                    "sample_count": 20,
                    "expectancy_r": -0.2,
                    "max_drawdown_r": 12.0,
                    "winrate_pct": 40.0,
                }
            }
        },
    }

    status = evaluate_rollout_status(
        report,
        env_values={"MT5_EXECUTE_TRADES": "False"},
        requested_threshold=0.40,
        min_samples=10,
    )

    assert status["status"] == "BLOCKED"
    assert status["metrics_source"] == "live_policy"
    assert status["metrics"]["expectancy_r"] == -0.2


def test_load_env_values_parses_comments_quotes_and_live_flag(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "# local config",
                "MT5_EXECUTE_TRADES='true'",
                'ML_ACCEPT_THRESHOLD="0.50"',
                "BROKEN_LINE",
            ]
        )
    )

    values = load_env_values(str(env_path))

    assert values["MT5_EXECUTE_TRADES"] == "true"
    assert values["ML_ACCEPT_THRESHOLD"] == "0.50"
    assert "BROKEN_LINE" not in values


def test_normalize_threshold_key_formats_numeric_values():
    assert normalize_threshold_key(0.5) == "0.50"
    assert normalize_threshold_key("0.6") == "0.60"
    assert normalize_threshold_key("bad") == "bad"
