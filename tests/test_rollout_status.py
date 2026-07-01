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

    status = evaluate_rollout_status(report, env_values={"MT5_EXECUTE_TRADES": "True"})

    assert status["status"] == "READY"
    assert status["threshold"] == "0.50"
    assert status["live_execution"] is True
    assert status["metrics"]["expectancy_r"] == 1.53
    assert {check["name"]: check["status"] for check in status["checks"]} == {
        "sample_count": "PASS",
        "expectancy_r": "PASS",
        "max_drawdown_r": "PASS",
        "live_execution": "WARN",
    }


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
