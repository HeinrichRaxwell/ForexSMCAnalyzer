import argparse
import json
import os
import sys


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve_project_path(path):
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    return os.path.join(BASE_DIR, path)


def normalize_threshold_key(value):
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)


def _parse_bool(value):
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def load_env_values(env_path=".env"):
    env_path = _resolve_project_path(env_path)
    values = {}
    if not env_path or not os.path.exists(env_path):
        return values

    with open(env_path, "r", encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, raw_value = line.split("=", 1)
            key = key.strip()
            value = raw_value.strip().strip("\"'")
            if key:
                values[key] = value
    return values


def _load_json(path):
    path = _resolve_project_path(path)
    with open(path, "r", encoding="utf-8") as json_file:
        return json.load(json_file)


def _check(name, passed, pass_message, fail_message):
    return {
        "name": name,
        "status": "PASS" if passed else "FAIL",
        "message": pass_message if passed else fail_message,
    }


def evaluate_rollout_status(
    report,
    env_values=None,
    requested_threshold=None,
    min_samples=50,
    min_expectancy_r=1.0,
    max_drawdown_r=3.0,
):
    env_values = env_values or {}
    recommendation = report.get("recommendation") or {}
    threshold = requested_threshold
    if threshold is None:
        threshold = recommendation.get("threshold", "0.50")
    threshold_key = normalize_threshold_key(threshold)

    metrics = (report.get("thresholds") or {}).get(threshold_key, {})
    sample_count = int(metrics.get("sample_count") or 0)
    expectancy = float(metrics.get("expectancy_r") or 0.0)
    max_drawdown = float(metrics.get("max_drawdown_r") or 0.0)
    live_execution = _parse_bool(env_values.get("MT5_EXECUTE_TRADES", "False"))

    checks = [
        _check(
            "sample_count",
            sample_count >= min_samples,
            f"{sample_count} >= min {min_samples}",
            f"{sample_count} < min {min_samples}",
        ),
        _check(
            "expectancy_r",
            expectancy >= min_expectancy_r,
            f"{expectancy}R >= min {min_expectancy_r}R",
            f"{expectancy}R < min {min_expectancy_r}R",
        ),
        _check(
            "max_drawdown_r",
            max_drawdown <= max_drawdown_r,
            f"{max_drawdown}R <= max {max_drawdown_r}R",
            f"{max_drawdown}R > max {max_drawdown_r}R",
        ),
        {
            "name": "live_execution",
            "status": "WARN" if live_execution else "PASS",
            "message": "MT5_EXECUTE_TRADES=True; scanner loop can place real orders."
            if live_execution
            else "MT5_EXECUTE_TRADES is not enabled.",
        },
    ]

    has_failures = any(check["status"] == "FAIL" for check in checks)
    return {
        "status": "BLOCKED" if has_failures else "READY",
        "threshold": threshold_key,
        "live_execution": live_execution,
        "metrics": metrics,
        "checks": checks,
        "recommendation": recommendation,
    }


def print_rollout_status(status):
    print(f"[Rollout] Status: {status['status']}")
    print(f"[Rollout] Threshold: {status['threshold']}")
    print(f"[Rollout] Live execution: {status['live_execution']}")
    metrics = status.get("metrics") or {}
    if metrics:
        print(
            "[Rollout] Metrics: "
            f"samples={metrics.get('sample_count')}, "
            f"winrate={metrics.get('winrate_pct')}%, "
            f"expectancy={metrics.get('expectancy_r')}R, "
            f"max_dd={metrics.get('max_drawdown_r')}R"
        )
    for check in status["checks"]:
        print(f"[{check['status']}] {check['name']}: {check['message']}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Offline rollout gate before live scanner.")
    parser.add_argument("--report", default="data/calibration_report.json")
    parser.add_argument("--env", default=".env")
    parser.add_argument("--threshold", default=None)
    parser.add_argument("--min-samples", type=int, default=50)
    parser.add_argument("--min-expectancy-r", type=float, default=1.0)
    parser.add_argument("--max-drawdown-r", type=float, default=3.0)
    args = parser.parse_args(argv)

    report = _load_json(args.report)
    env_values = load_env_values(args.env)
    status = evaluate_rollout_status(
        report,
        env_values=env_values,
        requested_threshold=args.threshold,
        min_samples=args.min_samples,
        min_expectancy_r=args.min_expectancy_r,
        max_drawdown_r=args.max_drawdown_r,
    )
    print_rollout_status(status)
    return 1 if status["status"] == "BLOCKED" else 0


if __name__ == "__main__":
    sys.exit(main())
