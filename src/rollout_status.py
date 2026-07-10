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


def _parse_int(value, default=None):
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def _parse_float(value, default=None):
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def _csv_items(value):
    if not value:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _csv_set(value):
    return set(_csv_items(value))


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


def _warn(name, passed, pass_message, warn_message):
    return {
        "name": name,
        "status": "PASS" if passed else "WARN",
        "message": pass_message if passed else warn_message,
    }


def _metric_float(metrics, name, default=0.0):
    try:
        return float(metrics.get(name))
    except (AttributeError, TypeError, ValueError):
        return default


def _metric_int(metrics, name, default=0):
    try:
        return int(metrics.get(name))
    except (AttributeError, TypeError, ValueError):
        return default


def _existing_paths(paths):
    missing = []
    for path in paths:
        resolved = _resolve_project_path(path)
        if not resolved or not os.path.exists(resolved):
            missing.append(path)
    return missing


def _report_live_policy_matches_env(report, env_values):
    live_policy = report.get("live_policy") or {}
    report_timeframes = set(live_policy.get("allowed_timeframes") or [])
    env_timeframes = _csv_set(env_values.get("MT5_ALLOWED_TIMEFRAMES"))
    report_allowlist = set(live_policy.get("allowed_strategies") or [])
    env_allowlist = _csv_set(env_values.get("MT5_LIVE_STRATEGY_ALLOWLIST"))
    report_blocklist = set(live_policy.get("blocked_strategies") or [])
    env_blocklist = _csv_set(env_values.get("MT5_LIVE_STRATEGY_BLOCKLIST"))

    checks = []
    if env_timeframes:
        checks.append(report_timeframes == env_timeframes)
    if env_allowlist:
        checks.append(report_allowlist == env_allowlist)
    if env_blocklist:
        checks.append(report_blocklist == env_blocklist or env_blocklist.issubset(report_blocklist))
    return all(checks) if checks else False


def evaluate_rollout_status(
    report,
    env_values=None,
    requested_threshold=None,
    min_samples=50,
    min_expectancy_r=1.0,
    max_drawdown_r=3.0,
    min_profit_factor=None,
    max_consecutive_losses=None,
    profile="paper",
    required_artifacts=None,
):
    env_values = env_values or {}
    profile = str(profile or "paper").strip().lower().replace("_", "-")
    recommendation = report.get("recommendation") or {}
    threshold = requested_threshold
    if threshold is None:
        threshold = recommendation.get("threshold", "0.50")
    threshold_key = normalize_threshold_key(threshold)

    metrics_source = "overall"
    metrics_pool = report.get("thresholds") or {}
    live_policy = report.get("live_policy") or {}
    live_policy_thresholds = live_policy.get("thresholds") or {}
    if threshold_key in live_policy_thresholds:
        metrics_pool = live_policy_thresholds
        metrics_source = "live_policy"

    metrics = metrics_pool.get(threshold_key, {})
    sample_count = int(metrics.get("sample_count") or 0)
    expectancy = float(metrics.get("expectancy_r") or 0.0)
    max_drawdown = float(metrics.get("max_drawdown_r") or 0.0)
    profit_factor = _metric_float(metrics, "profit_factor")
    consecutive_losses = _metric_int(metrics, "max_consecutive_losses")
    live_execution = _parse_bool(env_values.get("MT5_EXECUTE_TRADES", "False"))
    max_concurrent = _parse_int(env_values.get("MT5_MAX_CONCURRENT_TRADES"), 0)
    allowed_timeframes = _csv_items(env_values.get("MT5_ALLOWED_TIMEFRAMES"))
    entry_gate_enforced = _parse_bool(env_values.get("MT5_ENFORCE_ENTRY_GATE", "False"))
    daily_governor_enabled = _parse_bool(env_values.get("MT5_DAILY_GOVERNOR_ENABLED", "False"))
    strategy_allowlist = _csv_items(env_values.get("MT5_LIVE_STRATEGY_ALLOWLIST"))
    live_min_threshold = _parse_float(env_values.get("ML_LIVE_MIN_THRESHOLD"), 0.0)
    real_money_profile = profile in {"real-money", "real", "vps-real-money"}

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
        _check(
            "max_concurrent_trades",
            (not live_execution) or (max_concurrent is not None and max_concurrent > 0),
            f"MT5_MAX_CONCURRENT_TRADES={max_concurrent}",
            "MT5_MAX_CONCURRENT_TRADES must be > 0 before VPS/live trading.",
        ),
        _check(
            "allowed_timeframes",
            bool(allowed_timeframes),
            f"Allowed live timeframes: {', '.join(allowed_timeframes)}",
            "MT5_ALLOWED_TIMEFRAMES is empty; scanner has no explicit live timeframe policy.",
        ),
        _check(
            "entry_gate_not_enforced",
            not entry_gate_enforced,
            "Entry quality is telemetry-only; it will not block execution.",
            "MT5_ENFORCE_ENTRY_GATE=True conflicts with no-entry-gate live policy.",
        ),
    ]

    if min_profit_factor is not None:
        checks.append(
            _check(
                "profit_factor",
                profit_factor >= float(min_profit_factor),
                f"{profit_factor} >= min {min_profit_factor}",
                f"{profit_factor} < min {min_profit_factor}",
            )
        )

    if max_consecutive_losses is not None:
        checks.append(
            _check(
                "max_consecutive_losses",
                consecutive_losses <= int(max_consecutive_losses),
                f"{consecutive_losses} <= max {max_consecutive_losses}",
                f"{consecutive_losses} > max {max_consecutive_losses}",
            )
        )

    if required_artifacts:
        missing_artifacts = _existing_paths(required_artifacts)
        checks.append(
            _check(
                "required_artifacts",
                not missing_artifacts,
                f"Found required artifacts: {', '.join(required_artifacts)}",
                f"Missing required artifacts: {', '.join(missing_artifacts)}",
            )
        )

    if real_money_profile:
        live_policy = report.get("live_policy") or {}
        live_policy_sources = live_policy.get("sources") or {}
        real_source = live_policy_sources.get("real") or (report.get("sources") or {}).get("real") or {}
        real_expectancy = _metric_float(real_source, "expectancy_r")
        real_profit_factor = _metric_float(real_source, "profit_factor")
        live_policy_overall = live_policy.get("overall") or {}
        configured_allowlist = ", ".join(strategy_allowlist)

        checks.extend(
            [
                _warn(
                    "live_execution_enabled",
                    live_execution,
                    "MT5_EXECUTE_TRADES=True for explicit real-money run.",
                    "MT5_EXECUTE_TRADES=False; this is safe for preflight and must be enabled only after READY.",
                ),
                _check(
                    "strategy_allowlist",
                    bool(strategy_allowlist),
                    f"Live strategy allowlist: {configured_allowlist}",
                    "MT5_LIVE_STRATEGY_ALLOWLIST must be explicit for real-money VPS.",
                ),
                _check(
                    "daily_governor_enabled",
                    daily_governor_enabled,
                    "MT5_DAILY_GOVERNOR_ENABLED=True.",
                    "MT5_DAILY_GOVERNOR_ENABLED must be True for real-money VPS.",
                ),
                _check(
                    "live_policy_report_present",
                    bool(live_policy.get("thresholds")),
                    "Calibration report includes live_policy metrics.",
                    "Calibration report is missing live_policy metrics; regenerate it before VPS.",
                ),
                _check(
                    "live_policy_matches_env",
                    _report_live_policy_matches_env(report, env_values),
                    "Calibration live_policy matches current .env policy.",
                    "Regenerate calibration_report.json after changing live timeframes/strategy policy.",
                ),
                _check(
                    "live_min_threshold",
                    live_min_threshold >= float(threshold_key),
                    f"ML_LIVE_MIN_THRESHOLD={live_min_threshold:.2f} >= rollout threshold {threshold_key}.",
                    f"ML_LIVE_MIN_THRESHOLD={live_min_threshold:.2f} is below rollout threshold {threshold_key}.",
                ),
                _check(
                    "real_source_expectancy",
                    real_expectancy >= 0.0,
                    f"Real-source live-policy expectancy is non-negative: {real_expectancy}R.",
                    f"Real-source live-policy expectancy is negative: {real_expectancy}R.",
                ),
                _check(
                    "real_source_profit_factor",
                    real_profit_factor >= 1.0,
                    f"Real-source live-policy profit factor is >= 1.0: {real_profit_factor}.",
                    f"Real-source live-policy profit factor is < 1.0: {real_profit_factor}.",
                ),
                _warn(
                    "live_policy_overall",
                    _metric_float(live_policy_overall, "expectancy_r") >= 0.0,
                    "Overall live-policy expectancy is non-negative.",
                    "Overall live-policy expectancy is negative; threshold subset must carry the edge.",
                ),
            ]
        )

    has_failures = any(check["status"] == "FAIL" for check in checks)
    return {
        "status": "BLOCKED" if has_failures else "READY",
        "profile": profile,
        "threshold": threshold_key,
        "metrics_source": metrics_source,
        "live_execution": live_execution,
        "metrics": metrics,
        "checks": checks,
        "recommendation": recommendation,
    }


def print_rollout_status(status):
    print(f"[Rollout] Status: {status['status']}")
    print(f"[Rollout] Profile: {status.get('profile', 'paper')}")
    print(f"[Rollout] Threshold: {status['threshold']}")
    print(f"[Rollout] Metrics source: {status.get('metrics_source', 'overall')}")
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
    parser.add_argument("--profile", choices=["paper", "real-money"], default="paper")
    parser.add_argument("--min-samples", type=int, default=None)
    parser.add_argument("--min-expectancy-r", type=float, default=None)
    parser.add_argument("--max-drawdown-r", type=float, default=None)
    parser.add_argument("--min-profit-factor", type=float, default=None)
    parser.add_argument("--max-consecutive-losses", type=int, default=None)
    args = parser.parse_args(argv)

    min_samples = 50 if args.min_samples is None else args.min_samples
    min_expectancy_r = 1.0 if args.min_expectancy_r is None else args.min_expectancy_r
    max_drawdown_r = 3.0 if args.max_drawdown_r is None else args.max_drawdown_r
    min_profit_factor = args.min_profit_factor
    max_consecutive_losses = args.max_consecutive_losses
    required_artifacts = None
    if args.profile == "real-money":
        min_samples = 100 if args.min_samples is None else max(min_samples, 100)
        min_expectancy_r = 0.25 if args.min_expectancy_r is None else max(min_expectancy_r, 0.25)
        max_drawdown_r = 5.0 if args.max_drawdown_r is None else min(max_drawdown_r, 5.0)
        min_profit_factor = max(min_profit_factor or 0.0, 1.25)
        max_consecutive_losses = min(max_consecutive_losses or 999999, 5)
        required_artifacts = [
            "models/smc_xgb_classifier.joblib",
            "models/smc_lgb_classifier.joblib",
            "models/confidence_calibrator.joblib",
            "data/calibration_report.json",
        ]

    report = _load_json(args.report)
    env_values = load_env_values(args.env)
    status = evaluate_rollout_status(
        report,
        env_values=env_values,
        requested_threshold=args.threshold,
        min_samples=min_samples,
        min_expectancy_r=min_expectancy_r,
        max_drawdown_r=max_drawdown_r,
        min_profit_factor=min_profit_factor,
        max_consecutive_losses=max_consecutive_losses,
        profile=args.profile,
        required_artifacts=required_artifacts,
    )
    print_rollout_status(status)
    return 1 if status["status"] == "BLOCKED" else 0


if __name__ == "__main__":
    sys.exit(main())
