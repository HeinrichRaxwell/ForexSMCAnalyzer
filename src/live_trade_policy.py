import math
import os


DEFAULT_BLOCKED_STRATEGIES = {"Pivot", "SND", "Swapzone"}


def _csv_set(value: str | None) -> set[str]:
    if not value:
        return set()
    return {item.strip() for item in value.split(",") if item.strip()}


def _entry_policy_prefix(entry_type: str | None) -> str | None:
    """Return the environment-variable prefix for an execution path."""
    normalized = " ".join(str(entry_type or "").strip().upper().split())
    return {
        "WATCHZONE": "MT5_WATCH_ZONE",
        "WATCH ZONE": "MT5_WATCH_ZONE",
        "STANDARD LIMIT": "MT5_STANDARD_LIMIT",
    }.get(normalized)


def _matches_scoped_strategy_rule(
    rule: str,
    *,
    raw_strategy: str,
    normalized_strategy: str,
    timeframe: str | None,
) -> bool:
    """Match a policy rule written as ``TIMEFRAME:STRATEGY``.

    ``*`` matches any timeframe or strategy. A strategy-only rule remains
    supported for concise all-timeframe blocks.
    """
    item = str(rule or "").strip()
    if not item:
        return False
    if ":" in item:
        rule_timeframe, rule_strategy = (part.strip() for part in item.split(":", 1))
    else:
        rule_timeframe, rule_strategy = "*", item

    actual_timeframe = str(timeframe or "").strip().upper()
    timeframe_matches = rule_timeframe in {"", "*"} or rule_timeframe.upper() == actual_timeframe
    strategy_matches = rule_strategy in {"", "*"} or rule_strategy in {raw_strategy, normalized_strategy}
    return timeframe_matches and strategy_matches


def _matches_scoped_strategy_rules(
    rules: set[str],
    *,
    raw_strategy: str,
    normalized_strategy: str,
    timeframe: str | None,
) -> bool:
    return any(
        _matches_scoped_strategy_rule(
            rule,
            raw_strategy=raw_strategy,
            normalized_strategy=normalized_strategy,
            timeframe=timeframe,
        )
        for rule in rules
    )


def _read_float_env(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _as_float(value, default: float | None = None) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


_TIER_HIGH_DEFAULT = 0.65
_TIER_ULTRA_DEFAULT = 0.80


def confidence_tier(probability) -> str:
    """Return execution priority tier label for a given ML probability.

    - 'ULTRA'  : prob >= ML_ULTRA_CONFIDENCE_TIER  (default 0.80)
    - 'HIGH'   : prob >= ML_HIGH_CONFIDENCE_TIER   (default 0.65)
    - 'NORMAL' : anything below HIGH threshold, or non-numeric input

    Thresholds are configurable via env vars.
    Strategy blocklist is unchanged — this label does not override any block.
    """
    prob = _as_float(probability)
    if prob is None:
        return "NORMAL"
    high_threshold = _read_float_env("ML_HIGH_CONFIDENCE_TIER", _TIER_HIGH_DEFAULT)
    ultra_threshold = _read_float_env("ML_ULTRA_CONFIDENCE_TIER", _TIER_ULTRA_DEFAULT)
    if prob >= ultra_threshold:
        return "ULTRA"
    if prob >= high_threshold:
        return "HIGH"
    return "NORMAL"


def _as_int(value, default: int | None = None) -> int | None:
    try:
        result = int(float(value))
    except (TypeError, ValueError):
        return default
    return result


def _setup_features(setup: dict | None) -> dict:
    if not isinstance(setup, dict):
        return {}
    features = setup.get("features")
    return features if isinstance(features, dict) else {}


def _is_m15_timeframe(timeframe: str | None, setup: dict | None, features: dict) -> bool:
    raw_values = [
        timeframe,
        (setup or {}).get("timeframe") if isinstance(setup, dict) else None,
        features.get("timeframe"),
    ]
    for raw_value in raw_values:
        if raw_value is None:
            continue
        normalized = str(raw_value).strip().upper()
        if normalized in {"M15", "15", "15.0"}:
            return True
    return False


def _is_countertrend_knn_ic_trap(
    strategy: str,
    setup: dict | None,
    *,
    probability: float | None = None,
    timeframe: str | None = None,
) -> bool:
    if str(strategy or "").strip() != "IC":
        return False

    features = _setup_features(setup)
    if not _is_m15_timeframe(timeframe, setup, features):
        return False

    direction = _as_int((setup or {}).get("direction") if isinstance(setup, dict) else None)
    if direction is None:
        direction = _as_int(features.get("direction"))
    trend = _as_int(features.get("trend", (setup or {}).get("trend") if isinstance(setup, dict) else None))
    if direction not in {-1, 1} or trend not in {-1, 1} or direction == trend:
        return False

    prob = _as_float(probability)
    max_low_confidence = _read_float_env("ML_M15_IC_COUNTERTREND_MAX_CONFIDENCE", 0.50)
    if prob is None or prob >= max_low_confidence:
        return False

    knn_sig = _as_float(features.get("knn_prob_sig"))
    knn_opp = _as_float(features.get("knn_prob_opp"))
    if knn_sig is None or knn_opp is None:
        return False

    min_margin = _read_float_env("ML_M15_IC_COUNTERTREND_KNN_OPP_MARGIN", 0.10)
    return (knn_opp - knn_sig) >= min_margin


def normalize_strategy_name(strategy: str, setup: dict | None = None) -> str:
    setup = setup or {}
    name = str(strategy or "").strip()
    try:
        setup_type = int(setup.get("setup_type"))
    except (TypeError, ValueError):
        setup_type = None

    option_name = str(setup.get("option_name", ""))
    
    # Handle Pivot Rejection vs Legacy Pivot
    if name == "Pivot" and (setup_type == 2 or "Pivot" in option_name):
        return "PIVOT_REJECTION"
    if name == "PIVOT_REJECTION" or (not name and setup_type == 2):
        return "PIVOT_REJECTION"
        
    # Normalize FVG_OR_BPR
    if name in {"FVG", "BPR", "FVG_OR_BPR"} or (not name and setup_type == 0):
        return "FVG_OR_BPR"
        
    # Normalize OB_OR_SWAPZONE_IC_SND
    if name in {"OB", "Swapzone", "IC", "SND", "Breaker", "OB_OR_SWAPZONE_IC_SND"} or (not name and setup_type == 1):
        return "OB_OR_SWAPZONE_IC_SND"
        
    return name


def should_allow_live_strategy(
    strategy: str,
    setup: dict | None = None,
    *,
    probability: float | None = None,
    timeframe: str | None = None,
    entry_type: str | None = None,
) -> tuple[bool, str]:
    raw_name = str(strategy or "").strip()
    normalized = normalize_strategy_name(strategy, setup)
    allowlist = _csv_set(os.getenv("MT5_LIVE_STRATEGY_ALLOWLIST"))
    configured_blocklist = os.getenv("MT5_LIVE_STRATEGY_BLOCKLIST")
    blocklist = DEFAULT_BLOCKED_STRATEGIES if configured_blocklist is None else _csv_set(configured_blocklist)

    if allowlist and normalized not in allowlist and raw_name not in allowlist:
        return False, f"strategy_not_allowlisted:{normalized}"
    if raw_name in blocklist:
        return False, f"strategy_blocked:{raw_name}"
    if normalized in blocklist:
        return False, f"strategy_blocked:{normalized}"

    prefix = _entry_policy_prefix(entry_type)
    if prefix:
        scoped_allowlist = _csv_set(os.getenv(f"{prefix}_STRATEGY_ALLOWLIST"))
        scoped_blocklist = _csv_set(os.getenv(f"{prefix}_STRATEGY_BLOCKLIST"))
        policy_context = f"{entry_type}:{str(timeframe or '').upper()}:{raw_name}"
        if scoped_allowlist and not _matches_scoped_strategy_rules(
            scoped_allowlist,
            raw_strategy=raw_name,
            normalized_strategy=normalized,
            timeframe=timeframe,
        ):
            return False, f"entry_policy_not_allowlisted:{policy_context}"
        if _matches_scoped_strategy_rules(
            scoped_blocklist,
            raw_strategy=raw_name,
            normalized_strategy=normalized,
            timeframe=timeframe,
        ):
            return False, f"entry_policy_blocked:{policy_context}"

    if _is_countertrend_knn_ic_trap(raw_name, setup, probability=probability, timeframe=timeframe):
        return False, "execution_veto:m15_ic_countertrend_knn_opposition"
    return True, f"strategy_allowed:{normalized}"
