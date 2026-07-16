import pytest
import os
from src.live_trade_policy import normalize_strategy_name, should_allow_live_strategy


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    monkeypatch.delenv("MT5_LIVE_STRATEGY_ALLOWLIST", raising=False)
    monkeypatch.delenv("MT5_LIVE_STRATEGY_BLOCKLIST", raising=False)
    monkeypatch.delenv("MT5_WATCH_ZONE_STRATEGY_ALLOWLIST", raising=False)
    monkeypatch.delenv("MT5_WATCH_ZONE_STRATEGY_BLOCKLIST", raising=False)
    monkeypatch.delenv("MT5_STANDARD_LIMIT_STRATEGY_ALLOWLIST", raising=False)
    monkeypatch.delenv("MT5_STANDARD_LIMIT_STRATEGY_BLOCKLIST", raising=False)


def test_reaction_pivot_setup_is_normalized_separately_from_legacy_pivot():
    setup = {"setup_type": 2, "option_name": "Pivot Rejection BUY"}

    assert normalize_strategy_name("Pivot", setup) == "PIVOT_REJECTION"


def test_legacy_pivot_is_blocked_by_default():
    allowed, reason = should_allow_live_strategy("Pivot", {"setup_type": 1})

    assert allowed is False
    assert reason == "strategy_blocked:Pivot"


def test_reaction_pivot_respects_raw_pivot_blocklist():
    assert should_allow_live_strategy("Pivot", {"setup_type": 2}) == (False, "strategy_blocked:Pivot")


def test_core_smc_are_allowed_by_default():
    assert should_allow_live_strategy("FVG", {"setup_type": 0}) == (True, "strategy_allowed:FVG_OR_BPR")
    assert should_allow_live_strategy("BPR", {"setup_type": 0}) == (True, "strategy_allowed:FVG_OR_BPR")


def test_snd_and_swapzone_are_blocked_by_default_due_to_negative_forward_metrics():
    assert should_allow_live_strategy("SND", {"setup_type": 1}) == (False, "strategy_blocked:SND")
    assert should_allow_live_strategy("Swapzone", {"setup_type": 1}) == (False, "strategy_blocked:Swapzone")


def test_m15_ic_countertrend_knn_opposition_is_vetoed():
    setup = {
        "direction": 1,
        "features": {
            "timeframe": 15,
            "direction": 1,
            "trend": -1,
            "knn_prob_sig": 0.3871881624258358,
            "knn_prob_opp": 0.6128118375741641,
        },
    }

    assert should_allow_live_strategy("IC", setup, probability=0.45461783439490444, timeframe="M15") == (
        False,
        "execution_veto:m15_ic_countertrend_knn_opposition",
    )


def test_m15_ic_countertrend_with_neutral_knn_remains_allowed():
    setup = {
        "direction": 1,
        "features": {
            "timeframe": 15,
            "direction": 1,
            "trend": -1,
            "knn_prob_sig": 0.49895494672426194,
            "knn_prob_opp": 0.5010450532757379,
        },
    }

    assert should_allow_live_strategy("IC", setup, probability=0.45461783439490444, timeframe="M15") == (
        True,
        "strategy_allowed:OB_OR_SWAPZONE_IC_SND",
    )


def test_h1_ic_low_confidence_trend_aligned_is_not_vetoed_by_m15_rule():
    setup = {
        "direction": -1,
        "features": {
            "timeframe": 60,
            "direction": -1,
            "trend": -1,
            "knn_prob_sig": 0.715576857493041,
            "knn_prob_opp": 0.28442314250695905,
        },
    }

    assert should_allow_live_strategy("IC", setup, probability=0.45461783439490444, timeframe="H1") == (
        True,
        "strategy_allowed:OB_OR_SWAPZONE_IC_SND",
    )


def test_watch_zone_policy_blocks_m30_fvg_but_keeps_m30_ob_candidate(monkeypatch):
    monkeypatch.setenv("MT5_LIVE_STRATEGY_BLOCKLIST", "")
    monkeypatch.setenv("MT5_WATCH_ZONE_STRATEGY_ALLOWLIST", "M30:OB,H1:OB")

    assert should_allow_live_strategy("FVG", timeframe="M30", entry_type="WatchZone") == (
        False,
        "entry_policy_not_allowlisted:WatchZone:M30:FVG",
    )
    assert should_allow_live_strategy("OB", timeframe="M30", entry_type="WatchZone") == (
        True,
        "strategy_allowed:OB_OR_SWAPZONE_IC_SND",
    )


def test_watch_zone_policy_does_not_block_standard_limit(monkeypatch):
    monkeypatch.setenv("MT5_LIVE_STRATEGY_BLOCKLIST", "")
    monkeypatch.setenv("MT5_WATCH_ZONE_STRATEGY_ALLOWLIST", "M30:OB,H1:OB")

    assert should_allow_live_strategy("FVG", timeframe="M30", entry_type="Standard Limit") == (
        True,
        "strategy_allowed:FVG_OR_BPR",
    )


def test_standard_limit_policy_blocks_bb_and_h4_ob(monkeypatch):
    monkeypatch.setenv("MT5_LIVE_STRATEGY_BLOCKLIST", "")
    monkeypatch.setenv("MT5_STANDARD_LIMIT_STRATEGY_BLOCKLIST", "*:BB,H4:OB")

    assert should_allow_live_strategy("BB", timeframe="M30", entry_type="Standard Limit") == (
        False,
        "entry_policy_blocked:Standard Limit:M30:BB",
    )
    assert should_allow_live_strategy("OB", timeframe="H4", entry_type="Standard Limit") == (
        False,
        "entry_policy_blocked:Standard Limit:H4:OB",
    )
    assert should_allow_live_strategy("OB", timeframe="M30", entry_type="Standard Limit") == (
        True,
        "strategy_allowed:OB_OR_SWAPZONE_IC_SND",
    )


import os
import pytest
from src.live_trade_policy import confidence_tier


def test_tier_normal_below_high(monkeypatch):
    monkeypatch.setenv("ML_HIGH_CONFIDENCE_TIER", "0.65")
    monkeypatch.setenv("ML_ULTRA_CONFIDENCE_TIER", "0.80")
    assert confidence_tier(0.55) == "NORMAL"
    assert confidence_tier(0.50) == "NORMAL"
    assert confidence_tier(0.6499) == "NORMAL"


def test_tier_high_between_thresholds(monkeypatch):
    monkeypatch.setenv("ML_HIGH_CONFIDENCE_TIER", "0.65")
    monkeypatch.setenv("ML_ULTRA_CONFIDENCE_TIER", "0.80")
    assert confidence_tier(0.65) == "HIGH"
    assert confidence_tier(0.70) == "HIGH"
    assert confidence_tier(0.7999) == "HIGH"


def test_tier_ultra_at_and_above(monkeypatch):
    monkeypatch.setenv("ML_HIGH_CONFIDENCE_TIER", "0.65")
    monkeypatch.setenv("ML_ULTRA_CONFIDENCE_TIER", "0.80")
    assert confidence_tier(0.80) == "ULTRA"
    assert confidence_tier(0.95) == "ULTRA"
    assert confidence_tier(1.0) == "ULTRA"


def test_tier_none_returns_normal():
    assert confidence_tier(None) == "NORMAL"
    assert confidence_tier("bad") == "NORMAL"


def test_tier_uses_env_thresholds(monkeypatch):
    """Tier boundaries are fully configurable via env vars."""
    monkeypatch.setenv("ML_HIGH_CONFIDENCE_TIER", "0.55")
    monkeypatch.setenv("ML_ULTRA_CONFIDENCE_TIER", "0.70")
    assert confidence_tier(0.55) == "HIGH"
    assert confidence_tier(0.70) == "ULTRA"
    assert confidence_tier(0.54) == "NORMAL"
