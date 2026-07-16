from datetime import datetime, timedelta

from src.price_watch_zones import (
    WatchZone,
    _execution_retry_elapsed,
    build_watch_zone_execution_setup,
    save_watch_zones,
)


def _dual_zone(**overrides):
    data = {
        "zone_id": "XAUUSD_M30_FVG_BULL_4000_20260715T1200",
        "symbol": "XAUUSD",
        "timeframe": "M30",
        "strategy": "FVG",
        "direction": 1,
        "entry_price": 4000.0,
        "sl_price": 3995.0,
        "tp_price": 4010.0,
        "zone_top": 4000.0,
        "zone_bottom": 3998.0,
        "probability": 0.71,
        "probability_b": 0.77,
        "entry_price_b": 3998.0,
        "sl_price_b": 3994.0,
        "tp_price_b": 4008.0,
        "features": {},
        "features_b": {},
        "oscillator_label": "BUY",
        "is_dual": True,
        "created_at": "2026-07-15 12:00:00",
        "expires_at": "",
        "rejection_confirmed": True,
        "rejection_confirmed_b": True,
    }
    data.update(overrides)
    return WatchZone(**data)


def test_watch_zone_uses_option_b_risk_profile_when_hit_is_near_option_b():
    setup = build_watch_zone_execution_setup(_dual_zone(), current_price=3998.05)

    assert setup["entry_price"] == 3998.0
    assert setup["sl_price"] == 3994.0
    assert setup["tp_price"] == 4008.0
    assert setup["probability"] == 0.77
    assert setup["rejection_confirmed"] is True
    assert "Option B" in setup["option_name"]


def test_watch_zone_uses_option_a_risk_profile_when_hit_is_near_option_a():
    setup = build_watch_zone_execution_setup(_dual_zone(), current_price=3999.95)

    assert setup["entry_price"] == 4000.0
    assert setup["sl_price"] == 3995.0
    assert setup["tp_price"] == 4010.0
    assert "Option A" in setup["option_name"]


def test_watch_zone_retry_cooldown_blocks_immediate_repeat_attempt(monkeypatch):
    import src.price_watch_zones as zones

    monkeypatch.setattr(zones, "_ZONE_EXECUTION_RETRY_SECONDS", 30.0)
    now = datetime.now()
    blocked = {"last_execution_attempt": now.strftime("%Y-%m-%d %H:%M:%S")}
    eligible = {
        "last_execution_attempt": (now - timedelta(seconds=31)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    }

    assert _execution_retry_elapsed(blocked, now) is False
    assert _execution_retry_elapsed(eligible, now) is True


def test_watch_zone_refreshes_option_b_rejection_state(monkeypatch, tmp_path):
    import src.price_watch_zones as zones

    monkeypatch.setattr(zones, "_DATA_DIR", str(tmp_path))
    option_a = {
        "timeframe": "M30", "strategy": "FVG", "direction": 1,
        "entry_price": 4000.0, "sl_price": 3995.0, "tp_price": 4010.0,
        "time": "2026-07-15 12:00:00", "rejection_confirmed": False,
    }
    option_b = {
        **option_a,
        "entry_price": 3998.0, "sl_price": 3994.0, "tp_price": 4008.0,
        "rejection_confirmed": False,
    }
    candidate = {"is_dual": True, "opt_a": option_a, "opt_b": option_b, "prob_a": 0.7, "prob_b": 0.7}
    save_watch_zones("XAUUSD", [candidate])

    option_b["rejection_confirmed"] = True
    save_watch_zones("XAUUSD", [candidate])

    stored = next(iter(zones._load_zones("XAUUSD").values()))
    assert stored["rejection_confirmed_b"] is True
