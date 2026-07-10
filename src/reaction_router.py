"""Reaction-based routing for key-level pivot setups."""

import pandas as pd


ORDER_MARKET = 0
ORDER_LIMIT = 1
ORDER_STOP = 2

STATE_CONFIRMED = "CONFIRMED_REJECTION"
STATE_APPROACHING = "APPROACHING_LEVEL"
STATE_BREAKOUT = "BREAKOUT_CONFIRMATION"


def _as_float(candle: dict, name: str) -> float:
    return float(candle[name])


def reaction_strength(candle: dict, level: float, direction: int) -> float:
    """Return wick rejection strength in 0..1 for one candle at a key level."""
    open_ = _as_float(candle, "Open")
    high = _as_float(candle, "High")
    low = _as_float(candle, "Low")
    close = _as_float(candle, "Close")
    total_range = high - low
    if total_range <= 0.0:
        return 0.0

    if int(direction) == 1:
        body_top = max(open_, close)
        if not (low <= float(level) <= body_top):
            return 0.0
        wick = min(open_, close) - low
    else:
        body_bottom = min(open_, close)
        if not (body_bottom <= float(level) <= high):
            return 0.0
        wick = high - max(open_, close)

    return max(0.0, min(float(wick) / total_range, 1.0))


def _body_ratio(candle: dict) -> float:
    high = _as_float(candle, "High")
    low = _as_float(candle, "Low")
    total_range = high - low
    if total_range <= 0.0:
        return 0.0
    return abs(_as_float(candle, "Close") - _as_float(candle, "Open")) / total_range


def _touched(candle: dict, level: float, direction: int) -> bool:
    level = float(level)
    if int(direction) == 1:
        return _as_float(candle, "Low") <= level <= max(_as_float(candle, "Open"), _as_float(candle, "Close"))
    return min(_as_float(candle, "Open"), _as_float(candle, "Close")) <= level <= _as_float(candle, "High")


def _breakout_confirmed(previous: dict | None, candle: dict, level: float, direction: int, min_body_ratio: float) -> bool:
    if previous is None or _body_ratio(candle) < min_body_ratio:
        return False

    level = float(level)
    prev_close = _as_float(previous, "Close")
    close = _as_float(candle, "Close")
    if int(direction) == 1:
        return prev_close <= level and close > level
    return prev_close >= level and close < level


def classify_reaction(
    df: pd.DataFrame,
    level: float,
    direction: int,
    strong_wick_ratio: float = 0.5,
    breakout_body_ratio: float = 0.6,
) -> tuple[str, int, float]:
    """Classify recent market reaction and select order routing.

    Returns (state, order_type, strength):
    - confirmed rejection -> market order
    - approaching untouched level -> limit order
    - strong close through level -> stop order
    """
    if df.empty:
        return STATE_APPROACHING, ORDER_LIMIT, 0.0

    window = df.tail(2).reset_index(drop=True)
    previous = window.iloc[-2].to_dict() if len(window) >= 2 else None
    candle = window.iloc[-1].to_dict()
    strength = reaction_strength(candle, level, direction)

    if _touched(candle, level, direction) and strength >= float(strong_wick_ratio):
        return STATE_CONFIRMED, ORDER_MARKET, strength

    if _breakout_confirmed(previous, candle, level, direction, float(breakout_body_ratio)):
        return STATE_BREAKOUT, ORDER_STOP, strength

    return STATE_APPROACHING, ORDER_LIMIT, strength


def compute_levels(
    order_type: int,
    direction: int,
    confirm_price: float,
    level: float,
    wick_extreme: float,
    target: float,
    sl_buffer: float = 0.2,
) -> dict:
    """Return entry/SL/TP levels consistent with the chosen order type."""
    direction = int(direction)
    if order_type == ORDER_MARKET:
        entry = float(confirm_price)
        sl = float(wick_extreme) - sl_buffer if direction == 1 else float(wick_extreme) + sl_buffer
    elif order_type == ORDER_LIMIT:
        entry = float(level)
        sl = float(wick_extreme) - sl_buffer if direction == 1 else float(wick_extreme) + sl_buffer
    else:
        entry = float(confirm_price)
        sl = float(level) - sl_buffer if direction == 1 else float(level) + sl_buffer
    return {"entry": entry, "sl": sl, "tp": float(target)}
