import pandas as pd
import pytest

from src.reaction_router import (
    compute_levels,
    ORDER_LIMIT,
    ORDER_MARKET,
    ORDER_STOP,
    STATE_APPROACHING,
    STATE_BREAKOUT,
    STATE_CONFIRMED,
    classify_reaction,
    reaction_strength,
)


def _candle(open_, high, low, close):
    return {"Open": open_, "High": high, "Low": low, "Close": close}


def test_reaction_strength_pinbar_buy():
    strength = reaction_strength(_candle(2005, 2006, 1996, 2005), level=1997.0, direction=1)

    assert strength == pytest.approx(0.9)


def test_confirmed_buy_rejection_routes_market_order():
    df = pd.DataFrame([
        _candle(2002, 2004, 2001, 2003),
        _candle(2005, 2006, 1996, 2005),
    ])

    state, order, strength = classify_reaction(df, level=1997.0, direction=1)

    assert state == STATE_CONFIRMED
    assert order == ORDER_MARKET
    assert strength >= 0.5


def test_approaching_buy_level_routes_limit_order():
    df = pd.DataFrame([
        _candle(2008, 2010, 2007, 2009),
        _candle(2006, 2008, 2004, 2005),
    ])

    state, order, strength = classify_reaction(df, level=1997.0, direction=1)

    assert state == STATE_APPROACHING
    assert order == ORDER_LIMIT
    assert strength == 0.0


def test_breakout_buy_routes_stop_order():
    df = pd.DataFrame([
        _candle(1995, 1998, 1994, 1996),
        _candle(1997, 2005, 1996, 2004),
    ])

    state, order, strength = classify_reaction(df, level=1998.0, direction=1)

    assert state == STATE_BREAKOUT
    assert order == ORDER_STOP
    assert strength >= 0.0


def test_market_levels_buy():
    levels = compute_levels(
        ORDER_MARKET,
        direction=1,
        confirm_price=2003.0,
        level=1997.0,
        wick_extreme=1996.0,
        target=2010.0,
    )

    assert levels["entry"] == pytest.approx(2003.0)
    assert levels["sl"] < 1997.0
    assert levels["tp"] == pytest.approx(2010.0)


def test_limit_levels_buy():
    levels = compute_levels(
        ORDER_LIMIT,
        direction=1,
        confirm_price=2005.0,
        level=1997.0,
        wick_extreme=1996.0,
        target=2010.0,
    )

    assert levels["entry"] == pytest.approx(1997.0)
    assert levels["sl"] < 1997.0


def test_stop_levels_buy():
    levels = compute_levels(
        ORDER_STOP,
        direction=1,
        confirm_price=2002.0,
        level=1998.0,
        wick_extreme=1997.0,
        target=2010.0,
    )

    assert levels["entry"] >= 1998.0
    assert levels["sl"] < levels["entry"]
    assert levels["tp"] > levels["entry"]
