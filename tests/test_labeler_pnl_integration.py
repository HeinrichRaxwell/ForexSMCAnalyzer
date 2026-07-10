import os
import pandas as pd
import numpy as np
import pytest
from src.labeler import label_smc_setups

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HIST_CSV = os.path.join(BASE_DIR, "data", "historical_xauusdm_15.csv")


def _real_slice(n=900):
    """Slice nyata data historis XAUUSD M15 — pasti menghasilkan setup,
    sehingga test benar-benar menguji (bukan SKIP)."""
    if not os.path.exists(HIST_CSV):
        pytest.skip(f"historical data not found at {HIST_CSV}")
    df = pd.read_csv(HIST_CSV)
    df["time"] = pd.to_datetime(df["time"])
    # ambil potongan tengah agar ada cukup bar sebelum & sesudah untuk resolusi
    if len(df) > n:
        df = df.iloc[-n:].reset_index(drop=True)
    return df


@pytest.fixture(scope="module")
def labeled():
    df = _real_slice()
    out = label_smc_setups(df, symbol="XAUUSD")
    return out


def test_real_data_produces_setups(labeled):
    # Verifikasi nyata: data historis HARUS menghasilkan setup.
    assert not labeled.empty, "labeler tidak menghasilkan setup dari data nyata"


def test_labeled_setups_have_pnl_relative_column(labeled):
    assert "pnl_relative" in labeled.columns
    assert labeled["pnl_relative"].notna().all()


def test_winning_setups_pnl_below_raw_rr(labeled):
    """pnl_relative untuk WIN harus < RR mentah, karena cost dikurangi."""
    wins = labeled[labeled["label"] == 1]
    assert not wins.empty, "tidak ada winning setup pada slice ini"
    rr_raw = (wins["tp_price"] - wins["entry_price"]).abs() / \
             (wins["entry_price"] - wins["sl_price"]).abs()
    # setiap win harus strictly lebih kecil dari RR mentah (cost terpotong)
    assert (wins["pnl_relative"] < rr_raw + 1e-9).all()
    # dan harus benar-benar terpotong (bukan sama persis) untuk risk > 0
    assert (wins["pnl_relative"] < rr_raw).any()


def test_losing_setups_deeper_than_minus_one(labeled):
    """pnl_relative untuk LOSS harus <= -1.0 (lebih dalam karena cost)."""
    losses = labeled[labeled["label"] == 0]
    assert not losses.empty, "tidak ada losing setup pada slice ini"
    assert (losses["pnl_relative"] <= -1.0 + 1e-9).all()


def test_labeled_setups_have_new_features():
    df = _real_slice()
    out = label_smc_setups(df, symbol="XAUUSD")
    assert not out.empty, "real data slice must produce setups"
    for col in ["rr_ratio", "atr_percentile", "body_to_range_ratio",
                "dist_to_recent_swing", "htf_trend_aligned", "confluence_score"]:
        assert col in out.columns, f"missing feature {col}"
        assert out[col].notna().all(), f"NaN leaked in {col}"


def test_labeled_setups_have_order_routing_features():
    df = _real_slice()
    out = label_smc_setups(df, symbol="XAUUSD")
    assert not out.empty, "real data slice must produce setups"
    for col in ["order_type", "reaction_strength"]:
        assert col in out.columns, f"missing feature {col}"
        assert out[col].notna().all(), f"NaN leaked in {col}"
