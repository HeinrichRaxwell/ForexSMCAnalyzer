"""Pure feature functions for SMC setups. Each has an explicit hypothesis
for why it should separate winning trades (TP) from losing trades (SL).

Semua fungsi di sini PURE (tanpa side effect) dan testable terpisah, dipanggil
dari src/labeler.py saat membentuk fitur tiap setup. Tujuan: membuat fitur entry
lebih informatif sehingga confidence model menjadi predictive.
"""
import numpy as np
import pandas as pd


def rr_ratio(entry: float, sl: float, tp: float) -> float:
    """Reward-to-risk ratio = |tp-entry| / |entry-sl|.

    Hipotesis: setup RR tinggi butuh winrate lebih rendah untuk profit; RR adalah
    sinyal kualitas setup yang eksplisit dan belum ada sebagai fitur.
    Aman terhadap risk = 0 (return 0.0).
    """
    risk = abs(float(entry) - float(sl))
    if risk <= 0.0:
        return 0.0
    return abs(float(tp) - float(entry)) / risk


def atr_percentile(atr_window, current_atr: float) -> float:
    """Posisi volatilitas saat ini dalam distribusi ATR terkini (0..1).

    Hipotesis: regime volatilitas ekstrem mengubah reliabilitas pola SMC.
    Dihitung sebagai fraksi nilai pada window yang <= current_atr.
    Aman terhadap window kosong (return 0.0).
    """
    s = pd.to_numeric(pd.Series(atr_window), errors="coerce").dropna()
    if s.empty:
        return 0.0
    return float((s <= float(current_atr)).mean())


def body_to_range_ratio(open_: float, high: float, low: float, close: float) -> float:
    """Konviksi candle = |close-open| / (high-low).

    Hipotesis: body kuat di candle sinyal menandakan momentum/komitmen.
    Aman terhadap range = 0 (return 0.0).
    """
    rng = float(high) - float(low)
    if rng <= 0.0:
        return 0.0
    return abs(float(close) - float(open_)) / rng


def dist_to_recent_swing_norm(entry: float, swing_price: float, atr: float) -> float:
    """Jarak entry ke swing terdekat dinormalisasi ATR = |swing-entry| / atr.

    Hipotesis: cukup ruang menuju target (swing) sebelum kena rintangan.
    Aman terhadap atr = 0 (return 0.0).
    """
    if float(atr) <= 0.0:
        return 0.0
    return abs(float(swing_price) - float(entry)) / float(atr)


def htf_trend_aligned(direction: int, htf_trend: int) -> int:
    """1 jika arah trade searah trend higher-timeframe, selain itu 0.

    Hipotesis: setup searah HTF trend lebih sering tembus TP.
    htf_trend = 0 (netral) dianggap tidak aligned (return 0).
    """
    if int(htf_trend) == 0:
        return 0
    return 1 if int(direction) == int(htf_trend) else 0


def confluence_score(flags: list) -> int:
    """Jumlah elemen SMC yang menumpuk di entry (FVG+OB+pivot+...).

    Hipotesis: makin banyak konfluensi => setup makin berkualitas.
    """
    return int(sum(1 for f in flags if bool(f)))
