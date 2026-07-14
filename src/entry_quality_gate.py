import math
import os
from dataclasses import dataclass
from typing import Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# Oscillator level definitions  (reference / patokan only, NOT hard entry rules)
# RSI 8 + Stoch Osc composite zones:
#   composite <= 20          -> BUY   (strongly oversold)
#   20 < composite <= 30     -> REBUY (oversold, re-accumulation)
#   30 < composite < 70      -> WAIT  (neutral)
#   70 <= composite < 80     -> RESELL (overbought, re-distribution)
#   composite >= 80          -> SELL  (strongly overbought)
# ---------------------------------------------------------------------------
OSC_LEVEL_BUY    = 20.0
OSC_LEVEL_REBUY  = 30.0
OSC_LEVEL_RESELL = 70.0
OSC_LEVEL_SELL   = 80.0

CONF_BOOST_STRONG   = 0.05   # zone strongly aligns with trade direction
CONF_BOOST_MODERATE = 0.02   # zone moderately aligns
CONF_PENALTY_STRONG = 0.07   # zone strongly opposes trade direction
CONF_PENALTY_MOD    = 0.04   # zone moderately opposes


CORE_STRATEGIES = {"FVG", "BPR", "FVG_OR_BPR", "OB_OR_SWAPZONE_IC_SND", "PIVOT_REJECTION"}
WEAK_STRATEGY_THRESHOLDS = {
    "Pivot": 0.78,
    "SND": 0.68,
    "Swapzone": 0.68,
    "IC": 0.65,
    "OB": 0.65,
}


@dataclass(frozen=True)
class SpreadContext:
    spread_points: float | None = None
    spread_price: float | None = None
    point: float | None = None
    digits: int | None = None


@dataclass(frozen=True)
class OscillatorContext:
    """
    RSI 8 + Stochastic Oscillator (%K9, %D3, smoothing=3, High/Low, Exponential).
    All values on 0-100 scale.
    signal_label : BUY / REBUY / WAIT / RESELL / SELL  (patokan reference zones)
    confidence_delta : suggested confidence adjustment applied by evaluate_entry_quality
    """
    rsi_8: float | None = None
    stoch_k: float | None = None        # Stoch %K (0-100)
    stoch_d: float | None = None        # Stoch %D (0-100)
    signal_label: str | None = None     # 'BUY','REBUY','WAIT','RESELL','SELL'
    confidence_delta: float = 0.0       # + boost / - penalty

    def __init__(
        self,
        rsi_8: float | None = None,
        stoch_k: float | None = None,
        stoch_d: float | None = None,
        signal_label: str | None = None,
        confidence_delta: float = 0.0,
        stoch_rsi_k: float | None = None,
        stoch_rsi_d: float | None = None,
    ):
        object.__setattr__(self, "rsi_8", rsi_8)
        
        resolved_k = stoch_k
        if resolved_k is None and stoch_rsi_k is not None:
            resolved_k = stoch_rsi_k * 100.0 if stoch_rsi_k <= 1.0 else stoch_rsi_k
        object.__setattr__(self, "stoch_k", resolved_k)

        resolved_d = stoch_d
        if resolved_d is None and stoch_rsi_d is not None:
            resolved_d = stoch_rsi_d * 100.0 if stoch_rsi_d <= 1.0 else stoch_rsi_d
        object.__setattr__(self, "stoch_d", resolved_d)

        object.__setattr__(self, "signal_label", signal_label)
        object.__setattr__(self, "confidence_delta", confidence_delta)

    # Backward-compat aliases
    @property
    def stoch_rsi_k(self):
        return self.stoch_k / 100.0 if self.stoch_k is not None else None

    @property
    def stoch_rsi_d(self):
        return self.stoch_d / 100.0 if self.stoch_d is not None else None


# TF hierarchy order (ascending timeframe size)
_TF_ORDER = ["M1", "M5", "M15", "M30", "H1", "H4", "D1"]

# HTF / LTF map for oscillator multi-TF lookup
# Key: signal TF  ->  (htf_candidates, ltf_candidates)
_TF_OSC_MAP: dict[str, tuple[list[str], list[str]]] = {
    "M15": (["H1", "H4"],    ["M5", "M1"]),
    "M30": (["H1", "H4"],    ["M5", "M1"]),
    "H1":  (["H4", "D1"],    ["M15", "M5"]),
    "H4":  (["D1"],          ["H1", "M30"]),
    "D1":  (["D1"],          ["H4", "H1"]),
    "M5":  (["M30", "H1"],   ["M1"]),
    "M1":  (["M15", "M5"],   []),
}


@dataclass(frozen=True)
class MultiTFOscillatorContext:
    """
    3-layer oscillator confluence: HTF (bias) + Signal TF (setup) + LTF (trigger).

    Weights:  HTF=0.50  |  Signal TF=0.35  |  LTF=0.15
    All OscillatorContext values are on 0-100 scale.
    confluent_delta = weighted sum of all layer deltas (with direction sign applied).
    confluence_score = 0..3  (count of layers aligning with trade direction).
    """
    htf_tf: str | None = None               # name of the HTF used
    htf: OscillatorContext | None = None    # HTF oscillator
    signal_tf: str | None = None            # signal timeframe name
    signal: OscillatorContext | None = None # signal TF oscillator
    ltf_tf: str | None = None               # name of the LTF used
    ltf: OscillatorContext | None = None    # LTF oscillator (entry precision)
    confluent_delta: float = 0.0            # total confidence delta (weighted)
    confluence_score: int = 0               # 0-3: how many layers align

    # Expose the signal-TF oscillator as the primary single-TF context
    # (backwards-compat: code that expects OscillatorContext can use .primary)
    @property
    def primary(self) -> OscillatorContext:
        return self.signal or OscillatorContext()


@dataclass(frozen=True)
class EntryGateDecision:
    allowed: bool
    filtered_reason: str
    reason: str
    required_confidence: float
    spread_r: float | None = None


def _read_float_env(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _read_bool_env(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return str(raw_value).strip().lower() in {"1", "true", "yes", "on"}


def points_to_price(points: float, point: float) -> float:
    try:
        return float(points) * float(point)
    except (TypeError, ValueError):
        return 0.0


def build_spread_context(bid: float, ask: float, point: float, digits: int | None = None) -> SpreadContext:
    try:
        bid_value = float(bid)
        ask_value = float(ask)
        point_value = float(point)
    except (TypeError, ValueError):
        return SpreadContext(point=point, digits=digits)

    spread_price = max(0.0, ask_value - bid_value)
    spread_points = spread_price / point_value if point_value > 0 else None
    return SpreadContext(
        spread_points=spread_points,
        spread_price=spread_price,
        point=point_value,
        digits=digits,
    )


def _rsi(series: pd.Series, period: int) -> pd.Series:
    """Wilder-smoothed RSI via EWM (alpha = 1/period)."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, 0.00001)
    return 100 - (100 / (1 + rs))


def _stoch_oscillator(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 9,
    d_period: int = 3,
    smooth: int = 3,
) -> Tuple[pd.Series, pd.Series]:
    """
    Standard Stochastic Oscillator with Exponential smoothing.

    Parameters
    ----------
    k_period : lookback for raw %K  (default 9)
    d_period : EWM span for %D signal line  (default 3)
    smooth   : EWM span for smoothing raw %K -> fast %K  (default 3)
               Uses EWM (exponential) to match 'Exponential' variant in MT5/TV.

    Returns (stoch_k, stoch_d) on 0-100 scale.
    """
    lowest_low   = low.rolling(window=k_period, min_periods=max(1, k_period // 2)).min()
    highest_high = high.rolling(window=k_period, min_periods=max(1, k_period // 2)).max()
    range_hl     = (highest_high - lowest_low).replace(0, 0.00001)
    raw_k   = 100.0 * (close - lowest_low) / range_hl
    stoch_k = raw_k.ewm(span=smooth, adjust=False, min_periods=1).mean().clip(0.0, 100.0)
    stoch_d = stoch_k.ewm(span=d_period, adjust=False, min_periods=1).mean().clip(0.0, 100.0)
    return stoch_k, stoch_d


def _composite_level(rsi_val: float, stoch_val: float) -> float:
    """Average of RSI-8 and Stoch-%K (both 0-100) -> single composite for zone classification."""
    return (rsi_val + stoch_val) / 2.0


def classify_oscillator_zone(rsi_val, stoch_k) -> Tuple[str, float]:
    """
    Classify current oscillator state into one of 5 reference zones.
    Returns (label, confidence_delta) where delta is the 'with-direction' boost.
    """
    if rsi_val is None or stoch_k is None:
        return "WAIT", 0.0
    composite = _composite_level(rsi_val, stoch_k)
    if composite <= OSC_LEVEL_BUY:
        return "BUY", CONF_BOOST_STRONG
    elif composite <= OSC_LEVEL_REBUY:
        return "REBUY", CONF_BOOST_MODERATE
    elif composite < OSC_LEVEL_RESELL:
        return "WAIT", 0.0
    elif composite < OSC_LEVEL_SELL:
        return "RESELL", CONF_BOOST_MODERATE
    else:
        return "SELL", CONF_BOOST_STRONG


def _last_finite(series: pd.Series) -> float | None:
    cleaned = pd.to_numeric(series, errors="coerce").dropna()
    if cleaned.empty:
        return None
    value = float(cleaned.iloc[-1])
    return value if math.isfinite(value) else None


def build_oscillator_context(df: pd.DataFrame | None) -> OscillatorContext:
    """
    Compute RSI-8 and Stochastic Oscillator (%K9, %D3, smooth=3, High/Low, Exponential)
    from an OHLC DataFrame. Returns OscillatorContext with all values on 0-100 scale
    and a pre-classified signal_label / confidence_delta.
    """
    if df is None or df.empty or "Close" not in df.columns:
        return OscillatorContext()

    close = pd.to_numeric(df["Close"], errors="coerce")

    # RSI-8 (Wilder / EWM)
    rsi_8_val = _last_finite(_rsi(close, 8))

    # Stochastic Oscillator — prefer High/Low; fall back to Close if not present
    if "High" in df.columns and "Low" in df.columns:
        high = pd.to_numeric(df["High"], errors="coerce")
        low  = pd.to_numeric(df["Low"],  errors="coerce")
    else:
        high = close
        low  = close

    stoch_k_series, stoch_d_series = _stoch_oscillator(high, low, close)
    stoch_k_val = _last_finite(stoch_k_series)
    stoch_d_val = _last_finite(stoch_d_series)

    label, delta = classify_oscillator_zone(rsi_8_val, stoch_k_val)

    return OscillatorContext(
        rsi_8=rsi_8_val,
        stoch_k=stoch_k_val,
        stoch_d=stoch_d_val,
        signal_label=label,
        confidence_delta=delta,
    )


def build_multi_tf_oscillator(
    signal_tf: str,
    tf_osc_data: dict,           # {tf_name: OscillatorContext}, pre-computed for all TFs
) -> MultiTFOscillatorContext:
    """
    Build a 3-layer (HTF / Signal TF / LTF) oscillator context for a given signal TF.

    HTF weight  = 0.50  — overall market bias (e.g. H1/H4 for M15 signal)
    Signal TF   = 0.35  — setup context (the TF where the SMC signal was found)
    LTF weight  = 0.15  — entry precision trigger (e.g. M5/M1 for M15 signal)

    The confluent_delta is the SUM of weighted deltas, with sign flipped for opposing layers.
    confluence_score counts how many layers agree with 'their direction' (computed at call site).
    """
    htf_candidates, ltf_candidates = _TF_OSC_MAP.get(signal_tf, ([], []))

    # Pick best available HTF and LTF (first match in preferred order)
    htf_name = next((t for t in htf_candidates if tf_osc_data.get(t) is not None), None)
    ltf_name = next((t for t in ltf_candidates if tf_osc_data.get(t) is not None), None)

    htf_osc    = tf_osc_data.get(htf_name) if htf_name else None
    signal_osc = tf_osc_data.get(signal_tf)
    ltf_osc    = tf_osc_data.get(ltf_name) if ltf_name else None

    return MultiTFOscillatorContext(
        htf_tf=htf_name,
        htf=htf_osc,
        signal_tf=signal_tf,
        signal=signal_osc,
        ltf_tf=ltf_name,
        ltf=ltf_osc,
        # confluent_delta and confluence_score are computed at evaluation time
        # (they depend on trade direction which isn't known here)
    )


def _score_mtf_osc_layer(
    osc: "OscillatorContext | None",
    direction: int,   # 1=long, -1=short
    weight: float,
) -> tuple[float, int]:
    """
    Given one oscillator layer and trade direction, return (delta_contribution, aligns_count).
    delta_contribution is positive when the layer aligns, negative when it opposes.
    """
    if osc is None or osc.signal_label is None:
        return 0.0, 0
    label = osc.signal_label
    aligns  = (direction == 1 and label in ("BUY", "REBUY")) or \
               (direction == -1 and label in ("SELL", "RESELL"))
    opposes = (direction == 1 and label in ("SELL", "RESELL")) or \
               (direction == -1 and label in ("BUY", "REBUY"))
    if aligns:
        return osc.confidence_delta * weight, 1
    elif opposes:
        return -(osc.confidence_delta * weight), 0
    return 0.0, 0  # WAIT = neutral


def evaluate_multi_tf_osc_delta(
    mtf: "MultiTFOscillatorContext",
    direction: int,
) -> tuple[float, int]:
    """
    Compute total confidence delta and confluence_score for a MultiTFOscillatorContext
    given the trade direction.  Called at confidence-gate time.

    Returns (confluent_delta, confluence_score).
    """
    HTF_WEIGHT    = 0.50
    SIGNAL_WEIGHT = 0.35
    LTF_WEIGHT    = 0.15

    d_htf,    c_htf    = _score_mtf_osc_layer(mtf.htf,    direction, HTF_WEIGHT)
    d_signal, c_signal = _score_mtf_osc_layer(mtf.signal, direction, SIGNAL_WEIGHT)
    d_ltf,    c_ltf    = _score_mtf_osc_layer(mtf.ltf,    direction, LTF_WEIGHT)

    return d_htf + d_signal + d_ltf, c_htf + c_signal + c_ltf


_ZONE_EMOJI = {
    "BUY":    "\U0001f7e2",  # green circle
    "REBUY":  "\U0001f7e9",  # green square
    "WAIT":   "\U0001f7e1",  # yellow circle
    "RESELL": "\U0001f7e7",  # orange circle
    "SELL":   "\U0001f534",  # red circle
}


def format_oscillator_line(osc: "OscillatorContext | None") -> str:
    """
    Human-readable summary for Telegram signal messages (single TF).
    Example: RSI8: 24.3 | Stoch %K: 18.7 / %D: 21.4 -> [REBUY] comp 21.5 | Conf +2%
    """
    if osc is None or (osc.rsi_8 is None and osc.stoch_k is None):
        return "RSI8: n/a | Stoch: n/a"
    rsi_str = f"{osc.rsi_8:.1f}"  if osc.rsi_8   is not None else "n/a"
    stk_str = f"{osc.stoch_k:.1f}" if osc.stoch_k is not None else "n/a"
    std_str = f"{osc.stoch_d:.1f}" if osc.stoch_d is not None else "n/a"
    label   = osc.signal_label or "WAIT"
    emoji   = _ZONE_EMOJI.get(label, "\u26aa")
    comp_str = ""
    if osc.rsi_8 is not None and osc.stoch_k is not None:
        comp = _composite_level(osc.rsi_8, osc.stoch_k)
        comp_str = f" comp {comp:.1f}"
    delta_str = ""
    if osc.confidence_delta != 0.0:
        pct  = int(round(osc.confidence_delta * 100))
        sign = "+" if pct > 0 else ""
        delta_str = f" | Conf {sign}{pct}%"
    return f"RSI8: {rsi_str} | Stoch %K: {stk_str} / %D: {std_str} -> {emoji} [{label}]{comp_str}{delta_str}"


def format_multi_tf_oscillator_block(mtf: "MultiTFOscillatorContext | None", direction: int = 0) -> str:
    """
    Multi-line Telegram block showing RSI8+Stoch for HTF, Signal TF, and LTF.
    Includes weighted confluence summary.

    direction: 1=LONG, -1=SHORT, 0=unknown (suppress alignment indicator).
    """
    if mtf is None:
        return "RSI8 + Stoch: n/a"

    lines = []

    def _layer_line(tf_label: str, osc, direction: int) -> str:
        if osc is None:
            return f"{tf_label}: n/a"
        rsi_str = f"{osc.rsi_8:.1f}"  if osc.rsi_8   is not None else "n/a"
        stk_str = f"{osc.stoch_k:.1f}" if osc.stoch_k is not None else "n/a"
        std_str = f"{osc.stoch_d:.1f}" if osc.stoch_d is not None else "n/a"
        label   = osc.signal_label or "WAIT"
        emoji   = _ZONE_EMOJI.get(label, "\u26aa")
        align_indicator = ""
        if direction != 0:
            aligns  = (direction == 1 and label in ("BUY", "REBUY")) or \
                      (direction == -1 and label in ("SELL", "RESELL"))
            opposes = (direction == 1 and label in ("SELL", "RESELL")) or \
                      (direction == -1 and label in ("BUY", "REBUY"))
            if aligns:
                align_indicator = " \u2714"   # checkmark
            elif opposes:
                align_indicator = " \u26a0"   # warning
        return f"{tf_label}: RSI8={rsi_str} %K={stk_str} %D={std_str} {emoji}[{label}]{align_indicator}"

    # HTF layer (bias)
    htf_label = f"HTF ({mtf.htf_tf})" if mtf.htf_tf else "HTF"
    lines.append(_layer_line(htf_label, mtf.htf, direction))

    # Signal TF layer (setup)
    sig_label = f"Signal ({mtf.signal_tf})" if mtf.signal_tf else "Signal TF"
    lines.append(_layer_line(sig_label, mtf.signal, direction))

    # LTF layer (trigger)
    ltf_label = f"LTF ({mtf.ltf_tf})" if mtf.ltf_tf else "LTF"
    lines.append(_layer_line(ltf_label, mtf.ltf, direction))

    # Confluence summary
    if direction != 0:
        confluent_delta, confluence_score = evaluate_multi_tf_osc_delta(mtf, direction)
        layers_with_data = sum(1 for x in [mtf.htf, mtf.signal, mtf.ltf] if x is not None)
        score_str = f"{confluence_score}/{layers_with_data}"
        pct = int(round(confluent_delta * 100))
        sign = "+" if pct > 0 else ""
        lines.append(f"Confluence: {score_str} layers align | Net Conf {sign}{pct}%")

    return "\n".join(lines)


def _risk_price(setup: dict) -> float:
    try:
        return abs(float(setup.get("entry_price")) - float(setup.get("sl_price")))
    except (TypeError, ValueError):
        return 0.0


def _has_htf_support(setup: dict, features: dict) -> bool:
    if bool(setup.get("htf_prioritized", False)):
        return True
    if setup.get("matching_htf_fvgs"):
        return True

    for key in ("floop_trend_aligned", "htf_trend_aligned"):
        try:
            if int(features.get(key, 0)) == 1:
                return True
        except (TypeError, ValueError):
            pass

    try:
        return float(features.get("confluence_score", 0.0)) >= 2.0
    except (TypeError, ValueError):
        return False


def _required_confidence(strategy: str, accept_threshold: float, direction: int) -> float:
    runtime_threshold = float(accept_threshold)
    base = _read_float_env("ML_ENTRY_BASE_THRESHOLD", runtime_threshold)
    threshold = max(runtime_threshold, base)
    threshold = max(threshold, WEAK_STRATEGY_THRESHOLDS.get(str(strategy), threshold))

    if int(direction) == 1:
        threshold += _read_float_env("ML_ENTRY_BUY_CONFIDENCE_BONUS", 0.0)

    if str(strategy) not in CORE_STRATEGIES and str(strategy) not in WEAK_STRATEGY_THRESHOLDS:
        threshold = max(threshold, _read_float_env("ML_ENTRY_UNKNOWN_STRATEGY_THRESHOLD", 0.65))

    return min(threshold, 0.95)


def evaluate_entry_quality(
    setup: dict,
    *,
    strategy: str,
    probability: float,
    accept_threshold: float,
    spread: SpreadContext | None = None,
    oscillator: "OscillatorContext | MultiTFOscillatorContext | None" = None,
) -> EntryGateDecision:
    direction = int(setup.get("direction", 1))
    required_confidence = _required_confidence(strategy, accept_threshold, direction)

    try:
        prob = float(probability)
    except (TypeError, ValueError):
        prob = 0.0

    if not math.isfinite(prob) or prob < required_confidence:
        return EntryGateDecision(
            allowed=False,
            filtered_reason="entry_gate_below_required_confidence",
            reason=f"confidence {prob:.2%} below entry gate {required_confidence:.2%}",
            required_confidence=required_confidence,
        )

    spread_r = None
    risk = _risk_price(setup)
    if spread is not None and spread.spread_price is not None and risk > 0:
        spread_r = float(spread.spread_price) / risk

    # Spread checks to prevent entry on wide spreads (if enabled)
    if _read_bool_env("MT5_ENFORCE_SPREAD_FILTER", False):
        if spread is not None and spread.spread_price is not None:
            symbol = setup.get("symbol", "")
            from src.smc_detector import get_pip_multiplier
            pip_mult = get_pip_multiplier(symbol)
            spread_pips = spread.spread_price / pip_mult if pip_mult > 0 else 0.0
            
            # 1. Absolute spread check (MT5_RUNNER_MAX_SPREAD_PIPS in .env)
            max_spread_pips = _read_float_env("MT5_RUNNER_MAX_SPREAD_PIPS", 5.0)
            if spread_pips > max_spread_pips:
                return EntryGateDecision(
                    allowed=False,
                    filtered_reason="entry_gate_spread_too_high",
                    reason=f"Spread {spread_pips:.1f} pips exceeds max allowed {max_spread_pips:.1f} pips",
                    required_confidence=required_confidence,
                    spread_r=spread_r,
                )
                
            # 2. Relative spread check (spread relative to risk/SL distance, max ratio from env/default 20%)
            max_spread_ratio = _read_float_env("MT5_MAX_SPREAD_RATIO", 0.20)
            if spread_r is not None and spread_r > max_spread_ratio:
                return EntryGateDecision(
                    allowed=False,
                    filtered_reason="entry_gate_spread_ratio_too_high",
                    reason=f"Spread ratio {spread_r:.2%} exceeds max allowed {max_spread_ratio:.2%}",
                    required_confidence=required_confidence,
                    spread_r=spread_r,
                )

    features = setup.get("features") if isinstance(setup.get("features"), dict) else {}
    rr_ratio = features.get("rr_ratio", setup.get("rr_ratio"))
    try:
        rr_value = float(rr_ratio)
    except (TypeError, ValueError):
        rr_value = None
    min_rr = _read_float_env("ML_ENTRY_MIN_RR", 1.20)
    if rr_value is not None and rr_value > 0 and rr_value < min_rr:
        return EntryGateDecision(
            allowed=False,
            filtered_reason="entry_gate_rr_too_low",
            reason=f"RR {rr_value:.2f} below min {min_rr:.2f}",
            required_confidence=required_confidence,
            spread_r=spread_r,
        )

    # -----------------------------------------------------------------------
    # Oscillator confidence adjustment  (RSI8 + Stoch, 5-level zone patokan)
    # Supports both single-TF OscillatorContext and 3-layer MultiTFOscillatorContext.
    # Multi-TF weights:  HTF=0.50 | Signal TF=0.35 | LTF=0.15
    # Levels: BUY<=20, REBUY<=30, WAIT 30-70, RESELL>=70, SELL>=80
    # Effect: naikin/kurangin required_confidence — NOT hard entry/exit block.
    # -----------------------------------------------------------------------

    # Unpack oscillator into (primary_osc, weighted_delta, confluence_score)
    if isinstance(oscillator, MultiTFOscillatorContext):
        primary_osc       = oscillator.primary          # signal-TF layer
        weighted_delta, confluence_score = evaluate_multi_tf_osc_delta(oscillator, direction)
        is_multi_tf       = True
    else:
        primary_osc       = oscillator or OscillatorContext()
        weighted_delta    = None    # computed below from primary_osc
        confluence_score  = 0
        is_multi_tf       = False

    rsi_8   = primary_osc.rsi_8
    stoch_k = primary_osc.stoch_k

    if rsi_8 is not None and stoch_k is not None:
        label = primary_osc.signal_label or "WAIT"

        if is_multi_tf:
            # Multi-TF: use weighted delta directly
            delta = weighted_delta
            aligns  = delta > 0
            opposes = delta < 0
        else:
            # Single-TF fallback
            delta = primary_osc.confidence_delta
            aligns  = (direction == 1 and label in ("BUY", "REBUY")) or \
                      (direction == -1 and label in ("SELL", "RESELL"))
            opposes = (direction == 1 and label in ("SELL", "RESELL")) or \
                      (direction == -1 and label in ("BUY", "REBUY"))

        if aligns:
            # Zone supports the trade: lower required threshold (capped at -5%)
            new_req = max(required_confidence - abs(delta), required_confidence - 0.05)
            required_confidence = max(new_req, 0.45)
        elif opposes:
            # Zone opposes: raise required threshold (capped at +7%)
            penalty = min(abs(delta), 0.07)
            required_confidence = min(required_confidence + penalty, 0.90)

        # Legacy hard-gate: extreme opposition zones require extra confidence
        # (uses signal-TF oscillator values regardless of multi-TF mode)
        sell_oversold_rsi   = _read_float_env("ML_SELL_OVERSOLD_RSI8",       32.0)
        sell_oversold_stoch = _read_float_env("ML_SELL_OVERSOLD_STOCH_RSI",   0.20)
        buy_overbought_rsi  = _read_float_env("ML_BUY_OVERBOUGHT_RSI8",      68.0)
        buy_overbought_stoch= _read_float_env("ML_BUY_OVERBOUGHT_STOCH_RSI",  0.80)
        stoch_k_01 = stoch_k / 100.0   # convert 0-100 -> 0-1 for legacy thresholds

        oscillator_extreme_reason = None
        if direction == -1 and rsi_8 <= sell_oversold_rsi and stoch_k_01 <= sell_oversold_stoch:
            oscillator_extreme_reason = "entry_gate_oscillator_oversold_sell"
        elif direction == 1 and rsi_8 >= buy_overbought_rsi and stoch_k_01 >= buy_overbought_stoch:
            oscillator_extreme_reason = "entry_gate_oscillator_overbought_buy"

        if oscillator_extreme_reason is not None:
            htf_supported = _has_htf_support(setup, features)
            if htf_supported:
                required_confidence = max(
                    required_confidence,
                    _read_float_env("ML_OSCILLATOR_EXTREME_HTF_CONFIDENCE", 0.70),
                )
                pass_reason = "entry_gate_pass_htf_supported_oscillator_extreme"
            else:
                required_confidence = max(
                    required_confidence,
                    _read_float_env("ML_OSCILLATOR_EXTREME_UNSUPPORTED_CONFIDENCE", 0.80),
                )
                pass_reason = "entry_gate_pass_oscillator_extreme_high_confidence"

            mtf_note = f" [MTF confluence {confluence_score}/3]" if is_multi_tf else ""
            if prob < required_confidence:
                return EntryGateDecision(
                    allowed=False,
                    filtered_reason=oscillator_extreme_reason,
                    reason=(
                        f"oscillator extreme needs {required_confidence:.2%} confidence; "
                        f"got {prob:.2%} - RSI8 {rsi_8:.1f}, Stoch %K {stoch_k:.1f} [{label}]{mtf_note}"
                    ),
                    required_confidence=required_confidence,
                    spread_r=spread_r,
                )

            htf_txt = "HTF support" if htf_supported else "very high confidence"
            return EntryGateDecision(
                allowed=True,
                filtered_reason=pass_reason,
                reason=(
                    f"oscillator extreme accepted with {htf_txt}; "
                    f"RSI8 {rsi_8:.1f}, Stoch %K {stoch_k:.1f} [{label}]{mtf_note}"
                ),
                required_confidence=required_confidence,
                spread_r=spread_r,
            )

        # Non-extreme zone: apply boost/penalty and block only if below new threshold
        if opposes and prob < required_confidence:
            mtf_note = f" [MTF {confluence_score}/3 layers align]" if is_multi_tf else ""
            return EntryGateDecision(
                allowed=False,
                filtered_reason=f"oscillator_{label.lower()}_opposes",
                reason=(
                    f"confidence {prob:.2%} below gate {required_confidence:.2%} after "
                    f"oscillator [{label}] penalty - RSI8 {rsi_8:.1f}, Stoch %K {stoch_k:.1f}{mtf_note}"
                ),
                required_confidence=required_confidence,
                spread_r=spread_r,
            )

    return EntryGateDecision(
        allowed=True,
        filtered_reason="entry_gate_pass",
        reason="entry gate passed",
        required_confidence=required_confidence,
        spread_r=spread_r,
    )
