"""
price_watch_zones.py
====================
Pre-register ALL detected SMC zones (FVG, OB, BPR, etc.) after each full scan
so the realtime tick loop can detect when price enters a zone IMMEDIATELY,
without waiting for the next full scan cycle.

Flow
----
1. After full scan builds `all_setups`, call `save_watch_zones(symbol, all_setups)`.
2. In the tick loop, call `check_price_in_watch_zones(symbol, current_tick)`.
3. When a zone is hit, it returns a `WatchZoneHit` that the caller can use to
   immediately re-run evaluate_entry_quality + execute_trade/market_order.

Key design decisions
--------------------
- Zones are stored per-symbol in `data/watch_zones_{symbol}.json`.
- Each zone entry is keyed by a stable ID built from TF + strategy + bar index + entry_price.
- A zone is removed once it gets a live ticket or expires (beyond TTL_BARS candle counts).
- "Hit" means current_price is within [zone_low, zone_high] for BUY or vice-versa for SELL.
- No ML re-prediction at tick time — too expensive. We use the probability from the last scan.
  The hit triggers the FULL evaluation only if prob >= threshold (fast pre-filter).
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
)

# Zone expires after this many minutes regardless of hits (prevent stale zones)
_ZONE_TTL_MINUTES = int(os.getenv("WATCH_ZONE_TTL_MINUTES", "120"))

# Maximum number of active watch zones kept in memory / disk per symbol
_MAX_ZONES_PER_SYMBOL = int(os.getenv("WATCH_ZONE_MAX_PER_SYMBOL", "40"))

# How close (in price units, i.e. USD for XAUUSD) price needs to be to trigger
_ZONE_APPROACH_BUFFER = float(os.getenv("WATCH_ZONE_APPROACH_BUFFER", "0.30"))


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class WatchZone:
    zone_id: str            # Stable ID e.g. "M30_FVG_BULL_3285.500_3283.000_20260710T082000"
    symbol: str
    timeframe: str
    strategy: str
    direction: int          # 1=BULL, -1=BEAR
    entry_price: float      # Fib 0.5 or primary entry
    sl_price: float
    tp_price: float
    zone_top: float         # Upper edge of the FVG/OB zone
    zone_bottom: float      # Lower edge of the FVG/OB zone
    probability: float      # Latest ML prob from last scan
    probability_b: float    # Fib 0.618 prob (or same as probability for single-opt)
    entry_price_b: float    # Fib 0.618 entry (or same as entry_price)
    sl_price_b: float
    tp_price_b: float
    features: dict          # Full feature dict for re-evaluation
    features_b: dict
    oscillator_label: str   # e.g. "BUY" / "SELL" / "WAIT" for quick Telegram note
    is_dual: bool
    created_at: str         # ISO timestamp when zone was registered
    expires_at: str         # ISO timestamp when zone auto-expires
    hit_count: int = 0      # How many times price entered this zone
    last_hit_at: str = ""   # ISO timestamp of last hit
    triggered: bool = False # True once an order was placed from this zone
    htf_prioritized: bool = False
    rejection_confirmed: bool = False


@dataclass
class WatchZoneHit:
    zone: WatchZone
    current_price: float
    entry_triggered: bool = False   # True if the hit warrants immediate evaluation
    reason: str = ""


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------
def _zone_file(symbol: str) -> str:
    safe_sym = symbol.replace("/", "_").replace("\\", "_")
    return os.path.join(_DATA_DIR, f"watch_zones_{safe_sym}.json")


def _load_zones(symbol: str) -> dict[str, dict]:
    path = _zone_file(symbol)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def _save_zones(symbol: str, zones: dict[str, dict]) -> None:
    os.makedirs(_DATA_DIR, exist_ok=True)
    path = _zone_file(symbol)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(zones, f, indent=2, default=str)
    except Exception as e:
        print(f"[WatchZones] Failed to save zones for {symbol}: {e}")


def _zone_to_dict(z: WatchZone) -> dict:
    return asdict(z)


def _zone_from_dict(d: dict) -> WatchZone:
    # Be permissive about missing keys from older saves
    return WatchZone(
        zone_id=d.get("zone_id", ""),
        symbol=d.get("symbol", ""),
        timeframe=d.get("timeframe", ""),
        strategy=d.get("strategy", ""),
        direction=int(d.get("direction", 1)),
        entry_price=float(d.get("entry_price", 0)),
        sl_price=float(d.get("sl_price", 0)),
        tp_price=float(d.get("tp_price", 0)),
        zone_top=float(d.get("zone_top", 0)),
        zone_bottom=float(d.get("zone_bottom", 0)),
        probability=float(d.get("probability", 0.5)),
        probability_b=float(d.get("probability_b", 0.5)),
        entry_price_b=float(d.get("entry_price_b", d.get("entry_price", 0))),
        sl_price_b=float(d.get("sl_price_b", d.get("sl_price", 0))),
        tp_price_b=float(d.get("tp_price_b", d.get("tp_price", 0))),
        features=d.get("features") or {},
        features_b=d.get("features_b") or {},
        oscillator_label=d.get("oscillator_label", "WAIT"),
        is_dual=bool(d.get("is_dual", False)),
        created_at=d.get("created_at", ""),
        expires_at=d.get("expires_at", ""),
        hit_count=int(d.get("hit_count", 0)),
        last_hit_at=d.get("last_hit_at", ""),
        triggered=bool(d.get("triggered", False)),
        htf_prioritized=bool(d.get("htf_prioritized", False)),
        rejection_confirmed=bool(d.get("rejection_confirmed", False)),
    )


# ---------------------------------------------------------------------------
# Zone ID builder (stable across scans for same setup bar)
# ---------------------------------------------------------------------------
def _build_zone_id(setup: dict, symbol: str) -> str:
    tf = setup.get("timeframe", "XX")
    strat = setup.get("strategy", "UNK")
    direction = "BULL" if int(setup.get("direction", 1)) == 1 else "BEAR"
    entry = float(setup.get("entry_price", 0))
    bar_time = str(setup.get("time", ""))[:16].replace(" ", "T").replace(":", "")
    return f"{tf}_{strat}_{direction}_{entry:.3f}_{bar_time}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def save_watch_zones(symbol: str, all_setups_or_candidates: list, confidence_threshold: float = 0.50) -> int:
    """
    Called at the end of each full scan with the list of all detected setup candidates
    (both single and dual). Saves or updates watch zones to disk.

    Accepts either:
    - A list of raw `setup` dicts (from get_active_setups)
    - A list of scanner `candidate` dicts (with 'opt_a', 'opt_b', 'opt' keys)

    Returns the count of active zones after the update.
    """
    now = datetime.now()
    expires = now + timedelta(minutes=_ZONE_TTL_MINUTES)
    now_str = now.strftime("%Y-%m-%d %H:%M:%S")
    exp_str = expires.strftime("%Y-%m-%d %H:%M:%S")

    existing = _load_zones(symbol)

    # Purge expired and already-triggered zones first
    existing = {
        k: v for k, v in existing.items()
        if not _is_expired(v, now) and not v.get("triggered", False)
    }

    added = 0
    for item in all_setups_or_candidates:
        # Determine if this is a candidate dict or raw setup dict
        is_candidate = "opt_a" in item or "opt" in item

        if is_candidate:
            if item.get("is_dual") and item.get("opt_a") and item.get("opt_b"):
                opt_a = item["opt_a"]
                opt_b = item["opt_b"]
                zone = _build_zone_from_dual(symbol, opt_a, opt_b, item.get("prob_a", 0.5), item.get("prob_b", 0.5), now_str, exp_str)
            elif item.get("opt"):
                opt = item["opt"]
                zone = _build_zone_from_single(symbol, opt, item.get("max_prob", 0.5), now_str, exp_str)
            else:
                continue
        else:
            # Raw setup dict
            zone = _build_zone_from_single(symbol, item, item.get("probability", 0.5), now_str, exp_str)

        if zone is None:
            continue

        zid = zone.zone_id
        if zid in existing:
            # Update probability and oscillator label from latest scan, preserve hit state
            existing_zone = existing[zid]
            existing_zone["probability"] = zone.probability
            existing_zone["probability_b"] = zone.probability_b
            existing_zone["oscillator_label"] = zone.oscillator_label
            existing_zone["rejection_confirmed"] = zone.rejection_confirmed
            existing_zone["htf_prioritized"] = zone.htf_prioritized
            existing_zone["features"] = zone.features
            existing_zone["features_b"] = zone.features_b
            # Extend TTL each scan since zone is still active
            existing_zone["expires_at"] = exp_str
        else:
            existing[zid] = _zone_to_dict(zone)
            added += 1

    # Trim to max zones, keeping highest probability ones
    if len(existing) > _MAX_ZONES_PER_SYMBOL:
        sorted_zones = sorted(existing.items(), key=lambda kv: -kv[1].get("probability", 0))
        existing = dict(sorted_zones[:_MAX_ZONES_PER_SYMBOL])

    _save_zones(symbol, existing)
    print(f"[WatchZones] {symbol}: {len(existing)} active zones ({added} new) | TTL {_ZONE_TTL_MINUTES}min")
    return len(existing)


def check_price_in_watch_zones(
    symbol: str,
    current_tick,
    confidence_threshold: float = 0.50,
) -> list[WatchZoneHit]:
    """
    Called every tick. Returns list of WatchZoneHit for zones where price has
    entered or is very close to the entry zone. Caller should iterate and
    immediately trigger entry evaluation + execution for each hit.

    A "hit" happens when:
    - BULL zone: current ask is <= zone_top + buffer AND >= zone_bottom - buffer
    - BEAR zone: current bid is >= zone_bottom - buffer AND <= zone_top + buffer
    - Probability >= confidence_threshold (skip weak zones at tick time)
    - Zone not yet triggered
    """
    if current_tick is None:
        return []

    try:
        current_ask = float(current_tick.ask)
        current_bid = float(current_tick.bid)
    except (TypeError, ValueError, AttributeError):
        return []

    zones_data = _load_zones(symbol)
    if not zones_data:
        return []

    now = datetime.now()
    hits: list[WatchZoneHit] = []
    zones_updated = False

    for zid, zd in list(zones_data.items()):
        if _is_expired(zd, now):
            del zones_data[zid]
            zones_updated = True
            continue
        if zd.get("triggered", False):
            continue

        prob = float(zd.get("probability", 0.5))
        if prob < confidence_threshold * 0.90:
            # Skip zones well below threshold — no point watching them
            # Using 90% of threshold to allow zones that are close to qualifying
            continue

        direction = int(zd.get("direction", 1))
        zone_top = float(zd.get("zone_top", 0))
        zone_bottom = float(zd.get("zone_bottom", 0))
        buf = _ZONE_APPROACH_BUFFER

        if direction == 1:  # BULL: price should be near or inside the zone from above
            price = current_ask
            in_zone = (zone_bottom - buf) <= price <= (zone_top + buf)
        else:               # BEAR: price should be near or inside the zone from below
            price = current_bid
            in_zone = (zone_bottom - buf) <= price <= (zone_top + buf)

        if not in_zone:
            continue

        # We have a hit
        zone = _zone_from_dict(zd)
        should_trigger = prob >= confidence_threshold
        hit = WatchZoneHit(
            zone=zone,
            current_price=price,
            entry_triggered=should_trigger,
            reason=(
                f"price {price:.3f} entered {zone.timeframe} {zone.strategy} zone "
                f"[{zone_bottom:.3f}–{zone_top:.3f}] | prob {prob:.1%}"
            ),
        )
        hits.append(hit)

        # Update hit count in storage
        zones_data[zid]["hit_count"] = zd.get("hit_count", 0) + 1
        zones_data[zid]["last_hit_at"] = now.strftime("%Y-%m-%d %H:%M:%S")
        zones_updated = True

    if zones_updated:
        _save_zones(symbol, zones_data)

    return hits


def mark_zone_triggered(symbol: str, zone_id: str) -> None:
    """Mark a zone as triggered (order placed) so it won't fire again."""
    zones_data = _load_zones(symbol)
    if zone_id in zones_data:
        zones_data[zone_id]["triggered"] = True
        zones_data[zone_id]["triggered_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _save_zones(symbol, zones_data)


def clear_watch_zones(symbol: str) -> None:
    """Clear all watch zones for a symbol (call on clean restart if needed)."""
    _save_zones(symbol, {})


def get_active_zone_count(symbol: str) -> int:
    """Quick count of non-expired, non-triggered zones."""
    zones_data = _load_zones(symbol)
    now = datetime.now()
    return sum(
        1 for zd in zones_data.values()
        if not _is_expired(zd, now) and not zd.get("triggered", False)
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _is_expired(zone_dict: dict, now: datetime) -> bool:
    exp_str = zone_dict.get("expires_at", "")
    if not exp_str:
        return False
    try:
        exp_dt = datetime.strptime(str(exp_str)[:19], "%Y-%m-%d %H:%M:%S")
        return now > exp_dt
    except (ValueError, TypeError):
        return False


def _extract_oscillator_label(setup: dict) -> str:
    """Extract oscillator label string from setup's oscillator context (if any)."""
    osc = setup.get("oscillator")
    if osc is None:
        return "WAIT"
    # Try MultiTFOscillatorContext
    if hasattr(osc, "signal") and osc.signal is not None:
        return osc.signal.signal_label or "WAIT"
    # Try OscillatorContext directly
    if hasattr(osc, "signal_label"):
        return osc.signal_label or "WAIT"
    return "WAIT"


def _compute_zone_bounds(setup: dict) -> tuple[float, float]:
    """
    Compute [zone_bottom, zone_top] from setup.
    For FVG/BPR this is the FVG body; we fall back to SL↔entry range.
    """
    entry = float(setup.get("entry_price", 0))
    sl = float(setup.get("sl_price", entry))
    direction = int(setup.get("direction", 1))

    # Try to get the actual FVG/OB body if present
    fvg_top = setup.get("fvg_top") or setup.get("ob_top")
    fvg_bottom = setup.get("fvg_bottom") or setup.get("ob_bottom")

    if fvg_top is not None and fvg_bottom is not None:
        try:
            return float(fvg_bottom), float(fvg_top)
        except (TypeError, ValueError):
            pass

    # Fallback: use the SL–entry range as the "zone"
    lo = min(entry, sl)
    hi = max(entry, sl)
    return lo, hi


def _build_zone_from_single(
    symbol: str,
    setup: dict,
    probability: float,
    now_str: str,
    exp_str: str,
) -> WatchZone | None:
    try:
        entry = float(setup["entry_price"])
        sl = float(setup["sl_price"])
        tp = float(setup["tp_price"])
    except (KeyError, TypeError, ValueError):
        return None

    zone_bottom, zone_top = _compute_zone_bounds(setup)
    zid = _build_zone_id(setup, symbol)

    return WatchZone(
        zone_id=zid,
        symbol=symbol,
        timeframe=setup.get("timeframe", "M30"),
        strategy=setup.get("strategy", "FVG"),
        direction=int(setup.get("direction", 1)),
        entry_price=entry,
        sl_price=sl,
        tp_price=tp,
        zone_top=zone_top,
        zone_bottom=zone_bottom,
        probability=float(probability),
        probability_b=float(probability),
        entry_price_b=entry,
        sl_price_b=sl,
        tp_price_b=tp,
        features=setup.get("features") or {},
        features_b=setup.get("features") or {},
        oscillator_label=_extract_oscillator_label(setup),
        is_dual=False,
        created_at=now_str,
        expires_at=exp_str,
        htf_prioritized=bool(setup.get("htf_prioritized", False)),
        rejection_confirmed=bool(setup.get("rejection_confirmed", False)),
    )


def _build_zone_from_dual(
    symbol: str,
    opt_a: dict,
    opt_b: dict,
    prob_a: float,
    prob_b: float,
    now_str: str,
    exp_str: str,
) -> WatchZone | None:
    try:
        entry_a = float(opt_a["entry_price"])
        sl_a = float(opt_a["sl_price"])
        tp_a = float(opt_a["tp_price"])
        entry_b = float(opt_b["entry_price"])
        sl_b = float(opt_b["sl_price"])
        tp_b = float(opt_b["tp_price"])
    except (KeyError, TypeError, ValueError):
        return None

    # Zone spans from the deepest fib entry (opt_b) to sl of opt_a
    direction = int(opt_a.get("direction", 1))
    if direction == 1:
        # Bull: SL is below, entries are above SL
        zone_bottom = min(sl_a, sl_b)
        zone_top = max(entry_a, entry_b)
    else:
        # Bear: SL is above, entries are below SL
        zone_top = max(sl_a, sl_b)
        zone_bottom = min(entry_a, entry_b)

    zid = _build_zone_id(opt_a, symbol)
    max_prob = max(prob_a, prob_b)

    return WatchZone(
        zone_id=zid,
        symbol=symbol,
        timeframe=opt_a.get("timeframe", "M30"),
        strategy=opt_a.get("strategy", "FVG"),
        direction=direction,
        entry_price=entry_a,
        sl_price=sl_a,
        tp_price=tp_a,
        zone_top=zone_top,
        zone_bottom=zone_bottom,
        probability=max_prob,
        probability_b=prob_b,
        entry_price_b=entry_b,
        sl_price_b=sl_b,
        tp_price_b=tp_b,
        features=opt_a.get("features") or {},
        features_b=opt_b.get("features") or {},
        oscillator_label=_extract_oscillator_label(opt_a),
        is_dual=True,
        created_at=now_str,
        expires_at=exp_str,
        htf_prioritized=bool(opt_a.get("htf_prioritized", False)),
        rejection_confirmed=bool(opt_a.get("rejection_confirmed", False)),
    )
