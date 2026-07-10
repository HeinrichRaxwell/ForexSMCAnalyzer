from dataclasses import dataclass
from datetime import datetime

LIVE_ENTRY_TIMEFRAMES = frozenset({"M30", "H1", "H4", "D1"})


@dataclass(frozen=True)
class RealtimeReactionDecision:
    should_enter: bool
    reason: str
    current_price: float | None = None
    reaction_move: float | None = None


@dataclass(frozen=True)
class RealtimeWatchCandidate:
    signal_key: str
    leg: str
    setup: dict


@dataclass(frozen=True)
class RealtimeReactionPassResult:
    changed: bool
    executed_count: int
    checked_count: int


def _normalize_timeframe(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        numeric_map = {
            30: "M30",
            60: "H1",
            240: "H4",
            1440: "D1",
        }
        return numeric_map.get(int(value))
    normalized = str(value).strip().upper()
    aliases = {
        "30": "M30",
        "30.0": "M30",
        "30M": "M30",
        "M30": "M30",
        "60": "H1",
        "60.0": "H1",
        "1H": "H1",
        "H1": "H1",
        "240": "H4",
        "240.0": "H4",
        "4H": "H4",
        "H4": "H4",
        "1440": "D1",
        "1440.0": "D1",
        "1D": "D1",
        "D1": "D1",
    }
    return aliases.get(normalized, normalized)


def is_live_entry_timeframe(timeframe) -> bool:
    """Return True only for timeframes allowed to create new live entries."""
    return _normalize_timeframe(timeframe) in LIVE_ENTRY_TIMEFRAMES


def _tick_value(tick, field: str):
    try:
        value = float(getattr(tick, field))
    except (TypeError, ValueError, AttributeError):
        return None
    return value


def _direction_from_payload(payload: dict) -> int:
    raw_direction = payload.get("direction", 1)
    if isinstance(raw_direction, str):
        return -1 if raw_direction.upper().startswith("BEAR") or raw_direction.upper().startswith("SELL") else 1
    return -1 if int(raw_direction) < 0 else 1


def should_enter_on_realtime_reaction(
    setup: dict,
    *,
    previous_tick,
    current_tick,
    entry_buffer: float = 0.5,
    min_reaction_move: float = 0.10,
) -> RealtimeReactionDecision:
    """Confirm a fast tick reaction inside a known closed-candle setup entry zone."""
    if previous_tick is None or current_tick is None:
        return RealtimeReactionDecision(False, "missing_tick")

    direction = _direction_from_payload(setup)
    try:
        entry_price = float(setup["entry_price"])
        sl_price = float(setup["sl_price"])
    except (TypeError, ValueError, KeyError):
        return RealtimeReactionDecision(False, "invalid_setup_prices")

    if direction == 1:
        current_price = _tick_value(current_tick, "ask")
        previous_reaction_price = _tick_value(previous_tick, "bid")
        current_reaction_price = _tick_value(current_tick, "bid")
        in_zone = current_price is not None and sl_price + entry_buffer <= current_price <= entry_price + entry_buffer
        reaction_move = (
            current_reaction_price - previous_reaction_price
            if current_reaction_price is not None and previous_reaction_price is not None
            else None
        )
    else:
        current_price = _tick_value(current_tick, "bid")
        previous_reaction_price = _tick_value(previous_tick, "ask")
        current_reaction_price = _tick_value(current_tick, "ask")
        in_zone = current_price is not None and entry_price - entry_buffer <= current_price <= sl_price - entry_buffer
        reaction_move = (
            previous_reaction_price - current_reaction_price
            if current_reaction_price is not None and previous_reaction_price is not None
            else None
        )

    if not in_zone:
        return RealtimeReactionDecision(False, "price_outside_entry_zone", current_price, reaction_move)

    if reaction_move is None or reaction_move < float(min_reaction_move):
        return RealtimeReactionDecision(False, "realtime_reaction_not_confirmed", current_price, reaction_move)

    return RealtimeReactionDecision(True, "realtime_reaction_confirmed", current_price, reaction_move)


def _build_setup_from_record(record: dict, features_key: str = "features") -> dict:
    features = record.get(features_key)
    setup = dict(features) if isinstance(features, dict) else {}

    direction = setup.get("direction", record.get("direction", 1))
    setup["direction"] = _direction_from_payload({"direction": direction})

    if "entry_price" not in setup:
        setup["entry_price"] = record.get("price")
    if "sl_price" not in setup:
        setup["sl_price"] = record.get("sl_price")
    if "tp_price" not in setup:
        setup["tp_price"] = record.get("tp_price")

    setup["rejection_confirmed"] = bool(record.get("rejection_confirmed", True))
    setup.setdefault("timeframe", record.get("timeframe"))
    setup.setdefault("strategy", record.get("type"))
    return setup


def iter_realtime_watch_candidates(sent_signals: dict):
    """Yield accepted registry records without a live ticket for fast reaction monitoring."""
    for signal_key, record in (sent_signals or {}).items():
        if not isinstance(record, dict):
            continue
        if not is_live_entry_timeframe(record.get("timeframe")):
            continue
        if record.get("outcome_recorded") or record.get("is_low_confidence", False):
            continue

        if "price_0.5" in record or "features_0.5" in record:
            if record.get("ticket_a") is None and not record.get("outcome_a_recorded", False):
                yield RealtimeWatchCandidate(signal_key, "a", _build_setup_from_record(record, "features_0.5"))
            if record.get("ticket_b") is None and not record.get("outcome_b_recorded", False):
                yield RealtimeWatchCandidate(signal_key, "b", _build_setup_from_record(record, "features_0.618"))
            continue

        if record.get("ticket_id") is None:
            yield RealtimeWatchCandidate(signal_key, "single", _build_setup_from_record(record, "features"))


def _store_ticket(record: dict, leg: str, ticket_id: int, now: str, decision: RealtimeReactionDecision):
    if leg == "a":
        record["ticket_a"] = ticket_id
        record["outcome_a_recorded"] = False
    elif leg == "b":
        record["ticket_b"] = ticket_id
        record["outcome_b_recorded"] = False
    else:
        record["ticket_id"] = ticket_id
        record["outcome_recorded"] = False

    record["realtime_reaction_entry_at"] = now
    record["realtime_reaction_reason"] = decision.reason
    record["realtime_reaction_price"] = decision.current_price
    record["realtime_reaction_move"] = decision.reaction_move


def run_realtime_reaction_pass(
    sent_signals: dict,
    *,
    symbol: str,
    previous_tick,
    current_tick,
    execute_market_order,
    entry_buffer: float = 0.5,
    min_reaction_move: float = 0.10,
    now: str | None = None,
) -> RealtimeReactionPassResult:
    """Try fast market entries for already-registered setups that do not have tickets yet."""
    changed = False
    executed_count = 0
    checked_count = 0
    executed_signal_keys = set()
    timestamp = now or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for candidate in iter_realtime_watch_candidates(sent_signals):
        if candidate.signal_key in executed_signal_keys:
            continue
        checked_count += 1
        decision = should_enter_on_realtime_reaction(
            candidate.setup,
            previous_tick=previous_tick,
            current_tick=current_tick,
            entry_buffer=entry_buffer,
            min_reaction_move=min_reaction_move,
        )
        if not decision.should_enter:
            continue

        ticket_id, _message = execute_market_order(candidate.setup, symbol)
        if ticket_id is None:
            continue

        _store_ticket(sent_signals[candidate.signal_key], candidate.leg, ticket_id, timestamp, decision)
        changed = True
        executed_count += 1
        executed_signal_keys.add(candidate.signal_key)

    return RealtimeReactionPassResult(changed, executed_count, checked_count)
