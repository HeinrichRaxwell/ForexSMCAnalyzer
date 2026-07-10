from dataclasses import dataclass
from datetime import datetime, time


@dataclass(frozen=True)
class DailyRiskDecision:
    allowed: bool
    mode: str
    reason: str
    realized_pips: float
    consecutive_losses: int


@dataclass(frozen=True)
class DailyPipSummary:
    realized_pips: float
    closed_positions: int
    consecutive_losses: int


def evaluate_daily_risk(
    *,
    realized_pips: float,
    consecutive_losses: int,
    min_target_pips: float = 100.0,
    runner_target_pips: float = 300.0,
    max_loss_pips: float = 200.0,
    max_consecutive_losses: int = 3,
) -> DailyRiskDecision:
    realized = float(realized_pips)
    losses = int(consecutive_losses)
    if realized >= float(min_target_pips):
        return DailyRiskDecision(True, "protect_profit", "daily_min_target_reached", realized, losses)
    if realized <= -abs(float(max_loss_pips)) or losses >= int(max_consecutive_losses):
        return DailyRiskDecision(False, "halt", "daily_limits_hit", realized, losses)
    return DailyRiskDecision(True, "normal", "daily_limits_ok", realized, losses)


def summarize_daily_pips_from_deals(
    deals,
    *,
    symbol: str | None = None,
    magic: int,
    pip_multiplier: float,
    deal_entry_in,
    deal_entry_out,
    deal_type_buy,
    deal_type_sell,
) -> DailyPipSummary:
    entries = {}
    closed_results = []
    for deal in deals or []:
        if int(getattr(deal, "magic", magic)) != int(magic):
            continue
        deal_symbol = getattr(deal, "symbol", None)
        if symbol and deal_symbol is not None and str(deal_symbol) != str(symbol):
            continue
        position_id = getattr(deal, "position_id", None)
        if position_id is None:
            continue
        entry_kind = getattr(deal, "entry", None)
        deal_type = getattr(deal, "type", None)
        price = float(getattr(deal, "price", 0.0))

        if entry_kind == deal_entry_in:
            if deal_type == deal_type_buy:
                direction = 1
            elif deal_type == deal_type_sell:
                direction = -1
            else:
                continue
            entries[position_id] = (price, direction)
            continue

        if entry_kind == deal_entry_out and position_id in entries and pip_multiplier > 0:
            entry_price, direction = entries[position_id]
            pips = ((price - entry_price) * direction) / float(pip_multiplier)
            closed_results.append(float(pips))

    consecutive_losses = 0
    for pips in reversed(closed_results):
        if pips < 0:
            consecutive_losses += 1
        else:
            break

    return DailyPipSummary(
        realized_pips=sum(closed_results),
        closed_positions=len(closed_results),
        consecutive_losses=consecutive_losses,
    )


def _day_window(now=None):
    current = now or datetime.now()
    start = datetime.combine(current.date(), time.min)
    return start, current


def get_mt5_daily_pip_summary(mt5, symbol: str, magic: int, pip_multiplier: float, now=None) -> DailyPipSummary:
    start, end = _day_window(now)
    deals = mt5.history_deals_get(start, end)
    if deals is None:
        return DailyPipSummary(0.0, 0, 0)
    return summarize_daily_pips_from_deals(
        deals,
        symbol=symbol,
        magic=magic,
        pip_multiplier=pip_multiplier,
        deal_entry_in=getattr(mt5, "DEAL_ENTRY_IN", 0),
        deal_entry_out=getattr(mt5, "DEAL_ENTRY_OUT", 1),
        deal_type_buy=getattr(mt5, "DEAL_TYPE_BUY", 0),
        deal_type_sell=getattr(mt5, "DEAL_TYPE_SELL", 1),
    )
