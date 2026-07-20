import os
import sys
import time
import json
import argparse
from contextlib import contextmanager
from datetime import datetime
from html import escape
import re
import pandas as pd
import numpy as np
import MetaTrader5 as mt5

# Add project root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import connect_mt5, fetch_historical_data
from src.smc_detector import (
    detect_swing_points,
    detect_structures,
    detect_fvg_and_ob,
    detect_snr_and_swapzones,
    detect_bpr,
    get_pip_multiplier,
    detect_indecision_candles,
    detect_supply_demand_zones,
)
from src.labeler import get_killzone
from src.inference import predict_setup_probability, process_mt5_history_feedback
from src.rejection_detector import detect_rejection_at_level
from src.main import find_dynamic_tp, extract_active_htf_fvgs, get_active_setups, plot_smc_chart
from src.telegram_bot import send_telegram_alert
from src.indicators.knn_classifier import run_knn_classifier, calculate_knn_probability_at_bar
from src.indicators.volume_clusters import calculate_volume_clusters
from src.execution import (
    execute_trade_for_setup,
    execute_market_order_for_setup,
    manage_active_trades,
    get_active_broker_symbol,
    should_emergency_exit_on_reversal,
    _last_closed_trend,
)
from src.entry_quality_gate import (
    EntryGateDecision,
    MultiTFOscillatorContext,
    OscillatorContext,
    build_multi_tf_oscillator,
    build_oscillator_context,
    build_spread_context,
    evaluate_entry_quality,
    evaluate_multi_tf_osc_delta,
    format_multi_tf_oscillator_block,
    format_oscillator_line,
)
from src.live_trade_policy import confidence_tier, should_allow_live_strategy
from src.rollout_status import evaluate_rollout_status, load_env_values, _load_json as load_rollout_json
from src.realtime_reaction_watcher import (
    RealtimeReactionPassResult,
    is_live_entry_timeframe,
    run_realtime_reaction_pass,
)
from src.shadow_tracker import (
    build_shadow_signal_records,
    process_shadow_signal_outcomes,
    should_shadow_signal,
    upsert_shadow_signals,
)
from src.price_watch_zones import (
    WatchZoneHit,
    build_watch_zone_execution_setup,
    check_price_in_watch_zones,
    get_active_zone_count,
    mark_zone_execution_attempt,
    mark_zone_triggered,
    save_watch_zones,
)

# Storage for sent signal signatures
SENT_SIGNALS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sent_signals.json")
SHADOW_SIGNALS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "shadow_signals.json")

def get_sent_signals_file() -> str:
    from src.data_loader import get_active_account_login
    login = get_active_account_login()
    filename = f"sent_signals_{login}.json" if login else "sent_signals.json"
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", filename)

def get_shadow_signals_file() -> str:
    from src.data_loader import get_active_account_login
    login = get_active_account_login()
    filename = f"shadow_signals_{login}.json" if login else "shadow_signals.json"
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", filename)
DEFAULT_ACCEPT_THRESHOLD = 0.50
PRICE_TOO_FAR_MARKER = "price is too far from market"
PRICE_TOO_FAR_WATCH_REASON = "watch_price_too_far"


def configure_console_encoding(streams=None) -> int:
    """Prevent Windows console encodings from crashing scanner logging."""
    configured = 0
    for stream in streams or (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if not callable(reconfigure):
            continue
        try:
            reconfigure(errors="backslashreplace")
            configured += 1
        except (OSError, ValueError):
            continue
    return configured


LAST_KNOWN_LOGIN = None

def check_and_sync_active_account():
    """Detect if MT5 login has changed, and if so, synchronize active pending orders to the new account."""
    global LAST_KNOWN_LOGIN
    from src.data_loader import get_current_account_login
    current_login = get_current_account_login()
    if current_login is not None:
        if LAST_KNOWN_LOGIN is not None and current_login != LAST_KNOWN_LOGIN:
            print(f"[Account Change] Switched from #{LAST_KNOWN_LOGIN} to #{current_login}. Syncing active pending orders...")
            try:
                sync_pending_orders_on_account_change(LAST_KNOWN_LOGIN, current_login)
            except Exception as e:
                print(f"[Account Change] Failed to sync pending orders: {e}")
        LAST_KNOWN_LOGIN = current_login

def sync_pending_orders_on_account_change(old_login: int, new_login: int):
    """
    Read active pending orders from old_login's registry and place them on the new_login account.
    """
    import os
    import json
    from src.execution import execute_trade_for_setup
    from src.telegram_bot import send_telegram_alert
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    old_file = os.path.join(base_dir, "data", f"sent_signals_{old_login}.json")
    new_file = os.path.join(base_dir, "data", f"sent_signals_{new_login}.json")
    
    if not os.path.exists(old_file):
        print(f"[Account Sync] Old account registry {old_file} does not exist. Nothing to sync.")
        return
        
    try:
        with open(old_file, "r") as f:
            old_signals = json.load(f)
    except Exception as e:
        print(f"[Account Sync] Failed to read old account registry: {e}")
        return
        
    if not old_signals:
        return
        
    new_signals = {}
    if os.path.exists(new_file):
        try:
            with open(new_file, "r") as f:
                new_signals = json.load(f)
        except Exception:
            pass
            
    synced_count = 0
    
    for sig_key, sig_data in list(old_signals.items()):
        if sig_data.get("close_reason") or sig_data.get("manager_exit_trigger"):
            continue
            
        is_dual = "price_0.5" in sig_data or "ticket_a" in sig_data or "ticket_b" in sig_data
        
        if sig_key in new_signals:
            existing_sig = new_signals[sig_key]
            if is_dual:
                if (existing_sig.get("ticket_a") is not None) or (existing_sig.get("ticket_b") is not None):
                    continue
            else:
                if existing_sig.get("ticket") is not None:
                    continue
                    
        print(f"[Account Sync] Re-placing pending order for signal {sig_key} on new account #{new_login}...")
        
        def build_setup_dict(features, price, option_name):
            if not features:
                return None
            return {
                "timeframe": sig_data.get("timeframe"),
                "direction": 1 if sig_data.get("direction") == "BUY" else -1,
                "strategy_type": sig_data.get("type"),
                "entry_price": float(features.get("entry_price", price)),
                "sl_price": float(features.get("sl_price")),
                "tp_price": float(features.get("tp_price")),
                "option_name": option_name,
                "tp2_price": features.get("tp2_price"),
                "tp3_price": features.get("tp3_price"),
            }
            
        new_entry = dict(sig_data)
        placed_any = False
        
        if is_dual:
            if sig_data.get("ticket_a") is not None:
                setup_a = build_setup_dict(sig_data.get("features_0.5"), sig_data.get("price_0.5"), "Option A")
                if setup_a:
                    ticket_a, msg_a = execute_trade_for_setup(setup_a)
                    if ticket_a:
                        new_entry["ticket_a"] = ticket_a
                        placed_any = True
                    else:
                        print(f"[Account Sync] Option A failed: {msg_a}")
                        new_entry["ticket_a"] = None
            
            if sig_data.get("ticket_b") is not None:
                setup_b = build_setup_dict(sig_data.get("features_0.618"), sig_data.get("price_0.618"), "Option B")
                if setup_b:
                    ticket_b, msg_b = execute_trade_for_setup(setup_b)
                    if ticket_b:
                        new_entry["ticket_b"] = ticket_b
                        placed_any = True
                    else:
                        print(f"[Account Sync] Option B failed: {msg_b}")
                        new_entry["ticket_b"] = None
        else:
            if sig_data.get("ticket") is not None:
                setup_single = build_setup_dict(sig_data.get("features"), sig_data.get("price"), "Single Option")
                if setup_single:
                    ticket, msg = execute_trade_for_setup(setup_single)
                    if ticket:
                        new_entry["ticket"] = ticket
                        placed_any = True
                    else:
                        print(f"[Account Sync] Single order failed: {msg}")
                        new_entry["ticket"] = None
                        
        if placed_any:
            new_signals[sig_key] = new_entry
            synced_count += 1
            try:
                alert_text = (
                    f"🔄 <b>[Account Sync] Akun terdeteksi berubah ke #{new_login}</b>\n\n"
                    f"Memasang ulang pending limit order aktif dari akun lama:\n"
                    f"• <b>TF/Strategi:</b> <code>{sig_data.get('timeframe')} {sig_data.get('type')}</code>\n"
                    f"• <b>Arah:</b> <code>{sig_data.get('direction')}</code>\n"
                )
                if is_dual:
                    if new_entry.get("ticket_a"):
                        alert_text += f"• <b>Ticket Option A:</b> #{new_entry.get('ticket_a')}\n"
                    if new_entry.get("ticket_b"):
                        alert_text += f"• <b>Ticket Option B:</b> #{new_entry.get('ticket_b')}\n"
                else:
                    if new_entry.get("ticket"):
                        alert_text += f"• <b>Ticket:</b> #{new_entry.get('ticket')}\n"
                send_telegram_alert(alert_text)
            except Exception:
                pass
                
    if synced_count > 0:
        os.makedirs(os.path.dirname(new_file), exist_ok=True)
        with open(new_file, "w") as f:
            json.dump(new_signals, f, indent=4)
        print(f"[Account Sync] Successfully synchronized {synced_count} active pending orders to account #{new_login}.")


def is_cooldown_expired(last_attempt_str: str, cooldown_seconds: int = 60) -> bool:
    """Return True if the cooldown period has passed since the last execution attempt."""
    if not last_attempt_str:
        return True
    try:
        from datetime import datetime
        last_time = datetime.strptime(str(last_attempt_str), '%Y-%m-%d %H:%M:%S')
        elapsed = (datetime.now() - last_time).total_seconds()
        return elapsed >= cooldown_seconds
    except Exception:
        return True


def is_price_too_far_execution(message) -> bool:
    """Return True when MT5 execution skipped only because entry is currently far."""
    return PRICE_TOO_FAR_MARKER in str(message or "").lower()


def recovery_failure_action(message) -> str:
    """Classify a failed order so recovery only retries meaningful failures."""
    text = str(message or "").strip().lower()
    if is_price_too_far_execution(text):
        return "price_watch"
    if any(
        marker in text
        for marker in (
            "auto-execution disabled",
            "live strategy policy blocked",
            "strategy_not_allowlisted",
            "strategy_blocked",
            "entry_policy_",
            "timeframe ",
            "immediate emergency reversal",
        )
    ) and ("disabled" in text or "blocked" in text or "entry_policy_" in text or "emergency reversal" in text):
        return "blocked"
    if any(
        marker in text
        for marker in (
            "max concurrent trades reached",
            "max same-direction trades reached",
            "max same-direction exposure reached",
            "max pending orders reached",
            "daily risk governor",
            "daily governor unavailable",
            "blocked mixed strategy",
            "proximity",
            "market indicators check failed",
        )
    ):
        return "deferred"
    return "retry"


def record_recovery_failure(
    sig_data: dict,
    message,
    retries: int,
    max_retries: int,
    *,
    message_key: str,
    retries_key: str,
    outcome_key: str,
) -> str:
    """Persist a recovery result without consuming retries for deferred states."""
    action = recovery_failure_action(message)
    sig_data[message_key] = message
    if action == "price_watch":
        sig_data.update(_price_watch_metadata(message))
    elif action == "blocked":
        sig_data[outcome_key] = True
        sig_data["watch_status"] = "execution_blocked"
    elif action == "deferred":
        sig_data["watch_status"] = "execution_deferred"
    else:
        next_retries = retries + 1
        sig_data[retries_key] = next_retries
        if next_retries >= max_retries:
            sig_data[outcome_key] = True
            sig_data["watch_status"] = "execution_retry_exhausted"
    return action


def enforce_recovery_strategy_policy(
    sig_data: dict,
    *,
    strategy: str,
    setup: dict,
    probability: float,
    timeframe: str,
    outcome_keys: tuple[str, ...],
    message_keys: tuple[str, ...],
) -> tuple[bool, str]:
    """Apply the current policy before a historical signal can be recovered."""
    allowed, reason = should_allow_live_strategy(
        strategy,
        setup,
        probability=probability,
        timeframe=timeframe,
        entry_type="Standard Limit",
    )
    if allowed:
        return True, reason

    sig_data["watch_status"] = "execution_blocked"
    for key in outcome_keys:
        sig_data[key] = True
    for key in message_keys:
        sig_data[key] = f"Recovery blocked by live strategy policy: {reason}"
    return False, reason


def should_retry_unfilled_watch_record(sig_data: dict, ticket_fields, outcome_fields=None) -> bool:
    """Return True for accepted live records waiting for price to return near entry."""
    if not isinstance(sig_data, dict) or sig_data.get("is_low_confidence", False):
        return False

    watch_reason = str(sig_data.get("watch_reason", "")).lower()
    known_far_message = any(
        is_price_too_far_execution(sig_data.get(field))
        for field in (
            "execution_message",
            "execution_message_0.5",
            "execution_message_0.618",
            "watch_last_execution_message",
            "watch_last_execution_message_0.5",
            "watch_last_execution_message_0.618",
        )
    )
    if watch_reason != PRICE_TOO_FAR_WATCH_REASON and not known_far_message:
        return False

    if not any(sig_data.get(field) is None for field in ticket_fields):
        return False

    if outcome_fields is None:
        outcome_fields = ("outcome_recorded", "outcome_a_recorded", "outcome_b_recorded")
    return not any(sig_data.get(field, False) for field in outcome_fields)


def _price_watch_metadata(*messages) -> dict:
    metadata = {}
    if any(is_price_too_far_execution(message) for message in messages):
        metadata["watch_reason"] = PRICE_TOO_FAR_WATCH_REASON
        metadata["watch_status"] = "waiting_for_price_return"
        metadata["watch_last_checked_at"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    return metadata


def _read_float_env(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default


def _execute_trades_enabled() -> bool:
    return os.getenv("MT5_EXECUTE_TRADES", "False").strip().lower() in {"1", "true", "yes", "on"}


def _bool_env(name: str, default: bool = False) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return str(raw_value).strip().lower() in {"1", "true", "yes", "on"}


def get_accept_threshold(cli_threshold=None, default: float = DEFAULT_ACCEPT_THRESHOLD) -> float:
    """Resolve live execution confidence threshold from CLI, env, then default."""
    if cli_threshold is not None:
        threshold = float(cli_threshold)
    else:
        raw_value = os.getenv("ML_ACCEPT_THRESHOLD")
        if raw_value is None:
            threshold = default
        else:
            try:
                threshold = float(raw_value)
            except (TypeError, ValueError):
                print(f"[Scanner Config] Invalid ML_ACCEPT_THRESHOLD={raw_value!r}; using {default}.")
                threshold = default

    if _execute_trades_enabled():
        live_minimum = _read_float_env("ML_LIVE_MIN_THRESHOLD", threshold)
        if threshold < live_minimum:
            print(
                f"[Scanner Config] Raised live threshold from {threshold:.2f} "
                f"to {live_minimum:.2f} because MT5_EXECUTE_TRADES=True."
            )
            threshold = live_minimum
    return threshold


def assert_rollout_ready_for_live(threshold: float, *, report_path: str = "data/calibration_report.json", env_path: str = ".env"):
    """Fail fast before a VPS scanner can place real-money orders on a blocked rollout."""
    if not _execute_trades_enabled():
        return True, "MT5_EXECUTE_TRADES=False"
    if not _bool_env("MT5_REQUIRE_ROLLOUT_READY", True):
        return True, "MT5_REQUIRE_ROLLOUT_READY disabled"

    report = load_rollout_json(report_path)
    env_values = load_env_values(env_path)
    status = evaluate_rollout_status(
        report,
        env_values=env_values,
        requested_threshold=threshold,
        min_samples=100,
        min_expectancy_r=0.25,
        max_drawdown_r=5.0,
        min_profit_factor=1.25,
        max_consecutive_losses=5,
        profile="real-money",
        required_artifacts=[
            "models/smc_xgb_classifier.joblib",
            "models/smc_lgb_classifier.joblib",
            "models/confidence_calibrator.joblib",
            "data/calibration_report.json",
        ],
    )
    if status["status"] == "READY":
        return True, "real-money rollout preflight ready"

    failures = [
        f"{check['name']}={check['message']}"
        for check in status.get("checks", [])
        if check.get("status") == "FAIL"
    ]
    message = "; ".join(failures[:6])
    if len(failures) > 6:
        message = f"{message}; +{len(failures) - 6} more"
    return False, message or "real-money rollout preflight blocked"


def _scanner_lock_file(symbol: str, magic: int, lock_dir: str = None) -> str:
    if lock_dir is None:
        lock_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    # One magic number owns one account-level execution budget; separate symbol workers must not multiply exposure.
    return os.path.join(lock_dir, f"scanner_magic_{int(magic)}.lock")


def _pid_is_running(pid: int) -> bool:
    try:
        pid_value = int(pid)
    except (TypeError, ValueError):
        return False
    if pid_value <= 0:
        return False
    if pid_value == os.getpid():
        return True
    if os.name == "nt":
        try:
            import ctypes

            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            STILL_ACTIVE = 259
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid_value)
            if not handle:
                return False
            try:
                exit_code = ctypes.c_ulong()
                if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                    return False
                return exit_code.value == STILL_ACTIVE
            finally:
                kernel32.CloseHandle(handle)
        except Exception:
            return False
    try:
        os.kill(pid_value, 0)
        return True
    except OSError:
        return False


@contextmanager
def scanner_instance_lock(symbol: str, magic: int, lock_dir: str = None):
    """Prevent multiple live scanner workers from racing the same symbol/magic."""
    lock_path = _scanner_lock_file(symbol, magic, lock_dir=lock_dir)
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    fd = None
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            try:
                with open(lock_path, "r", encoding="utf-8") as lock_file:
                    existing_pid = lock_file.read().strip()
            except OSError:
                existing_pid = ""
            if existing_pid and not _pid_is_running(existing_pid):
                try:
                    os.remove(lock_path)
                    continue
                except OSError:
                    pass
            raise RuntimeError(f"scanner already running for {symbol} magic {magic}: {lock_path}")

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as lock_file:
            lock_file.write(str(os.getpid()))
        fd = None
        yield lock_path
    finally:
        if fd is not None:
            os.close(fd)
        try:
            with open(lock_path, "r", encoding="utf-8") as lock_file:
                owner_pid = lock_file.read().strip()
            if owner_pid == str(os.getpid()):
                os.remove(lock_path)
        except OSError:
            pass


def get_live_spread_context(symbol: str):
    """Build spread context from live MT5 tick and symbol precision."""
    try:
        import MetaTrader5 as mt5
        from src.execution import get_active_broker_symbol

        broker_symbol = get_active_broker_symbol(symbol)
        tick = mt5.symbol_info_tick(broker_symbol)
        info = mt5.symbol_info(broker_symbol)
        if tick is None or info is None:
            return None

        digits = getattr(info, "digits", None)
        point = getattr(info, "point", None)
        if not point:
            point = 10 ** -int(digits) if digits is not None else 0.001

        return build_spread_context(
            bid=getattr(tick, "bid", None),
            ask=getattr(tick, "ask", None),
            point=point,
            digits=digits,
        )
    except Exception as exc:
        print(f"[Entry Gate] Unable to read live spread for {symbol}: {exc}")
        return None


def _entry_gate_to_dict(decision, *, enforced: bool = True) -> dict:
    return {
        "allowed": bool(decision.allowed) if enforced else True,
        "filtered_reason": decision.filtered_reason,
        "reason": decision.reason,
        "required_confidence": decision.required_confidence,
        "spread_r": decision.spread_r,
        "enforced": bool(enforced),
        "would_have_allowed": bool(decision.allowed),
    }


def evaluate_live_entry_gate(
    setup: dict,
    *,
    strategy: str,
    probability: float,
    accept_threshold: float,
    symbol: str,
    timeframe: str,
    timeframes_data: dict,
    oscillator=None,
):
    """Attach entry-quality diagnostics without blocking live execution."""
    tf_df = (timeframes_data or {}).get(timeframe)
    # Use pre-computed oscillator if available; fall back to per-call computation
    osc = oscillator if oscillator is not None else build_oscillator_context(tf_df)
    observed_decision = evaluate_entry_quality(
        setup,
        strategy=strategy,
        probability=probability,
        accept_threshold=accept_threshold,
        spread=get_live_spread_context(symbol),
        oscillator=osc,
    )

    setup["entry_gate"] = _entry_gate_to_dict(observed_decision, enforced=False)

    if observed_decision.allowed:
        return observed_decision

    return EntryGateDecision(
        allowed=True,
        filtered_reason="entry_gate_observer_only",
        reason=f"entry gate observer-only; would have blocked: {observed_decision.reason}",
        required_confidence=observed_decision.required_confidence,
        spread_r=observed_decision.spread_r,
    )


STRATEGY_DISPLAY_NAMES = {
    'FVG': "Fair Value Gap",
    'OB': "Order Block",
    'BPR': "Balanced Price Range",
    'IC': "Indecision Candle",
    'SND': "Supply/Demand Zone",
    'Pivot': "Pivot Rejection",
    'Swapzone': "Swapzone (SBR/RBS)",
    'Breaker': "Breaker Block",
}


def _html_text(value) -> str:
    return escape(str(value), quote=False)


def _html_text_multiline(value) -> str:
    """Escape HTML but preserve newlines as Telegram-compatible line breaks."""
    return escape(str(value), quote=False).replace("\n", "\n")


def _format_price(value) -> str:
    try:
        return f"<code>{float(value):.3f}</code>"
    except (TypeError, ValueError):
        return "<code>-</code>"


def _format_percent(value) -> str:
    try:
        return f"<code>{float(value):.2%}</code>"
    except (TypeError, ValueError):
        return "<code>-</code>"


def get_strategy_display_name(strategy: str) -> str:
    return STRATEGY_DISPLAY_NAMES.get(strategy, strategy or "SMC Setup")


def format_direction_label(direction: int) -> str:
    return "BUY / Long" if int(direction) == 1 else "SELL / Short"


def format_rejection_status(rejection_sources) -> str:
    clean_sources = sorted({
        str(src)
        for src in rejection_sources
        if src not in (None, "", "None")
    })
    if clean_sources:
        return f"Confirmed ({'/'.join(clean_sources)})"
    return "Not confirmed on current LTF touch"


def format_htf_priority_status(is_prioritized: bool) -> str:
    return "Confirmed" if is_prioritized else "Not active"


def format_execution_status(ticket_id, execution_message: str, skipped_peer: str = None, monitoring_only: bool = False) -> str:
    if monitoring_only:
        return "Monitoring only (MT5_EXECUTE_TRADES disabled)"
    if ticket_id and "MARKET" in str(execution_message).upper():
        return f"Market order active (ticket #{ticket_id})"
    if ticket_id:
        return f"Pending order placed (ticket #{ticket_id})"
    if is_price_too_far_execution(execution_message):
        return "Watching price return (will retry when closer)"
    if skipped_peer:
        return f"Skipped ({skipped_peer} market order active)"
    return f"Failed ({_html_text(execution_message)})"


def format_entry_policy_status(timeframe: str) -> str:
    if is_live_entry_timeframe(timeframe):
        return "Live-entry timeframe"
    return "Monitoring-only timeframe (no live order)"


def _format_signal_source_note() -> str:
    return "Source: closed-candle scanner data and MT5 execution response."


def _format_confluence_lines(confluences) -> str:
    if not confluences:
        return "No extra confluence detail recorded."
    return "\n".join(f"{idx}. {_html_text(reason)}" for idx, reason in enumerate(confluences, start=1))


def _format_htf_match_lines(htf_matches) -> str:
    if not htf_matches:
        return ""
    lines = ["", "<b>HTF Match</b>"]
    for match in htf_matches:
        timeframe = _html_text(match.get('timeframe', '-'))
        lines.append(f"- <code>{timeframe}</code> FVG {_format_price(match.get('bottom'))}-{_format_price(match.get('top'))}")
    return "\n".join(lines)


def format_dual_signal_message(
    *,
    symbol: str,
    timeframe: str,
    direction: int,
    setup_desc: str,
    probability_a: float,
    probability_b: float,
    confidence_threshold: float,
    opt_a: dict,
    opt_b: dict,
    execution_status_a: str,
    execution_status_b: str,
    htf_priority_status: str,
    rejection_status: str,
    confluences,
    htf_matches,
    oscillator_line: str = "",
) -> str:
    """Build the Telegram body for a dual-fib trade signal."""
    # Oscillator section — preserve multi-line formatting from format_multi_tf_oscillator_block
    osc_section = f"<b>Oscillator (RSI8 + Stoch)</b>\n{_html_text_multiline(oscillator_line)}\n\n" if oscillator_line else ""

    # Lot sizes — read from actual env / dynamic lot config (no hardcode)
    from src.execution import resolve_lot_size
    lot_a = resolve_lot_size("0.5", symbol)
    lot_b = resolve_lot_size("0.618", symbol)
    lot_a_str = f"{lot_a:.2f} lot" if lot_a else "lot n/a"
    lot_b_str = f"{lot_b:.2f} lot" if lot_b else "lot n/a"

    # TP levels — only show if actually computed (avoid misleading '-')
    def _tp_line(label: str, key: str, opt: dict) -> str:
        val = opt.get(key)
        try:
            fval = float(val)
            if fval > 0:
                return f"{label}: {_format_price(fval)}\n"
        except (TypeError, ValueError):
            pass
        return ""

    tp_lines_a = (
        f"TP1: {_format_price(opt_a.get('tp_price'))}\n"
        + _tp_line("TP2 dynamic", "tp2_price", opt_a)
        + _tp_line("TP3 extension", "tp3_price", opt_a)
    )

    return (
        f"<b>SMC Trade Signal - {_html_text(symbol)}</b>\n\n"
        f"<b>Signal</b>\n"
        f"Symbol: <code>{_html_text(symbol)}</code>\n"
        f"Timeframe: <code>{_html_text(timeframe)}</code>\n"
        f"Direction: <b>{format_direction_label(direction)}</b>\n"
        f"Setup: {_html_text(setup_desc)}\n\n"
        f"<b>Model Confidence</b>\n"
        f"0.500 entry ({lot_a_str}): {_format_percent(probability_a)}\n"
        f"0.618 entry ({lot_b_str}): {_format_percent(probability_b)}\n"
        f"Accept threshold: {_format_percent(confidence_threshold)}\n\n"
        f"{osc_section}"
        f"<b>Execution</b>\n"
        f"Entry policy: {_html_text(format_entry_policy_status(timeframe))}\n"
        f"HTF priority: {_html_text(htf_priority_status)}\n"
        f"LTF rejection: {_html_text(rejection_status)}\n"
        f"Order 0.500: {execution_status_a}\n"
        f"Order 0.618: {execution_status_b}\n\n"
        f"<b>Levels</b>\n"
        f"Entry 0.500 ({lot_a_str}): {_format_price(opt_a.get('entry_price'))}\n"
        f"Entry 0.618 ({lot_b_str}): {_format_price(opt_b.get('entry_price'))}\n"
        f"Stop Loss: {_format_price(opt_a.get('sl_price'))}\n"
        f"{tp_lines_a}\n"
        f"<b>Confluence</b>\n"
        f"{_format_confluence_lines(confluences)}"
        f"{_format_htf_match_lines(htf_matches)}\n\n"
        f"<i>{_html_text(_format_signal_source_note())}</i>"
    )


def format_single_signal_message(
    *,
    symbol: str,
    timeframe: str,
    direction: int,
    setup_desc: str,
    probability: float,
    confidence_threshold: float,
    setup: dict,
    execution_status: str,
    htf_priority_status: str,
    rejection_status: str,
    confluences,
    htf_matches,
    oscillator_line: str = "",
) -> str:
    """Build the Telegram body for a single-entry trade signal."""
    # Oscillator section — preserve multi-line formatting from format_multi_tf_oscillator_block
    osc_section = f"<b>Oscillator (RSI8 + Stoch)</b>\n{_html_text_multiline(oscillator_line)}\n\n" if oscillator_line else ""

    # Lot size — read from actual env / dynamic lot config
    from src.execution import resolve_lot_size
    lot = resolve_lot_size(setup.get('option_name', ''), symbol)
    lot_str = f"{lot:.2f} lot" if lot else "lot n/a"

    # TP levels — only show if actually computed
    def _tp_line(label: str, key: str) -> str:
        val = setup.get(key)
        try:
            fval = float(val)
            if fval > 0:
                return f"{label}: {_format_price(fval)}\n"
        except (TypeError, ValueError):
            pass
        return ""

    tp_lines = (
        f"TP1: {_format_price(setup.get('tp_price'))}\n"
        + _tp_line("TP2 dynamic", "tp2_price")
        + _tp_line("TP3 extension", "tp3_price")
    )

    return (
        f"<b>SMC Trade Signal - {_html_text(symbol)}</b>\n\n"
        f"<b>Signal</b>\n"
        f"Symbol: <code>{_html_text(symbol)}</code>\n"
        f"Timeframe: <code>{_html_text(timeframe)}</code>\n"
        f"Direction: <b>{format_direction_label(direction)}</b>\n"
        f"Setup: {_html_text(setup_desc)}\n\n"
        f"<b>Model Confidence</b>\n"
        f"Entry confidence ({lot_str}): {_format_percent(probability)}\n"
        f"Accept threshold: {_format_percent(confidence_threshold)}\n\n"
        f"{osc_section}"
        f"<b>Execution</b>\n"
        f"Entry policy: {_html_text(format_entry_policy_status(timeframe))}\n"
        f"HTF priority: {_html_text(htf_priority_status)}\n"
        f"LTF rejection: {_html_text(rejection_status)}\n"
        f"Order status: {execution_status}\n\n"
        f"<b>Levels</b>\n"
        f"Entry ({lot_str}): {_format_price(setup.get('entry_price'))}\n"
        f"Stop Loss: {_format_price(setup.get('sl_price'))}\n"
        f"{tp_lines}\n"
        f"<b>Confluence</b>\n"
        f"{_format_confluence_lines(confluences)}"
        f"{_format_htf_match_lines(htf_matches)}\n\n"
        f"<i>{_html_text(_format_signal_source_note())}</i>"
    )


def send_recovery_alert_with_chart(
    message: str,
    *,
    timeframes_data: dict,
    timeframe: str,
    symbol: str,
    direction_name: str,
    strategy: str,
    setups: list,
    image_suffix: str,
) -> bool:
    """Send a recovery alert with the same chart snapshot flow as new entries."""
    image_filename = None
    try:
        tf_df = timeframes_data.get(timeframe)
        if tf_df is None:
            raise KeyError(f"timeframe data not found: {timeframe}")

        setup_index = "unknown"
        if setups:
            setup_index = setups[0].get("index", "unknown")
        image_filename = f"temp_alert_{timeframe}_{image_suffix}_{setup_index}.png"
        title = f"{symbol} {timeframe} - {direction_name} Recovery {get_strategy_display_name(strategy)} Confluence"
        plot_smc_chart(tf_df, title=title, active_setups=setups, output_filename=image_filename)
    except Exception as e:
        print(f"Failed to generate recovery chart image: {e}")
        image_filename = None

    try:
        return send_telegram_alert(message, image_filename)
    finally:
        if image_filename and os.path.exists(image_filename):
            try:
                os.remove(image_filename)
            except Exception:
                pass


def _market_entry_has_immediate_emergency_reversal(
    setup: dict,
    *,
    timeframe: str = None,
    timeframes_data: dict = None,
) -> bool:
    if not timeframe or not timeframes_data:
        return False

    df_tf = timeframes_data.get(timeframe)
    if df_tf is None:
        return False

    try:
        direction = int(setup.get("direction", 1))
    except (TypeError, ValueError):
        direction = 1

    h1_trend = _last_closed_trend(timeframes_data.get("H1")) if "H1" in timeframes_data else None
    h4_trend = _last_closed_trend(timeframes_data.get("H4")) if "H4" in timeframes_data else None
    entry_val = float(setup.get("entry_price", 0.0))
    return should_emergency_exit_on_reversal(df_tf, timeframe, direction, entry_val, entry_val, h1_trend, h4_trend)


def should_market_enter_setup(
    setup: dict,
    current_price: float,
    entry_buffer: float = 0.5,
    *,
    timeframe: str = None,
    timeframes_data: dict = None,
) -> bool:
    """Return True when price is inside the instant-entry zone after confirmed rejection."""
    if current_price is None or not setup.get("rejection_confirmed", False):
        return False

    if _market_entry_has_immediate_emergency_reversal(
        setup,
        timeframe=timeframe,
        timeframes_data=timeframes_data,
    ):
        setup["market_entry_blocked_reason"] = "immediate_emergency_reversal"
        return False

    direction = int(setup.get("direction", 1))
    entry_price = float(setup["entry_price"])
    sl_price = float(setup["sl_price"])
    current_price = float(current_price)

    lower_bound = entry_price - entry_buffer
    upper_bound = entry_price + entry_buffer
    return lower_bound <= current_price <= upper_bound


def should_place_pending_setup(
    setup: dict,
    *,
    timeframe: str = None,
    timeframes_data: dict = None,
) -> bool:
    """Block pending entries that the trade manager would immediately close as reversal."""
    if _market_entry_has_immediate_emergency_reversal(
        setup,
        timeframe=timeframe,
        timeframes_data=timeframes_data,
    ):
        setup["pending_entry_blocked_reason"] = "immediate_emergency_reversal"
        return False
    return True


def get_watch_zone_reversal_context(symbol: str, timeframe: str) -> dict:
    """Load fresh closed candles only when a WatchZone is about to enter market."""
    mt5_timeframes = {
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
    }
    context = {}
    for tf in dict.fromkeys((timeframe, "H1", "H4")):
        mt5_timeframe = mt5_timeframes.get(tf)
        if mt5_timeframe is None:
            continue
        try:
            frame = fetch_historical_data(symbol, mt5_timeframe, 80)
            if frame is not None and not frame.empty:
                context[tf] = apply_smc_detectors(frame, symbol=symbol, closed_only=True)
        except Exception as exc:
            print(f"[WatchZones] Could not refresh {tf} reversal guard: {exc}")
    return context


def refresh_watch_zone_rejection(symbol: str, setup: dict) -> tuple[bool, str]:
    """Confirm the selected WatchZone leg from fresh lower-timeframe candles."""
    checks = (
        ("M5", mt5.TIMEFRAME_M5, 15, 3),    # last 15 mins (3 candles)
        ("M1", mt5.TIMEFRAME_M1, 10, 5),    # last 10 mins (5 candles)
        ("M15", mt5.TIMEFRAME_M15, 15, 2),  # last 30 mins (2 candles)
    )
    for source, mt5_timeframe, candles, lookback in checks:
        if source == "M15" and setup.get("timeframe") == "M15":
            continue
        try:
            frame = fetch_historical_data(symbol, mt5_timeframe, candles)
            if frame is not None and not frame.empty and detect_rejection_at_level(
                frame,
                float(setup["entry_price"]),
                int(setup["direction"]),
                lookback=lookback,
                symbol=symbol,
            ):
                return True, source
        except Exception as exc:
            print(f"[WatchZones] Could not refresh {source} rejection guard: {exc}")
    return False, "None"


def choose_recovery_execution_mode(
    setup: dict,
    current_price: float,
    *,
    timeframe: str = None,
    timeframes_data: dict = None,
) -> str:
    """Choose market recovery when price has returned to a confirmed rejection zone."""
    return (
        "market"
        if should_market_enter_setup(
            setup,
            current_price,
            timeframe=timeframe,
            timeframes_data=timeframes_data,
        )
        else "pending"
    )


def should_promote_low_confidence_record(sig_data: dict, ticket_fields) -> bool:
    """Return True when a shadow-tracked setup can fall through to live execution."""
    if not sig_data or not sig_data.get("is_low_confidence", False):
        return False

    if any(sig_data.get(field) is not None for field in ticket_fields):
        return False

    outcome_fields = ("outcome_recorded", "outcome_a_recorded", "outcome_b_recorded")
    return not any(sig_data.get(field, False) for field in outcome_fields)


def choose_dual_market_entry_option(
    opt_a: dict,
    opt_b: dict,
    current_price: float,
    entry_buffer: float = 0.5,
    *,
    timeframe: str = None,
    timeframes_data: dict = None,
):
    """Choose which dual fib layer should become a market order, preserving 0.5 priority near entry."""
    if current_price is None:
        return None

    if _market_entry_has_immediate_emergency_reversal(
        opt_a,
        timeframe=timeframe,
        timeframes_data=timeframes_data,
    ):
        opt_a["market_entry_blocked_reason"] = "immediate_emergency_reversal"
        opt_b["market_entry_blocked_reason"] = "immediate_emergency_reversal"
        return None

    direction = int(opt_a.get("direction", 1))
    current_price = float(current_price)

    if direction == 1:
        if (
            opt_a.get("rejection_confirmed", False)
            and float(opt_b["entry_price"]) <= current_price <= float(opt_a["entry_price"]) + entry_buffer
        ):
            return "a"
        if (
            opt_b.get("rejection_confirmed", False)
            and float(opt_b["entry_price"]) - entry_buffer <= current_price < float(opt_b["entry_price"])
        ):
            return "b"
    else:
        if (
            opt_a.get("rejection_confirmed", False)
            and float(opt_a["entry_price"]) - entry_buffer <= current_price <= float(opt_b["entry_price"])
        ):
            return "a"
        if (
            opt_b.get("rejection_confirmed", False)
            and float(opt_b["entry_price"]) < current_price <= float(opt_b["entry_price"]) + entry_buffer
        ):
            return "b"

    return None


def choose_dual_recovery_execution_mode(
    opt_a: dict,
    opt_b: dict,
    current_price: float,
    option: str,
    *,
    timeframe: str = None,
    timeframes_data: dict = None,
) -> str:
    """Use the same dual-fib market priority during recovery as during first execution."""
    market_option = choose_dual_market_entry_option(
        opt_a,
        opt_b,
        current_price,
        timeframe=timeframe,
        timeframes_data=timeframes_data,
    )
    if market_option == option:
        return "market"
    if market_option in {"a", "b"}:
        return "skip"
    return "pending"


def drop_latest_forming_candle(df_tf: pd.DataFrame) -> pd.DataFrame:
    """Return only fully closed candles from an MT5 OHLC frame."""
    if df_tf is None or df_tf.empty:
        return df_tf
    closed_df = df_tf.iloc[:-1].copy()
    closed_df.attrs["closed_only"] = True
    return closed_df


def apply_smc_detectors(df_tf: pd.DataFrame, symbol: str, closed_only: bool = False) -> pd.DataFrame:
    """Run the full live SMC detector pipeline for one timeframe."""
    is_source_closed_only = bool(getattr(df_tf, "attrs", {}).get("closed_only", False))
    
    # In live scanning, if the source data is not already closed-only,
    # we mark the dataframe as having a running candle so the detectors don't detect new setups on it.
    if not is_source_closed_only:
        df_tf.attrs["has_running_candle"] = True

    df_tf = detect_swing_points(df_tf)
    df_tf = detect_structures(df_tf)
    df_tf = detect_fvg_and_ob(df_tf, symbol=symbol)
    df_tf = detect_snr_and_swapzones(df_tf)
    df_tf = detect_bpr(df_tf, symbol=symbol)
    df_tf = detect_indecision_candles(df_tf, symbol=symbol)
    df_tf = detect_supply_demand_zones(df_tf, symbol=symbol)
    
    if closed_only:
        # Drop the latest forming candle after detectors (mitigation check) have run on it
        df_tf = drop_latest_forming_candle(df_tf)
    elif is_source_closed_only:
        df_tf.attrs["closed_only"] = True
        
    return df_tf

def load_sent_signals() -> dict:
    """Load the registry of already alerted signals."""
    path = get_sent_signals_file()
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_sent_signals(sent_dict: dict):
    """Save the registry of alerted signals to disk."""
    path = get_sent_signals_file()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(sent_dict, f, indent=4)


def get_realtime_tick(symbol: str, ensure_connection: bool = True):
    """Read the current broker tick for lightweight reaction watching."""
    if ensure_connection:
        connect_mt5()
    broker_symbol = get_active_broker_symbol(symbol)
    return mt5.symbol_info_tick(broker_symbol)


def run_realtime_reaction_cycle(
    symbol: str,
    *,
    previous_tick=None,
    current_tick=None,
    entry_buffer: float = 0.5,
    min_reaction_move: float = 0.10,
) -> RealtimeReactionPassResult:
    """Run one lightweight tick-reaction pass without recalculating closed-candle SMC setups."""
    if current_tick is None:
        current_tick = get_realtime_tick(symbol)
    if previous_tick is None or current_tick is None:
        return RealtimeReactionPassResult(changed=False, executed_count=0, checked_count=0)

    sent_signals = load_sent_signals()
    result = run_realtime_reaction_pass(
        sent_signals,
        symbol=symbol,
        previous_tick=previous_tick,
        current_tick=current_tick,
        execute_market_order=execute_market_order_for_setup,
        entry_buffer=entry_buffer,
        min_reaction_move=min_reaction_move,
    )
    if result.changed:
        save_sent_signals(sent_signals)
        print(
            f"[Realtime Reaction] Executed {result.executed_count} market order(s) "
            f"from {result.checked_count} watched setup(s)."
        )

    magic = int(os.getenv("MT5_MAGIC_NUMBER", "202606"))
    manage_active_trades(symbol, magic, {})
    return result


def register_shadow_candidate(
    sig_key: str,
    symbol: str,
    timeframe: str,
    strategy: str,
    direction_name: str,
    accept_threshold: float,
    opt: dict = None,
    probability: float = None,
    opt_a: dict = None,
    probability_a: float = None,
    opt_b: dict = None,
    probability_b: float = None,
    shadow_signals_file: str = None,
    now: str = None,
    force: bool = False,
    filtered_reason: str = "below_accept_threshold",
) -> bool:
    """Store below-threshold candidates for virtual outcome tracking without executing them."""
    if shadow_signals_file is None:
        shadow_signals_file = get_shadow_signals_file()
    if opt is not None and not force:
        if not should_shadow_signal(probability, accept_threshold):
            return False
    elif opt is None and not force:
        probs = [p for p in (probability_a, probability_b) if p is not None]
        if not probs or not any(should_shadow_signal(p, accept_threshold) for p in probs):
            return False

    records = build_shadow_signal_records(
        signal_id=sig_key,
        symbol=symbol,
        timeframe=timeframe,
        strategy=strategy,
        direction_name=direction_name,
        accept_threshold=accept_threshold,
        opt=opt,
        probability=probability,
        opt_a=opt_a,
        probability_a=probability_a,
        opt_b=opt_b,
        probability_b=probability_b,
        now=now,
        filtered_reason=filtered_reason,
    )
    return upsert_shadow_signals(records, shadow_signals_file=shadow_signals_file)


def register_low_confidence_lead(
    lead: dict,
    sent_signals: dict,
    symbol: str,
    timeframe: str,
    strategy: str,
    direction_name: str,
    accept_threshold: float,
    shadow_signals_file: str = None,
    now: str = None,
) -> bool:
    """Register a below-threshold lead in the silent registry and shadow tracker."""
    if shadow_signals_file is None:
        shadow_signals_file = get_shadow_signals_file()
    if lead['is_dual']:
        opt_a = lead['opt_a']
        opt_b = lead['opt_b']
        prob_a = lead['prob_a']
        prob_b = lead['prob_b']

        setup_time_str = str(opt_a['time'])
        sig_key = f"{timeframe}_{strategy}_DUAL_{direction_name}_{opt_a['entry_price']:.3f}_{opt_b['entry_price']:.3f}_{setup_time_str.replace(' ', '_')}"

        opt_a['status'] = "FILTERED (Low Confidence)"
        opt_b['status'] = "FILTERED (Low Confidence)"

        changed = False
        if sig_key not in sent_signals:
            print(f"[Scanner Registry] Registering low confidence dual {timeframe} {strategy} at {opt_a['time']} for manual tracking.")
            sent_signals[sig_key] = {
                'time_sent': now or datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'timeframe': timeframe,
                'direction': direction_name,
                'type': strategy,
                'price_0.5': opt_a['entry_price'],
                'price_0.618': opt_b['entry_price'],
                'probability_0.5': prob_a,
                'probability_0.618': prob_b,
                'ticket_a': None,
                'ticket_b': None,
                'reentries_count': 0,
                'features_0.5': opt_a['features'],
                'features_0.618': opt_b['features'],
                'is_low_confidence': True,
            }
            changed = True

        shadow_changed = register_shadow_candidate(
            sig_key=sig_key,
            symbol=symbol,
            timeframe=timeframe,
            strategy=strategy,
            direction_name=direction_name,
            accept_threshold=accept_threshold,
            opt_a=opt_a,
            probability_a=prob_a,
            opt_b=opt_b,
            probability_b=prob_b,
            shadow_signals_file=shadow_signals_file,
            now=now,
        )
        return changed or shadow_changed

    opt = lead['opt']
    prob = lead['max_prob']

    setup_time_str = str(opt['time'])
    sig_key = f"{timeframe}_{strategy}_SINGLE_{direction_name}_{opt['entry_price']:.3f}_{setup_time_str.replace(' ', '_')}"

    opt['status'] = "FILTERED (Low Confidence)"

    changed = False
    if sig_key not in sent_signals:
        print(f"[Scanner Registry] Registering low confidence single {timeframe} {strategy} at {opt['time']} for manual tracking.")
        sent_signals[sig_key] = {
            'time_sent': now or datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'timeframe': timeframe,
            'direction': direction_name,
            'type': strategy,
            'price': opt['entry_price'],
            'probability': prob,
            'reentries_count': 0,
            'features': opt['features'],
            'is_low_confidence': True,
        }
        changed = True

    shadow_changed = register_shadow_candidate(
        sig_key=sig_key,
        symbol=symbol,
        timeframe=timeframe,
        strategy=strategy,
        direction_name=direction_name,
        accept_threshold=accept_threshold,
        opt=opt,
        probability=prob,
        shadow_signals_file=shadow_signals_file,
        now=now,
    )
    return changed or shadow_changed


def register_entry_gate_filtered_lead(
    lead: dict,
    sent_signals: dict,
    symbol: str,
    timeframe: str,
    strategy: str,
    direction_name: str,
    accept_threshold: float,
    filtered_reason: str = None,
    shadow_signals_file: str = None,
    now: str = None,
) -> bool:
    """Track below-threshold candidates rejected by entry quality checks."""
    if shadow_signals_file is None:
        shadow_signals_file = get_shadow_signals_file()
    if lead["is_dual"]:
        opt_a = lead["opt_a"]
        opt_b = lead["opt_b"]
        reason = filtered_reason or opt_a.get("filtered_reason", "entry_gate_filtered")
        sig_key = (
            f"{timeframe}_{strategy}_DUAL_{direction_name}_"
            f"{opt_a['entry_price']:.3f}_{opt_b['entry_price']:.3f}_"
            f"{str(opt_a['time']).replace(' ', '_')}"
        )
        return register_shadow_candidate(
            sig_key=sig_key,
            symbol=symbol,
            timeframe=timeframe,
            strategy=strategy,
            direction_name=direction_name,
            accept_threshold=accept_threshold,
            opt_a=opt_a,
            probability_a=lead["prob_a"],
            opt_b=opt_b,
            probability_b=lead["prob_b"],
            shadow_signals_file=shadow_signals_file,
            now=now,
            force=False,
            filtered_reason=reason,
        )

    opt = lead["opt"]
    reason = filtered_reason or opt.get("filtered_reason", "entry_gate_filtered")
    sig_key = (
        f"{timeframe}_{strategy}_SINGLE_{direction_name}_"
        f"{opt['entry_price']:.3f}_{str(opt['time']).replace(' ', '_')}"
    )
    return register_shadow_candidate(
        sig_key=sig_key,
        symbol=symbol,
        timeframe=timeframe,
        strategy=strategy,
        direction_name=direction_name,
        accept_threshold=accept_threshold,
        opt=opt,
        probability=lead["max_prob"],
        shadow_signals_file=shadow_signals_file,
        now=now,
        force=False,
        filtered_reason=reason,
    )


def register_entry_gate_filtered_option(
    sig_key: str,
    opt: dict,
    probability: float,
    *,
    symbol: str,
    timeframe: str,
    strategy: str,
    direction_name: str,
    accept_threshold: float,
    leg: str = None,
    shadow_signals_file: str = None,
    now: str = None,
) -> bool:
    """Track one rejected leg from an otherwise live-eligible setup."""
    if shadow_signals_file is None:
        shadow_signals_file = get_shadow_signals_file()
    signal_id = f"{sig_key}_{leg}" if leg else sig_key
    return register_shadow_candidate(
        sig_key=signal_id,
        symbol=symbol,
        timeframe=timeframe,
        strategy=strategy,
        direction_name=direction_name,
        accept_threshold=accept_threshold,
        opt=opt,
        probability=probability,
        shadow_signals_file=shadow_signals_file,
        now=now,
        force=False,
        filtered_reason=opt.get("filtered_reason", "entry_gate_filtered"),
    )


def process_existing_shadow_outcomes(
    timeframes_data: dict,
    shadow_signals_file: str = None,
    shadow_labeled_data_path: str = None,
    now: str = None,
    trigger_retrain: bool = True,
) -> dict:
    """Resolve previously tracked shadow signals using the latest fetched candles."""
    if shadow_signals_file is None:
        shadow_signals_file = get_shadow_signals_file()
    kwargs = {
        "shadow_signals_file": shadow_signals_file,
        "now": now,
    }
    if shadow_labeled_data_path is not None:
        kwargs["shadow_labeled_data_path"] = shadow_labeled_data_path
    result = process_shadow_signal_outcomes(timeframes_data, **kwargs)
    result["retrain_result"] = None

    labeled_rows = int(result.get("labeled_rows_appended", 0) or 0)
    if trigger_retrain and labeled_rows > 0:
        from src.inference import check_and_trigger_retraining

        result["retrain_result"] = check_and_trigger_retraining(labeled_rows)

    return result


def prune_invalid_pending_orders(symbol: str, magic: int, active_high_confidence_setups: list):
    """
    Cancel pending orders on the MT5 account that are no longer active or mitigated.
    Valid limit orders are allowed to wait for price to return to the entry zone.
    To prevent premature cancellations of valid pending orders during normal pullbacks:
    1. We check the setup age in the sent signals registry. If it is young (< 4 hours), we keep it.
    2. We check if the price has violated the stop loss level of the pending order. If violated, we prune it immediately.
    """
    import MetaTrader5 as mt5
    import os
    from datetime import datetime
    from src.execution import _orders_for_symbol_magic, get_active_broker_symbol, load_sent_signals
    
    broker_symbol = get_active_broker_symbol(symbol)
    orders = _orders_for_symbol_magic(broker_symbol, magic)
    if len(orders) == 0:
        return
        
    tick = mt5.symbol_info_tick(broker_symbol)
    if tick is None:
        return
    current_price = tick.ask if len(orders) > 0 else tick.bid
    
    try:
        sent_signals = load_sent_signals()
    except Exception:
        sent_signals = {}
        
    # We check if there is an active high-confidence setup with a matching direction and close entry price
    cancelled_tickets = []
    for o in orders:
        o_price = o.price_open
        o_type = o.type
        
        is_still_valid = False
        tf = None
        for s in active_high_confidence_setups:
            s_o_type = 2 if s['direction'] == 1 else 3
            if s_o_type == o_type:
                try:
                    raw_entry = float(s['entry_price'])
                    spread = max(0.0, float(tick.ask) - float(tick.bid))
                    broker_entry = float(
                        s.get(
                            "broker_entry_price",
                            raw_entry + spread if int(s['direction']) == 1 else raw_entry - spread,
                        )
                    )
                except (TypeError, ValueError):
                    raw_entry = None
                    broker_entry = None

                # Execution offsets pending entries by live spread; compare against
                # that broker price so valid orders are not immediately pruned.
                candidate_prices = [price for price in (raw_entry, broker_entry) if price is not None]
                if any(abs(price - o_price) < 0.15 for price in candidate_prices):
                    is_still_valid = True
                    tf = s['timeframe']
                    break
                    
        if not is_still_valid:
            # Check if this ticket is registered in sent_signals and is still "young"
            found_in_registry = False
            time_placed_str = None
            for sig_key, sig_data in sent_signals.items():
                if sig_data.get('ticket_a') == o.ticket or sig_data.get('ticket_b') == o.ticket or sig_data.get('ticket_id') == o.ticket:
                    found_in_registry = True
                    time_placed_str = sig_data.get('time_sent')
                    break
            
            is_young = False
            if found_in_registry and time_placed_str:
                try:
                    placed_time = datetime.strptime(time_placed_str, "%Y-%m-%d %H:%M:%S")
                    age_seconds = (datetime.now() - placed_time).total_seconds()
                    min_age_hours = float(os.getenv("MT5_PENDING_PRUNE_MIN_AGE_HOURS", "4.0"))
                    if age_seconds < min_age_hours * 3600:
                        is_young = True
                except Exception:
                    pass
            
            # Check if price has violated the SL of this pending order
            sl_violated = False
            if getattr(o, "sl", 0.0) > 0.0:
                o_sl = float(o.sl)
                # mt5.ORDER_TYPE_BUY_LIMIT = 2, mt5.ORDER_TYPE_SELL_LIMIT = 3
                if o_type == 2 and float(tick.bid) <= o_sl:
                    sl_violated = True
                elif o_type == 3 and float(tick.ask) >= o_sl:
                    sl_violated = True
            
            if is_young and not sl_violated:
                # Do not prune young pending orders; let them wait for retracement
                continue

            reason = "SL violated" if sl_violated else "structure mitigated/invalid (age expired)"
            print(f"[Risk Management] Cancelling zombie/invalid pending order #{o.ticket} ({reason}).")
            
            request = {
                "action": mt5.TRADE_ACTION_REMOVE,
                "order": o.ticket,
            }
            res = mt5.order_send(request)
            if res is not None and res.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"[Risk Management] Order #{o.ticket} successfully cancelled.")
                cancelled_tickets.append((o.ticket, reason))
                
    if cancelled_tickets:
        try:
            lines = [f"🧹 <b>[Risk Management] Cleaned up {len(cancelled_tickets)} zombie pending orders:</b>"]
            for ticket, reason in cancelled_tickets:
                lines.append(f"• Order #{ticket} ({reason})")
            send_telegram_alert("\n".join(lines))
        except Exception:
            pass

def is_good_fvg(df: pd.DataFrame, idx: int, setup: dict, symbol: str, timeframes_data: dict) -> tuple:
    """
    Applies high-quality SMC FVG filters to ensure only high-probability setups are taken.
    Returns: (is_good, reason)
    """
    # 1. HTF Trend Confluence (with D1 Trend) - Disabled per user request (Option 2 - Aggressive)
    # d1_df = timeframes_data.get('D1')
    # if d1_df is not None and not d1_df.empty:
    #     d1_trend = d1_df['Trend'].iloc[-1]
    #     if setup['direction'] != d1_trend:
    #         return False, f"Trend conflict: Setup is {'BULL' if setup['direction'] == 1 else 'BEAR'} but D1 Trend is {'BULL' if d1_trend == 1 else 'BEAR'}"
            
    # 2. Displacement Candle Volume (Buyer/Seller Pressure)
    # The candle at idx-1 is the middle candle that created the gap.
    if idx >= 1:
        vol = df['Volume'].iloc[idx-1]
        high = df['High'].iloc[idx-1]
        low = df['Low'].iloc[idx-1]
        close = df['Close'].iloc[idx-1]
        
        candle_range = high - low
        if candle_range > 0:
            if setup['direction'] == 1:  # Buy (Bullish) - We want Buyer Volume
                buyer_vol = vol * (close - low) / candle_range
                buyer_ratio = buyer_vol / vol
                if buyer_ratio < 0.5:
                    return False, f"Weak buyer volume pressure ({buyer_ratio:.2f} < 0.50)"
            else:  # Sell (Bearish) - We want Seller Volume
                seller_vol = vol * (high - close) / candle_range
                seller_ratio = seller_vol / vol
                if seller_ratio < 0.5:
                    return False, f"Weak seller volume pressure ({seller_ratio:.2f} < 0.50)"
                    
        # Calculate 20-period average volume
        # avg_vol = df['Volume'].rolling(window=20).mean().iloc[idx-1]
        # if pd.notna(avg_vol) and avg_vol > 0:
        #     if vol < 1.1 * avg_vol:
        #         return False, f"Low displacement volume ({vol:.0f} < 1.1x avg {avg_vol:.0f})"
        pass
                
    # 3. Displacement Candle Body Size
    if idx >= 1:
        high = df['High'].iloc[idx-1]
        low = df['Low'].iloc[idx-1]
        close = df['Close'].iloc[idx-1]
        open_val = df['Open'].iloc[idx-1]
        
        candle_range = high - low
        body_size = abs(close - open_val)
        if candle_range > 0:
            body_ratio = body_size / candle_range
            if body_ratio < 0.5:
                return False, f"Weak displacement candle (body/range {body_ratio:.2f} < 0.5)"
                
    # 4. FVG Width Constraints
    width = setup['fvg_width']
    atr = setup['atr_14']
    if atr > 0:
        rel_width = width / atr
        if rel_width < 0.25:
            return False, f"FVG width too narrow ({rel_width:.2f} < 0.25 ATR)"
        # Max width constraint removed per user request
        pass
            
    # Absolute width check for Gold (XAUUSD)
    symbol_upper = symbol.upper()
    if "XAUUSD" in symbol_upper or "GOLD" in symbol_upper:
        tf = setup.get('timeframe', 'H4')
        min_width = 1.0 if tf == 'M30' else 1.5
        if width < min_width:
            return False, f"FVG width too narrow ({width:.2f} USD < {min_width} USD)"
        # Max width constraint removed per user request
        pass
            
    return True, "Valid High-Quality FVG"

def run_scan(symbol: str, confidence_threshold: float):
    """Run a single scan cycle across all timeframes and send new signals to Telegram."""
    print(f"\n--- Starting Scan Cycle for {symbol} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
    registry_changed = False
    
    # 1. Try to connect to MT5 Exness terminal
    import MetaTrader5 as mt5
    if not connect_mt5():
        print("[Scanner Error] Failed to connect to MetaTrader 5 terminal. Skipping cycle.")
        return
        
    check_and_sync_active_account()
        
    # 1.5. Run feedback loop to process MT5 history outcomes and retrain model
    try:
        feedback_result = process_mt5_history_feedback(return_details=True)
        new_feedback_count = feedback_result.get('feedback_count', 0)
        retrain_result = feedback_result.get('retrain_result') or {}
        if new_feedback_count > 0:
            if retrain_result.get('retrained'):
                print(f"[Feedback Loop] Learned {new_feedback_count} new outcomes and retrained model.")
                send_telegram_alert(
                    f"🔄 <b>Bot AI mempelajari {new_feedback_count} hasil trade baru dan melakukan retraining otomatis.</b>"
                )
            elif retrain_result.get('status') == 'ERROR':
                print(f"[Feedback Loop] Learned {new_feedback_count} new outcomes, but retraining failed: {retrain_result.get('error')}")
                send_telegram_alert(
                    f"⚠️ <b>Bot AI mempelajari {new_feedback_count} hasil trade baru, tetapi retraining gagal:</b> "
                    f"<code>{retrain_result.get('error')}</code>"
                )
            else:
                threshold = retrain_result.get('threshold', '?')
                accumulated = retrain_result.get('new_trades_since_last_train', new_feedback_count)
                print(f"[Feedback Loop] Learned {new_feedback_count} new outcomes. Retraining deferred ({accumulated}/{threshold}).")
                send_telegram_alert(
                    f"🧠 <b>Bot AI mempelajari {new_feedback_count} hasil trade baru.</b>\n"
                    f"Retraining ditunda sampai akumulasi mencapai <b>{threshold}</b> trade."
                )
    except Exception as e:
        print(f"[Feedback Loop Error] {e}")
        
    timeframes_data = {}
    mt5_active = True
    
    try:
        # Fetch multi-timeframe data with expanded lookback bars
        print("Fetching multi-timeframe data from MT5...")
        timeframes_data['D1'] = fetch_historical_data(symbol, mt5.TIMEFRAME_D1, 100)
        timeframes_data['H4'] = fetch_historical_data(symbol, mt5.TIMEFRAME_H4, 250)
        timeframes_data['H1'] = fetch_historical_data(symbol, mt5.TIMEFRAME_H1, 300)
        timeframes_data['M30'] = fetch_historical_data(symbol, mt5.TIMEFRAME_M30, 400)
        timeframes_data['M15'] = fetch_historical_data(symbol, mt5.TIMEFRAME_M15, 500)
        timeframes_data['M5'] = fetch_historical_data(symbol, mt5.TIMEFRAME_M5, 500)
        timeframes_data['M1'] = fetch_historical_data(symbol, mt5.TIMEFRAME_M1, 500)
    except Exception as e:
        print(f"[Scanner Error] Error loading data from MT5: {e}")
        import MetaTrader5 as mt5
        mt5.shutdown()
        return

    d1_pivot_source = timeframes_data.get('D1')

    try:
        shadow_result = process_existing_shadow_outcomes(timeframes_data)
        if shadow_result.get("resolved_count", 0) or shadow_result.get("expired_count", 0):
            print(
                "[Shadow Tracker] Resolved "
                f"{shadow_result.get('resolved_count', 0)} shadow signals, "
                f"expired {shadow_result.get('expired_count', 0)}, "
                f"appended {shadow_result.get('labeled_rows_appended', 0)} labeled rows."
            )
    except Exception as e:
        print(f"[Shadow Tracker Error] {e}")

    # Draw daily pivots on MT5 charts
    from src.indicators.pivots import draw_pivots_on_mt5
    try:
        df_d1 = d1_pivot_source
        if df_d1 is not None and not df_d1.empty:
            draw_pivots_on_mt5(symbol, df_d1)
            print("[Pivots Visualizer] Drew/updated Daily Pivots on MT5 charts.")
    except Exception as e:
        print(f"[Pivots Visualizer Error] {e}")
    
    # 2. Run SMC detection algorithms on all timeframes
    for tf_name in timeframes_data:
        df_tf = timeframes_data[tf_name]
        df_tf = apply_smc_detectors(df_tf, symbol=symbol, closed_only=True)
         # Calculate ATR_14
        close_prev = df_tf['Close'].shift(1).fillna(df_tf['Open'])
        tr = np.maximum(
            df_tf['High'] - df_tf['Low'],
            np.maximum(
                np.abs(df_tf['High'] - close_prev),
                np.abs(df_tf['Low'] - close_prev)
            )
        )
        df_tf['ATR_14'] = tr.rolling(window=14, min_periods=1).mean()
        timeframes_data[tf_name] = df_tf
        
    # Pre-calculate trends for FLOOP Pro MTF/HTF
    from src.indicators.floop import calculate_atr, calculate_range_filter
    tf_trends = {}
    for tf_name in timeframes_data:
        df_tf = timeframes_data[tf_name]
        try:
            df_tf_copy = df_tf.copy()
            df_tf_copy['time'] = pd.to_datetime(df_tf_copy['time'])
            df_tf_copy.set_index('time', inplace=True)
            
            atr_floop = calculate_atr(df_tf_copy, 14)
            _, trend_floop, _ = calculate_range_filter(df_tf_copy['Close'], atr_floop, sensitivity=6, atr_multiplier=0.8)
            tf_trends[tf_name] = pd.Series(trend_floop, index=df_tf_copy.index)
        except Exception as e:
            print(f"Error calculating RF trend for TF {tf_name}: {e}")
            tf_trends[tf_name] = None
            
    # Pre-calculate KNN, Volume Profile, and Oscillator (RSI8+Stoch) data per timeframe
    print("Pre-calculating KNN, Volume Profile, and Oscillator features for live scanner...")
    tf_knn_data = {}
    tf_vp_data = {}
    tf_osc_data = {}   # OscillatorContext per timeframe
    for tf_name, df_tf in timeframes_data.items():
        # KNN
        try:
            pc1, pc2, pc3, pc4, target_clean = run_knn_classifier(
                df_tf,
                atr_period=10, factor=2.0,
                k_neighbors=10, sampling_window_size=1000, momentum_window=10,
                normalizing_window_size=1000,
                lazy=True
            )
            t_last = len(df_tf) - 1
            knn_up, knn_down = calculate_knn_probability_at_bar(
                t_last, pc1.values, pc2.values, pc3.values, pc4.values, target_clean.values,
                k=10, sampling_window=1000, stride=10
            )
            tf_knn_data[tf_name] = (knn_up, knn_down)
        except Exception as e:
            print(f"Error computing live KNN for TF {tf_name}: {e}")
            tf_knn_data[tf_name] = (0.0, 0.0)

        # Volume profile
        try:
            clusters_data = calculate_volume_clusters(
                df_tf, lookback=200, k=5, iterations=20, rows=20
            )
            tf_vp_data[tf_name] = clusters_data
        except Exception as e:
            print(f"Error computing live Volume Clusters for {tf_name}: {e}")
            tf_vp_data[tf_name] = {}

        # Oscillator (RSI8 + Stochastic — compute per-TF for multi-TF lookup table)
        try:
            tf_osc_data[tf_name] = build_oscillator_context(df_tf)
        except Exception as e:
            print(f"Error computing Oscillator for TF {tf_name}: {e}")
            tf_osc_data[tf_name] = OscillatorContext()
            
    # 3. Extract active HTF FVGs for hierarchy prioritization
    active_fvgs_by_tf = {}
    for tf_name in ['M15', 'M30', 'H1', 'H4', 'D1']:
        active_fvgs_by_tf[tf_name] = extract_active_htf_fvgs(timeframes_data[tf_name])
        
    def get_strategy_name(option_name: str) -> str:
        if "OB" in option_name:
            return "OB"
        elif "BPR" in option_name:
            return "BPR"
        elif "IC" in option_name:
            return "IC"
        elif "Swap" in option_name:
            return "Swapzone"
        elif "Breaker" in option_name:
            return "Breaker"
        elif "SND" in option_name:
            return "SND"
        elif "Pivot" in option_name:
            return "Pivot"
        else:
            return "FVG"
 
    # Extract setups (FVG, OB, BPR, IC, SND, Pivot)
    all_setups = []
    for tf_name in ['D1', 'H4', 'H1', 'M30', 'M15']:
        tf_setups = get_active_setups(timeframes_data[tf_name], symbol=symbol, tf_trends=tf_trends, df_d1=d1_pivot_source)
        for s in tf_setups:
            s['timeframe'] = tf_name
            s['strategy'] = get_strategy_name(s['option_name'])
            s['symbol'] = symbol
            if s['strategy'] in ['FVG', 'OB', 'BPR', 'IC', 'SND', 'Pivot', 'Swapzone', 'Breaker']:
                all_setups.append(s)
            
    # 4. Multi-Timeframe Alignment, Suppression, and Rejection Checks
    tf_weights = {'M15': 1, 'M30': 2, 'H1': 3, 'H4': 4, 'D1': 5}
    tf_minutes_map = {'M15': 15, 'M30': 30, 'H1': 60, 'H4': 240, 'D1': 1440}
    
    for setup in all_setups:
        setup['htf_prioritized'] = False
        setup['matching_htf_fvgs'] = []
        setup['suppressed'] = False
        setup['htf_conflict_reason'] = ""
        setup_tf = setup['timeframe']
        
        for htf_name in ['M15', 'M30', 'H1', 'H4', 'D1']:
            if tf_weights[htf_name] > tf_weights[setup_tf]:
                # HTF Prioritization (same direction)
                for htf_fvg in active_fvgs_by_tf[htf_name]:
                    is_same = (setup['direction'] == 1 and htf_fvg['type'] == 'BULLISH') or \
                              (setup['direction'] == -1 and htf_fvg['type'] == 'BEARISH')
                    if is_same:
                        entry = setup['entry_price']
                        if entry >= htf_fvg['bottom'] and entry <= htf_fvg['top']:
                            setup['htf_prioritized'] = True
                            fvg_info = htf_fvg.copy()
                            fvg_info['timeframe'] = htf_name
                            setup['matching_htf_fvgs'].append(fvg_info)
                            
                # Conflict Suppression (opposite direction) - only if entry is inside the opposite HTF FVG
                for htf_fvg in active_fvgs_by_tf[htf_name]:
                    is_opp = (setup['direction'] == 1 and htf_fvg['type'] == 'BEARISH') or \
                              (setup['direction'] == -1 and htf_fvg['type'] == 'BULLISH')
                    if is_opp:
                        entry = setup['entry_price']
                        if entry >= htf_fvg['bottom'] and entry <= htf_fvg['top']:
                            setup['suppressed'] = True
                            setup['htf_conflict_reason'] = f"Entry inside opposite active {htf_name} FVG"
                            break
                        
        # Check Rejection on lower timeframes (M15, M5, M1) for the setup
        rej_confirmed = False
        rej_tf_source = "None"
        
        # 1. Check on M5 (lookback 30 candles)
        m5_df = timeframes_data.get('M5')
        if m5_df is not None and not m5_df.empty:
            if detect_rejection_at_level(m5_df, setup['entry_price'], setup['direction'], lookback=30):
                rej_confirmed = True
                rej_tf_source = "M5"
                
        # 2. Check on M1 (lookback 90 candles) if not already confirmed on M5
        if not rej_confirmed:
            m1_df = timeframes_data.get('M1')
            if m1_df is not None and not m1_df.empty:
                if detect_rejection_at_level(m1_df, setup['entry_price'], setup['direction'], lookback=90):
                    rej_confirmed = True
                    rej_tf_source = "M1"
                    
        # 3. Fallback to M15 (lookback 15 candles) if not confirmed on M5/M1 and setup is on timeframe higher than M15
        if not rej_confirmed and setup_tf != 'M15':
            m15_df = timeframes_data.get('M15')
            if m15_df is not None and not m15_df.empty:
                if detect_rejection_at_level(m15_df, setup['entry_price'], setup['direction'], lookback=15):
                    rej_confirmed = True
                    rej_tf_source = "M15"
                    
        setup['rejection_confirmed'] = rej_confirmed
        setup['rejection_source'] = rej_tf_source

    # 4.3. Calculate ML Probability for all setups
    tf_minutes_map = {'M1': 1, 'M5': 5, 'M15': 15, 'M30': 30, 'H1': 60, 'H4': 240, 'D1': 1440}
    for setup in all_setups:
        if setup.get('suppressed', False):
            setup['probability'] = 0.0
            continue
            
        tf = setup['timeframe']
        knn_up_tf, knn_down_tf = tf_knn_data.get(tf, (0.0, 0.0))
        knn_prob_sig = knn_up_tf if setup['direction'] == 1 else knn_down_tf
        knn_prob_opp = knn_down_tf if setup['direction'] == 1 else knn_up_tf
        
        clusters_data_tf = tf_vp_data.get(tf, {})
        dist_entry_to_poc = 0.0
        dist_entry_to_nearest_poc = 0.0
        if clusters_data_tf and 'current_poc' in clusters_data_tf:
            curr_poc = clusters_data_tf['current_poc']
            entry = setup['entry_price']
            dist_entry_to_poc = (entry - curr_poc) / curr_poc if curr_poc > 0 else 0.0
            
            pocs = clusters_data_tf.get('pocs', [])
            if pocs:
                dist_entry_to_nearest_poc = min(abs(entry - poc) for poc in pocs) / entry
                
        features = {
            'timeframe': tf_minutes_map[setup['timeframe']],
            'hour': setup['hour'],
            'day_of_week': setup['day_of_week'],
            'setup_type': setup['setup_type'],
            'direction': setup['direction'],
            'entry_price': setup['entry_price'],
            'sl_price': setup['sl_price'],
            'tp_price': setup['tp_price'],
            'risk_pips': setup['risk_pips'],
            'atr_14': setup['atr_14'],
            'trend': setup['trend'],
            'relative_risk': setup['relative_risk'],
            'killzone': setup['killzone'],
            'fvg_width': setup['fvg_width'],
            'relative_fvg_width': setup['relative_fvg_width'],
            'near_psychological_level': setup['near_psychological_level'],
            'knn_prob_sig': knn_prob_sig,
            'knn_prob_opp': knn_prob_opp,
            'dist_entry_to_poc': dist_entry_to_poc,
            'dist_entry_to_nearest_poc': dist_entry_to_nearest_poc,
            'dist_entry_to_pp': setup.get('dist_entry_to_pp', 0.0),
            'dist_entry_to_nearest_pivot': setup.get('dist_entry_to_nearest_pivot', 0.0),
            'floop_signal': setup['floop_signal'],
            'floop_strength': setup['floop_strength'],
            'floop_trend': setup.get('floop_trend', 0),
            'floop_trend_aligned': 1 if setup.get('floop_trend', 0) == setup['direction'] else 0
        }

        # Attach 3-layer (HTF / Signal TF / LTF) oscillator context to the setup.
        # NOT fed into ML model (avoids feature mismatch); used for confidence gate + Telegram display.
        setup['oscillator'] = build_multi_tf_oscillator(tf, tf_osc_data)
        
        try:
            prob = predict_setup_probability(features)
        except Exception as e:
            print(f"Error predicting probability for {tf} {setup['strategy']}: {e}")
            prob = 0.5
            
        setup['probability'] = prob
        setup['features'] = features

    # 4.5. SMC Setup Confluence Clustering & Deduplication
    execution_groups = {}  # key: (timeframe, index, strategy), value: list of setups
    other_setups = []
    
    for setup in all_setups:
        if setup.get('suppressed', False):
            continue
            
        tf = setup['timeframe']
        strat = setup['strategy']
        if is_live_entry_timeframe(tf):
            key = (tf, setup['index'], strat)
            if key not in execution_groups:
                execution_groups[key] = []
            execution_groups[key].append(setup)
        else:
            other_setups.append(setup)
            
    # Build candidates (combining dual options)
    candidates = []
    for key, setups_list in execution_groups.items():
        tf, idx, strat = key
        
        # Check if it's a dual strategy with both options present
        opt_a = None
        opt_b = None
        for s in setups_list:
            if "Option A" in s['option_name'] or "Midpoint" in s['option_name'] or "0.5" in s['option_name']:
                opt_a = s
            elif "Option B" in s['option_name'] or "Golden Pocket" in s['option_name'] or "0.618" in s['option_name']:
                opt_b = s
                
        if opt_a is not None and opt_b is not None:
            max_prob = max(opt_a['probability'], opt_b['probability'])
            candidates.append({
                'id': f"DUAL_{tf}_{strat}_{idx}",
                'timeframe': tf,
                'strategy': strat,
                'direction': opt_a['direction'],
                'opt_a': opt_a,
                'opt_b': opt_b,
                'prob_a': opt_a['probability'],
                'prob_b': opt_b['probability'],
                'max_prob': max_prob,
                'entry_price': opt_a['entry_price'],  # Anchoring on Midpoint entry
                'is_dual': True
            })
        else:
            for s in setups_list:
                candidates.append({
                    'id': f"SINGLE_{tf}_{strat}_{idx}_{s['entry_price']:.3f}",
                    'timeframe': tf,
                    'strategy': strat,
                    'direction': s['direction'],
                    'opt': s,
                    'max_prob': s['probability'],
                    'entry_price': s['entry_price'],
                    'is_dual': False
                })
                
    # Sort ALL candidates by max_prob descending
    candidates.sort(key=lambda x: -x['max_prob'])
    
    clusters = []
    processed_ids = set()
    
    for c in candidates:
        if c['id'] in processed_ids:
            continue
            
        cluster = {
            'lead': c,
            'members': [c]
        }
        processed_ids.add(c['id'])
        
        # Look for other candidates in the same price zone
        cluster_proximity = _read_float_env("MT5_CLUSTER_PROXIMITY_USD", 1.5)
        for other_c in candidates:
            if other_c['id'] in processed_ids:
                continue

            if other_c['direction'] == c['direction']:
                if abs(other_c['entry_price'] - c['entry_price']) <= cluster_proximity:
                    cluster['members'].append(other_c)
                    processed_ids.add(other_c['id'])
                    
        clusters.append(cluster)
        
    # Mark non-lead members as suppressed
    for cluster in clusters:
        lead = cluster['lead']
        for member in cluster['members']:
            if member['id'] == lead['id']:
                continue
            # Mark its internal setups as suppressed
            if member['is_dual']:
                member['opt_a']['suppressed'] = True
                member['opt_a']['htf_conflict_reason'] = f"Clustered into Lead Setup {lead['id']}"
                member['opt_b']['suppressed'] = True
                member['opt_b']['htf_conflict_reason'] = f"Clustered into Lead Setup {lead['id']}"
            else:
                member['opt']['suppressed'] = True
                member['opt']['htf_conflict_reason'] = f"Clustered into Lead Setup {lead['id']}"
                
    # 5. Model Inference, Clustering Notification & Dispatch
    sent_signals = load_sent_signals()
    signals_sent_this_cycle = 0
    active_high_confidence = []
    registry_changed = False
    
    for cluster in clusters:
        lead = cluster['lead']
        tf = lead['timeframe']
        strat = lead['strategy']
        dir_name = "BULL" if lead['direction'] == 1 else "BEAR"
        
        # Build confluences/reasons list for this cluster
        reasons = []
        
        # 1. Structure & ML Probs
        for member in cluster['members']:
            m_tf = member['timeframe']
            m_strat = member['strategy']
            m_prob = member['max_prob']
            
            strat_desc = get_strategy_display_name(m_strat)
            reasons.append(f"{strat_desc} {m_tf} (Model confidence: {m_prob:.1%})")
            
            # Check rejection source of the member
            if member['is_dual']:
                rej_src_a = member['opt_a'].get('rejection_source', 'None')
                rej_src_b = member['opt_b'].get('rejection_source', 'None')
                if rej_src_a != 'None':
                    reasons.append(f"Wick Rejection touch confirmed on {rej_src_a}")
                if rej_src_b != 'None':
                    reasons.append(f"Wick Rejection touch confirmed on {rej_src_b}")
            else:
                rej_src = member['opt'].get('rejection_source', 'None')
                if rej_src != 'None':
                    reasons.append(f"Wick Rejection touch confirmed on {rej_src}")
                    
        # 2. HTF Priority, Psych levels, FLOOP, Volume POC, and Oscillator (from Lead candidate)
        lead_opt = lead['opt_a'] if lead['is_dual'] else lead['opt']

        if lead_opt.get('htf_prioritized', False):
            reasons.append("Aligned inside active Higher Timeframe (HTF) structure")

        if lead_opt.get('near_psychological_level', 0) == 1:
            reasons.append("Entry zone near Psychological Round Level (ends in 0 or 5)")

        # FLOOP Pro — trend alignment
        floop_trend = lead_opt.get('floop_trend', 0)
        floop_signal = lead_opt.get('floop_signal', 0)
        floop_strength = lead_opt.get('floop_strength', 0)
        if floop_trend == lead_opt['direction']:
            strength_desc = "strong" if floop_strength >= 10 else ("moderate" if floop_strength >= 6 else "mild")
            reasons.append(f"Supported by FLOOP Pro Trend Filter ({strength_desc} strength={int(floop_strength)})")
        elif floop_signal == lead_opt['direction']:
            reasons.append("FLOOP Pro fired a directional signal this bar")

        # Volume Profile — POC proximity
        dist_poc = lead_opt.get('dist_entry_to_poc', 0.0)
        dist_nearest_poc = lead_opt.get('dist_entry_to_nearest_poc', 0.0)
        if abs(dist_poc) <= 0.005 and dist_poc != 0.0:
            reasons.append("Zone overlaps with high-volume POC cluster (current cluster)")
        elif abs(dist_nearest_poc) <= 0.003 and dist_nearest_poc != 0.0:
            reasons.append("Entry zone near high-volume POC cluster (cross-cluster)")

        # KNN Classifier signal alignment
        knn_sig = lead_opt.get('features', {}).get('knn_prob_sig', 0.0)
        knn_opp = lead_opt.get('features', {}).get('knn_prob_opp', 0.0)
        if knn_sig >= 0.60:
            reasons.append(f"KNN Classifier bullish alignment: {knn_sig:.1%} directional probability")
        elif knn_sig >= 0.50:
            reasons.append(f"KNN Classifier mild directional bias: {knn_sig:.1%}")
        if knn_opp >= 0.60:
            reasons.append(f"KNN Classifier opposite-direction signal detected ({knn_opp:.1%}) — contra caution")

        # Oscillator Zone — multi-TF RSI8 + Stoch (HTF / Signal TF / LTF)
        # Patokan reference only — NOT hard entry rule. Used to boost/penalise confidence.
        mtf_osc = lead_opt.get('oscillator')
        if isinstance(mtf_osc, MultiTFOscillatorContext):
            direction = lead_opt['direction']
            confluent_delta, conf_score = evaluate_multi_tf_osc_delta(mtf_osc, direction)
            layers_total = sum(1 for x in [mtf_osc.htf, mtf_osc.signal, mtf_osc.ltf] if x is not None)

            def _layer_desc(osc, tf_name):
                if osc is None:
                    return None
                label = osc.signal_label or "WAIT"
                rsi_s = f"{osc.rsi_8:.1f}" if osc.rsi_8 is not None else "n/a"
                stk_s = f"{osc.stoch_k:.1f}" if osc.stoch_k is not None else "n/a"
                aligns = (direction == 1 and label in ("BUY", "REBUY")) or (direction == -1 and label in ("SELL", "RESELL"))
                tag = " [ALIGN]" if aligns else (" [OPPOSE]" if label not in ("WAIT",) else "")
                return f"{tf_name}: RSI8={rsi_s} %K={stk_s} [{label}]{tag}"

            parts = [p for p in [
                _layer_desc(mtf_osc.htf,    f"HTF({mtf_osc.htf_tf})"),
                _layer_desc(mtf_osc.signal, f"Sig({mtf_osc.signal_tf})"),
                _layer_desc(mtf_osc.ltf,    f"LTF({mtf_osc.ltf_tf})"),
            ] if p is not None]

            pct = int(round(confluent_delta * 100))
            sign = "+" if pct > 0 else ""
            score_str = f"{conf_score}/{layers_total} layers align, net {sign}{pct}% conf"

            if confluent_delta > 0:
                reasons.append(f"Oscillator MTF ({score_str}) — {' | '.join(parts)}")
            elif confluent_delta < 0:
                reasons.append(f"Oscillator MTF caution ({score_str}) — {' | '.join(parts)}")
            else:
                reasons.append(f"Oscillator MTF neutral ({score_str}) — {' | '.join(parts)}")
        elif mtf_osc is not None and hasattr(mtf_osc, 'signal_label'):
            # Fallback: single-TF OscillatorContext
            label = mtf_osc.signal_label
            direction = lead_opt['direction']
            rsi_str = f"{mtf_osc.rsi_8:.1f}" if mtf_osc.rsi_8 is not None else "n/a"
            stk_str = f"{mtf_osc.stoch_k:.1f}" if mtf_osc.stoch_k is not None else "n/a"
            osc_base = f"Oscillator RSI8={rsi_str} Stoch%K={stk_str} [{label}]"
            if (direction == 1 and label in ("BUY", "REBUY")) or (direction == -1 and label in ("SELL", "RESELL")):
                reasons.append(f"{osc_base} — zone ALIGNS with trade direction (confidence +)")
            elif label == "WAIT":
                reasons.append(f"{osc_base} — oscillator in neutral zone")
            else:
                reasons.append(f"{osc_base} — zone OPPOSES trade direction (confidence -)") 
            
        # Filter duplicates
        unique_reasons = []
        seen_reasons = set()
        for r in reasons:
            if r not in seen_reasons:
                seen_reasons.add(r)
                unique_reasons.append(r)
                
        is_high_conf = lead['max_prob'] >= confidence_threshold
        
        if is_high_conf:
            strategy_allowed, strategy_reason = should_allow_live_strategy(
                strat,
                lead_opt,
                probability=lead['max_prob'],
                timeframe=tf,
                entry_type="Standard Limit",
            )
            if not strategy_allowed:
                lead_opt['filtered_reason'] = strategy_reason
                print(f"[Live Strategy Policy] {tf} {strat} at {lead_opt.get('time')} skipped: {strategy_reason}")
                if register_entry_gate_filtered_lead(
                    lead=lead,
                    sent_signals=sent_signals,
                    symbol=symbol,
                    timeframe=tf,
                    strategy=strat,
                    direction_name=dir_name,
                    accept_threshold=confidence_threshold,
                    filtered_reason=strategy_reason,
                ):
                    registry_changed = True
                continue

            if lead['is_dual']:
                opt_a = lead['opt_a']
                opt_b = lead['opt_b']
                prob_a = lead['prob_a']
                prob_b = lead['prob_b']
                
                # 1. Apply FVG Quality Filters (only for FVG strategy)
                if strat == 'FVG':
                    is_good, quality_reason = is_good_fvg(timeframes_data[tf], opt_a['index'], opt_a, symbol, timeframes_data)
                    if not is_good:
                        print(f"[Quality Filter] {tf} FVG at index {opt_a['index']} rejected: {quality_reason}")
                        continue
                        
                setup_time_str = str(opt_a['time'])
                sig_key = f"{tf}_{strat}_DUAL_{dir_name}_{opt_a['entry_price']:.3f}_{opt_b['entry_price']:.3f}_{setup_time_str.replace(' ', '_')}"
                
                opt_a['status'] = "HIGH CONFIDENCE SIGNAL"
                opt_b['status'] = "HIGH CONFIDENCE SIGNAL"
                
                allow_entry = True
                reentries_count = 0

                if sig_key in sent_signals and should_promote_low_confidence_record(
                    sent_signals[sig_key],
                    ("ticket_a", "ticket_b"),
                ):
                    promoted_record = sent_signals.pop(sig_key)
                    reentries_count = promoted_record.get('reentries_count', 0)
                    print(f"[Scanner Registry] Promoting low confidence dual {tf} {strat} at {opt_a['time']} to live execution.")
                
                if sig_key in sent_signals:
                    sig_data = sent_signals[sig_key]
                    reentries_count = sig_data.get('reentries_count', 0)
                    
                    execute_enabled = os.getenv("MT5_EXECUTE_TRADES", "False").strip().lower() == "true"
                    if execute_enabled:
                        magic = int(os.getenv("MT5_MAGIC_NUMBER", "202606"))
                        from src.execution import get_active_broker_symbol
                        broker_symbol = get_active_broker_symbol(symbol)
                        tick = mt5.symbol_info_tick(broker_symbol)
                        current_price = None
                        if tick is not None:
                            current_price = tick.ask if opt_a['direction'] == 1 else tick.bid
                        
                        def is_ticket_active(t):
                            if t is None:
                                return False
                            orders_act = mt5.orders_get(ticket=t)
                            if orders_act and len(orders_act) > 0:
                                return True
                            positions_act = mt5.positions_get(ticket=t)
                            if positions_act and len(positions_act) > 0:
                                return True
                            return False
                            
                        def has_history_deals(t):
                            if t is None:
                                return False
                            deals = mt5.history_deals_get(position=t)
                            return deals is not None and len(deals) > 0
                            
                        ticket_a = sig_data.get('ticket_a')
                        ticket_b = sig_data.get('ticket_b')
                        recovery_allowed, recovery_reason = enforce_recovery_strategy_policy(
                            sig_data,
                            strategy=strat,
                            setup=opt_a,
                            probability=lead['max_prob'],
                            timeframe=tf,
                            outcome_keys=('outcome_a_recorded', 'outcome_b_recorded', 'outcome_recorded'),
                            message_keys=('watch_last_execution_message_0.5', 'watch_last_execution_message_0.618'),
                        )
                        if not recovery_allowed:
                            print(f"[Recovery Engine] {tf} {strat} recovery blocked by live policy: {recovery_reason}")
                            registry_changed = True
                            continue
                        
                        ignore_outcome_rec = (ticket_a is None and ticket_b is None)
                        
                        retry_watch_a = should_retry_unfilled_watch_record(
                            sig_data,
                            ("ticket_a",),
                            ("outcome_a_recorded", "outcome_recorded"),
                        )
                        recover_inactive_a = (
                            ticket_a is not None
                            and not is_ticket_active(ticket_a)
                            and not has_history_deals(ticket_a)
                            and not sig_data.get('outcome_a_recorded', False)
                            and (ignore_outcome_rec or not sig_data.get('outcome_recorded', False))
                        )
                        try:
                            max_retries = int(os.getenv("MT5_RECOVERY_MAX_RETRIES", "3"))
                        except (TypeError, ValueError):
                            max_retries = 3
                        retries_a = int(sig_data.get('execution_retries_0.5', 0))
                        failed_execution_a = (
                            ticket_a is None
                            and not sig_data.get('outcome_a_recorded', False)
                            and retries_a < max_retries
                            and is_cooldown_expired(sig_data.get('last_execution_attempt_0.5'), 60)
                        )
                        
                        if recover_inactive_a or retry_watch_a or failed_execution_a:
                            if retry_watch_a:
                                print(f"[Price Watch] Option A (0.5) {tf} {strat} at {opt_a['time']} is near enough to retry.")
                            elif failed_execution_a:
                                print(f"[Recovery Engine] Option A (0.5) was not successfully placed. Retrying execution...")
                            else:
                                print(f"[Recovery Engine] Option A (0.5) Ticket #{ticket_a} is inactive. Re-placing...")
                            gate_a = evaluate_live_entry_gate(
                                opt_a,
                                strategy=strat,
                                probability=prob_a,
                                accept_threshold=confidence_threshold,
                                symbol=symbol,
                                timeframe=tf,
                                timeframes_data=timeframes_data,
                            )
                            if not gate_a.allowed:
                                opt_a['filtered_reason'] = gate_a.filtered_reason
                                if register_entry_gate_filtered_option(
                                    sig_key,
                                    opt_a,
                                    prob_a,
                                    symbol=symbol,
                                    timeframe=tf,
                                    strategy=strat,
                                    direction_name=dir_name,
                                    accept_threshold=confidence_threshold,
                                    leg="0.5",
                                ):
                                    registry_changed = True
                                print(f"[Entry Gate] Recovery Option A skipped: {gate_a.reason}")
                                new_ticket_a, exec_msg_a = None, f"Skipped ({gate_a.reason})"
                                continue
                            recovery_mode_a = choose_dual_recovery_execution_mode(
                                opt_a,
                                opt_b,
                                current_price,
                                option="a",
                                timeframe=tf,
                                timeframes_data=timeframes_data,
                            )
                            if recovery_mode_a == "market":
                                new_ticket_a, exec_msg_a = execute_market_order_for_setup(opt_a, symbol)
                            elif recovery_mode_a == "skip":
                                new_ticket_a, exec_msg_a = None, "Skipped (Option B market recovery active)"
                            elif not should_place_pending_setup(
                                opt_a,
                                timeframe=tf,
                                timeframes_data=timeframes_data,
                            ):
                                new_ticket_a, exec_msg_a = None, "Skipped (immediate emergency reversal)"
                            else:
                                new_ticket_a, exec_msg_a = execute_trade_for_setup(opt_a, symbol)
                            
                            sig_data['last_execution_attempt_0.5'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            registry_changed = True
                            
                            if new_ticket_a is not None:
                                sig_data['ticket_a'] = new_ticket_a
                                sig_data['outcome_a_recorded'] = False
                                sig_data['outcome_recorded'] = False
                                sig_data['watch_status'] = "ticket_placed"
                                sig_data['watch_last_execution_message_0.5'] = exec_msg_a
                                sig_data['execution_retries_0.5'] = 0
                                try:
                                    recovery_title = "Market Order Recovery Executed" if recovery_mode_a == "market" else "Pending Order Re-placed"
                                    recovery_msg_a = (
                                        f"🔄 <b>[Order Recovery] {recovery_title}</b> 🔄\n\n"
                                        f"Setup 0.5 yang sebelumnya inactive telah dieksekusi ulang.\n"
                                        f"• <b>Price:</b> {opt_a['entry_price']:.3f}\n"
                                        f"• <b>Execution:</b> {exec_msg_a}\n"
                                        f"• <b>New Ticket:</b> #{new_ticket_a}"
                                    )
                                    send_recovery_alert_with_chart(
                                        recovery_msg_a,
                                        timeframes_data=timeframes_data,
                                        timeframe=tf,
                                        symbol=symbol,
                                        direction_name=dir_name,
                                        strategy=strat,
                                        setups=[opt_a, opt_b],
                                        image_suffix="recovery_dual_a",
                                    )
                                except Exception:
                                    pass
                            else:
                                failure_action = record_recovery_failure(
                                    sig_data,
                                    exec_msg_a,
                                    retries_a,
                                    max_retries,
                                    message_key='watch_last_execution_message_0.5',
                                    retries_key='execution_retries_0.5',
                                    outcome_key='outcome_a_recorded',
                                )
                                if failure_action == "price_watch":
                                    print(f"[Recovery Engine] Option A (0.5) waiting for price: {exec_msg_a}")
                                elif failure_action == "deferred":
                                    print(f"[Recovery Engine] Option A (0.5) execution deferred: {exec_msg_a}")
                                elif failure_action == "blocked":
                                    print(f"[Recovery Engine] Option A (0.5) recovery stopped: {exec_msg_a}")
                                else:
                                    print(f"[Recovery Engine] Option A (0.5) placement failed: {exec_msg_a}")
                                if failure_action == "retry" and retries_a + 1 >= max_retries:
                                    print(f"[Recovery Engine] Option A (0.5) reached max retry limit ({max_retries}) for {symbol}. Disabling further retries.")
                                    
                        retry_watch_b = should_retry_unfilled_watch_record(
                            sig_data,
                            ("ticket_b",),
                            ("outcome_b_recorded", "outcome_recorded"),
                        )
                        recover_inactive_b = (
                            ticket_b is not None
                            and not is_ticket_active(ticket_b)
                            and not has_history_deals(ticket_b)
                            and not sig_data.get('outcome_b_recorded', False)
                            and (ignore_outcome_rec or not sig_data.get('outcome_recorded', False))
                        )
                        retries_b = int(sig_data.get('execution_retries_0.618', 0))
                        failed_execution_b = (
                            ticket_b is None
                            and not sig_data.get('outcome_b_recorded', False)
                            and retries_b < max_retries
                            and is_cooldown_expired(sig_data.get('last_execution_attempt_0.618'), 60)
                        )
                        
                        if recover_inactive_b or retry_watch_b or failed_execution_b:
                            if retry_watch_b:
                                print(f"[Price Watch] Option B (0.618) {tf} {strat} at {opt_b['time']} is near enough to retry.")
                            elif failed_execution_b:
                                print(f"[Recovery Engine] Option B (0.618) was not successfully placed. Retrying execution...")
                            else:
                                print(f"[Recovery Engine] Option B (0.618) Ticket #{ticket_b} is inactive. Re-placing...")
                            gate_b = evaluate_live_entry_gate(
                                opt_b,
                                strategy=strat,
                                probability=prob_b,
                                accept_threshold=confidence_threshold,
                                symbol=symbol,
                                timeframe=tf,
                                timeframes_data=timeframes_data,
                            )
                            if not gate_b.allowed:
                                opt_b['filtered_reason'] = gate_b.filtered_reason
                                if register_entry_gate_filtered_option(
                                    sig_key,
                                    opt_b,
                                    prob_b,
                                    symbol=symbol,
                                    timeframe=tf,
                                    strategy=strat,
                                    direction_name=dir_name,
                                    accept_threshold=confidence_threshold,
                                    leg="0.618",
                                ):
                                    registry_changed = True
                                print(f"[Entry Gate] Recovery Option B skipped: {gate_b.reason}")
                                new_ticket_b, exec_msg_b = None, f"Skipped ({gate_b.reason})"
                                continue
                            recovery_mode_b = choose_dual_recovery_execution_mode(
                                opt_a,
                                opt_b,
                                current_price,
                                option="b",
                                timeframe=tf,
                                timeframes_data=timeframes_data,
                            )
                            if recovery_mode_b == "market":
                                new_ticket_b, exec_msg_b = execute_market_order_for_setup(opt_b, symbol)
                            elif recovery_mode_b == "skip":
                                new_ticket_b, exec_msg_b = None, "Skipped (Option A market recovery active)"
                            elif not should_place_pending_setup(
                                opt_b,
                                timeframe=tf,
                                timeframes_data=timeframes_data,
                            ):
                                new_ticket_b, exec_msg_b = None, "Skipped (immediate emergency reversal)"
                            else:
                                new_ticket_b, exec_msg_b = execute_trade_for_setup(opt_b, symbol)
                            
                            sig_data['last_execution_attempt_0.618'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            registry_changed = True
                            
                            if new_ticket_b is not None:
                                sig_data['ticket_b'] = new_ticket_b
                                sig_data['outcome_b_recorded'] = False
                                sig_data['outcome_recorded'] = False
                                sig_data['watch_status'] = "ticket_placed"
                                sig_data['watch_last_execution_message_0.618'] = exec_msg_b
                                sig_data['execution_retries_0.618'] = 0
                                try:
                                    recovery_title = "Market Order Recovery Executed" if recovery_mode_b == "market" else "Pending Order Re-placed"
                                    recovery_msg_b = (
                                        f"🔄 <b>[Order Recovery] {recovery_title}</b> 🔄\n\n"
                                        f"Setup 0.618 yang sebelumnya inactive telah dieksekusi ulang.\n"
                                        f"• <b>Price:</b> {opt_b['entry_price']:.3f}\n"
                                        f"• <b>Execution:</b> {exec_msg_b}\n"
                                        f"• <b>New Ticket:</b> #{new_ticket_b}"
                                    )
                                    send_recovery_alert_with_chart(
                                        recovery_msg_b,
                                        timeframes_data=timeframes_data,
                                        timeframe=tf,
                                        symbol=symbol,
                                        direction_name=dir_name,
                                        strategy=strat,
                                        setups=[opt_a, opt_b],
                                        image_suffix="recovery_dual_b",
                                    )
                                except Exception:
                                    pass
                            else:
                                failure_action = record_recovery_failure(
                                    sig_data,
                                    exec_msg_b,
                                    retries_b,
                                    max_retries,
                                    message_key='watch_last_execution_message_0.618',
                                    retries_key='execution_retries_0.618',
                                    outcome_key='outcome_b_recorded',
                                )
                                if failure_action == "price_watch":
                                    print(f"[Recovery Engine] Option B (0.618) waiting for price: {exec_msg_b}")
                                elif failure_action == "deferred":
                                    print(f"[Recovery Engine] Option B (0.618) execution deferred: {exec_msg_b}")
                                elif failure_action == "blocked":
                                    print(f"[Recovery Engine] Option B (0.618) recovery stopped: {exec_msg_b}")
                                else:
                                    print(f"[Recovery Engine] Option B (0.618) placement failed: {exec_msg_b}")
                                if failure_action == "retry" and retries_b + 1 >= max_retries:
                                    print(f"[Recovery Engine] Option B (0.618) reached max retry limit ({max_retries}) for {symbol}. Disabling further retries.")
                                    
                    # Mark as active to protect from pruning, then continue
                    active_high_confidence.append(opt_a)
                    active_high_confidence.append(opt_b)
                    continue
                    
                # Auto-execute trades on MT5 (market execution if price inside setup entry zone and rejection confirmed, else limit orders)
                ticket_a, ticket_b = None, None
                exec_msg_a, exec_msg_b = "", ""

                gate_a = evaluate_live_entry_gate(
                    opt_a,
                    strategy=strat,
                    probability=prob_a,
                    accept_threshold=confidence_threshold,
                    symbol=symbol,
                    timeframe=tf,
                    timeframes_data=timeframes_data,
                    oscillator=opt_a.get('oscillator'),
                )
                gate_b = evaluate_live_entry_gate(
                    opt_b,
                    strategy=strat,
                    probability=prob_b,
                    accept_threshold=confidence_threshold,
                    symbol=symbol,
                    timeframe=tf,
                    timeframes_data=timeframes_data,
                    oscillator=opt_b.get('oscillator'),
                )
                opt_a['filtered_reason'] = gate_a.filtered_reason
                opt_b['filtered_reason'] = gate_b.filtered_reason

                if not gate_a.allowed:
                    if register_entry_gate_filtered_option(
                        sig_key,
                        opt_a,
                        prob_a,
                        symbol=symbol,
                        timeframe=tf,
                        strategy=strat,
                        direction_name=dir_name,
                        accept_threshold=confidence_threshold,
                        leg="0.5",
                    ):
                        registry_changed = True
                if not gate_b.allowed:
                    if register_entry_gate_filtered_option(
                        sig_key,
                        opt_b,
                        prob_b,
                        symbol=symbol,
                        timeframe=tf,
                        strategy=strat,
                        direction_name=dir_name,
                        accept_threshold=confidence_threshold,
                        leg="0.618",
                    ):
                        registry_changed = True
                if not gate_a.allowed and not gate_b.allowed:
                    print(f"[Entry Gate] {tf} {strat} at index {opt_a['index']} skipped: {gate_a.reason} / {gate_b.reason}")
                    continue
                
                from src.execution import get_active_broker_symbol
                broker_symbol = get_active_broker_symbol(symbol)
                tick = mt5.symbol_info_tick(broker_symbol)
                current_price = None
                if tick is not None:
                    current_price = tick.ask if opt_a['direction'] == 1 else tick.bid
                    
                is_market_entry = False
                market_option = choose_dual_market_entry_option(
                    opt_a,
                    opt_b,
                    current_price,
                    timeframe=tf,
                    timeframes_data=timeframes_data,
                )
                if market_option == "a" and gate_a.allowed:
                    ticket_a, exec_msg_a = execute_market_order_for_setup(opt_a, symbol)
                    ticket_b, exec_msg_b = None, "Skipped (Option A Market Order placed)"
                    is_market_entry = True
                elif market_option == "b" and gate_b.allowed:
                    ticket_b, exec_msg_b = execute_market_order_for_setup(opt_b, symbol)
                    ticket_a, exec_msg_a = None, "Skipped (Option B Market Order placed)"
                    is_market_entry = True
                            
                if not is_market_entry:
                    if gate_a.allowed and should_place_pending_setup(
                        opt_a,
                        timeframe=tf,
                        timeframes_data=timeframes_data,
                    ):
                        ticket_a, exec_msg_a = execute_trade_for_setup(opt_a, symbol)
                    elif gate_a.allowed:
                        ticket_a, exec_msg_a = None, "Skipped (immediate emergency reversal)"
                    else:
                        ticket_a, exec_msg_a = None, f"Skipped ({gate_a.reason})"
                    if gate_b.allowed and should_place_pending_setup(
                        opt_b,
                        timeframe=tf,
                        timeframes_data=timeframes_data,
                    ):
                        ticket_b, exec_msg_b = execute_trade_for_setup(opt_b, symbol)
                    elif gate_b.allowed:
                        ticket_b, exec_msg_b = None, "Skipped (immediate emergency reversal)"
                    else:
                        ticket_b, exec_msg_b = None, f"Skipped ({gate_b.reason})"
                
                is_placed = (ticket_a is not None) or (ticket_b is not None)
                is_monitoring_only = (
                    (ticket_a is None and ticket_b is None)
                    and ("disabled" in exec_msg_a.lower() or "disabled" in exec_msg_b.lower())
                )
                is_price_watch = is_price_too_far_execution(exec_msg_a) or is_price_too_far_execution(exec_msg_b)
                should_alert = is_placed or is_monitoring_only or is_price_watch
                
                if not should_alert:
                    print(f"[Execution Engine] {tf} {strat} at index {opt_a['index']} skipped: {exec_msg_a} / {exec_msg_b}")
                    continue

                if is_price_watch and not is_placed and not is_monitoring_only:
                    print(
                        f"[Price Watch] Registered {tf} {strat} at {opt_a['time']} "
                        f"for retry when price returns near entry: {exec_msg_a} / {exec_msg_b}"
                    )
                    if gate_a.allowed:
                        active_high_confidence.append(opt_a)
                    if gate_b.allowed:
                        active_high_confidence.append(opt_b)
                    sent_signals[sig_key] = {
                        'time_sent': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'timeframe': tf,
                        'direction': dir_name,
                        'type': strat,
                        'price_0.5': opt_a['entry_price'],
                        'price_0.618': opt_b['entry_price'],
                        'probability_0.5': prob_a,
                        'probability_0.618': prob_b,
                        'ticket_a': ticket_a,
                        'ticket_b': ticket_b,
                        'outcome_a_recorded': not gate_a.allowed,
                        'outcome_b_recorded': not gate_b.allowed,
                        'entry_gate_0.5': opt_a.get('entry_gate'),
                        'entry_gate_0.618': opt_b.get('entry_gate'),
                        'reentries_count': reentries_count + 1,
                        'features_0.5': opt_a['features'],
                        'features_0.618': opt_b['features'],
                        'watch_last_execution_message_0.5': exec_msg_a if is_price_too_far_execution(exec_msg_a) else None,
                        'watch_last_execution_message_0.618': exec_msg_b if is_price_too_far_execution(exec_msg_b) else None,
                        'last_execution_attempt_0.5': datetime.now().strftime('%Y-%m-%d %H:%M:%S') if ticket_a is None else None,
                        'last_execution_attempt_0.618': datetime.now().strftime('%Y-%m-%d %H:%M:%S') if ticket_b is None else None,
                        'confidence_tier': confidence_tier(lead['max_prob']),
                        **_price_watch_metadata(exec_msg_a, exec_msg_b),
                    }
                    signals_sent_this_cycle += 1
                    registry_changed = True
                    continue
                    
                if gate_a.allowed:
                    active_high_confidence.append(opt_a)
                if gate_b.allowed:
                    active_high_confidence.append(opt_b)
                
                print(f"[New Cluster Signal Triggered] {tf} Dual {strat} at {opt_a['time']} | Direction: {dir_name} | Win Probs: 0.5={prob_a:.2%}, 0.618={prob_b:.2%}")
                
                # Generate chart
                tf_df = timeframes_data[tf]
                tf_setups = [opt_a, opt_b]
                title = f"{symbol} {tf} - {dir_name} Dual {strat} Confluence"
                image_filename = f"temp_alert_{tf}_dual_{opt_a['index']}.png"
                
                try:
                    plot_smc_chart(tf_df, title=title, active_setups=tf_setups, output_filename=image_filename)
                except Exception as e:
                    print(f"Failed to generate chart image: {e}")
                    image_filename = None
                    
                # Format Telegram message using HTML
                rej_src_a = opt_a.get('rejection_source', 'None')
                rej_src_b = opt_b.get('rejection_source', 'None')
                rej_status = format_rejection_status([rej_src_a, rej_src_b])
                htf_prior_status = format_htf_priority_status(opt_a['htf_prioritized'] or opt_b['htf_prioritized'])
                exec_status_a = format_execution_status(
                    ticket_a,
                    exec_msg_a,
                    skipped_peer="0.618" if "Skipped" in exec_msg_a else None,
                    monitoring_only=is_monitoring_only,
                )
                exec_status_b = format_execution_status(
                    ticket_b,
                    exec_msg_b,
                    skipped_peer="0.500" if "Skipped" in exec_msg_b else None,
                    monitoring_only=is_monitoring_only,
                )

                # De-duplicate matching HTF structures for the Telegram formatter.
                matching_fvgs = opt_a['matching_htf_fvgs'] + opt_b['matching_htf_fvgs']
                seen_fvgs = set()
                unique_matching = []
                for f in matching_fvgs:
                    f_key = (f['timeframe'], f['bottom'], f['top'])
                    if f_key not in seen_fvgs:
                        seen_fvgs.add(f_key)
                        unique_matching.append(f)
                
                setup_desc = f"{get_strategy_display_name(strat)} (Dual Fibonacci Entry)"
                # Reuse pre-computed oscillator from lead setup (opt_a) for Telegram display
                _osc_ctx = opt_a.get('oscillator') or build_oscillator_context(timeframes_data.get(tf))
                msg = format_dual_signal_message(
                    symbol=symbol,
                    timeframe=tf,
                    direction=opt_a['direction'],
                    setup_desc=setup_desc,
                    probability_a=prob_a,
                    probability_b=prob_b,
                    confidence_threshold=confidence_threshold,
                    opt_a=opt_a,
                    opt_b=opt_b,
                    execution_status_a=exec_status_a,
                    execution_status_b=exec_status_b,
                    htf_priority_status=htf_prior_status,
                    rejection_status=rej_status,
                    confluences=unique_reasons,
                    htf_matches=unique_matching,
                    oscillator_line=format_multi_tf_oscillator_block(
                        _osc_ctx if isinstance(_osc_ctx, MultiTFOscillatorContext) else None,
                        direction=opt_a.get('direction', 0),
                    ) if isinstance(_osc_ctx, MultiTFOscillatorContext) else format_oscillator_line(
                        _osc_ctx.primary if isinstance(_osc_ctx, MultiTFOscillatorContext) else _osc_ctx
                    ),
                )

                success = send_telegram_alert(msg, image_filename)
                
                if image_filename and os.path.exists(image_filename):
                    try:
                        os.remove(image_filename)
                    except Exception:
                        pass
                        
                if success or is_placed or is_price_watch:
                    sent_signals[sig_key] = {
                        'time_sent': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'timeframe': tf,
                        'direction': dir_name,
                        'type': strat,
                        'price_0.5': opt_a['entry_price'],
                        'price_0.618': opt_b['entry_price'],
                        'probability_0.5': prob_a,
                        'probability_0.618': prob_b,
                        'ticket_a': ticket_a,
                        'ticket_b': ticket_b,
                        'outcome_a_recorded': not gate_a.allowed,
                        'outcome_b_recorded': not gate_b.allowed,
                        'entry_gate_0.5': opt_a.get('entry_gate'),
                        'entry_gate_0.618': opt_b.get('entry_gate'),
                        'reentries_count': reentries_count + 1,
                        'features_0.5': opt_a['features'],
                        'features_0.618': opt_b['features'],
                        'watch_last_execution_message_0.5': exec_msg_a if is_price_too_far_execution(exec_msg_a) else None,
                        'watch_last_execution_message_0.618': exec_msg_b if is_price_too_far_execution(exec_msg_b) else None,
                        'last_execution_attempt_0.5': datetime.now().strftime('%Y-%m-%d %H:%M:%S') if ticket_a is None else None,
                        'last_execution_attempt_0.618': datetime.now().strftime('%Y-%m-%d %H:%M:%S') if ticket_b is None else None,
                        'confidence_tier': confidence_tier(lead['max_prob']),
                        **_price_watch_metadata(exec_msg_a, exec_msg_b),
                    }
                    signals_sent_this_cycle += 1
                    registry_changed = True

            else:
                opt = lead['opt']
                prob = lead['max_prob']
                
                setup_time_str = str(opt['time'])
                sig_key = f"{tf}_{strat}_SINGLE_{dir_name}_{opt['entry_price']:.3f}_{setup_time_str.replace(' ', '_')}"
                
                opt['status'] = "HIGH CONFIDENCE SIGNAL"
                
                allow_entry = True
                reentries_count = 0

                if sig_key in sent_signals and should_promote_low_confidence_record(
                    sent_signals[sig_key],
                    ("ticket_id",),
                ):
                    promoted_record = sent_signals.pop(sig_key)
                    reentries_count = promoted_record.get('reentries_count', 0)
                    print(f"[Scanner Registry] Promoting low confidence single {tf} {strat} at {opt['time']} to live execution.")
                
                if sig_key in sent_signals:
                    sig_data = sent_signals[sig_key]
                    reentries_count = sig_data.get('reentries_count', 0)
                    
                    execute_enabled = os.getenv("MT5_EXECUTE_TRADES", "False").strip().lower() == "true"
                    if execute_enabled:
                        from src.execution import get_active_broker_symbol
                        broker_symbol = get_active_broker_symbol(symbol)
                        tick = mt5.symbol_info_tick(broker_symbol)
                        current_price = None
                        if tick is not None:
                            current_price = tick.ask if opt['direction'] == 1 else tick.bid

                        def is_ticket_active(t):
                            if t is None:
                                return False
                            orders_act = mt5.orders_get(ticket=t)
                            if orders_act and len(orders_act) > 0:
                                return True
                            positions_act = mt5.positions_get(ticket=t)
                            if positions_act and len(positions_act) > 0:
                                return True
                            return False

                        def has_history_deals(t):
                            if t is None:
                                return False
                            deals = mt5.history_deals_get(position=t)
                            return deals is not None and len(deals) > 0

                        try:
                            max_retries = int(os.getenv("MT5_RECOVERY_MAX_RETRIES", "3"))
                        except (TypeError, ValueError):
                            max_retries = 3

                        ticket_id = sig_data.get('ticket_id')
                        recovery_allowed, recovery_reason = enforce_recovery_strategy_policy(
                            sig_data,
                            strategy=strat,
                            setup=opt,
                            probability=prob,
                            timeframe=tf,
                            outcome_keys=('outcome_recorded',),
                            message_keys=('watch_last_execution_message',),
                        )
                        if not recovery_allowed:
                            print(f"[Recovery Engine] {tf} {strat} recovery blocked by live policy: {recovery_reason}")
                            registry_changed = True
                            continue
                        retry_watch = should_retry_unfilled_watch_record(
                            sig_data,
                            ("ticket_id",),
                            ("outcome_recorded",),
                        )
                        recover_inactive = (
                            ticket_id is not None
                            and not is_ticket_active(ticket_id)
                            and not has_history_deals(ticket_id)
                            and not sig_data.get('outcome_recorded', False)
                        )
                        retries = int(sig_data.get('execution_retries', 0))
                        failed_execution = (
                            ticket_id is None
                            and not sig_data.get('outcome_recorded', False)
                            and retries < max_retries
                            and is_cooldown_expired(sig_data.get('last_execution_attempt'), 60)
                        )
                        
                        if recover_inactive or retry_watch or failed_execution:
                            if retry_watch:
                                print(f"[Price Watch] Single {tf} {strat} at {opt['time']} is near enough to retry.")
                            elif failed_execution:
                                print(f"[Recovery Engine] Single setup was not successfully placed. Retrying execution...")
                            else:
                                print(f"[Recovery Engine] Single Ticket #{ticket_id} is inactive. Re-placing...")
                            gate = evaluate_live_entry_gate(
                                opt,
                                strategy=strat,
                                probability=prob,
                                accept_threshold=confidence_threshold,
                                symbol=symbol,
                                timeframe=tf,
                                timeframes_data=timeframes_data,
                            )
                            if not gate.allowed:
                                opt['filtered_reason'] = gate.filtered_reason
                                if register_entry_gate_filtered_lead(
                                    lead=lead,
                                    sent_signals=sent_signals,
                                    symbol=symbol,
                                    timeframe=tf,
                                    strategy=strat,
                                    direction_name=dir_name,
                                    accept_threshold=confidence_threshold,
                                    filtered_reason=gate.filtered_reason,
                                ):
                                    registry_changed = True
                                print(f"[Entry Gate] Recovery single skipped: {gate.reason}")
                                continue
                            recovery_mode = choose_recovery_execution_mode(
                                opt,
                                current_price,
                                timeframe=tf,
                                timeframes_data=timeframes_data,
                            )
                            if recovery_mode == "market":
                                new_ticket, exec_msg = execute_market_order_for_setup(opt, symbol)
                            elif not should_place_pending_setup(
                                opt,
                                timeframe=tf,
                                timeframes_data=timeframes_data,
                            ):
                                new_ticket, exec_msg = None, "Skipped (immediate emergency reversal)"
                            else:
                                new_ticket, exec_msg = execute_trade_for_setup(opt, symbol)
                            
                            sig_data['last_execution_attempt'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            registry_changed = True
                            
                            if new_ticket is not None:
                                sig_data['ticket_id'] = new_ticket
                                sig_data['outcome_recorded'] = False
                                sig_data['watch_status'] = "ticket_placed"
                                sig_data['watch_last_execution_message'] = exec_msg
                                sig_data['execution_retries'] = 0
                                try:
                                    recovery_title = "Market Order Recovery Executed" if recovery_mode == "market" else "Pending Order Re-placed"
                                    recovery_msg = (
                                        f"🔄 <b>[Order Recovery] {recovery_title}</b> 🔄\n\n"
                                        f"Setup yang sebelumnya inactive telah dieksekusi ulang.\n"
                                        f"• <b>Price:</b> {opt['entry_price']:.3f}\n"
                                        f"• <b>Execution:</b> {exec_msg}\n"
                                        f"• <b>New Ticket:</b> #{new_ticket}"
                                    )
                                    send_recovery_alert_with_chart(
                                        recovery_msg,
                                        timeframes_data=timeframes_data,
                                        timeframe=tf,
                                        symbol=symbol,
                                        direction_name=dir_name,
                                        strategy=strat,
                                        setups=[opt],
                                        image_suffix="recovery_single",
                                    )
                                except Exception:
                                    pass
                            else:
                                failure_action = record_recovery_failure(
                                    sig_data,
                                    exec_msg,
                                    retries,
                                    max_retries,
                                    message_key='watch_last_execution_message',
                                    retries_key='execution_retries',
                                    outcome_key='outcome_recorded',
                                )
                                if failure_action == "price_watch":
                                    print(f"[Recovery Engine] Single waiting for price: {exec_msg}")
                                elif failure_action == "deferred":
                                    print(f"[Recovery Engine] Single execution deferred: {exec_msg}")
                                elif failure_action == "blocked":
                                    print(f"[Recovery Engine] Single recovery stopped: {exec_msg}")
                                else:
                                    print(f"[Recovery Engine] Single setup placement failed: {exec_msg}")
                                if failure_action == "retry" and retries + 1 >= max_retries:
                                    print(f"[Recovery Engine] Single reached max retry limit ({max_retries}) for {symbol}. Disabling further retries.")
                                registry_changed = True
                                    
                    # Mark as active to protect from pruning, then continue
                    active_high_confidence.append(opt)
                    continue
                    
                # Auto-execute single trade on MT5 (market execution if price inside setup entry zone and rejection confirmed, else limit orders)
                ticket_id = None
                exec_msg = ""
                is_market_entry = False

                gate = evaluate_live_entry_gate(
                    opt,
                    strategy=strat,
                    probability=prob,
                    accept_threshold=confidence_threshold,
                    symbol=symbol,
                    timeframe=tf,
                    timeframes_data=timeframes_data,
                    oscillator=opt.get('oscillator'),
                )
                opt['filtered_reason'] = gate.filtered_reason
                if not gate.allowed:
                    print(f"[Entry Gate] Single {tf} {strat} at index {opt['index']} skipped: {gate.reason}")
                    if register_entry_gate_filtered_lead(
                        lead=lead,
                        sent_signals=sent_signals,
                        symbol=symbol,
                        timeframe=tf,
                        strategy=strat,
                        direction_name=dir_name,
                        accept_threshold=confidence_threshold,
                        filtered_reason=gate.filtered_reason,
                    ):
                        registry_changed = True
                    continue
                
                from src.execution import get_active_broker_symbol
                broker_symbol = get_active_broker_symbol(symbol)
                tick = mt5.symbol_info_tick(broker_symbol)
                current_price = None
                if tick is not None:
                    current_price = tick.ask if opt['direction'] == 1 else tick.bid
                    
                if should_market_enter_setup(
                    opt,
                    current_price,
                    timeframe=tf,
                    timeframes_data=timeframes_data,
                ):
                    ticket_id, exec_msg = execute_market_order_for_setup(opt, symbol)
                    is_market_entry = True
                            
                if not is_market_entry:
                    if should_place_pending_setup(
                        opt,
                        timeframe=tf,
                        timeframes_data=timeframes_data,
                    ):
                        ticket_id, exec_msg = execute_trade_for_setup(opt, symbol)
                    else:
                        ticket_id, exec_msg = None, "Skipped (immediate emergency reversal)"
                    
                is_placed = ticket_id is not None
                is_monitoring_only = "disabled" in exec_msg.lower()
                is_price_watch = is_price_too_far_execution(exec_msg)
                should_alert = is_placed or is_monitoring_only or is_price_watch
                
                if not should_alert:
                    print(f"[Execution Engine] Single {tf} {strat} at index {opt['index']} skipped: {exec_msg}")
                    continue

                if is_price_watch and not is_placed and not is_monitoring_only:
                    print(
                        f"[Price Watch] Registered single {tf} {strat} at {opt['time']} "
                        f"for retry when price returns near entry: {exec_msg}"
                    )
                    active_high_confidence.append(opt)
                    sent_signals[sig_key] = {
                        'time_sent': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'timeframe': tf,
                        'direction': dir_name,
                        'type': strat,
                        'price': opt['entry_price'],
                        'probability': prob,
                        'ticket_id': ticket_id,
                        'entry_gate': opt.get('entry_gate'),
                        'reentries_count': reentries_count + 1,
                        'features': opt['features'],
                        'watch_last_execution_message': exec_msg if is_price_too_far_execution(exec_msg) else None,
                        'last_execution_attempt': datetime.now().strftime('%Y-%m-%d %H:%M:%S') if ticket_id is None else None,
                        'confidence_tier': confidence_tier(prob),
                        **_price_watch_metadata(exec_msg),
                    }
                    signals_sent_this_cycle += 1
                    registry_changed = True
                    continue
                    
                active_high_confidence.append(opt)
                print(f"[New Cluster Signal Triggered] Single {tf} {strat} at {opt['time']} | Direction: {dir_name} | Win Prob: {prob:.2%}")
                
                # Generate chart
                tf_df = timeframes_data[tf]
                tf_setups = [opt]
                title = f"{symbol} {tf} - {dir_name} Single {strat} Confluence"
                image_filename = f"temp_alert_{tf}_single_{opt['index']}.png"
                
                try:
                    plot_smc_chart(tf_df, title=title, active_setups=tf_setups, output_filename=image_filename)
                except Exception as e:
                    print(f"Failed to generate chart image: {e}")
                    image_filename = None
                    
                # Format Telegram message using HTML
                rej_src = opt.get('rejection_source', 'None')
                rej_status = format_rejection_status([rej_src])
                htf_prior_status = format_htf_priority_status(opt['htf_prioritized'])
                exec_status = format_execution_status(
                    ticket_id,
                    exec_msg,
                    monitoring_only=is_monitoring_only,
                )

                matching_fvgs = opt['matching_htf_fvgs']
                setup_desc = opt.get('option_name') or get_strategy_display_name(strat)
                # Reuse pre-computed oscillator from setup dict for Telegram display
                _osc_ctx = opt.get('oscillator') or build_oscillator_context(timeframes_data.get(tf))
                msg = format_single_signal_message(
                    symbol=symbol,
                    timeframe=tf,
                    direction=opt['direction'],
                    setup_desc=setup_desc,
                    probability=prob,
                    confidence_threshold=confidence_threshold,
                    setup=opt,
                    execution_status=exec_status,
                    htf_priority_status=htf_prior_status,
                    rejection_status=rej_status,
                    confluences=unique_reasons,
                    htf_matches=matching_fvgs,
                    oscillator_line=format_multi_tf_oscillator_block(
                        _osc_ctx if isinstance(_osc_ctx, MultiTFOscillatorContext) else None,
                        direction=opt.get('direction', 0),
                    ) if isinstance(_osc_ctx, MultiTFOscillatorContext) else format_oscillator_line(
                        _osc_ctx.primary if isinstance(_osc_ctx, MultiTFOscillatorContext) else _osc_ctx
                    ),
                )
                
                success = send_telegram_alert(msg, image_filename)
                
                if image_filename and os.path.exists(image_filename):
                    try:
                        os.remove(image_filename)
                    except Exception:
                        pass
                        
                if success or is_placed or is_price_watch:
                    sent_signals[sig_key] = {
                        'time_sent': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'timeframe': tf,
                        'direction': dir_name,
                        'type': strat,
                        'price': opt['entry_price'],
                        'probability': prob,
                        'ticket_id': ticket_id,
                        'entry_gate': opt.get('entry_gate'),
                        'reentries_count': reentries_count + 1,
                        'features': opt['features'],
                        'watch_last_execution_message': exec_msg if is_price_too_far_execution(exec_msg) else None,
                        'last_execution_attempt': datetime.now().strftime('%Y-%m-%d %H:%M:%S') if ticket_id is None else None,
                        'confidence_tier': confidence_tier(prob),
                        **_price_watch_metadata(exec_msg),
                    }
                    signals_sent_this_cycle += 1
                    registry_changed = True
        else:
            # Low confidence cluster - Silent Registration for Lead Setup
            if register_low_confidence_lead(
                lead=lead,
                sent_signals=sent_signals,
                symbol=symbol,
                timeframe=tf,
                strategy=strat,
                direction_name=dir_name,
                accept_threshold=confidence_threshold,
            ):
                registry_changed = True
    for setup in other_setups:
        tf = setup['timeframe']
        knn_up_tf, knn_down_tf = tf_knn_data.get(tf, (0.0, 0.0))
        knn_prob_sig = knn_up_tf if setup['direction'] == 1 else knn_down_tf
        knn_prob_opp = knn_down_tf if setup['direction'] == 1 else knn_up_tf
        
        clusters_data_tf = tf_vp_data.get(tf, {})
        dist_entry_to_poc = 0.0
        dist_entry_to_nearest_poc = 0.0
        if clusters_data_tf and 'current_poc' in clusters_data_tf:
            curr_poc = clusters_data_tf['current_poc']
            entry = setup['entry_price']
            dist_entry_to_poc = (entry - curr_poc) / curr_poc if curr_poc > 0 else 0.0
            
            pocs = clusters_data_tf.get('pocs', [])
            if pocs:
                dist_entry_to_nearest_poc = min(abs(entry - poc) for poc in pocs) / entry

        features = {
            'timeframe': tf_minutes_map[setup['timeframe']],
            'hour': setup['hour'],
            'day_of_week': setup['day_of_week'],
            'setup_type': setup['setup_type'],
            'direction': setup['direction'],
            'entry_price': setup['entry_price'],
            'sl_price': setup['sl_price'],
            'tp_price': setup['tp_price'],
            'risk_pips': setup['risk_pips'],
            'atr_14': setup['atr_14'],
            'trend': setup['trend'],
            'relative_risk': setup['relative_risk'],
            'killzone': setup['killzone'],
            'fvg_width': setup['fvg_width'],
            'relative_fvg_width': setup['relative_fvg_width'],
            'near_psychological_level': setup['near_psychological_level'],
            'knn_prob_sig': knn_prob_sig,
            'knn_prob_opp': knn_prob_opp,
            'dist_entry_to_poc': dist_entry_to_poc,
            'dist_entry_to_nearest_poc': dist_entry_to_nearest_poc,
            'dist_entry_to_pp': setup.get('dist_entry_to_pp', 0.0),
            'dist_entry_to_nearest_pivot': setup.get('dist_entry_to_nearest_pivot', 0.0),
            'floop_signal': setup['floop_signal'],
            'floop_strength': setup['floop_strength'],
            'floop_trend': setup.get('floop_trend', 0),
            'floop_trend_aligned': 1 if setup.get('floop_trend', 0) == setup['direction'] else 0
        }
        try:
            prob = predict_setup_probability(features)
        except Exception:
            prob = 0.5
        setup['probability'] = prob
        setup['status'] = "HIGH CONFIDENCE SIGNAL" if prob >= confidence_threshold else "FILTERED (LTF/Other)"
                
    if registry_changed:
        save_sent_signals(sent_signals)
        
    if signals_sent_this_cycle > 0:
        print(f"Sent {signals_sent_this_cycle} new alerts this cycle.")
    else:
        print("No new high confidence trade signals triggered this cycle.")
        
    # 8. Clean up invalid/old pending orders from MT5 and manage active positions
    execute_enabled = os.getenv("MT5_EXECUTE_TRADES", "False").strip().lower() == "true"
    if execute_enabled:
        try:
            magic = int(os.getenv("MT5_MAGIC_NUMBER", "202606"))
            prune_invalid_pending_orders(symbol, magic, active_high_confidence)
            manage_active_trades(symbol, magic, timeframes_data)
        except Exception as e:
            print(f"[Scanner Error] Error during pending orders pruning / active trade management: {e}")
            
    # 9. Save all detected candidates as Price Watch Zones so the tick loop can react
    #    immediately when price enters any zone — even ones below confidence threshold.
    #    This eliminates the "late entry" problem caused by waiting for the next full scan.
    if os.getenv("MT5_WATCH_ZONE_ENABLED", "True").strip().lower() == "true":
        try:
            zone_candidates = []
            for c in clusters:
                zone_candidates.append(c["lead"])
            # Also include any non-clustered setups from other_setups (LTF reference zones)
            for s in other_setups:
                if not s.get("suppressed", False):
                    zone_candidates.append(s)
            import MetaTrader5 as mt5
            from src.execution import get_active_broker_symbol
            broker_symbol = get_active_broker_symbol(symbol)
            tick = mt5.symbol_info_tick(broker_symbol)
            curr_price = 0.0
            if tick is not None:
                curr_price = (tick.bid + tick.ask) / 2.0
            n_zones = save_watch_zones(symbol, zone_candidates, confidence_threshold, current_price=curr_price)
            print(f"[WatchZones] Registered {n_zones} active price watch zones for {symbol}.")
        except Exception as e:
            print(f"[WatchZones] Failed to save watch zones: {e}")

    # Free MT5 connection at the very end of the cycle
    import MetaTrader5 as mt5
    mt5.shutdown()
    print("--- Scan Cycle Finished ---")

def main():
    configure_console_encoding()
    parser = argparse.ArgumentParser(description="Forex SMC Scanner background worker with Telegram Alerts.")
    parser.add_argument("--symbol", type=str, default="XAUUSD", help="Trading symbol or comma-separated list of symbols, or 'all'/'marketwatch' (default: XAUUSD)")
    parser.add_argument("--threshold", type=float, default=None, help="Confidence threshold to alert/order (default: ML_ACCEPT_THRESHOLD or 0.50)")
    parser.add_argument("--loop", action="store_true", help="Run continuously in a loop")
    parser.add_argument("--interval", type=int, default=5, help="Scan interval in minutes (default: 5)")
    parser.add_argument("--realtime-reaction", action="store_true", help="Watch existing setups with fast tick reaction checks between full scans")
    parser.add_argument("--tick-interval", type=float, default=1.0, help="Realtime reaction tick interval in seconds (default: 1.0)")
    parser.add_argument("--min-reaction-move", type=float, default=0.10, help="Minimum bid/ask reaction move needed for realtime market entry")
    
    args = parser.parse_args()
    confidence_threshold = get_accept_threshold(args.threshold)
    rollout_ready, rollout_message = assert_rollout_ready_for_live(confidence_threshold)
    if not rollout_ready:
        print("[Scanner Guard] Real-money rollout preflight BLOCKED.")
        print(f"[Scanner Guard] {rollout_message}")
        print("[Scanner Guard] Run: python -m src.rollout_status --profile real-money --threshold "
              f"{confidence_threshold:.2f}")
        return
    if _execute_trades_enabled():
        print(f"[Scanner Guard] {rollout_message}")
    
    # Verify environment file variables exist
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id or token.startswith("YOUR_") or chat_id.startswith("YOUR_"):
        print("\n[WARNING] Telegram credentials are not configured in your .env file.")
        print("Alerts will print in the console but will NOT be sent to Telegram.")
        print("Please configure TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env to enable alerts.\n")

    # Resolve symbols list
    sym_input = args.symbol
    if sym_input.lower() in ["all", "marketwatch"]:
        if not connect_mt5():
            print("[Scanner Error] Failed to connect to MT5 to fetch Market Watch symbols.")
            return
        import MetaTrader5 as mt5
        symbols = [s.name for s in mt5.symbols_get() if s.select]
        mt5.shutdown()
        print(f"Loaded {len(symbols)} selected symbols from Market Watch: {symbols}")
    else:
        symbols = [s.strip() for s in sym_input.split(",") if s.strip()]

    if not symbols:
        print("[Scanner Error] No symbols specified or found in Market Watch.")
        return

    magic = int(os.getenv("MT5_MAGIC_NUMBER", "202606"))
    lock_name = "multi" if (len(symbols) > 1 or sym_input.lower() in ["all", "marketwatch"]) else symbols[0]
    lock_ctx = scanner_instance_lock(lock_name, magic)
    
    try:
        lock_ctx.__enter__()
    except RuntimeError as exc:
        print(f"[Scanner Lock] {exc}")
        return

    if args.loop:
        try:
            print(f"Starting background worker loop for symbols: {symbols}. Scanning every {args.interval} minutes...")
            if args.realtime_reaction:
                print(
                    "[Realtime Reaction] Enabled. "
                    f"Watching every {args.tick_interval:.2f}s between full scans."
                )
            try:
                while True:
                    for sym in symbols:
                        run_scan(sym, confidence_threshold)
                        
                    sleep_seconds = max(0, args.interval * 60)
                    if not args.realtime_reaction:
                        print(f"Sleeping for {args.interval} minutes...")
                        time.sleep(sleep_seconds)
                        continue

                    deadline = time.time() + sleep_seconds
                    previous_ticks = {sym: None for sym in symbols}
                    print(f"Realtime reaction watching for {sleep_seconds:.0f} seconds before next full scan...")
                    if not connect_mt5():
                        print("[Realtime Reaction] MT5 connection unavailable; falling back to regular sleep.")
                        time.sleep(sleep_seconds)
                        continue
                    try:
                        import MetaTrader5 as mt5
                        while time.time() < deadline:
                            check_and_sync_active_account()
                            for sym in symbols:
                                try:
                                    current_tick = get_realtime_tick(sym, ensure_connection=False)
                                    prev_tick = previous_ticks.get(sym)
                                    if prev_tick is not None and current_tick is not None:
                                        # --- Existing: react to setups already in sent_signals ---
                                        run_realtime_reaction_cycle(
                                            sym,
                                            previous_tick=prev_tick,
                                            current_tick=current_tick,
                                            min_reaction_move=args.min_reaction_move,
                                        )

                                        # --- NEW: check all pre-registered price watch zones ---
                                        if os.getenv("MT5_WATCH_ZONE_ENABLED", "True").strip().lower() == "true":
                                            try:
                                                zone_hits = check_price_in_watch_zones(
                                                    sym, current_tick, confidence_threshold
                                                )
                                                for hit in zone_hits:
                                                    if not hit.entry_triggered:
                                                        # Price is near zone but confidence not yet at threshold
                                                        # Just log at debug level (don't spam)
                                                        continue
                                                    zone = hit.zone
                                                    
                                                    # Enforce live strategy policy (blocklist/allowlist)
                                                    strategy_allowed, strategy_reason = should_allow_live_strategy(
                                                        zone.strategy,
                                                        None,
                                                        probability=zone.probability,
                                                        timeframe=zone.timeframe,
                                                        entry_type="WatchZone",
                                                    )
                                                    if not strategy_allowed:
                                                        # Mark it triggered/disabled so it won't keep checking it
                                                        mark_zone_triggered(sym, zone.zone_id)
                                                        print(
                                                            f"[WatchZones] ⚠️ Strategy {zone.strategy} on {zone.timeframe} "
                                                            f"blocked by live policy: {strategy_reason}"
                                                        )
                                                        continue

                                                    # WatchZone instant entry is restricted to OB, BPR, FVG, and Pivot.
                                                    # SND, Swapzone, and IC must not enter instantly via WatchZone,
                                                    # as they require confirmation or passive pending orders to be safe.
                                                    if zone.strategy not in ["OB", "BPR", "FVG", "Pivot"]:
                                                        mark_zone_triggered(sym, zone.zone_id)
                                                        print(
                                                            f"[WatchZones] 🛡️ Strategy {zone.strategy} on {zone.timeframe} "
                                                            f"skipped for instant entry (only allowed for pending limits)"
                                                        )
                                                        continue

                                                    setup_dict = build_watch_zone_execution_setup(
                                                        zone, hit.current_price
                                                    )
                                                    if setup_dict["probability"] < confidence_threshold:
                                                        reason = (
                                                            f"selected WatchZone leg confidence "
                                                            f"{setup_dict['probability']:.1%} below threshold"
                                                        )
                                                        mark_zone_execution_attempt(sym, zone.zone_id, reason)
                                                        print(f"[WatchZones] Zone hit but {reason}")
                                                        continue
                                                    rejection_confirmed, rejection_source = refresh_watch_zone_rejection(
                                                        sym, setup_dict
                                                    )
                                                    setup_dict["rejection_confirmed"] = rejection_confirmed
                                                    setup_dict["rejection_source"] = rejection_source
                                                    watch_zone_context = get_watch_zone_reversal_context(
                                                        sym, zone.timeframe
                                                    )
                                                    if not should_market_enter_setup(
                                                        setup_dict,
                                                        hit.current_price,
                                                        timeframe=zone.timeframe,
                                                        timeframes_data=watch_zone_context,
                                                    ):
                                                        reason = "waiting for rejection confirmation or valid entry range"
                                                        mark_zone_execution_attempt(sym, zone.zone_id, reason)
                                                        print(
                                                            f"[WatchZones] Zone hit but market guard deferred: {reason} "
                                                            f"({zone.timeframe} {zone.strategy})"
                                                        )
                                                        continue

                                                    # Keep diagnostics current without restoring the entry gate as
                                                    # a live-execution blocker.
                                                    evaluate_live_entry_gate(
                                                        setup_dict,
                                                        strategy=zone.strategy,
                                                        probability=setup_dict["probability"],
                                                        accept_threshold=confidence_threshold,
                                                        symbol=sym,
                                                        timeframe=zone.timeframe,
                                                        timeframes_data=watch_zone_context,
                                                    )
                                                    mark_zone_execution_attempt(sym, zone.zone_id)
                                                    # Immediately attempt market order at current price
                                                    ticket_id, exec_msg = execute_market_order_for_setup(setup_dict, sym)
                                                    if ticket_id is not None:
                                                        mark_zone_triggered(sym, zone.zone_id)
                                                        print(
                                                            f"[WatchZones] ✅ ZONE HIT → Market order #{ticket_id} placed! "
                                                            f"{zone.timeframe} {zone.strategy} {hit.reason}"
                                                        )
                                                        try:
                                                            from src.telegram_bot import send_telegram_alert
                                                            wz_msg = (
                                                                f"⚡ <b>[WatchZone Hit] Immediate Entry!</b>\n\n"
                                                                f"Harga masuk ke zona yang dipantau sebelum scan berikutnya.\n"
                                                                f"• <b>Zone:</b> <code>{zone.timeframe} {zone.strategy}</code>\n"
                                                                f"• <b>Harga masuk:</b> <code>{hit.current_price:.3f}</code>\n"
                                                                f"• <b>Entry:</b> <code>{zone.entry_price:.3f}</code> | "
                                                                f"SL: <code>{zone.sl_price:.3f}</code> | "
                                                                f"TP: <code>{zone.tp_price:.3f}</code>\n"
                                                                f"• <b>Confidence:</b> <code>{zone.probability:.1%}</code>\n"
                                                                f"• <b>Ticket:</b> #{ticket_id}"
                                                            )
                                                            send_telegram_alert(wz_msg)
                                                        except Exception:
                                                            pass
                                                    elif "duplicate" not in exec_msg.lower() and "disabled" not in exec_msg.lower():
                                                        print(
                                                            f"[WatchZones] ⚠ Zone hit but order not placed: {exec_msg} "
                                                            f"({zone.timeframe} {zone.strategy})"
                                                        )
                                            except Exception as wz_exc:
                                                print(f"[WatchZones - {sym}] Zone check error: {wz_exc}")

                                    previous_ticks[sym] = current_tick
                                except Exception as exc:
                                    print(f"[Realtime Reaction - {sym}] Tick cycle error: {exc}")
                            time.sleep(max(0.10, float(args.tick_interval)))
                    finally:
                        mt5.shutdown()
            except KeyboardInterrupt:
                print("\nScanner stopped by user.")
        finally:
            lock_ctx.__exit__(None, None, None)
    else:
        try:
            # Run once for all symbols
            for sym in symbols:
                run_scan(sym, confidence_threshold)
        finally:
            lock_ctx.__exit__(None, None, None)

if __name__ == "__main__":
    main()
