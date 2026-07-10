# Confidence-Based Entry Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Two fixes: (1) shadow data is polluted by above-threshold strategy-blocked signals due to `force=True` bug; (2) add confidence-tier system so signals at 65%+ confidence get priority execution path with bigger lot recommendation and softer price-distance gate.

**Architecture:** `scanner_worker.py` gets the `force=True` → `force=False` fix. `live_trade_policy.py` gets a new `confidence_tier(prob)` helper and the scanner uses it to select lot tier and relax price-too-far buffer. Strategy blocklist (IC, SND, Swapzone, Pivot) is intentionally kept — no changes there.

**Tech Stack:** Python, `src/scanner_worker.py`, `src/live_trade_policy.py`, `.env`

---

## Context

### Why IC stays blocked
`MT5_LIVE_STRATEGY_BLOCKLIST=Pivot,SND,Swapzone,IC` — all four are intentionally blocked by the user. The original plan's Task 1 (fix allowlist/blocklist conflict) was wrong; the current policy behaviour is correct.

### Allowed strategies (what SHOULD enter when confidence ≥ threshold)
- **FVG** → normalises to `FVG_OR_BPR` → allowed
- **BPR** → normalises to `FVG_OR_BPR` → allowed
- **OB** (Order Block) → normalises to `OB_OR_SWAPZONE_IC_SND` → allowed
- **Breaker** → normalises to `OB_OR_SWAPZONE_IC_SND` → allowed

Threshold in `.env`: `ML_ACCEPT_THRESHOLD=0.50`

### Bug found — shadow pollution
`register_entry_gate_filtered_lead` and `register_entry_gate_filtered_option` call
`register_shadow_candidate(force=True)`. This bypasses `should_shadow_signal` guard
(`min_conf <= prob < threshold`). Result: signals with confidence=0.65 (above threshold=0.50)
end up in `shadow_labeled_setups.csv`. 285 of 1 143 shadow rows have `confidence ≥ accept_threshold`,
corrupting winrate analysis.

### Missing feature — confidence tiers
No way to express "this signal is 80% confident, execute it more aggressively." All above-threshold
signals are treated identically. Adding tiers:
- **NORMAL** (threshold ≤ prob < 0.65): standard lot, standard price-distance buffer
- **HIGH** (0.65 ≤ prob < 0.80): 1.5× lot multiplier recommendation, wider price-distance buffer
- **ULTRA** (prob ≥ 0.80): 2× lot multiplier recommendation, widest price-distance buffer

The lot multiplier is a recommendation embedded in the Telegram alert text. Actual MT5 lot size
is unchanged (that is the execution layer's responsibility). The scanner already has
`watch_too_far_execution_message` / price-distance logic — the tier relaxes its buffer.

---

## File Structure

| File | Change |
|------|--------|
| `src/live_trade_policy.py` | Add `confidence_tier(prob)` + `ConfidenceTier` constants |
| `src/scanner_worker.py` | (1) `force=False` fix; (2) use `confidence_tier` for lot-tier label in signal dict |
| `tests/test_live_trade_policy.py` | Tests for `confidence_tier` |
| `tests/test_scanner_shadow_signals.py` | Tests for shadow pollution fix |

---

## Task 1: Fix shadow pollution — `force=True` → `force=False`

**Files:**
- Modify: `src/scanner_worker.py` — functions `register_entry_gate_filtered_lead` (around line 1009) and `register_entry_gate_filtered_option` (around line 1069)
- Test: `tests/test_scanner_shadow_signals.py`

The two functions call `register_shadow_candidate(..., force=True)`.
Change both to `force=False` so `should_shadow_signal` guard applies:
only signals with `min_conf <= prob < threshold` get shadow-tracked.
Above-threshold blocked signals are logged only, not shadow-tracked.

- [ ] **Step 1: Write failing tests**

Create `tests/test_scanner_shadow_signals.py`:

```python
import json
import os
import pytest
from src.scanner_worker import register_entry_gate_filtered_lead, register_entry_gate_filtered_option


def _make_single_lead(prob, entry_price=1900.0, tp_price=1920.0, sl_price=1890.0):
    opt = {
        "time": "2026-07-01 10:00:00",
        "entry_price": entry_price,
        "tp_price": tp_price,
        "sl_price": sl_price,
        "direction": 1,
        "index": 0,
        "option_name": "test_opt",
        "features": {"timeframe": 60, "direction": 1},
        "filtered_reason": "strategy_not_allowlisted",
    }
    return {"is_dual": False, "opt": opt, "max_prob": prob}


def test_above_threshold_not_shadow_tracked(tmp_path):
    """Signal with confidence above accept_threshold must NOT be shadow-tracked."""
    shadow_file = str(tmp_path / "shadow_signals.json")
    sent_signals = {}
    lead = _make_single_lead(prob=0.70)  # above threshold=0.50

    register_entry_gate_filtered_lead(
        lead=lead,
        sent_signals=sent_signals,
        symbol="XAUUSD",
        timeframe="H1",
        strategy="IC",
        direction_name="SHORT",
        accept_threshold=0.50,
        shadow_signals_file=shadow_file,
    )

    if os.path.exists(shadow_file):
        with open(shadow_file) as f:
            data = json.load(f)
        assert data == {}, (
            f"Above-threshold signal (conf=0.70, threshold=0.50) must not be "
            f"shadow-tracked, but found: {list(data.keys())}"
        )


def test_below_threshold_is_shadow_tracked(tmp_path):
    """Signal with confidence below accept_threshold MUST be shadow-tracked."""
    shadow_file = str(tmp_path / "shadow_signals.json")
    sent_signals = {}
    lead = _make_single_lead(prob=0.40)  # below threshold=0.50

    register_entry_gate_filtered_lead(
        lead=lead,
        sent_signals=sent_signals,
        symbol="XAUUSD",
        timeframe="H1",
        strategy="IC",
        direction_name="SHORT",
        accept_threshold=0.50,
        shadow_signals_file=shadow_file,
    )

    assert os.path.exists(shadow_file), "Below-threshold signal must be shadow-tracked"
    with open(shadow_file) as f:
        data = json.load(f)
    assert len(data) == 1, f"Expected 1 shadow signal, got {len(data)}"


def test_at_threshold_not_shadow_tracked(tmp_path):
    """Signal exactly at threshold is above-or-equal, NOT below — must not be shadow-tracked."""
    shadow_file = str(tmp_path / "shadow_signals.json")
    sent_signals = {}
    lead = _make_single_lead(prob=0.50)  # exactly at threshold=0.50

    register_entry_gate_filtered_lead(
        lead=lead,
        sent_signals=sent_signals,
        symbol="XAUUSD",
        timeframe="H1",
        strategy="IC",
        direction_name="SHORT",
        accept_threshold=0.50,
        shadow_signals_file=shadow_file,
    )

    if os.path.exists(shadow_file):
        with open(shadow_file) as f:
            data = json.load(f)
        assert data == {}, (
            "Signal at exactly threshold=0.50 must not be shadow-tracked "
            "(shadow is strictly below threshold)"
        )
```

- [ ] **Step 2: Run to verify the bug exists (tests should FAIL)**

```bash
cd "C:/Users/WINDOWS 11 PRO/forex-smc-analyzer"
python -m pytest tests/test_scanner_shadow_signals.py -v
```

Expected: `test_above_threshold_not_shadow_tracked` and `test_at_threshold_not_shadow_tracked` FAIL because `force=True` bypasses the guard.

- [ ] **Step 3: Find and fix the two callers**

In `src/scanner_worker.py`, find `register_entry_gate_filtered_lead`. It contains a call to `register_shadow_candidate` with `force=True`. Change it to `force=False`:

```python
# BEFORE (around the existing call):
    return register_shadow_candidate(
        sig_key=sig_key,
        ...
        force=True,
        filtered_reason=filtered_reason,
    )

# AFTER:
    return register_shadow_candidate(
        sig_key=sig_key,
        ...
        force=False,
        filtered_reason=filtered_reason,
    )
```

Do the **same** change in `register_entry_gate_filtered_option` (the nearby function that also calls `register_shadow_candidate(force=True)`).

Do NOT change `register_shadow_candidate` itself. Only these two callers.

- [ ] **Step 4: Run tests — all three must PASS**

```bash
python -m pytest tests/test_scanner_shadow_signals.py -v
```

Expected: All 3 PASS.

- [ ] **Step 5: Run full suite to check for regressions**

```bash
python -m pytest tests/ -v --tb=short 2>&1 | tail -40
```

Expected: No new FAILs compared to before this change.

- [ ] **Step 6: Commit**

```bash
git add src/scanner_worker.py tests/test_scanner_shadow_signals.py
git commit -m "fix: stop shadow-tracking above-threshold strategy-blocked signals

register_entry_gate_filtered_lead and register_entry_gate_filtered_option
called register_shadow_candidate with force=True, bypassing the
should_shadow_signal guard. This caused signals with confidence above
accept_threshold to be written into shadow_labeled_setups.csv, polluting
shadow winrate analysis (285 of 1143 shadow rows were above-threshold).

Changed force=True -> force=False in both callers. Shadow tracking now
applies only to signals with min_conf <= prob < threshold, as intended."
```

---

## Task 2: Add confidence-tier system for priority execution

**Files:**
- Modify: `src/live_trade_policy.py` — add `ConfidenceTier` enum and `confidence_tier(prob)` function
- Modify: `src/scanner_worker.py` — attach tier label to outgoing signal dict
- Test: `tests/test_live_trade_policy.py`

**What the tier does:**
- Embeds a `confidence_tier` key in the signal dict sent to execution/Telegram
- The tier is `"NORMAL"`, `"HIGH"`, or `"ULTRA"`
- Thresholds come from env vars: `ML_HIGH_CONFIDENCE_TIER=0.65`, `ML_ULTRA_CONFIDENCE_TIER=0.80`
- The scanner uses the tier to include a lot-size recommendation label in the Telegram message text (e.g., `"⚡ HIGH CONFIDENCE — consider 1.5× lot"`)
- Does **not** change actual MT5 lot size (execution layer responsibility)
- Does **not** bypass strategy blocklist — IC/SND/Swapzone/Pivot remain blocked regardless of confidence

- [ ] **Step 1: Write failing tests in `tests/test_live_trade_policy.py`**

Create or append to `tests/test_live_trade_policy.py`:

```python
import os
import pytest
from src.live_trade_policy import confidence_tier


def test_tier_normal_below_high(monkeypatch):
    monkeypatch.setenv("ML_HIGH_CONFIDENCE_TIER", "0.65")
    monkeypatch.setenv("ML_ULTRA_CONFIDENCE_TIER", "0.80")
    assert confidence_tier(0.55) == "NORMAL"
    assert confidence_tier(0.50) == "NORMAL"
    assert confidence_tier(0.64) == "NORMAL"


def test_tier_high_between_thresholds(monkeypatch):
    monkeypatch.setenv("ML_HIGH_CONFIDENCE_TIER", "0.65")
    monkeypatch.setenv("ML_ULTRA_CONFIDENCE_TIER", "0.80")
    assert confidence_tier(0.65) == "HIGH"
    assert confidence_tier(0.70) == "HIGH"
    assert confidence_tier(0.799) == "HIGH"


def test_tier_ultra_at_and_above(monkeypatch):
    monkeypatch.setenv("ML_HIGH_CONFIDENCE_TIER", "0.65")
    monkeypatch.setenv("ML_ULTRA_CONFIDENCE_TIER", "0.80")
    assert confidence_tier(0.80) == "ULTRA"
    assert confidence_tier(0.95) == "ULTRA"
    assert confidence_tier(1.0) == "ULTRA"


def test_tier_none_returns_normal():
    """None or non-float probability defaults to NORMAL."""
    assert confidence_tier(None) == "NORMAL"
    assert confidence_tier("bad") == "NORMAL"


def test_tier_uses_env_thresholds(monkeypatch):
    """Tier boundaries are configurable via env."""
    monkeypatch.setenv("ML_HIGH_CONFIDENCE_TIER", "0.55")
    monkeypatch.setenv("ML_ULTRA_CONFIDENCE_TIER", "0.70")
    assert confidence_tier(0.55) == "HIGH"
    assert confidence_tier(0.70) == "ULTRA"
    assert confidence_tier(0.54) == "NORMAL"
```

- [ ] **Step 2: Run to verify they fail (function doesn't exist yet)**

```bash
python -m pytest tests/test_live_trade_policy.py -k "tier" -v
```

Expected: ImportError or NameError — `confidence_tier` not defined.

- [ ] **Step 3: Add `confidence_tier` to `src/live_trade_policy.py`**

Add after the existing `_read_float_env` and `_as_float` helpers (before `_setup_features`):

```python
_TIER_HIGH_DEFAULT = 0.65
_TIER_ULTRA_DEFAULT = 0.80


def confidence_tier(probability) -> str:
    """Return execution priority tier for a given ML probability.

    Returns 'ULTRA', 'HIGH', or 'NORMAL'.
    Thresholds are configurable via ML_HIGH_CONFIDENCE_TIER and
    ML_ULTRA_CONFIDENCE_TIER env vars.
    """
    prob = _as_float(probability)
    if prob is None:
        return "NORMAL"
    high_threshold = _read_float_env("ML_HIGH_CONFIDENCE_TIER", _TIER_HIGH_DEFAULT)
    ultra_threshold = _read_float_env("ML_ULTRA_CONFIDENCE_TIER", _TIER_ULTRA_DEFAULT)
    if prob >= ultra_threshold:
        return "ULTRA"
    if prob >= high_threshold:
        return "HIGH"
    return "NORMAL"
```

- [ ] **Step 4: Run tier tests — all must PASS**

```bash
python -m pytest tests/test_live_trade_policy.py -k "tier" -v
```

Expected: All 5 PASS.

- [ ] **Step 5: Wire confidence tier into scanner signal dict**

In `src/scanner_worker.py`, find where a new signal is written to `sent_signals` after passing the strategy check (the block that sets keys like `'time_sent'`, `'probability'`, `'ticket_id'`, etc.).

Add one import at the top of the file (with the other live_trade_policy imports):
```python
from src.live_trade_policy import confidence_tier, should_allow_live_strategy
```
(If `should_allow_live_strategy` is already imported, add only `confidence_tier` to the same import line.)

Then in each place where a new **high-confidence** signal dict is built (search for `'is_low_confidence'` absent — these are the accepted signal dicts), add:

```python
'confidence_tier': confidence_tier(prob),
```

There are multiple signal-creation sites (single-leg, dual-leg options). Add the key in **all** of them. The value is always `confidence_tier(lead['max_prob'])` or `confidence_tier(prob)` depending on context.

- [ ] **Step 6: Add `.env` variables**

Append to `.env`:
```
ML_HIGH_CONFIDENCE_TIER=0.65
ML_ULTRA_CONFIDENCE_TIER=0.80
```

- [ ] **Step 7: Run full suite**

```bash
python -m pytest tests/ -v --tb=short 2>&1 | tail -40
```

Expected: No new FAILs.

- [ ] **Step 8: Commit**

```bash
git add src/live_trade_policy.py src/scanner_worker.py tests/test_live_trade_policy.py .env
git commit -m "feat: add confidence-tier system (NORMAL/HIGH/ULTRA) to signal flow

Signals at 65%+ confidence are tagged HIGH; 80%+ are tagged ULTRA.
The tier is stored as confidence_tier in the signal dict for use by
execution and alerting layers.

Thresholds are configurable via ML_HIGH_CONFIDENCE_TIER (default 0.65)
and ML_ULTRA_CONFIDENCE_TIER (default 0.80).

Strategy blocklist is unchanged — IC/SND/Swapzone/Pivot remain blocked
regardless of confidence tier."
```

---

## Self-Review

**Spec coverage:**
- "threshold 50%, 60% should enter" — Task 1 removes the data pollution; Task 2 makes 60%+ signals visibly prioritised
- "80-90% even 100% should enter" — Task 2 ULTRA tier marks these clearly
- "IC gw blok" — IC stays blocked; no changes to strategy policy
- Shadow winrate integrity — Task 1 fixes the force=True pollution

**Placeholder scan:** None.

**Type consistency:**
- `confidence_tier(probability) -> str` — uses existing `_as_float` pattern
- `force=False` — same type, just corrected value
- `'confidence_tier': confidence_tier(prob)` — str value, consistent

**Potential risk:** After Task 1, signals that were previously force-shadow-tracked (above-threshold strategy-blocked) will no longer appear in shadow data. This is the correct behaviour but means `shadow_labeled_setups.csv` will accumulate fewer rows going forward — only genuinely below-threshold signals will be tracked.
