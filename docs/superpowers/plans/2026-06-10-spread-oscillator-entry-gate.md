# Spread Oscillator Entry Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a stricter but still frequent-entry live gate that keeps FVG/BPR eligible, normalizes Exness XAUUSD spread, and blocks counter-oscillator entries before MT5 execution.

**Architecture:** Add a focused `src/entry_quality_gate.py` module for spread, RSI 8, Stoch RSI (9,3,3), and strategy threshold decisions. Integrate it in `src/scanner_worker.py` immediately before live order placement and route blocked high-confidence candidates into shadow tracking with explicit reasons.

**Tech Stack:** Python, pandas, pytest, existing scanner/shadow tracker modules.

---

### Task 1: Entry Gate Unit Tests

**Files:**
- Create: `tests/test_entry_quality_gate.py`
- Create: `src/entry_quality_gate.py`

- [ ] Write tests for Exness point-to-price conversion, live spread context, FVG/BPR eligibility, spread risk rejection, and RSI/Stoch RSI direction guards.
- [ ] Run `python -m pytest tests\test_entry_quality_gate.py -q` and confirm import failure before implementation.
- [ ] Implement the minimal gate module to satisfy the tests.
- [ ] Re-run `python -m pytest tests\test_entry_quality_gate.py -q`.

### Task 2: Shadow Reason Tests

**Files:**
- Modify: `tests/test_scanner_shadow_signals.py`
- Modify: `src/shadow_tracker.py`
- Modify: `src/scanner_worker.py`

- [ ] Add a failing test proving high-confidence, entry-gate-filtered candidates are stored in shadow with `filtered_reason`.
- [ ] Add `force` and `filtered_reason` parameters to scanner/shadow helpers.
- [ ] Re-run `python -m pytest tests\test_scanner_shadow_signals.py -q`.

### Task 3: Scanner Integration

**Files:**
- Modify: `src/scanner_worker.py`

- [ ] Import the entry gate helpers.
- [ ] Evaluate gate decisions after confidence and existing FVG quality filters, before `execute_market_order_for_setup()` or `execute_trade_for_setup()`.
- [ ] Keep FVG and BPR live-eligible at the stricter base threshold when spread/oscillator context is valid.
- [ ] Route blocked high-confidence candidates to shadow with reason.
- [ ] Run scanner decision, market order, and shadow tests.

### Task 4: Verification And Docs

**Files:**
- Modify: `docs/superpowers/PROGRESS_ml-confidence-predictive.md`

- [ ] Run targeted tests and `py_compile`.
- [ ] Update progress notes with the new gate behavior and honest limitation: this improves filtering but does not guarantee profit.
