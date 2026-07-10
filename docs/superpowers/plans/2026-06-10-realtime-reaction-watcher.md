# Realtime Reaction Watcher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fast tick watcher so already-known valid setups can react with market entry and active SL protection between full closed-candle scanner cycles.

**Architecture:** Keep the main SMC/FVG/BPR detector closed-candle based. Add `src.realtime_reaction_watcher` for pure tick-reaction decisions and registry updates, then let `scanner_worker --realtime-reaction` run lightweight tick passes between normal full scans.

**Tech Stack:** Python, MetaTrader5 tick API, pytest, existing scanner/execution registry.

---

### Task 1: Pure Realtime Reaction Decisions

**Files:**
- Create: `src/realtime_reaction_watcher.py`
- Create: `tests/test_realtime_reaction_watcher.py`

- [ ] Write failing tests for BUY and SELL reaction ticks inside the entry zone.
- [ ] Write failing tests that reject falling/noisy ticks inside the zone.
- [ ] Implement `should_enter_on_realtime_reaction()`.
- [ ] Run `python -m pytest tests\test_realtime_reaction_watcher.py -q`.

### Task 2: Registry Pass For Market Execution

**Files:**
- Modify: `src/realtime_reaction_watcher.py`
- Modify: `tests/test_realtime_reaction_watcher.py`

- [ ] Write failing tests proving a no-ticket single registry record can trigger a market order on a valid reaction tick.
- [ ] Write failing tests proving an existing-ticket record is not duplicated.
- [ ] Implement candidate extraction and `run_realtime_reaction_pass()`.
- [ ] Run `python -m pytest tests\test_realtime_reaction_watcher.py -q`.

### Task 3: Scanner Loop Integration

**Files:**
- Modify: `src/scanner_worker.py`
- Modify: `docs/superpowers/PROGRESS_ml-confidence-predictive.md`

- [ ] Add CLI flags `--realtime-reaction` and `--tick-interval`.
- [ ] Between full scans, run a lightweight tick loop that calls `run_realtime_reaction_pass()` and `manage_active_trades(symbol, magic, {})`.
- [ ] Preserve closed-candle full scan behavior.
- [ ] Run targeted scanner tests, compile, and full suite.
