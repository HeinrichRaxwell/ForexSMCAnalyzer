import os
import pytest

@pytest.fixture(autouse=True)
def clean_test_env(monkeypatch):
    """Ensure environment variables don't bleed from the local .env into tests."""
    monkeypatch.setenv("MT5_ENFORCE_SPREAD_FILTER", "False")
    monkeypatch.setenv("MT5_ENFORCE_ENTRY_GATE", "False")
