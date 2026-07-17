import os
import sys
import dotenv

# 1. Mock load_dotenv globally before any code imports it, so local .env is not loaded
dotenv.load_dotenv = lambda *args, **kwargs: False

# 2. Clear all MT5_ and ML_ configuration variables from the environment to ensure test isolation
saved_env = {}
for key in list(os.environ.keys()):
    if key.startswith("MT5_") or key.startswith("ML_"):
        saved_env[key] = os.environ[key]
        del os.environ[key]

# Provide a fixture to restore them if needed, though pytest runs in a single process
import pytest

@pytest.fixture(scope="session", autouse=True)
def restore_env_after_session():
    yield
    # Restore variables after the entire test session is complete
    for key, val in saved_env.items():
        os.environ[key] = val
