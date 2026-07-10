import os

import pytest

from src.scanner_worker import _pid_is_running, _scanner_lock_file, scanner_instance_lock


def test_pid_is_running_handles_current_and_invalid_pid():
    assert _pid_is_running(os.getpid()) is True
    assert _pid_is_running(-1) is False
    assert _pid_is_running("not-a-pid") is False


def test_scanner_instance_lock_blocks_second_worker_for_same_symbol_and_magic(tmp_path):
    with scanner_instance_lock("XAUUSD", 202606, lock_dir=str(tmp_path)):
        with pytest.raises(RuntimeError, match="already running"):
            with scanner_instance_lock("XAUUSD", 202606, lock_dir=str(tmp_path)):
                pass


def test_scanner_instance_lock_releases_lock_file_on_exit(tmp_path):
    with scanner_instance_lock("XAUUSD", 202606, lock_dir=str(tmp_path)) as lock_path:
        assert os.path.exists(lock_path)

    assert not os.path.exists(lock_path)


def test_scanner_instance_lock_replaces_stale_dead_pid_lock(tmp_path):
    lock_path = _scanner_lock_file("XAUUSD", 202606, lock_dir=str(tmp_path))
    with open(lock_path, "w", encoding="utf-8") as lock_file:
        lock_file.write("999999999")

    with scanner_instance_lock("XAUUSD", 202606, lock_dir=str(tmp_path)) as acquired_path:
        assert acquired_path == lock_path
        with open(lock_path, "r", encoding="utf-8") as lock_file:
            assert lock_file.read().strip() == str(os.getpid())

    assert not os.path.exists(lock_path)
