import pytest
from unittest.mock import patch, MagicMock
import os
from src.telegram_bot import (
    PHOTO_ALERT_CAPTION,
    send_telegram_alert,
    TELEGRAM_MAX_CAPTION_LEN,
    TELEGRAM_MAX_MESSAGE_LEN,
)


@pytest.fixture(autouse=True)
def disable_delivery_journal(monkeypatch):
    """Keep unit-test alert fixtures out of the local runtime audit journal."""
    monkeypatch.setenv("TELEGRAM_EVENT_LOG_ENABLED", "False")
    monkeypatch.delenv("TELEGRAM_EVENT_LOG_PATH", raising=False)


@patch('src.telegram_bot.requests.post')
def test_telegram_alert_records_secret_free_delivery_event(mock_post, monkeypatch, tmp_path):
    mock_post.return_value = MagicMock(status_code=200)
    journal = tmp_path / "telegram_delivery_events.jsonl"
    monkeypatch.setenv("TELEGRAM_EVENT_LOG_ENABLED", "True")
    monkeypatch.setenv("TELEGRAM_EVENT_LOG_PATH", str(journal))
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "12345:abcde")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "98765")

    assert send_telegram_alert("<b>WatchZone</b> Ticket #1") is True

    event = __import__("json").loads(journal.read_text(encoding="utf-8"))
    assert event["channel"] == "text"
    assert event["delivered"] is True
    assert event["message"] == "<b>WatchZone</b> Ticket #1"
    assert "12345:abcde" not in journal.read_text(encoding="utf-8")

def test_telegram_alert_placeholder_fails():
    """Verify that using placeholder token/chat_id returns False and warns user."""
    with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "YOUR_TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID": "YOUR_CHAT_ID"}):
        res = send_telegram_alert("Test message")
        assert res is False

@patch('src.telegram_bot.requests.post')
def test_telegram_alert_text_success(mock_post):
    """Test successful text alert message dispatch."""
    # Mock requests response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_post.return_value = mock_response
    
    with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "12345:abcde", "TELEGRAM_CHAT_ID": "98765"}):
        res = send_telegram_alert("Test message")
        assert res is True
        mock_post.assert_called_once_with(
            "https://api.telegram.org/bot12345:abcde/sendMessage",
            data={'chat_id': '98765', 'text': 'Test message', 'parse_mode': 'HTML', 'disable_web_page_preview': True},
            timeout=15
        )

@patch('src.telegram_bot.requests.post')
def test_telegram_alert_photo_success(mock_post, tmp_path):
    """Test successful image dispatch sends a clean chart caption plus full details."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_post.return_value = mock_response
    
    # Create dummy image file
    dummy_img = tmp_path / "chart.png"
    dummy_img.write_bytes(b"dummy image data")
    
    with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "12345:abcde", "TELEGRAM_CHAT_ID": "98765"}):
        res = send_telegram_alert("Test message", image_path=str(dummy_img))
        assert res is True
        assert mock_post.call_count == 2
        args, kwargs = mock_post.call_args_list[0]
        assert args[0] == "https://api.telegram.org/bot12345:abcde/sendPhoto"
        assert kwargs['data']['chat_id'] == '98765'
        assert kwargs['data']['caption'] == PHOTO_ALERT_CAPTION
        assert kwargs['data']['parse_mode'] == 'HTML'
        assert 'photo' in kwargs['files']
        text_args, text_kwargs = mock_post.call_args_list[1]
        assert text_args[0] == "https://api.telegram.org/bot12345:abcde/sendMessage"
        assert text_kwargs['data']['text'] == "Test message"

@patch('src.telegram_bot.requests.post')
def test_telegram_alert_long_photo_keeps_chart_and_sends_full_text(mock_post, tmp_path):
    """Long alerts should not lose the chart because Telegram captions are shorter than messages."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_post.return_value = mock_response

    dummy_img = tmp_path / "chart.png"
    dummy_img.write_bytes(b"dummy image data")
    long_message = "<b>Alert</b>\n" + ("Detailed line with setup context.\n" * 80)

    with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "12345:abcde", "TELEGRAM_CHAT_ID": "98765"}):
        res = send_telegram_alert(long_message, image_path=str(dummy_img))

    assert res is True
    assert mock_post.call_count == 2
    photo_args, photo_kwargs = mock_post.call_args_list[0]
    text_args, text_kwargs = mock_post.call_args_list[1]

    assert photo_args[0] == "https://api.telegram.org/bot12345:abcde/sendPhoto"
    assert photo_kwargs['data']['caption'] == PHOTO_ALERT_CAPTION
    assert len(photo_kwargs['data']['caption']) <= TELEGRAM_MAX_CAPTION_LEN
    assert text_args[0] == "https://api.telegram.org/bot12345:abcde/sendMessage"
    assert text_kwargs['data']['text'] == long_message

@patch('src.telegram_bot.requests.post')
def test_telegram_alert_text_splits_over_telegram_message_limit(mock_post):
    """Text alerts above Telegram's message limit should be split into multiple sends."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_post.return_value = mock_response

    long_message = "line\n" * ((TELEGRAM_MAX_MESSAGE_LEN // 5) + 50)

    with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "12345:abcde", "TELEGRAM_CHAT_ID": "98765"}):
        res = send_telegram_alert(long_message)

    assert res is True
    assert mock_post.call_count >= 2
    for call in mock_post.call_args_list:
        _, kwargs = call
        assert len(kwargs['data']['text']) <= TELEGRAM_MAX_MESSAGE_LEN
