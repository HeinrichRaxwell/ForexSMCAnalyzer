import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

TELEGRAM_MAX_CAPTION_LEN = 1024
TELEGRAM_MAX_MESSAGE_LEN = 4096
PHOTO_ALERT_CAPTION = (
    "<b>Forex SMC AI Analyzer</b>\n"
    "Chart snapshot attached. Full signal details follow below."
)
LONG_ALERT_PHOTO_CAPTION = PHOTO_ALERT_CAPTION

def _split_message(message: str, max_len: int = TELEGRAM_MAX_MESSAGE_LEN) -> list:
    if len(message) <= max_len:
        return [message]

    chunks = []
    current = []
    current_len = 0

    for line in message.splitlines(keepends=True):
        if len(line) > max_len:
            if current:
                chunks.append("".join(current))
                current = []
                current_len = 0
            for start in range(0, len(line), max_len):
                chunks.append(line[start:start + max_len])
            continue

        if current_len + len(line) > max_len:
            chunks.append("".join(current))
            current = [line]
            current_len = len(line)
        else:
            current.append(line)
            current_len += len(line)

    if current:
        chunks.append("".join(current))

    return chunks

def _send_text_messages(token: str, chat_id: str, message: str) -> bool:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    chunks = _split_message(message)

    for idx, chunk in enumerate(chunks, start=1):
        try:
            data = {
                'chat_id': chat_id,
                'text': chunk,
                'parse_mode': 'HTML',
                'disable_web_page_preview': True
            }
            res = requests.post(url, data=data, timeout=15)
            if res.status_code != 200:
                print(f"[Telegram Bot] Error sending text message part {idx}/{len(chunks)} (HTTP {res.status_code}): {res.text}")
                return False
        except Exception as e:
            print(f"[Telegram Bot] Exception sending text message part {idx}/{len(chunks)}: {e}")
            return False

    print("[Telegram Bot] Alert text message sent successfully.")
    return True

def send_telegram_alert(message: str, image_path: str = None) -> bool:
    """
    Send a message and optional chart image to the configured Telegram chat.
    
    Args:
        message (str): The text message to send (supports Telegram HTML).
        image_path (str): Optional path to the chart image file.
        
    Returns:
        bool: True if sent successfully, False otherwise.
    """
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    
    if not token or not chat_id or token.startswith("YOUR_") or chat_id.startswith("YOUR_"):
        print("[Telegram Bot] Warning: Telegram credentials not set or placeholder used in .env file.")
        return False
        
    # Send image if path is provided and exists
    if image_path and os.path.exists(image_path):
        url = f"https://api.telegram.org/bot{token}/sendPhoto"
        try:
            with open(image_path, 'rb') as img:
                files = {'photo': img}
                data = {
                    'chat_id': chat_id,
                    'caption': PHOTO_ALERT_CAPTION,
                    'parse_mode': 'HTML'  # Use HTML parsing for robust tag styling
                }
                res = requests.post(url, data=data, files=files, timeout=20)
                if res.status_code == 200:
                    print("[Telegram Bot] Alert sent successfully with chart image.")
                    return _send_text_messages(token, chat_id, message)
                else:
                    print(f"[Telegram Bot] Error sending photo (HTTP {res.status_code}): {res.text}")
        except Exception as e:
            print(f"[Telegram Bot] Exception sending photo: {e}")
            
    # Fallback to plain text message if no image is provided or sending image failed
    return _send_text_messages(token, chat_id, message)
