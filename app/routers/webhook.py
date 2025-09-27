from fastapi import APIRouter
import requests
import os
import logging

router = APIRouter()
logger = logging.getLogger("webhook")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

@router.post("/webhook")
async def telegram_webhook(update: dict):
    try: 
        chat_id = update["message"]["chat"]["id"]
        text = update["message"].get("text", "")

        # TODO: Call orchestrator agent here
        reply_text = f"You said: {text}"

        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": chat_id, "text": reply_text}
        )
        logger.info(f"Received message: {text} from chat_id: {chat_id}")

    except Exception as e:
        logger.error(f"Error processing webhook: {e}")

    return {"ok": True}
