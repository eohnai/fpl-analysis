from fastapi import FastAPI
from app.routers import suggest_transfers, webhook
from dotenv import load_dotenv
import os
import requests
import logging
from contextlib import asynccontextmanager

logger = logging.getLogger("fastapi")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events."""
    # check if TELEGRAM_BOT_TOKEN is set
    if not TELEGRAM_BOT_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN is not set. Exiting.")
        raise ValueError("TELEGRAM_BOT_TOKEN is not set in environment variables.")
    
    # startup: register webhook with Telegram using ngrok URL
    try:
        tunnels = requests.get("http://localhost:4040/api/tunnels").json()
        public_url = None
        for tunnel in tunnels["tunnels"]:
            if tunnel["proto"] == "https":
                public_url = tunnel["public_url"]
                break

        if not public_url:
            logger.warning("No ngrok HTTPS tunnel found. Make sure you ran ngrok first.")
        else:
            webhook_url = f"{public_url}/webhook"
            response = requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/setWebhook",
                params={"url": webhook_url}
            )
            if response.status_code == 200:
                logger.info(f"Telegram webhook set: {webhook_url}")
            else:
                logger.warning(f"Failed to set webhook: {response.text}")
    except Exception as e:
        logger.error(f"Error setting webhook: {e}")

    yield

    # shutdown: remove webhook
    try:
        response = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/deleteWebhook"
        )
        if response.status_code == 200:
            logger.info("Telegram webhook deleted.")
        else:
            logger.warning(f"Failed to delete webhook: {response.text}")
    except Exception as e:
        logger.error(f"Error deleting webhook: {e}")

app = FastAPI(lifespan=lifespan)

app.include_router(suggest_transfers.router)
app.include_router(webhook.router)

@app.get("/")
def root():
    return {"message": "FPL Agent is running!"}
