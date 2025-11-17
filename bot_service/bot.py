import logging
import requests
import os
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
MODEL_SERVICE_URL = "http://model_service:5001/predict"

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO)
logger = logging.getLogger(__name__)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text('Send an image, please')


async def classify_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    photo_file_id = update.message.photo[-1].file_id

    photo_file = await context.bot.get_file(photo_file_id)

    image_bytes = await photo_file.download_as_bytearray()

    files = {'image': image_bytes}

    try:
        response = requests.post(MODEL_SERVICE_URL, files=files)
        if response.status_code == 200:
            prediction = response.json().get("prediction")
            await update.message.reply_text(f'Classification result: {prediction}')
        else:
            await update.message.reply_text(
                f'Model service error: {response.status_code}. Check log.')
    except requests.exceptions.RequestException as e:
        logger.error(f"Error with connection to service model: {e}")
        await update.message.reply_text('Error with connection to service model.')


def main():
    if not TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN environment is not set")
        return

    application = Application.builder().token(TOKEN).build()

    application.add_handler(CommandHandler("start", start))

    application.add_handler(MessageHandler(filters.PHOTO & ~filters.COMMAND, classify_image))

    application.run_polling(poll_interval=3.0)


if __name__ == "__main__":
    main()
