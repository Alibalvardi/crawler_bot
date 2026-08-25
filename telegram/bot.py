import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from telegram.request import HTTPXRequest


load_dotenv(Path(__file__).with_name(".env"))

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_PROXY = os.getenv("TELEGRAM_PROXY", "").strip() or None
DEFAULT_GENERATION_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")


EMBEDDING_MODELS = [
    item.strip()
    for item in os.getenv(
        "EMBEDDING_MODELS",
        "text-embedding-3-small,text-embedding-3-large",
    ).split(",")
    if item.strip()
]
GENERATION_MODELS = [
    item.strip()
    for item in os.getenv(
        "GENERATION_MODELS",
        "gpt-5,gpt-5-mini,gpt-4.1-mini",
    ).split(",")
    if item.strip()
]
if DEFAULT_GENERATION_MODEL not in GENERATION_MODELS:
    GENERATION_MODELS.insert(0, DEFAULT_GENERATION_MODEL)

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN is not set.")

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def main_menu() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        [
            ["شروع مکالمه"],
            ["مدل های embedding", "مدل های generation"],
            ["انتخاب عمق crawler", "تعداد صفحات crawler"],
        ],
        resize_keyboard=True,
    )


def conversation_menu() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup([["پایان مکالمه"]], resize_keyboard=True)


def model_keyboard(kind: str) -> InlineKeyboardMarkup:
    models = EMBEDDING_MODELS if kind == "embedding" else GENERATION_MODELS
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton(model, callback_data=f"model:{kind}:{model}")]
            for model in models
        ]
    )


def value_keyboard(kind: str) -> InlineKeyboardMarkup:
    values = {
        "depth": [1, 2, 3, 4, 5],
        "pages": [1, 3, 5, 10, 20, 50],
    }[kind]
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    str(value),
                    callback_data=f"config:{kind}:{value}",
                )
            ]
            for value in values
        ]
    )


def user_config(context: ContextTypes.DEFAULT_TYPE) -> dict[str, Any]:
    config = context.user_data.setdefault(
        "config",
        {
            "embedding_model": EMBEDDING_MODELS[0],
            "generation_model": DEFAULT_GENERATION_MODEL,
            "crawler_depth": 2,
            "crawler_pages": 5,
        },
    )
    return config


def clear_conversation(context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data["state"] = "idle"
    context.user_data.pop("site", None)
    context.user_data.pop("history", None)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    clear_conversation(context)
    await update.message.reply_text(
        "سلام! از منوی زیر یک گزینه را انتخاب کن.",
        reply_markup=main_menu(),
    )


async def show_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    clear_conversation(context)
    await update.message.reply_text("منوی اصلی:", reply_markup=main_menu())


async def send_long_message(update: Update, text: str) -> None:
    # Telegram messages have a size limit; split without losing the answer.
    for start_index in range(0, len(text), 4000):
        await update.message.reply_text(text[start_index : start_index + 4000])


async def begin_conversation(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    context.user_data["state"] = "waiting_for_site"
    context.user_data["history"] = []
    await update.message.reply_text(
        "آدرس سایت را بفرست.\n"
        "توجه: در این نسخه crawler به ربات وصل نیست و آدرس سایت فقط برای "
        "زمینه پاسخ‌گویی استفاده می‌شود.",
        reply_markup=conversation_menu(),
    )


async def ask_embedding_models(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    selected = user_config(context)["embedding_model"]
    await update.message.reply_text(
        f"مدل embedding را انتخاب کن.\nمدل فعلی: {selected}",
        reply_markup=model_keyboard("embedding"),
    )


async def ask_generation_models(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    selected = user_config(context)["generation_model"]
    await update.message.reply_text(
        f"مدل generation را انتخاب کن.\nمدل فعلی: {selected}",
        reply_markup=model_keyboard("generation"),
    )


async def ask_depth(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    selected = user_config(context)["crawler_depth"]
    await update.message.reply_text(
        f"عمق crawler را انتخاب کن.\nمقدار فعلی: {selected}",
        reply_markup=value_keyboard("depth"),
    )


async def ask_pages(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    selected = user_config(context)["crawler_pages"]
    await update.message.reply_text(
        f"تعداد صفحات crawler را انتخاب کن.\nمقدار فعلی: {selected}",
        reply_markup=value_keyboard("pages"),
    )


async def answer_query(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    site = context.user_data["site"]
    config = user_config(context)
    history: list[dict[str, str]] = context.user_data.setdefault("history", [])
    query = update.message.text.strip()

    history.append(
        {
            "role": "user",
            "content": (
                f"Website: {site}\n"
                f"Selected embedding model (not active yet): {config['embedding_model']}\n"
                f"Configured crawler depth (not active yet): {config['crawler_depth']}\n"
                f"Configured crawler pages (not active yet): {config['crawler_pages']}\n"
                f"User query: {query}"
            ),
        }
    )
    history[:] = history[-500:]

    try:
        await update.message.chat.send_action(ChatAction.TYPING)


        answer = "مدل پاسخ متنی برنگرداند؛ دوباره تلاش کن."

        await send_long_message(update, answer)
        await update.message.reply_text(
            "query بعدی را بفرست یا «پایان مکالمه» را بزن.",
            reply_markup=conversation_menu(),
        )
    except Exception:
        logger.exception("Generation failed for Telegram user %s", update.effective_user.id)
        await update.message.reply_text(
            "در ارتباط با مدل مشکلی پیش آمد. مدل انتخابی یا کلید API را بررسی کن.",
            reply_markup=conversation_menu(),
        )


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return

    text = update.message.text.strip()
    state = context.user_data.get("state", "idle")

    if text == "شروع مکالمه":
        await begin_conversation(update, context)
    elif text == "مدل های embedding":
        await ask_embedding_models(update, context)
    elif text == "مدل های generation":
        await ask_generation_models(update, context)
    elif text == "انتخاب عمق crawler":
        await ask_depth(update, context)
    elif text == "تعداد صفحات crawler":
        await ask_pages(update, context)
    elif text == "پایان مکالمه":
        clear_conversation(context)
        await update.message.reply_text(
            "مکالمه پایان یافت و به منوی اصلی برگشتی.",
            reply_markup=main_menu(),
        )
    elif state == "waiting_for_site":
        context.user_data["site"] = text
        context.user_data["state"] = "waiting_for_query"
        await update.message.reply_text(
            "سایت دریافت شد. حالا query را بفرست.",
            reply_markup=conversation_menu(),
        )
    elif state == "waiting_for_query":
        await answer_query(update, context)
    else:
        await update.message.reply_text(
            "از منوی زیر یک گزینه را انتخاب کن.",
            reply_markup=main_menu(),
        )


async def handle_callback(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    query = update.callback_query
    await query.answer()
    data = query.data or ""
    config = user_config(context)

    if data.startswith("model:"):
        _, kind, model = data.split(":", 2)
        allowed = EMBEDDING_MODELS if kind == "embedding" else GENERATION_MODELS
        if model in allowed:
            config[f"{kind}_model"] = model
            await query.edit_message_text(f"مدل {kind} روی «{model}» تنظیم شد.")
    elif data.startswith("config:"):
        _, kind, value = data.split(":", 2)
        config_key = "crawler_depth" if kind == "depth" else "crawler_pages"
        config[config_key] = int(value)
        label = "عمق crawler" if kind == "depth" else "تعداد صفحات crawler"
        await query.edit_message_text(f"{label} روی {value} تنظیم شد.")

    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="منوی اصلی:",
        reply_markup=main_menu(),
    )


def build_application() -> Application:
    builder = Application.builder().token(TELEGRAM_BOT_TOKEN)
    if TELEGRAM_PROXY:
        builder = builder.request(
            HTTPXRequest(proxy=TELEGRAM_PROXY)
        ).get_updates_request(
            HTTPXRequest(proxy=TELEGRAM_PROXY)
        )
        logger.info("Using Telegram proxy %s", TELEGRAM_PROXY)

    application = builder.build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("menu", show_menu))
    application.add_handler(CallbackQueryHandler(handle_callback))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text)
    )
    return application


if __name__ == "__main__":
    logger.info("Starting Telegram bot")
    build_application().run_polling(allowed_updates=Update.ALL_TYPES)
