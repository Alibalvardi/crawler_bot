import asyncio
import logging
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv
from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    ReplyKeyboardMarkup,
    Update,
)
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from database import Database
from crawler import acrawl, to_absolute_url
from chunker import Chunk, chunk_crawl_pages
from embedder import embed_texts
from generator import (
    DEFAULT_BASE_URL as DEFAULT_GENERATION_BASE_URL,
    build_prompt,
    generate_answer,
)
from vector_store import (
    delete_user_collection,
    retrieve_user_chunks,
    save_user_chunks,
)


load_dotenv(Path(__file__).with_name(".env"))

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_PROXY = os.getenv("TELEGRAM_PROXY", "").strip() or None
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GENERATION_BASE_URL = os.getenv(
    "GENERATION_BASE_URL",
    DEFAULT_GENERATION_BASE_URL,
)
DEFAULT_GENERATION_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "20"))
CRAWL_CONCURRENCY = int(os.getenv("CRAWL_CONCURRENCY", "8"))
CRAWL_DELAY = float(os.getenv("CRAWL_DELAY", "0.1"))
CRAWL_TIMEOUT = float(os.getenv("CRAWL_TIMEOUT", "30"))
CRAWL_RETRIES = int(os.getenv("CRAWL_RETRIES", "1"))
CRAWL_VERIFY_SSL = os.getenv("CRAWL_VERIFY_SSL", "true").lower() not in {
    "0",
    "false",
    "no",
}
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
MIN_CHUNK_CHARS = int(os.getenv("MIN_CHUNK_CHARS", "30"))
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "32"))
RETRIEVE_TOP_K = int(os.getenv("RETRIEVE_TOP_K", "10"))
VECTOR_DB_PATH = os.getenv(
    "VECTOR_DB_PATH",
    str(PROJECT_ROOT / "data" / "vector_db"),
)

DEPTH_SETTING = "تنظیم عمق خزنده"
PAGES_SETTING = "تعداد صفحات خزنده"
TOP_K_SETTING = "تعداد نتایج بازیابی"
CONCURRENCY_SETTING = "تعداد درخواست‌های هم‌زمان خزنده"
BATCH_SIZE_SETTING = "اندازه دستهٔ بردارسازی"

EMBEDDING_MODELS = [
    item.strip()
    for item in os.getenv(
        "EMBEDDING_MODELS",
        "text-embedding-3-small,text-embedding-3-large",
    ).split(",")
    if item.strip()
]
if "local" not in EMBEDDING_MODELS:
    EMBEDDING_MODELS.insert(0, "local")
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
database = Database()
user_vector_locks: dict[int, asyncio.Lock] = {}
user_pipeline_locks: dict[int, asyncio.Lock] = {}
user_query_locks: dict[int, asyncio.Lock] = {}
user_cleanup_tasks: dict[int, asyncio.Task] = {}


def rtl_text(text: str) -> str:
    lines = []
    for line in str(text).splitlines(keepends=True):
        body = line.rstrip("\r\n")
        ending = line[len(body):]
        body = re.sub(
            r"(?<![\u2066])([A-Za-z0-9][A-Za-z0-9_./:?&=%+#@~,-]*)(?![\u2069])",
            lambda match: f"\u2066{match.group(1)}\u2069",
            body,
        )
        lines.append(f"\u200f{body}{ending}")
    return "".join(lines)


async def reply_rtl(update: Update, text: str, **kwargs: Any) -> Any:
    return await update.message.reply_text(rtl_text(text), **kwargs)


def main_menu() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        [
            ["شروع مکالمه"],
            ["مدل های embedding", "مدل های generation"],
            ["تنظیمات"],
        ],
        resize_keyboard=True,
    )


def settings_menu() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        [
            [DEPTH_SETTING, PAGES_SETTING],
            [TOP_K_SETTING, CONCURRENCY_SETTING],
            [BATCH_SIZE_SETTING],
            ["بازگشت به منوی اصلی"],
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


def default_settings() -> dict[str, Any]:
    return {
        "embedding_model": "local",
        "generation_model": DEFAULT_GENERATION_MODEL,
        "crawler_depth": 5,
        "crawler_pages": 100,
        "retrieve_top_k": RETRIEVE_TOP_K,
        "crawl_concurrency": CRAWL_CONCURRENCY,
        "embed_batch_size": EMBED_BATCH_SIZE,
    }


def ensure_user(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user = update.effective_user
    if user is None:
        raise RuntimeError("Telegram user is missing.")
    database.upsert_user(user)
    context.user_data["telegram_id"] = user.id
    return user.id


def user_config(context: ContextTypes.DEFAULT_TYPE) -> dict[str, Any]:
    telegram_id = context.user_data["telegram_id"]
    config = database.get_settings(telegram_id, default_settings())
    context.user_data["config"] = config
    return config


def clear_conversation(context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data["state"] = "idle"
    context.user_data.pop("site", None)
    context.user_data.pop("history", None)
    context.user_data.pop("crawl_result", None)
    context.user_data.pop("crawl_chunks", None)
    context.user_data.pop("embeddings", None)
    context.user_data.pop("query_count", None)


def raise_if_cancelled(cancel_event: threading.Event) -> None:
    if cancel_event.is_set():
        raise asyncio.CancelledError


def get_user_vector_lock(telegram_id: int) -> asyncio.Lock:
    return user_vector_locks.setdefault(telegram_id, asyncio.Lock())


def get_user_pipeline_lock(telegram_id: int) -> asyncio.Lock:
    return user_pipeline_locks.setdefault(telegram_id, asyncio.Lock())


def get_user_query_lock(telegram_id: int) -> asyncio.Lock:
    return user_query_locks.setdefault(telegram_id, asyncio.Lock())


async def cancel_background_crawl(context: ContextTypes.DEFAULT_TYPE) -> None:
    task = context.user_data.pop("crawl_task", None)
    cancel_event = context.user_data.pop("pipeline_cancel_event", None)
    await stop_pipeline_task(task, cancel_event)


async def stop_pipeline_task(
    task: asyncio.Task | None,
    cancel_event: threading.Event | None = None,
) -> None:
    if cancel_event is not None:
        cancel_event.set()
    if task is None or task.done():
        return
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    except Exception:
        logger.exception("Background site pipeline failed while stopping")


async def cleanup_conversation_resources(
    telegram_id: int,
    pipeline_task: asyncio.Task | None,
    cancel_event: threading.Event | None,
) -> None:
    try:
        await stop_pipeline_task(pipeline_task, cancel_event)
        await delete_user_vectors(telegram_id)
        logger.info("Conversation data cleanup completed for user %s", telegram_id)
    except Exception:
        logger.exception("Conversation data cleanup failed for user %s", telegram_id)


def schedule_conversation_cleanup(
    telegram_id: int,
    pipeline_task: asyncio.Task | None,
    cancel_event: threading.Event | None,
) -> None:
    previous = user_cleanup_tasks.get(telegram_id)

    async def cleanup_after_previous() -> None:
        if previous is not None and not previous.done():
            await previous
        await cleanup_conversation_resources(
            telegram_id,
            pipeline_task,
            cancel_event,
        )

    task = asyncio.create_task(cleanup_after_previous())
    user_cleanup_tasks[telegram_id] = task

    def remove_finished(done_task: asyncio.Task) -> None:
        if user_cleanup_tasks.get(telegram_id) is done_task:
            user_cleanup_tasks.pop(telegram_id, None)

    task.add_done_callback(remove_finished)


async def wait_for_conversation_cleanup(telegram_id: int) -> None:
    task = user_cleanup_tasks.get(telegram_id)
    if task is not None and not task.done():
        await task


async def run_uncancellable_thread(
    function: Callable[..., Any], *args: Any, **kwargs: Any
) -> Any:
    worker = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        try:
            await worker
        except BaseException:
            pass
        raise


async def delete_user_vectors(telegram_id: int) -> None:
    async with get_user_vector_lock(telegram_id):
        await asyncio.to_thread(
            delete_user_collection,
            telegram_id,
            persist_dir=VECTOR_DB_PATH,
        )



async def crawl_site(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    site: str,
    config: dict[str, Any],
    cancel_event: threading.Event,
) -> bool:
    try:
        normalized_site = to_absolute_url(site)
        await reply_rtl(
            update,
            "در حال جستجو سایت هستم؛ لطفاً کمی صبر کن...",
            reply_markup=conversation_menu(),
        )
        result = await acrawl(
            normalized_site,
            max_pages=int(config["crawler_pages"]),
            max_depth=int(config["crawler_depth"]),
            concurrency=int(config["crawl_concurrency"]),
            delay=CRAWL_DELAY,
            timeout=CRAWL_TIMEOUT,
            retries=CRAWL_RETRIES,
            verify_ssl=CRAWL_VERIFY_SSL,
            cancel_check=lambda: raise_if_cancelled(cancel_event),
        )
        context.user_data["site"] = normalized_site
        context.user_data["crawl_result"] = result
        await reply_rtl(
            update,
            f"جستجو انجام شد.\n"
            f"تعداد صفحات یافت شده : {len(result.pages)}\n"
            f"تعداد صفحات ناموفق: {len(result.errors)}",
            reply_markup=conversation_menu(),
        )
        if not result.pages:
            context.user_data["state"] = "waiting_for_site"
            context.user_data.pop("embeddings", None)
            context.user_data.pop("crawl_result", None)
            await reply_rtl(
                update,
                "هیچ صفحه‌ای با موفقیت دریافت نشد؛ لطفاً آدرس سایت را مجدد "
                "وارد کن یا «پایان مکالمه» را انتخاب کن.",
                reply_markup=conversation_menu(),
            )
            return False
        return True
    except Exception:
        logger.exception("Crawling failed for Telegram user %s", update.effective_user.id)
        context.user_data["state"] = "waiting_for_site"
        await reply_rtl(
            update,
            "دریافت سایت ناموفق بود؛ لطفاً آدرس سایت را مجدد وارد کن یا "
            "«پایان مکالمه» را انتخاب کن.",
            reply_markup=conversation_menu(),
        )
        return False


def embed_chunks_in_memory(
    chunks: list[Chunk],
    embedding_model: str,
    batch_size: int,
    progress_callback: Callable[[int, int], None] | None = None,
    cancel_check: Callable[[], None] | None = None,
) -> Any:
    texts = [chunk.text for chunk in chunks]
    if embedding_model == "local":
        return embed_texts(
            texts,
            backend="local",
            batch_size=batch_size,
            show_progress=True,
            progress_callback=progress_callback,
            cancel_check=cancel_check,
        )
    return embed_texts(
        texts,
        backend="openai",
        model=embedding_model,
        batch_size=batch_size,
        show_progress=True,
        progress_callback=progress_callback,
        cancel_check=cancel_check,
    )

async def prepare_knowledge_base(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    config: dict[str, Any],
    cancel_event: threading.Event,
) -> bool:
    result = context.user_data.get("crawl_result")
    if result is None or not result.pages:
        return False

    cancel_check = lambda: raise_if_cancelled(cancel_event)
    chunks = await run_uncancellable_thread(
        chunk_crawl_pages,
        result.pages,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        min_chunk_chars=MIN_CHUNK_CHARS,
        cancel_check=cancel_check,
    )
    context.user_data["crawl_chunks"] = chunks
    await reply_rtl(
        update,
        f"chunk‌بندی انجام شد.\nتعداد chunkهای ساخته‌شده: {len(chunks)}",
        reply_markup=conversation_menu(),
    )
    if not chunks:
        await reply_rtl(
            update,
            "از محتوای صفحات chunk قابل استفاده‌ای ساخته نشد.",
            reply_markup=conversation_menu(),
        )
        return False

    embedding_model = str(config["embedding_model"])
    progress_message = await reply_rtl(
        update,
        f"در حال ساخت embedding با مدل «{embedding_model}» هستم...\n"
        "پیشرفت: ۰٪",
    )

    loop = asyncio.get_running_loop()
    progress_state = {"percent": 0, "last_update": 0.0}
    progress_lock = asyncio.Lock()
    progress_finished = False

    async def update_embedding_progress(percent: int) -> None:
        if progress_finished:
            return
        try:
            async with progress_lock:
                if progress_finished:
                    return
                await progress_message.edit_text(
                    rtl_text(
                        f"در حال ساخت embedding با مدل «{embedding_model}» هستم...\n"
                        f"پیشرفت: {percent}٪"
                    )
                )
        except Exception:
            logger.debug("Could not update embedding progress", exc_info=True)

    def on_embedding_progress(done: int, total: int) -> None:
        if cancel_event.is_set():
            return
        if total <= 0:
            return
        percent = min(100, int(done * 100 / total))
        now = time.monotonic()
        if percent >= 100:
            return
        if (
            percent - progress_state["percent"] < 5
            and now - progress_state["last_update"] < 1
        ):
            return
        if percent == progress_state["percent"] and percent < 100:
            return
        progress_state["percent"] = percent
        progress_state["last_update"] = now
        loop.call_soon_threadsafe(
            lambda: asyncio.create_task(update_embedding_progress(percent))
        )

    embeddings = await run_uncancellable_thread(
        embed_chunks_in_memory,
        chunks,
        embedding_model,
        int(config["embed_batch_size"]),
        on_embedding_progress,
        cancel_check,
    )
    raise_if_cancelled(cancel_event)
    context.user_data["embeddings"] = embeddings
    progress_finished = True
    final_embedding_text = rtl_text(
        f"embedding با موفقیت ساخته شد.\n"
        f"مدل: {embedding_model}\n"
    )
    try:
        async with progress_lock:
            await progress_message.edit_text(final_embedding_text)
    except Exception:
        logger.warning(
            "Could not edit embedding progress message; sending final status instead",
            exc_info=True,
        )
        await reply_rtl(update, final_embedding_text)


    telegram_id = context.user_data["telegram_id"]
    raise_if_cancelled(cancel_event)
    async with get_user_vector_lock(telegram_id):
        raise_if_cancelled(cancel_event)
        stored_count = await run_uncancellable_thread(
            save_user_chunks,
            telegram_id,
            chunks,
            embeddings,
            persist_dir=VECTOR_DB_PATH,
        )
    await reply_rtl(
        update,
        f"chunkها و embeddingها با موفقیت در ChromaDB ذخیره شدند.\n"
        f"تعداد رکوردهای ذخیره‌شده: {stored_count}",
        reply_markup=conversation_menu(),
    )
    return True


async def prepare_site_pipeline(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    site: str,
    config: dict[str, Any],
    cancel_event: threading.Event,
) -> bool:
    telegram_id = context.user_data["telegram_id"]
    try:
        async with get_user_pipeline_lock(telegram_id):
            if not await crawl_site(update, context, site, config, cancel_event):
                return False
            return await prepare_knowledge_base(
                update,
                context,
                config,
                cancel_event,
            )
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("Site pipeline failed for Telegram user %s", telegram_id)
        await reply_rtl(
            update,
            "آماده‌سازی سایت (chunk، embedding یا ذخیره‌سازی) با خطا مواجه شد.",
            reply_markup=conversation_menu(),
        )
        return False


async def send_long_message(update: Update, text: str) -> None:
    for index in range(0, len(text), 4000):
        await reply_rtl(update, text[index : index + 4000])


def log_retrieved_chunks(
    query: str,
    hits: list[dict],
    query_number: int,
) -> None:
    if not hits:
        logger.info(
            "query شماره %s | chunk مرتبطی پیدا نشد | query=%r",
            query_number,
            query,
        )
        return

    logger.info(
        "chunkهای ارسال‌شده به generator | query شماره=%s | query=%r | تعداد chunkها=%s",
        query_number,
        query,
        len(hits),
    )

    for index, hit in enumerate(hits, start=1):
        logger.info(
            "chunk %s | title=%r | url=%r | distance=%s | text=%s",
            index,
            hit.get("title") or "-",
            hit.get("url") or "-",
            hit.get("distance", "-"),
            hit.get("text", ""),
        )


def log_generator_debug_prompt(
    query: str,
    hits: list[dict],
    query_number: int,
) -> None:
    prompt = build_prompt(query, hits)
    logger.info(
        "متن دقیق ارسالی به generator.py | query شماره=%s | query=%r\n%s",
        query_number,
        query,
        prompt,
    )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    telegram_id = ensure_user(update, context)
    await cancel_background_crawl(context)
    await wait_for_conversation_cleanup(telegram_id)
    await delete_user_vectors(telegram_id)
    clear_conversation(context)
    await reply_rtl(
        update,
        "سلام! از منوی زیر یک گزینه را انتخاب کن.",
        reply_markup=main_menu(),
    )


async def show_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    telegram_id = ensure_user(update, context)
    await cancel_background_crawl(context)
    await wait_for_conversation_cleanup(telegram_id)
    await delete_user_vectors(telegram_id)
    clear_conversation(context)
    await reply_rtl(update, "منوی اصلی:", reply_markup=main_menu())


async def begin_conversation(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    telegram_id = ensure_user(update, context)
    await cancel_background_crawl(context)
    await wait_for_conversation_cleanup(telegram_id)
    await delete_user_vectors(telegram_id)
    clear_conversation(context)
    context.user_data["state"] = "waiting_for_site"
    context.user_data["history"] = []
    config = user_config(context)
    await reply_rtl(
        update,
        "آدرس سایت را بفرست.\n"
        f"سایت در {config['crawler_pages']} صفحه و عمق {config['crawler_depth']} "
        "جستجو می‌شود.\n"
        "بعد از ارسال اولین سوال، عملیات پاسخگویی بر اساس اطلاعات موجود شروع می‌شود.",
        reply_markup=conversation_menu(),
    )


async def ask_embedding_models(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    selected = user_config(context)["embedding_model"]
    await reply_rtl(
        update,
        f"مدل embedding را انتخاب کن.\nمدل فعلی: {selected}",
        reply_markup=model_keyboard("embedding"),
    )


async def ask_generation_models(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    selected = user_config(context)["generation_model"]
    await reply_rtl(
        update,
        f"مدل generation را انتخاب کن.\nمدل فعلی: {selected}",
        reply_markup=model_keyboard("generation"),
    )


async def ask_depth(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    selected = user_config(context)["crawler_depth"]
    context.user_data["state"] = "waiting_for_depth"
    context.user_data.pop("numeric_setting", None)
    await reply_rtl(
        update,
        f"عمق خزنده را به‌صورت عدد وارد کن.\n"
        f"مقدار فعلی: {selected}\n"
        "حداقل: 1 | حداکثر: 10",
        reply_markup=settings_menu(),
    )


async def ask_pages(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    selected = user_config(context)["crawler_pages"]
    context.user_data["state"] = "waiting_for_pages"
    context.user_data.pop("numeric_setting", None)
    await reply_rtl(
        update,
        f"تعداد صفحات خزنده را به‌صورت عدد وارد کن.\n"
        f"مقدار فعلی: {selected}\n"
        "حداقل: 1 | حداکثر: 1000",
        reply_markup=settings_menu(),
    )


async def ask_numeric_setting(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    *,
    setting_key: str,
    state: str,
    title: str,
    minimum: int,
    maximum: int,
) -> None:
    selected = user_config(context)[setting_key]
    context.user_data["state"] = state
    context.user_data["numeric_setting"] = {
        "key": setting_key,
        "title": title,
        "minimum": minimum,
        "maximum": maximum,
    }
    await reply_rtl(
        update,
        f"{title} را به‌صورت عدد وارد کن.\n"
        f"مقدار فعلی: {selected}\n"
        f"حداقل: {minimum} | حداکثر: {maximum}",
        reply_markup=settings_menu(),
    )


async def answer_query(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    telegram_id = ensure_user(update, context)
    config = user_config(context)
    site = context.user_data.get("site", "")
    history: list[dict[str, str]] = context.user_data.setdefault("history", [])
    query = update.message.text.strip()
    query_number = int(context.user_data.get("query_count", 0)) + 1
    context.user_data["query_count"] = query_number
    query_lock = get_user_query_lock(telegram_id)
    async with query_lock:
        crawl_task = context.user_data.get("crawl_task")
        if crawl_task is not None:
            await reply_rtl(
                update,
                "سوال دریافت شد؛ "
                "منتظر بمانید...",
                reply_markup=conversation_menu(),
            )
            try:
                pipeline_ready = await crawl_task
            except asyncio.CancelledError:
                return
            except Exception:
                logger.exception(
                    "Background site pipeline task failed for user %s",
                    telegram_id,
                )
                await reply_rtl(
                    update,
                    "آماده‌سازی سایت کامل نشد؛ لطفاً یک مکالمه جدید شروع کن.",
                    reply_markup=conversation_menu(),
                )
                return
            context.user_data.pop("crawl_task", None)
            if not pipeline_ready:
                return
            site = context.user_data["site"]
        elif "embeddings" not in context.user_data:
            if not site:
                await reply_rtl(
                    update,
                    "ابتدا باید آدرس سایت را بفرستی.",
                    reply_markup=conversation_menu(),
                )
                return
            cancel_event = context.user_data.setdefault(
                "pipeline_cancel_event",
                threading.Event(),
            )
            if not await prepare_site_pipeline(
                update,
                context,
                site,
                config,
                cancel_event,
            ):
                return
            site = context.user_data["site"]

        if "embeddings" not in context.user_data:
            cancel_event = context.user_data.setdefault(
                "pipeline_cancel_event",
                threading.Event(),
            )
            if not await prepare_knowledge_base(
                update,
                context,
                config,
                cancel_event,
            ):
                return

    prompt = (
        f"Website: {site}\n"
        f"Selected embedding model: {config['embedding_model']}\n"
        f"Crawler depth: {config['crawler_depth']}\n"
        f"Crawler pages: {config['crawler_pages']}\n"
        f"User query: {query}"
    )
    history.append({"role": "user", "content": prompt})
    history[:] = history[-MAX_HISTORY_MESSAGES:]

    try:
        await update.message.chat.send_action(ChatAction.TYPING)
        query_embedding = await asyncio.to_thread(
            embed_chunks_in_memory,
            [Chunk("query", site, "", 0, 0, query, len(query))],
            str(config["embedding_model"]),
            int(config["embed_batch_size"]),
        )
        async with get_user_vector_lock(telegram_id):
            hits = await asyncio.to_thread(
                retrieve_user_chunks,
                telegram_id,
                query_embedding[0],
                top_k=int(config["retrieve_top_k"]),
                persist_dir=VECTOR_DB_PATH,
            )
        log_retrieved_chunks(query, hits, query_number)
        log_generator_debug_prompt(query, hits, query_number)
        answer = await asyncio.to_thread(
            generate_answer,
            query,
            hits,
            api_key=OPENAI_API_KEY,
            base_url=GENERATION_BASE_URL,
            model=config["generation_model"],
        )
        if not answer:
            answer = "مدل پاسخ متنی برنگرداند؛ دوباره تلاش کن."
        history.append({"role": "assistant", "content": answer})
        history[:] = history[-MAX_HISTORY_MESSAGES:]
        await send_long_message(update, answer)
        await reply_rtl(
            update,
            "سوال بعدی را بفرست یا «پایان مکالمه» را بزن.",
            reply_markup=conversation_menu(),
        )
    except Exception:
        logger.exception("Generation failed for Telegram user %s", telegram_id)
        await reply_rtl(
            update,
            "در ارتباط با مدل مشکلی پیش آمد. کلید API و مدل را بررسی کن.",
            reply_markup=conversation_menu(),
        )


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return
    ensure_user(update, context)
    text = update.message.text.strip()
    state = context.user_data.get("state", "idle")

    if text == "شروع مکالمه":
        await begin_conversation(update, context)
    elif text == "تنظیمات":
        context.user_data["state"] = "idle"
        context.user_data.pop("numeric_setting", None)
        await reply_rtl(update, "تنظیمات:", reply_markup=settings_menu())
    elif text == "بازگشت به منوی اصلی":
        context.user_data["state"] = "idle"
        context.user_data.pop("numeric_setting", None)
        await reply_rtl(update, "منوی اصلی:", reply_markup=main_menu())
    elif text == "مدل های embedding":
        await ask_embedding_models(update, context)
    elif text == "مدل های generation":
        await ask_generation_models(update, context)
    elif text == DEPTH_SETTING:
        await ask_depth(update, context)
    elif text == PAGES_SETTING:
        await ask_pages(update, context)
    elif text == TOP_K_SETTING:
        await ask_numeric_setting(
            update,
            context,
            setting_key="retrieve_top_k",
            state="waiting_for_retrieve_top_k",
            title=TOP_K_SETTING,
            minimum=1,
            maximum=50,
        )
    elif text == CONCURRENCY_SETTING:
        await ask_numeric_setting(
            update,
            context,
            setting_key="crawl_concurrency",
            state="waiting_for_crawl_concurrency",
            title=CONCURRENCY_SETTING,
            minimum=1,
            maximum=32,
        )
    elif text == BATCH_SIZE_SETTING:
        await ask_numeric_setting(
            update,
            context,
            setting_key="embed_batch_size",
            state="waiting_for_embed_batch_size",
            title=BATCH_SIZE_SETTING,
            minimum=1,
            maximum=256,
        )
    elif state == "waiting_for_depth":
        try:
            depth = int(text)
        except ValueError:
            await reply_rtl(
                update,
                "عمق خزنده باید یک عدد صحیح باشد. دوباره وارد کن (۱ تا ۱۰):",
                reply_markup=settings_menu(),
            )
            return
        if not 1 <= depth <= 10:
            await reply_rtl(
                update,
                "عمق واردشده معتبر نیست. عددی بین ۱ تا ۱۰ وارد کن:",
                reply_markup=settings_menu(),
            )
            return
        database.update_setting(
            context.user_data["telegram_id"], "crawler_depth", depth
        )
        context.user_data["state"] = "idle"
        await reply_rtl(
            update,
            f"عمق خزنده روی {depth} تنظیم شد.",
            reply_markup=settings_menu(),
        )
    elif state == "waiting_for_pages":
        try:
            pages = int(text)
        except ValueError:
            await reply_rtl(
                update,
                "تعداد صفحات خزنده باید یک عدد صحیح باشد. دوباره وارد کن (۱ تا ۱۰۰۰):",
                reply_markup=settings_menu(),
            )
            return
        if not 1 <= pages <= 1000:
            await reply_rtl(
                update,
                "تعداد صفحات واردشده معتبر نیست. عددی بین ۱ تا ۱۰۰۰ وارد کن:",
                reply_markup=settings_menu(),
            )
            return
        database.update_setting(
            context.user_data["telegram_id"], "crawler_pages", pages
        )
        context.user_data["state"] = "idle"
        await reply_rtl(
            update,
            f"تعداد صفحات خزنده روی {pages} تنظیم شد.",
            reply_markup=settings_menu(),
        )
    elif state in {
        "waiting_for_retrieve_top_k",
        "waiting_for_crawl_concurrency",
        "waiting_for_embed_batch_size",
    }:
        numeric_setting = context.user_data.get("numeric_setting")
        if not numeric_setting:
            context.user_data["state"] = "idle"
            await reply_rtl(update, "لطفاً دوباره از منوی تنظیمات گزینه را انتخاب کن.", reply_markup=settings_menu())
            return
        try:
            value = int(text)
        except ValueError:
            await reply_rtl(
                update,
                f"{numeric_setting['title']} باید یک عدد صحیح باشد؛ دوباره وارد کن:",
                reply_markup=settings_menu(),
            )
            return
        minimum = numeric_setting["minimum"]
        maximum = numeric_setting["maximum"]
        if not minimum <= value <= maximum:
            await reply_rtl(
                update,
                f"مقدار واردشده معتبر نیست. عددی بین {minimum} تا {maximum} وارد کن:",
                reply_markup=settings_menu(),
            )
            return
        database.update_setting(
            context.user_data["telegram_id"], numeric_setting["key"], value
        )
        context.user_data["state"] = "idle"
        context.user_data.pop("numeric_setting", None)
        await reply_rtl(
            update,
            f"{numeric_setting['title']} روی {value} تنظیم شد.",
            reply_markup=settings_menu(),
        )
    elif text == "پایان مکالمه":
        pipeline_task = context.user_data.pop("crawl_task", None)
        cancel_event = context.user_data.pop("pipeline_cancel_event", None)
        if cancel_event is not None:
            cancel_event.set()
        schedule_conversation_cleanup(
            context.user_data["telegram_id"],
            pipeline_task,
            cancel_event,
        )
        clear_conversation(context)
        await reply_rtl(
            update,
            "مکالمه پایان یافت و به منوی اصلی برگشتی. پاک‌سازی داده‌ها در حال انجام است.",
            reply_markup=main_menu(),
        )
    elif state == "waiting_for_site":
        context.user_data["site"] = text
        context.user_data["state"] = "waiting_for_query"
        cancel_event = threading.Event()
        context.user_data["pipeline_cancel_event"] = cancel_event
        context.user_data["crawl_task"] = asyncio.create_task(
            prepare_site_pipeline(
                update,
                context,
                text,
                user_config(context),
                cancel_event,
            )
        )
        await reply_rtl(
            update,
            "سایت دریافت شد و crawl، chunking، embedding و ذخیره‌سازی "
            "در پس‌زمینه شروع شد. اولین سوال را بفرست؛ تا آماده‌شدن کامل "
            "پایگاه دانش منتظر می‌مانم.",
            reply_markup=conversation_menu(),
        )
    elif state == "waiting_for_query":
        await answer_query(update, context)
    else:
        await reply_rtl(
            update,
            "از منوی زیر یک گزینه را انتخاب کن.",
            reply_markup=main_menu(),
        )


async def handle_callback(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    telegram_id = ensure_user(update, context)
    query = update.callback_query
    await query.answer()
    data = query.data or ""

    if data.startswith("model:"):
        _, kind, model = data.split(":", 2)
        allowed = EMBEDDING_MODELS if kind == "embedding" else GENERATION_MODELS
        if model in allowed:
            key = f"{kind}_model"
            database.update_setting(telegram_id, key, model)
            await query.edit_message_text(
                rtl_text(f"مدل {kind} روی «{model}» تنظیم شد.")
            )
    elif data.startswith("config:"):
        _, kind, value = data.split(":", 2)
        key = "crawler_depth" if kind == "depth" else "crawler_pages"
        database.update_setting(telegram_id, key, int(value))
        label = "عمق crawler" if kind == "depth" else "تعداد صفحات crawler"
        await query.edit_message_text(rtl_text(f"{label} روی {value} تنظیم شد."))

    await context.bot.send_message(
        chat_id=telegram_id,
        text=rtl_text("منوی اصلی:"),
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
