from __future__ import annotations

from dotenv import load_dotenv
from openai import OpenAI

import logging
import os

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_BASE_URL = "https://api.gapgpt.app/v1"

SYSTEM_PROMPT = """تو یک دستیار پرسش‌وپاسخ هستی که فقط بر اساس متن‌هایی که در ادامه به تو داده می‌شود پاسخ می‌دهی.
قوانین:
- فقط از اطلاعات موجود در «متن‌های مرجع» استفاده کن، نه دانش عمومی خودت.
- اگر پاسخ سوال در متن‌های مرجع نبود، صادقانه بگو که این اطلاعات در محتوای وب‌سایت موجود نیست.
- در پایان پاسخ، منبع(های) استفاده‌شده را با ذکر آدرس (URL) بیاور.
- پاسخ را به زبان فارسی و به‌صورت روان و مختصر بنویس."""


def build_prompt(query: str, hits: list[dict]) -> str:

    if not hits:
        context = "(هیچ متن مرتبطی پیدا نشد)"
    else:
        context_parts = []
        for i, hit in enumerate(hits, 1):
            context_parts.append(
                f"[منبع {i}: {hit['title'] or hit['url']} | {hit['url']}]\n{hit['text']}"
            )
        context = "\n\n".join(context_parts)

    return f"""متن‌های مرجع:
{context}

سوال کاربر: {query}
"""


def generate_answer(
    query: str,
    hits: list[dict],
    *,
    api_key: str | None = None,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
) -> str:
    load_dotenv()
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "باید api_key را پاس بدهی یا متغیر محیطی OPENAI_API_KEY را تنظیم کنی."
        )

    client = OpenAI(base_url=base_url, api_key=api_key)
    prompt = build_prompt(query, hits)

    logger.info("generating answer with model=%s (context chunks=%d)", model, len(hits))
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content


def answer_question(
    query: str,
    *,
    site_id: str,
    embedding_backend: str = "local",
    persist_dir: str = "data/vector_db",
    top_k: int = 5,
    llm_api_key: str | None = None,
    llm_base_url: str = DEFAULT_BASE_URL,
    llm_model: str = DEFAULT_MODEL,
) -> dict:
    from vector_store import retrieve

    hits = retrieve(
        query,
        site_id=site_id,
        backend=embedding_backend,
        persist_dir=persist_dir,
        top_k=top_k,
    )

    answer = generate_answer(
        query,
        hits,
        api_key=llm_api_key,
        base_url=llm_base_url,
        model=llm_model,
    )

    return {
        "query": query,
        "answer": answer,
        "sources": [{"url": h["url"], "title": h["title"]} for h in hits],
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Retrieve + generate an answer for a query against a site's vector store.")
    parser.add_argument("query", help="سوال کاربر")
    parser.add_argument("--site-id", required=True)
    parser.add_argument("--embedding-backend", choices=["local", "gemini", "openai"], default="local")
    parser.add_argument("--persist-dir", default="data/vector_db")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--llm-model", default=DEFAULT_MODEL)
    parser.add_argument("--llm-base-url", default=DEFAULT_BASE_URL)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    result = answer_question(
        args.query,
        site_id=args.site_id,
        embedding_backend=args.embedding_backend,
        persist_dir=args.persist_dir,
        top_k=args.top_k,
        llm_model=args.llm_model,
        llm_base_url=args.llm_base_url,
    )

    print(f"\nسوال: {result['query']}\n")
    print(f"پاسخ:\n{result['answer']}\n")
    print("منابع:")
    for src in result["sources"]:
        print(f"- {src['title'] or src['url']} ({src['url']})")
