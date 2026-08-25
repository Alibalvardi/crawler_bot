from __future__ import annotations

import argparse
import logging
from pathlib import Path
from urllib.parse import urlparse

from chunker import chunk_crawl_result, save_chunks
from crawler import crawl, to_absolute_url
from embedder import embed_chunks, load_chunks, save_embeddings
from generator import generate_answer
from vector_store import build_vector_store, retrieve

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ساخت پایگاه دانش از سایت و پاسخ‌گویی به سؤال کاربر"
    )
    parser.add_argument("url", nargs="?", help="آدرس سایت؛ مثلاً example.com")
    parser.add_argument("-q", "--query", help="سؤال کاربر")

    parser.add_argument("--max-pages", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--insecure", action="store_true", help="غیرفعال کردن بررسی SSL")

    parser.add_argument("--output", default="data/crawled.json")
    parser.add_argument("--chunks-output", default="data/chunks.jsonl")
    parser.add_argument("--embeddings-dir", default="data/embeddings")
    parser.add_argument("--persist-dir", default="data/vector_db")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunk-overlap", type=int, default=50)
    parser.add_argument("--min-chunk-chars", type=int, default=30)

    parser.add_argument("--embedding-backend", choices=("local", "openai"), default="local")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--distance-threshold", type=float, default=None)
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument(
        "--reuse-crawl",
        action="store_true",
        help="به‌جای crawl مجدد، از فایل crawl موجود استفاده کن",
    )
    parser.add_argument(
        "--reuse-index",
        action="store_true",
        help="embedding و vector store را دوباره نساز",
    )
    parser.add_argument("--reset-index", action="store_true")
    return parser


def _value(value: str | None, prompt: str) -> str:
    result = (value or input(prompt)).strip()
    if not result:
        raise SystemExit("ورودی نمی‌تواند خ+++++++++. الی باشد.")
    return result


def _site_id(url: str) -> str:
    host = urlparse(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def run_pipeline(args: argparse.Namespace) -> str:
    url = to_absolute_url(_value(args.url, "آدرس وب‌سایت را وارد کن: "))
    query = _value(args.query, "سؤال خود را وارد کن: ")
    site_id = _site_id(url)

    crawl_path = Path(args.output)
    if args.reuse_crawl:
        if not crawl_path.is_file():
            raise SystemExit(f"فایل crawl پیدا نشد: {crawl_path}")
        print(f"[۱/۶] استفاده از crawl موجود: {crawl_path}")
    else:
        print(f"[۱/۶] در حال دریافت سایت: {url}")
        result = crawl(
            url,
            max_pages=args.max_pages,
            max_depth=args.max_depth,
            concurrency=args.concurrency,
            delay=args.delay,
            timeout=args.timeout,
            retries=args.retries,
            verify_ssl=not args.insecure,
        )
        result.save(crawl_path)
        print(f"      {len(result.pages)} صفحه ذخیره شد: {crawl_path}")

    print("[۲/۶] در حال ساخت chunkها...")
    chunks = chunk_crawl_result(
        crawl_path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        min_chunk_chars=args.min_chunk_chars,
    )
    if not chunks:
        raise SystemExit("از صفحات دریافت‌شده هیچ chunk قابل استفاده‌ای ساخته نشد.")
    chunks_path = save_chunks(chunks, args.chunks_output)
    print(f"      {len(chunks)} chunk ذخیره شد: {chunks_path}")

    embeddings_path = Path(args.embeddings_dir) / "embeddings.npy"
    if args.reuse_index and embeddings_path.is_file():
        print(f"[۳/۶] استفاده از embedding موجود: {embeddings_path}")
    else:
        print(f"[۳/۶] در حال ساخت embedding با backend={args.embedding_backend}...")
        chunk_dicts = load_chunks(chunks_path)
        embeddings = embed_chunks(
            chunk_dicts,
            backend=args.embedding_backend,
            batch_size=args.batch_size,
        )
        vectors_path, _ = save_embeddings(chunk_dicts, embeddings, args.embeddings_dir)
        embeddings_path = vectors_path
        print(f"      embeddingها ذخیره شدند: {embeddings_path}")

    if args.reuse_index:
        print("[۴/۶] استفاده از vector store موجود...")
    else:
        print("[۴/۶] در حال ساخت vector store...")
        indexed_count = build_vector_store(
            embeddings_path,
            chunks_path,
            site_id=site_id,
            persist_dir=args.persist_dir,
            reset=True,
        )
        print(f"      {indexed_count} chunk در vector store ذخیره شد.")

    print("[۵/۶] در حال جست‌وجوی متن‌های مرتبط...")
    hits = retrieve(
        query,
        site_id=site_id,
        backend=args.embedding_backend,
        persist_dir=args.persist_dir,
        top_k=args.top_k,
        distance_threshold=args.distance_threshold,
    )
    print(f"      {len(hits)} نتیجه پیدا شد.")

    print("[۶/۶] در حال تولید پاسخ...")
    generation_kwargs = {}
    if args.llm_model:
        generation_kwargs["model"] = args.llm_model
    if args.llm_base_url:
        generation_kwargs["base_url"] = args.llm_base_url
    answer = generate_answer(query, hits, **generation_kwargs)

    print("\n" + "=" * 60)
    print(f"سؤال: {query}")
    print(f"\nپاسخ:\n{answer}")
    print("\nمنابع:")
    seen_sources: set[str] = set()
    for hit in hits:
        if hit["url"] in seen_sources:
            continue
        seen_sources.add(hit["url"])
        print(f"- {hit['title'] or hit['url']} ({hit['url']})")
    print("=" * 60)
    return answer


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args()
    try:
        run_pipeline(args)
    except KeyboardInterrupt:
        raise SystemExit("\nعملیات توسط کاربر متوقف شد.")
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"خطا: {exc}") from exc


if __name__ == "__main__":
    main()
