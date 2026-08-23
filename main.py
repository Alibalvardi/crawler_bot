from __future__ import annotations

import argparse
import logging
from pathlib import Path

from chunker import chunk_crawl_result, save_chunks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="مرحله ۱: دریافت مطالب یک وب‌سایت (crawl همان دامنه)",
    )
    parser.add_argument("url", nargs="?", help="آدرس یا دامنه وب‌سایت، مثلاً example.com")
    parser.add_argument("--max-pages", type=int, default=500, help="حداکثر تعداد صفحات")
    parser.add_argument("--max-depth", type=int, default=5, help="حداکثر عمق لینک‌ها")
    parser.add_argument("--concurrency", type=int, default=16, help="تعداد درخواست همزمان")
    parser.add_argument("--delay", type=float, default=0.0, help="وقفه بین درخواست‌های هر worker (ثانیه)")
    parser.add_argument("--retries", type=int, default=2, help="چند بار URLهای خطا را دوباره امتحان کند")
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="به‌جای crawl از اول، فقط خطاهای فایل خروجی را دوباره بگیرد",
    )
    parser.add_argument(
        "--output",
        default="data/crawled.json",
        help="مسیر ذخیره خروجی JSON",
    )
    parser.add_argument(
        "--chunks-output",
        default="data/chunks.jsonl",
        help="مسیر ذخیره خروجی chunk ها (JSONL)",
    )
    parser.add_argument("--chunk-size", type=int, default=500, help="حداکثر طول هر chunk (کاراکتر)")
    parser.add_argument("--chunk-overlap", type=int, default=50, help="مقدار هم‌پوشانی بین chunk های متوالی")
    parser.add_argument(
        "--no-chunk",
        action="store_true",
        help="فقط crawl انجام شود، مرحله chunking اجرا نشود",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    output = Path(args.output)

    # if args.retry_errors:
    #     if not output.exists():
    #         raise SystemExit(f"فایل خروجی پیدا نشد: {output}")
    #     result = crawl(
    #         resume_from=output,
    #         max_pages=args.max_pages,
    #         max_depth=args.max_depth,
    #         concurrency=args.concurrency,
    #         delay=args.delay,
    #         retries=args.retries,
    #     )
    # else:
    #     url = args.url or input("آدرس وب‌سایت را وارد کن: ").strip()
    #     if not url:
    #         raise SystemExit("آدرس وب‌سایت خالی است.")
    #     result = crawl(
    #         url,
    #         max_pages=args.max_pages,
    #         max_depth=args.max_depth,
    #         concurrency=args.concurrency,
    #         delay=args.delay,
    #         retries=args.retries,
    #     )

    # saved = result.save(output)

    if not output.exists():
        raise SystemExit(f"فایل crawl شده پیدا نشد: {output}")
    saved = output

    # print(f"\nسایت: {result.seed_url}")
    # print(f"تعداد صفحات: {len(result.pages)}")
    # print(f"خطاها: {len(result.errors)}")
    print(f"خروجی: {saved.resolve()}")
    # for page in result.pages[:10]:
    #     preview = page.text.replace("\n", " ")[:80]
    #     print(f"- [d{page.depth}] {page.title or page.url} | {preview}")
    # if result.errors:
    #     print("نمونه خطاها:")
    #     for item in result.errors[:5]:
    #         print(f"- {item.url} | {item.reason}")

    if args.no_chunk:
        return

    print("\nدر حال chunk کردن متن صفحات...")
    chunks = chunk_crawl_result(
        saved,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    chunks_path = save_chunks(chunks, args.chunks_output)

    print(f"تعداد chunk تولیدشده: {len(chunks)}")
    print(f"خروجی chunk ها: {chunks_path.resolve()}")
    for chunk in chunks[:5]:
        preview = chunk.text.replace("\n", " ")[:80]
        print(f"- [{chunk.chunk_id}] {preview}")


if __name__ == "__main__":
    main()