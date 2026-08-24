from __future__ import annotations

import argparse
import logging
from pathlib import Path

from chunker import chunk_crawl_result, save_chunks
from crawler import parse_args


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