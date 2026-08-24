from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

# ترتیب اولویت جداکننده‌ها برای شکستن متن: اول پاراگراف، بعد خط، بعد جمله، بعد کلمه.
# اینجوری تا حد امکان وسط یک جمله یا کلمه بریده نمی‌شه.
DEFAULT_SEPARATORS = ["\n\n", "\n", "۔", ". ", "؟ ", "! ", "، ", " "]


@dataclass
class Chunk:
    chunk_id: str
    url: str
    title: str
    depth: int
    chunk_index: int
    text: str
    char_count: int


def split_text(
    text: str,
    *,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    separators: list[str] | None = None,
) -> list[str]:
    """متن را به‌صورت بازگشتی با جداکننده‌های داده‌شده می‌شکند تا قطعاتی
    نزدیک به chunk_size کاراکتر (و در صورت امکان از مرز جمله/پاراگراف) تولید شود،
    سپس قطعات را با overlap مشخص با هم ادغام می‌کند."""
    text = text.strip()
    if not text:
        return []
    if separators is None:
        separators = DEFAULT_SEPARATORS

    pieces = _recursive_split(text, chunk_size, separators)
    return _merge_with_overlap(pieces, chunk_size, chunk_overlap)


def _recursive_split(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    if len(text) <= chunk_size:
        return [text]

    if not separators:
        # دیگر جداکننده‌ای نمانده؛ به‌ناچار از روی طول کاراکتر می‌بریم.
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    sep, *rest_separators = separators
    if sep not in text:
        return _recursive_split(text, chunk_size, rest_separators)

    parts = [p for p in text.split(sep) if p.strip()]
    result: list[str] = []
    for part in parts:
        if len(part) > chunk_size:
            result.extend(_recursive_split(part, chunk_size, rest_separators))
        else:
            result.append(part)
    return result


def _merge_with_overlap(pieces: list[str], chunk_size: int, chunk_overlap: int) -> list[str]:
    """قطعات کوچک را کنار هم می‌چیند تا نزدیک chunk_size شوند، با overlap بین chunk متوالی."""
    if not pieces:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for piece in pieces:
        piece_len = len(piece) + 1  # +1 برای فاصله‌ای که موقع join اضافه می‌شود
        if current and current_len + piece_len > chunk_size:
            chunks.append(" ".join(current).strip())
            # overlap: از انتهای chunk قبلی چند تکه نگه می‌داریم تا در chunk بعدی هم بیایند
            overlap_pieces: list[str] = []
            overlap_len = 0
            for prev_piece in reversed(current):
                overlap_len += len(prev_piece) + 1
                overlap_pieces.insert(0, prev_piece)
                if overlap_len >= chunk_overlap:
                    break
            current = overlap_pieces
            current_len = sum(len(p) + 1 for p in current)

        current.append(piece)
        current_len += piece_len

    if current:
        chunks.append(" ".join(current).strip())

    return [c for c in chunks if c]


def chunk_crawl_result(
    crawl_json_path: str | Path,
    *,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    min_chunk_chars: int = 30,
) -> list[Chunk]:
    """فایل JSON خروجی crawler.py را می‌خواند و برای همه‌ی صفحات، chunk تولید می‌کند."""
    payload = json.loads(Path(crawl_json_path).read_text(encoding="utf-8"))
    all_chunks: list[Chunk] = []

    for page in payload.get("pages", []):
        url = page["url"]
        title = page.get("title", "")
        depth = page.get("depth", 0)
        text = page.get("text", "")

        pieces = split_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for idx, piece in enumerate(pieces):
            if len(piece) < min_chunk_chars:
                # قطعه‌ی خیلی کوتاه (مثلاً باقی‌مانده‌ی ناچیز آخر متن) ارزش embed شدن جدا ندارد
                continue
            chunk_id = f"{_slugify(url)}::{idx}"
            all_chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    url=url,
                    title=title,
                    depth=depth,
                    chunk_index=idx,
                    text=piece,
                    char_count=len(piece),
                )
            )

    return all_chunks


def _slugify(url: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", url).strip("_")[:150]


def save_chunks(chunks: list[Chunk], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Chunk crawled pages into overlapping text pieces.")
    parser.add_argument("crawl_json", help="path to crawl output JSON (from crawler.py)")
    parser.add_argument("-o", "--output", default="data/chunks.jsonl", help="output JSONL path")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunk-overlap", type=int, default=50)
    args = parser.parse_args()

    chunks = chunk_crawl_result(
        args.crawl_json,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    save_chunks(chunks, args.output)
    print(f"{len(chunks)} chunk از {args.crawl_json} تولید شد -> {args.output}")
