from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from openai import APIError, APITimeoutError, OpenAI, RateLimitError

logger = logging.getLogger(__name__)

# text-embedding-3-small: بردار ۱۵۳۶ بعدی، ارزان و برای فارسی/چندزبانه کیفیت خوبی دارد.
DEFAULT_MODEL = "text-embedding-3-small"
DEFAULT_BATCH_SIZE = 100
MAX_RETRIES = 5


@dataclass
class EmbeddedChunk:
    chunk_id: str
    url: str
    title: str
    depth: int
    chunk_index: int
    text: str
    embedding: list[float]


def load_chunks(path: str | Path) -> list[dict]:
    """فایل chunks.jsonl تولیدشده توسط chunker.py را می‌خواند."""
    chunks: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def _batched(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def embed_texts(
    client: OpenAI,
    texts: list[str],
    *,
    model: str = DEFAULT_MODEL,
) -> list[list[float]]:
    """یک batch متن را با API اوپن‌ای‌آی امبد می‌کند، با retry برای خطاهای موقتی."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.embeddings.create(model=model, input=texts)
            # پاسخ API لزوماً هم‌ترتیب با ورودی است، ولی برای اطمینان بر اساس index مرتب می‌کنیم.
            ordered = sorted(response.data, key=lambda item: item.index)
            return [item.embedding for item in ordered]
        except (RateLimitError, APITimeoutError, APIError) as exc:
            wait = min(2 ** attempt, 30)
            logger.warning("embedding batch failed (attempt %s/%s): %s -> retry in %ss", attempt, MAX_RETRIES, exc, wait)
            time.sleep(wait)
    raise RuntimeError(f"embedding batch failed after {MAX_RETRIES} retries")


def embed_chunks(
    chunks: list[dict],
    *,
    model: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    api_key: str | None = None,
) -> list[EmbeddedChunk]:
    client = OpenAI(base_url="https://api.gapgpt.app/v1",api_key=api_key)
    embedded: list[EmbeddedChunk] = []

    for batch in _batched(chunks, batch_size):
        texts = [item["text"] for item in batch]
        vectors = embed_texts(client, texts, model=model)
        for item, vector in zip(batch, vectors):
            embedded.append(
                EmbeddedChunk(
                    chunk_id=item["chunk_id"],
                    url=item["url"],
                    title=item.get("title", ""),
                    depth=item.get("depth", 0),
                    chunk_index=item.get("chunk_index", 0),
                    text=item["text"],
                    embedding=vector,
                )
            )
        logger.info("embedded %s/%s chunks", len(embedded), len(chunks))

    return embedded


def save_embeddings(embedded: list[EmbeddedChunk], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for item in embedded:
            f.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")
    return output


if __name__ == "__main__":
    import argparse

    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Embed chunks.jsonl with the OpenAI embeddings API.")
    parser.add_argument("--chunks_path",default="data/chunks.jsonl", help="path to chunks.jsonl (from chunker.py)")
    parser.add_argument("-o", "--output", default="data/embeddings_openai.jsonl", help="output JSONL path")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api_key", default="")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()

    chunks = load_chunks(args.chunks_path)
    embedded = embed_chunks(chunks, model=args.model, batch_size=args.batch_size,api_key=args.api_key)
    saved = save_embeddings(embedded, args.output)
    print(f"{len(embedded)} embedding تولید شد -> {saved.resolve()}")
