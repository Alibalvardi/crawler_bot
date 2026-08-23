from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from sentence_transformers import SentenceTransformer
import httpx
import numpy as np
from openai import OpenAI

logger = logging.getLogger(__name__)

DEFAULT_LOCAL_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
DEFAULT_GEMINI_MODEL = "gemini-embedding-001"
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_OPENAI_MODEL = "text-embedding-3-small"
OPENAI_API_BASE = "https://api.openai.com/v1"

# نگه‌داشته شده برای سازگاری با کدهای قبلی که از این نام استفاده می‌کردند
DEFAULT_MODEL = DEFAULT_LOCAL_MODEL


@dataclass
class EmbeddedChunk:
    chunk_id: str
    url: str
    title: str
    depth: int
    chunk_index: int
    text: str
    embedding: list[float] | None = None


def load_chunks(chunks_jsonl_path: str | Path) -> list[dict]:
    chunks = []
    with open(chunks_jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def _batched(items: list, batch_size: int):
    """items را به دسته‌های batch_size تایی تقسیم می‌کند (generator)."""
    for i in range(0, len(items), batch_size):
        yield items[i: i + batch_size]


def embed_texts(
        texts: list[str],
        *,
        backend: str = "local",
        model_name: str | None = None,
        batch_size: int = 32,
        show_progress: bool = True,
        gemini_api_key: str | None = None,
        openai_api_key: str | None = None,
        openai_base_url: str | None = None,
) -> np.ndarray:
    if backend == "local":
        return _embed_texts_local(
            texts,
            model_name=model_name or DEFAULT_LOCAL_MODEL,
            batch_size=batch_size,
            show_progress=show_progress,
        )
    if backend == "gemini":
        return _embed_texts_gemini(
            texts,
            model_name=model_name or DEFAULT_GEMINI_MODEL,
            api_key=gemini_api_key,
            show_progress=show_progress,
        )
    if backend == "openai":
        return _embed_texts_openai(
            texts,
            model_name=model_name or DEFAULT_OPENAI_MODEL,
            api_key=openai_api_key,
            base_url=openai_base_url or OPENAI_API_BASE,
            batch_size=batch_size,
            show_progress=show_progress,
        )
    raise ValueError(f"backend ناشناخته: {backend!r} (باید 'local'، 'gemini' یا 'openai' باشد)")


def _embed_texts_local(
        texts: list[str],
        *,
        model_name: str,
        batch_size: int,
        show_progress: bool,
) -> np.ndarray:
    logger.info("loading local embedding model: %s", model_name)
    model = SentenceTransformer(model_name)

    logger.info("embedding %d texts locally (batch_size=%d)", len(texts), batch_size)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,  # برای cosine similarity بهتره بردارها نرمال شده باشند
    )
    return embeddings


def _embed_texts_gemini(
        texts: list[str],
        *,
        model_name: str,
        api_key: str | None,
        show_progress: bool,
        batch_size: int = 20,  # سقف batchEmbedContents در حال حاضر ۱۰۰ است؛ ۲۰ محافظه‌کارانه و امن است
        max_retries: int = 3,
) -> np.ndarray:
    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "برای backend='gemini' باید GEMINI_API_KEY را به‌عنوان متغیر محیطی تنظیم کنی "
            "یا مستقیماً gemini_api_key را پاس بدهی."
        )

    url = f"{GEMINI_API_BASE}/models/{model_name}:batchEmbedContents"
    all_vectors: list[list[float]] = []

    with httpx.Client(timeout=30.0) as client:
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            payload = {
                "requests": [
                    {
                        "model": f"models/{model_name}",
                        "content": {"parts": [{"text": t}]},
                    }
                    for t in batch
                ]
            }

            for attempt in range(max_retries):
                response = client.post(url, params={"key": api_key}, json=payload)
                if response.status_code == 429:
                    wait = 2 ** attempt
                    logger.warning("Gemini rate limit hit, retrying in %ss", wait)
                    time.sleep(wait)
                    continue
                response.raise_for_status()
                break
            else:
                raise RuntimeError(f"Gemini embedding API پس از {max_retries} تلاش شکست خورد.")

            data = response.json()
            for item in data["embeddings"]:
                all_vectors.append(item["values"])

            if show_progress:
                logger.info("gemini embedded %d/%d", min(i + batch_size, len(texts)), len(texts))

    embeddings = np.array(all_vectors, dtype="float32")
    # نرمال‌سازی برای هماهنگی با خروجی backend محلی (که normalize_embeddings=True دارد)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms


def _openai_embed_batch(client, texts: list[str], model: str) -> list[list[float]]:
    response = client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in response.data]


def _embed_texts_openai(
        texts: list[str],
        *,
        model_name: str,
        api_key: str | None,
        base_url: str,
        batch_size: int,
        show_progress: bool,
) -> np.ndarray:
    from openai import OpenAI

    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "برای backend='openai' باید OPENAI_API_KEY را به‌عنوان متغیر محیطی تنظیم کنی "
            "یا مستقیماً openai_api_key را پاس بدهی."
        )

    client = OpenAI(base_url=base_url, api_key=api_key)

    all_vectors: list[list[float]] = []
    done = 0
    for batch in _batched(texts, batch_size):
        vectors = _openai_embed_batch(client, batch, model=model_name)
        all_vectors.extend(vectors)
        done += len(batch)
        if show_progress:
            logger.info("openai embedded %d/%d", done, len(texts))

    return np.array(all_vectors, dtype="float32")


def embed_chunks_openai(
        chunks: list[dict],
        *,
        api_key: str | None = None,
        base_url: str = OPENAI_API_BASE,
        model: str = DEFAULT_OPENAI_MODEL,
        batch_size: int = 32,
) -> list[EmbeddedChunk]:
    """معادل embed_chunks ولی مخصوص backend اوپن‌ای‌آی (یا پروکسی‌های سازگار مثل gapgpt.app)،
    که به‌جای آرایه‌ی numpy، مستقیماً لیستی از EmbeddedChunk (شامل خود بردار) برمی‌گرداند.
    این شکل خروجی برای مواقعی مفید است که می‌خواهی متن + متادیتا + بردار را با هم
    در یک ساختار واحد نگه داری (مثلاً قبل از نوشتن در فایل یا دیتابیس).
    """

    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "باید api_key را پاس بدهی یا متغیر محیطی OPENAI_API_KEY را تنظیم کنی."
        )

    client = OpenAI(base_url=base_url, api_key=api_key)

    embedded: list[EmbeddedChunk] = []
    for batch in _batched(chunks, batch_size):
        texts = [item["text"] for item in batch]
        vectors = _openai_embed_batch(client, texts, model=model)
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


def embed_chunks(
        chunks: list[dict],
        *,
        backend: str = "local",
        model_name: str | None = None,
        batch_size: int = 32,
        show_progress: bool = True,
        gemini_api_key: str | None = None,
        openai_api_key: str | None = None,
        openai_base_url: str | None = None,
) -> np.ndarray:
    """متن هر chunk را با backend انتخابی به بردار تبدیل می‌کند.

    خروجی یک آرایه‌ی numpy به شکل (تعداد chunk, ابعاد بردار) است که
    ترتیبش دقیقاً با ترتیب لیست chunks یکی است.
    """
    texts = [chunk["text"] for chunk in chunks]
    return embed_texts(
        texts,
        backend=backend,
        model_name=model_name,
        batch_size=batch_size,
        show_progress=show_progress,
        gemini_api_key=gemini_api_key,
        openai_api_key=openai_api_key,
        openai_base_url=openai_base_url,
    )


def save_embeddings(
        chunks: list[dict],
        embeddings: np.ndarray,
        output_dir: str | Path,
) -> tuple[Path, Path]:
    """بردارها را در یک فایل npy و متادیتای هر chunk را در یک فایل jsonl کنارش ذخیره می‌کند.
    این دو فایل با هم یک به یک (بر اساس ایندکس ردیف) مطابقت دارند."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vectors_path = output_dir / "embeddings.npy"
    metadata_path = output_dir / "embeddings_metadata.jsonl"

    np.save(vectors_path, embeddings)

    with metadata_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    return vectors_path, metadata_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Embed chunks.jsonl using a local or Gemini embedding backend.")
    parser.add_argument("chunks_jsonl", help="path to chunks.jsonl (from chunker.py)")
    parser.add_argument("-o", "--output-dir", default="data/embeddings", help="directory to save vectors + metadata")
    parser.add_argument("--backend", choices=["local", "gemini", "openai"], default="local")
    parser.add_argument("--model", default=None, help="model name (defaults depend on backend)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--openai-base-url",
        default=None,
        help="در صورت استفاده از پروکسی (مثل gapgpt.app) به‌جای api.openai.com",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    chunks = load_chunks(args.chunks_jsonl)
    if not chunks:
        raise SystemExit(f"هیچ chunk ای در {args.chunks_jsonl} پیدا نشد.")

    embeddings = embed_chunks(
        chunks,
        backend=args.backend,
        model_name=args.model,
        batch_size=args.batch_size,
        openai_base_url=args.openai_base_url,
    )
    # پوشه‌ی خروجی جدا برای هر backend، تا نتایج local و gemini با هم قاطی نشوند
    out_dir = Path(args.output_dir) / args.backend
    vectors_path, metadata_path = save_embeddings(chunks, embeddings, out_dir)

    print(f"\n{len(chunks)} chunk با backend={args.backend} embed شد.")
    print(f"ابعاد هر بردار: {embeddings.shape[1]}")
    print(f"بردارها: {vectors_path.resolve()}")
    print(f"متادیتا: {metadata_path.resolve()}")
