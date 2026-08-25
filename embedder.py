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
from sympy import true
from torch.version import cuda

logger = logging.getLogger(__name__)

DEFAULT_LOCAL_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
DEFAULT_OPENAI_MODEL = "text-embedding-3-small"
OPENAI_API_BASE = "https://api.openai.com/v1"



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
        batch_size: int = 32,
        show_progress: bool = True,
) -> np.ndarray:
    if backend == "local":
        return _embed_texts_local(
            texts,
            batch_size=batch_size,
            show_progress=show_progress,
        )

    if backend == "openai":
        return _embed_texts_openai(
            texts,
            batch_size=batch_size,
            show_progress=show_progress,
        )
    raise ValueError(f"backend ناشناخته: {backend!r} (باید 'local'،  یا 'openai' باشد)")


def _embed_texts_local(
        texts: list[str],
        *,
        batch_size: int,
        show_progress: bool,
) -> np.ndarray:
    logger.info("loading local embedding model: %s", DEFAULT_LOCAL_MODEL)
    model = SentenceTransformer(DEFAULT_LOCAL_MODEL,local_files_only=true,device="cuda")

    logger.info("embedding %d texts locally (batch_size=%d)", len(texts), batch_size)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings


def _openai_embed_batch(client, texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(model=DEFAULT_OPENAI_MODEL, input=texts)
    return [item.embedding for item in response.data]


def _embed_texts_openai(
        texts: list[str],
        *,
        batch_size: int,
        show_progress: bool,
) -> np.ndarray:
    from openai import OpenAI

    api_key = "sk-ydXsKQkQKbM7KmW3N3r7HY8KqOcelJ8Oj45Xuj78cvv2XkLa"
    if not api_key:
        raise RuntimeError(
            "برای backend='openai' باید OPENAI_API_KEY را به‌عنوان متغیر محیطی تنظیم کنی "
            "یا مستقیماً openai_api_key را پاس بدهی."
        )

    client = OpenAI(base_url="https://api.gapgpt.app/v1", api_key=api_key)

    all_vectors: list[list[float]] = []
    done = 0
    for batch in _batched(texts, batch_size):
        vectors = _openai_embed_batch(client, batch)
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
        batch_size: int = 32,
        show_progress: bool = True,
) -> np.ndarray:
    texts = [chunk["text"] for chunk in chunks]
    return embed_texts(
        texts,
        backend=backend,
        batch_size=batch_size,
        show_progress=show_progress,
    )


def save_embeddings(
        chunks: list[dict],
        embeddings: np.ndarray,
        output_dir: str | Path,
) -> tuple[Path, Path]:
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

    parser = argparse.ArgumentParser(description="Embed chunks.jsonl using a local or openai embedding backend.")
    parser.add_argument("chunks_jsonl", help="path to chunks.jsonl (from chunker.py)")
    parser.add_argument("-o", "--output-dir", default="data/embeddings", help="directory to save vectors + metadata")
    parser.add_argument("--backend", choices=["local", "openai"], default="openai")
    parser.add_argument("--model", default=None, help="model name (defaults depend on backend)")
    parser.add_argument("--batch-size", type=int, default=32)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    chunks = load_chunks(args.chunks_jsonl)
    if not chunks:
        raise SystemExit(f"هیچ chunk ای در {args.chunks_jsonl} پیدا نشد.")

    embeddings = embed_chunks(
        chunks,
        backend=args.backend,
        batch_size=args.batch_size,
    )

    out_dir = Path(args.output_dir)
    vectors_path, metadata_path = save_embeddings(chunks, embeddings, out_dir)

    print(f"\n{len(chunks)} chunk با backend={args.backend} embed شد.")
    print(f"ابعاد هر بردار: {embeddings.shape[1]}")
    print(f"بردارها: {vectors_path.resolve()}")
    print(f"متادیتا: {metadata_path.resolve()}")
