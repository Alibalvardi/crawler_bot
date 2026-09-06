from __future__ import annotations

import logging
import re
from pathlib import Path

import chromadb
import numpy as np

from embedder import embed_texts, load_chunks

logger = logging.getLogger(__name__)

DEFAULT_PERSIST_DIR = "data/vector_db"


def _collection_name(site_id: str) -> str:
    safe_site = "".join(c if c.isalnum() else "_" for c in site_id)[:50]
    return f"site_{safe_site}"


def get_client(persist_dir: str | Path = DEFAULT_PERSIST_DIR):
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(persist_dir))


def _user_collection_name(telegram_id: int) -> str:
    return f"user_{int(telegram_id)}"


def _unique_indices(ids: list[str]) -> list[int]:
    """Return the indices of the first occurrence of every ID."""
    seen: set[str] = set()
    unique_indices: list[int] = []
    for index, chunk_id in enumerate(ids):
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        unique_indices.append(index)
    return unique_indices


def save_user_chunks(
    telegram_id: int,
    chunks: list[object],
    embeddings: np.ndarray,
    *,
    collection_name: str | None = None,
    persist_dir: str | Path = DEFAULT_PERSIST_DIR,
) -> int:
    """Replace the user's temporary collection with chunk text + vectors."""
    if not chunks:
        raise ValueError("chunks نمی‌تواند خالی باشد.")
    if not isinstance(embeddings, np.ndarray) or embeddings.ndim != 2:
        raise ValueError("embeddings باید آرایه دوبعدی NumPy باشد.")
    if len(chunks) != len(embeddings):
        raise ValueError("تعداد chunkها و embeddingها برابر نیست.")

    client = get_client(persist_dir)
    name = collection_name or _user_collection_name(telegram_id)
    try:
        client.delete_collection(name)
    except Exception:
        pass
    collection = client.create_collection(
        name=name,
        metadata={"telegram_id": int(telegram_id)},
    )

    all_ids = [str(getattr(chunk, "chunk_id")) for chunk in chunks]
    unique_indices = _unique_indices(all_ids)
    duplicate_count = len(all_ids) - len(unique_indices)
    if duplicate_count:
        logger.warning(
            "Ignoring %d duplicate chunk IDs while saving user %s.",
            duplicate_count,
            telegram_id,
        )

    ids = [all_ids[index] for index in unique_indices]
    documents = [str(getattr(chunks[index], "text")) for index in unique_indices]
    metadatas = [
        {
            "url": str(getattr(chunks[index], "url", "")),
            "title": str(getattr(chunks[index], "title", "") or ""),
            "depth": int(getattr(chunks[index], "depth", 0)),
            "chunk_index": int(getattr(chunks[index], "chunk_index", 0)),
            "telegram_id": int(telegram_id),
        }
        for index in unique_indices
    ]
    vectors = embeddings[unique_indices].astype("float32").tolist()

    try:
        max_batch = client.get_max_batch_size()
    except AttributeError:
        max_batch = 4000

    total = len(ids)
    for start in range(0, total, max_batch):
        end = min(start + max_batch, total)
        collection.add(
            ids=ids[start:end],
            embeddings=vectors[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )

    return collection.count()

def retrieve_user_chunks(
    telegram_id: int,
    query_embedding: np.ndarray,
    *,
    collection_name: str | None = None,
    top_k: int = 5,
    persist_dir: str | Path = DEFAULT_PERSIST_DIR,
    distance_threshold: float | None = None,
) -> list[dict]:
    if top_k <= 0:
        raise ValueError("top_k باید بزرگ‌تر از صفر باشد.")
    vector = np.asarray(query_embedding, dtype="float32").reshape(-1).tolist()
    client = get_client(persist_dir)
    collection = client.get_collection(
        collection_name or _user_collection_name(telegram_id)
    )
    result = collection.query(
        query_embeddings=[vector],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    return [
        {
            "text": document,
            "url": metadata.get("url", ""),
            "title": metadata.get("title", ""),
            "distance": distance,
        }
        for document, metadata, distance in zip(
            documents, metadatas, distances
        )
    ]


def delete_user_collection(
    telegram_id: int,
    *,
    collection_name: str | None = None,
    persist_dir: str | Path = DEFAULT_PERSIST_DIR,
) -> None:
    client = get_client(persist_dir)
    try:
        client.delete_collection(collection_name or _user_collection_name(telegram_id))
    except Exception:
        pass


def build_vector_store(
    embeddings: np.ndarray | str | Path,
    chunks_jsonl: str | Path,
    *,
    site_id: str,
    persist_dir: str | Path = DEFAULT_PERSIST_DIR,
    reset: bool = False,
) -> int:
    chunks = load_chunks(chunks_jsonl)
    if not chunks:
        raise ValueError(f"هیچ chunk ای در {chunks_jsonl} پیدا نشد.")

    if isinstance(embeddings, (str, Path)):
        embeddings_path = Path(embeddings)
        if not embeddings_path.is_file():
            raise FileNotFoundError(f"فایل embedding پیدا نشد: {embeddings_path}")
        embeddings = np.load(embeddings_path)

    if not isinstance(embeddings, np.ndarray):
        raise TypeError("embeddings باید آرایه‌ی NumPy یا مسیر فایل .npy باشد.")

    if embeddings.ndim != 2:
        raise ValueError(
            f"embeddings باید دوبعدی باشد؛ شکل فعلی: {embeddings.shape}"
        )

    if len(embeddings) != len(chunks):
        raise ValueError(
            f"تعداد embeddingها ({len(embeddings)}) با تعداد chunkها "
            f"({len(chunks)}) برابر نیست."
        )


    client = get_client(persist_dir)
    name = _collection_name(site_id)

    if reset:
        try:
            client.delete_collection(name)
        except Exception:
            pass  # اگر از قبل وجود نداشت، مشکلی نیست

    collection = client.get_or_create_collection(name, metadata={ "site_id": site_id})

    all_ids = [str(c["chunk_id"]) for c in chunks]
    unique_indices = _unique_indices(all_ids)
    duplicate_count = len(all_ids) - len(unique_indices)
    if duplicate_count:
        logger.warning(
            "Ignoring %d duplicate chunk IDs while building site '%s'.",
            duplicate_count,
            site_id,
        )

    collection.add(
        ids=[all_ids[index] for index in unique_indices],
        embeddings=embeddings[unique_indices].tolist(),
        documents=[chunks[index]["text"] for index in unique_indices],
        metadatas=[
            {
                "url": chunks[index]["url"],
                "title": chunks[index].get("title", ""),
                "depth": chunks[index].get("depth", 0),
                "chunk_index": chunks[index].get("chunk_index", 0),
            }
            for index in unique_indices
        ],
    )

    stored_count = len(unique_indices)
    logger.info("%d chunk در collection '%s' ذخیره شد.", stored_count, name)
    return stored_count


def retrieve(
    query: str,
    *,
    site_id: str,
    backend: str = "local",
    persist_dir: str | Path = DEFAULT_PERSIST_DIR,
    top_k: int = 5,
    distance_threshold: float | None = None,
) -> list[dict]:
    if top_k <= 0:
        raise ValueError("top_k باید بزرگ‌تر از صفر باشد.")
    if distance_threshold is not None and distance_threshold < 0:
        raise ValueError("distance_threshold نمی‌تواند منفی باشد.")

    query_embedding = embed_texts(
        [query],
        backend=backend,
        show_progress=False,
    )[0]

    client = get_client(persist_dir)
    name = _collection_name(site_id)
    try:
        collection = client.get_collection(name)
    except Exception as exc:
        raise ValueError(
            f"collection '{name}' پیدا نشد. اول باید build_vector_store را برای این site_id/backend اجرا کنی."
        ) from exc

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
    )

    hits = []
    for doc, meta, dist, chunk_id in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
        results["ids"][0],
    ):
        if distance_threshold is not None and dist > distance_threshold:
            continue
        hits.append(
            {
                "chunk_id": chunk_id,
                "text": doc,
                "url": meta["url"],
                "title": meta["title"],
                "distance": dist,
            }
        )
    return hits


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build or query a ChromaDB vector store from chunks.jsonl")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_p = subparsers.add_parser("build", help="embed chunks و در ChromaDB ذخیره کن")
    build_p.add_argument("chunks_jsonl")
    build_p.add_argument("embeddings")
    build_p.add_argument("--site-id", required=True, help="شناسه‌ی یکتای سایت (مثلاً دامنه)")
    build_p.add_argument("--backend", choices=["local", ], default="local")
    build_p.add_argument("--persist-dir", default=DEFAULT_PERSIST_DIR)

    query_p = subparsers.add_parser("query", help="جست‌وجوی شباهت در یک collection موجود")
    query_p.add_argument("query_text")
    query_p.add_argument("--site-id", required=True)
    query_p.add_argument("--backend", choices=["local", "openai"], default="local")
    query_p.add_argument("--persist-dir", default=DEFAULT_PERSIST_DIR)
    query_p.add_argument("--top-k", type=int, default=5)
    query_p.add_argument(
        "--distance-threshold",
        type=float,
        default=None,
        help="حداکثر فاصله‌ی قابل‌قبول؛ فاصله‌ی بیشتر یعنی عدم وجود اطلاعات مرتبط",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.command == "build":
        n = build_vector_store(
            chunks_jsonl=args.chunks_jsonl,
            embeddings=args.embeddings,
            site_id=args.site_id,
            persist_dir=args.persist_dir,
        )
        print(f"\n{n} chunk در vector store ذخیره شد (site_id={args.site_id}, backend={args.backend}).")

    elif args.command == "query":
        hits = retrieve(
            args.query_text,
            site_id=args.site_id,
            backend=args.backend,
            persist_dir=args.persist_dir,
            top_k=args.top_k,
            distance_threshold=args.distance_threshold,
        )
        if not hits:
            print(f"\nاطلاعات مرتبطی برای {args.query_text!r} در این سایت وجود ندارد.\n")
        else:
            print(f"\n{len(hits)} نتیجه برای: {args.query_text!r}\n")
            for i, hit in enumerate(hits, 1):
                print(f"{i}. [{hit['distance']:.4f}] {hit['title']} ({hit['url']})")
                print(f"   {hit['text']}...\n")
