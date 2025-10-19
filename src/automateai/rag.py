"""RAG utilities for automateAI.

Refactored from the original demo application. Provides functions for:
 - Initializing Qdrant (in-memory) and embedding model
 - Text chunking using tiktoken
 - Embedding generation via fastembed
 - Upserting and searching vectors in Qdrant
 - Deleting file-associated vectors
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Tuple


def _lazy_imports() -> tuple[Any, ...]:  # pragma: no cover - trivial import wrapper
    # Import heavy deps lazily to keep library import light
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    from fastembed import TextEmbedding
    import tiktoken

    return QdrantClient, Distance, VectorParams, PointStruct, TextEmbedding, tiktoken


def init_qdrant() -> tuple[Any, str]:
    """Initialize Qdrant client with in-memory storage and a default collection."""
    QdrantClient, Distance, VectorParams, _, _, _ = _lazy_imports()

    client = QdrantClient(":memory:")
    collection_name = "documents"

    try:
        client.get_collection(collection_name)
    except Exception:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )

    return client, collection_name


def init_embedding_model():
    """Initialize the embedding model (BAAI/bge-small-en-v1.5)."""
    _, _, _, _, TextEmbedding, _ = _lazy_imports()
    return TextEmbedding(model_name="BAAI/bge-small-en-v1.5")


def extract_text_from_pdf(file) -> str:
    from pypdf import PdfReader  # lazy import

    try:
        pdf_reader = PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception:
        return ""


def extract_text_from_docx(file) -> str:
    from docx import Document  # lazy import

    try:
        doc = Document(file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception:
        return ""


def extract_text_from_txt(file) -> str:
    try:
        return str(file.read(), "utf-8")
    except Exception:
        return ""


def process_uploaded_file(uploaded_file) -> str:
    extension = uploaded_file.name.split(".")[-1].lower()
    if extension == "pdf":
        return extract_text_from_pdf(uploaded_file)
    if extension == "docx":
        return extract_text_from_docx(uploaded_file)
    if extension == "txt":
        return extract_text_from_txt(uploaded_file)
    return ""


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks using tiktoken."""
    *_, tiktoken = _lazy_imports()

    try:
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    except Exception:
        encoding = tiktoken.get_encoding("cl100k_base")

    tokens = encoding.encode(text)
    chunks: List[str] = []

    for i in range(0, len(tokens), chunk_size - chunk_overlap):
        chunk_tokens = tokens[i : i + chunk_size]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)

    return chunks


def generate_embeddings(embedding_model, texts: List[str]) -> List[List[float]]:
    """Generate embeddings for text chunks using provided embedding model."""
    embeddings = list(embedding_model.embed(texts))
    return embeddings


def store_in_qdrant(
    qdrant_client,
    collection_name: str,
    chunks: List[str],
    embeddings: List[List[float]],
    filename: str,
    file_id: str,
) -> int:
    """Store embeddings in Qdrant and return number of inserted points."""
    _, _, _, PointStruct, _, _ = _lazy_imports()

    points = []
    for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        point_id = hashlib.md5(f"{file_id}_{idx}".encode()).hexdigest()
        points.append(
            PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "text": chunk,
                    "filename": filename,
                    "file_id": file_id,
                    "chunk_index": idx,
                },
            )
        )

    qdrant_client.upsert(collection_name=collection_name, points=points)
    return len(points)


def search_similar_chunks(
    qdrant_client,
    collection_name: str,
    embedding_model,
    query: str,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """Search for similar chunks in Qdrant and return payload with scores."""
    query_embedding = list(embedding_model.embed([query]))[0]
    search_results = qdrant_client.search(
        collection_name=collection_name, query_vector=query_embedding, limit=limit
    )

    results: List[Dict[str, Any]] = []
    for result in search_results:
        results.append(
            {
                "text": result.payload["text"],
                "filename": result.payload["filename"],
                "score": result.score,
            }
        )
    return results


def delete_file_from_qdrant(qdrant_client, collection_name: str, file_id: str) -> int:
    """Delete all points associated with a file id and return the number removed."""
    # The scroll filter API accepts model objects; dict also works in current versions
    points, _next_page = qdrant_client.scroll(
        collection_name=collection_name,
        scroll_filter={
            "must": [
                {
                    "key": "file_id",
                    "match": {
                        "value": file_id,
                    },
                }
            ]
        },
        limit=1000,
    )

    point_ids = [p.id for p in points]
    if point_ids:
        qdrant_client.delete(collection_name=collection_name, points_selector=point_ids)
    return len(point_ids)


__all__ = [
    "init_qdrant",
    "init_embedding_model",
    "extract_text_from_pdf",
    "extract_text_from_docx",
    "extract_text_from_txt",
    "process_uploaded_file",
    "chunk_text",
    "generate_embeddings",
    "store_in_qdrant",
    "search_similar_chunks",
    "delete_file_from_qdrant",
]


