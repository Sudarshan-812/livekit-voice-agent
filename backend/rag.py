import io
import os
from typing import List

import chromadb
import PyPDF2
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

COLLECTION_NAME = "knowledge_base"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3

# Persist to disk so uploads survive backend restarts.
_CHROMA_PATH = os.path.join(os.path.dirname(__file__), "chroma_data")
_chroma_client = chromadb.PersistentClient(path=_CHROMA_PATH)
# DefaultEmbeddingFunction uses all-MiniLM-L6-v2 via chromadb's bundled ONNX runtime —
# no separate sentence-transformers install required.
_collection = _chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=DefaultEmbeddingFunction(),
)


def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c for c in chunks if c.strip()]


def store_chunks(chunks: List[str], source_name: str) -> None:
    existing = _collection.get()
    offset = len(existing["ids"])
    ids = [f"{source_name}_{i + offset}" for i in range(len(chunks))]
    _collection.add(documents=chunks, ids=ids)


def search_knowledge_base(query: str) -> str:
    try:
        count = _collection.count()
        if count == 0:
            return "No documents have been uploaded yet."
        results = _collection.query(query_texts=[query], n_results=min(TOP_K, count))
        docs = results.get("documents", [[]])[0]
        if not docs:
            return "No relevant information found in the uploaded documents."
        return "\n\n---\n\n".join(docs)
    except Exception as e:
        return f"Search failed: {e}"
