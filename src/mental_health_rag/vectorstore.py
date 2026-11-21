"""Build, save and load the FAISS vector index."""

from __future__ import annotations

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from .config import CONFIG
from .data import load_dataset, split_documents, to_documents


def get_embeddings(model_name: str = CONFIG.embedding_model) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=model_name, encode_kwargs={"normalize_embeddings": True})


def build_index(index_dir: Path = CONFIG.index_dir, limit: int | None = None) -> FAISS:
    df = load_dataset(limit=limit)
    chunks = split_documents(to_documents(df))
    print(f"Embedding {len(chunks):,} chunks from {len(df):,} conversations...")
    store = FAISS.from_documents(chunks, get_embeddings())
    index_dir.mkdir(parents=True, exist_ok=True)
    store.save_local(str(index_dir))
    print(f"Saved index to {index_dir}")
    return store


def load_index(index_dir: Path = CONFIG.index_dir) -> FAISS:
    if not (index_dir / "index.faiss").exists():
        raise FileNotFoundError(f"No index at {index_dir}. Run `python -m mental_health_rag build-index` first.")
    # The index is created locally by build_index, so deserializing it is safe.
    return FAISS.load_local(str(index_dir), get_embeddings(), allow_dangerous_deserialization=True)
