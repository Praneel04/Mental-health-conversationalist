"""Dataset loading and text preparation."""

from __future__ import annotations

import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import CONFIG

TEXT_COLUMNS = ("instruction", "input", "output")


def load_dataset(url: str = CONFIG.dataset_url, limit: int | None = None) -> pd.DataFrame:
    """Load the mental-health conversations dataset from Hugging Face (or a local parquet path)."""
    df = pd.read_parquet(url)
    missing = [c for c in TEXT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")
    return df.head(limit) if limit else df


def to_documents(df: pd.DataFrame) -> list[Document]:
    """Merge each conversation into one document.

    The raw text is kept as-is (not lowercased or stopword-stripped): sentence-transformer
    embeddings work best on natural language, and the LLM needs readable context.
    """
    combined = (
        df[list(TEXT_COLUMNS)]
        .fillna("")
        .astype(str)
        .agg(lambda row: "\n".join(part.strip() for part in row if part.strip()), axis=1)
    )
    return [
        Document(page_content=text, metadata={"row": int(idx)})
        for idx, text in combined.items()
        if text.strip()
    ]


def split_documents(
    docs: list[Document],
    chunk_size: int = CONFIG.chunk_size,
    chunk_overlap: int = CONFIG.chunk_overlap,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)
