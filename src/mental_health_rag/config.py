"""Central configuration. Override any value with an environment variable of the same name."""

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Config:
    dataset_url: str = _env(
        "DATASET_URL",
        "hf://datasets/CalebE/new_mental_health_conversations_all1/data/train-00000-of-00001.parquet",
    )
    embedding_model: str = _env("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    llm_model: str = _env("LLM_MODEL", "HuggingFaceH4/zephyr-7b-beta")
    index_dir: Path = field(default_factory=lambda: Path(_env("INDEX_DIR", "artifacts/faiss_index")))

    chunk_size: int = int(_env("CHUNK_SIZE", "500"))
    chunk_overlap: int = int(_env("CHUNK_OVERLAP", "50"))
    top_k: int = int(_env("TOP_K", "4"))

    max_new_tokens: int = int(_env("MAX_NEW_TOKENS", "512"))
    temperature: float = float(_env("TEMPERATURE", "0.7"))
    top_p: float = float(_env("TOP_P", "0.95"))


CONFIG = Config()
