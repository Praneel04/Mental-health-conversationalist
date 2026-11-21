"""Phase 1: classic NLP preprocessing and analysis (cleaning, tokenization, POS tags, BERT embeddings).

These features are for exploration/analysis. The RAG pipeline itself retrieves on raw text.
Optional dependencies: install with `pip install -r requirements-nlp.txt`.
"""

from __future__ import annotations

import re
from functools import lru_cache

import pandas as pd


@lru_cache(maxsize=1)
def _stopwords() -> frozenset[str]:
    import nltk
    from nltk.corpus import stopwords

    for pkg in ("punkt", "punkt_tab", "stopwords"):
        nltk.download(pkg, quiet=True)
    return frozenset(stopwords.words("english"))


def clean_text(text: object) -> str:
    """Lowercase, strip non-letters and remove English stopwords."""
    from nltk.tokenize import word_tokenize

    if not isinstance(text, str):
        return ""
    text = re.sub(r"[^a-z\s]", " ", text.lower())
    stop = _stopwords()
    return " ".join(tok for tok in word_tokenize(text) if tok not in stop)


def add_clean_columns(df: pd.DataFrame, columns=("instruction", "input", "output")) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        out[f"clean_{col}"] = out[col].map(clean_text)
    return out


def pos_tags(texts: pd.Series, model: str = "en_core_web_sm") -> list[list[tuple[str, str]]]:
    """Part-of-speech tags using spaCy, processed in batches with nlp.pipe."""
    import spacy

    nlp = spacy.load(model, disable=["ner", "parser", "lemmatizer"])
    return [[(t.text, t.pos_) for t in doc] for doc in nlp.pipe(texts.fillna("").tolist(), batch_size=64)]


def bert_embeddings(texts: pd.Series, model_name: str = "bert-base-uncased", batch_size: int = 32):
    """Mean-pooled BERT embeddings (padding tokens excluded), computed in batches."""
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    items = texts.fillna("").tolist()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            batch = tokenizer(
                items[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)
            hidden = model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            chunks.append(pooled.cpu().numpy())
    return np.vstack(chunks) if chunks else np.empty((0, model.config.hidden_size))
