# Mental Health RAG Chatbot

A retrieval-augmented generation (RAG) chatbot that answers mental-health questions using a dataset of real
counselling conversations, and compares three reasoning-prompting strategies:

- **Chain of Thought (CoT):** reasons step by step.
- **Tree of Thoughts (ToT):** explores alternative explanations, then picks the strongest.
- **Graph of Thoughts (GoT):** connects ideas across the retrieved passages into one answer.

> ⚠️ **Disclaimer:** This is an educational project, not a medical device or a substitute for professional care.
> If you or someone you know is in crisis, contact local emergency services or a crisis line
> (in the US, call or text **988**).

## How it works

```
 question ──► embed (MiniLM) ──► FAISS (MMR, top-k) ──► retrieved conversations
                                                              │
                         CoT / ToT / GoT prompt  ◄────────────┘
                                   │
                                   ▼
                        Zephyr-7B (local, HF Transformers) ──► answer + sources
```

| Component | Choice |
|---|---|
| Dataset | [`CalebE/new_mental_health_conversations_all1`](https://huggingface.co/datasets/CalebE/new_mental_health_conversations_all1) (instruction / input / output) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | FAISS, max-marginal-relevance retrieval |
| LLM | `HuggingFaceH4/zephyr-7b-beta` through its chat template |
| Orchestration | LangChain (LCEL) |

The project grew out of two phases, both kept in [`notebooks/exploration.ipynb`](notebooks/exploration.ipynb):

1. **Phase 1, NLP analysis:** text cleaning, tokenization, spaCy POS tagging and BERT sentence embeddings
   (now in `src/mental_health_rag/preprocessing.py`).
2. **Phase 2, RAG chatbot:** vector retrieval and LLM generation with CoT/ToT/GoT prompts
   (now a reusable package and CLI).

## Project structure

```
.
├── notebooks/
│   └── exploration.ipynb        # original Phase 1 + Phase 2 experiments
├── src/mental_health_rag/
│   ├── config.py                # model names, chunking and generation settings (env-overridable)
│   ├── data.py                  # dataset loading → LangChain documents → chunks
│   ├── preprocessing.py         # Phase 1: cleaning, tokenization, POS tags, BERT embeddings
│   ├── vectorstore.py           # build / save / load the FAISS index
│   ├── prompts.py               # CoT, ToT, GoT prompt templates
│   ├── llm.py                   # loads Zephyr-7B once (fp16 or 4-bit)
│   ├── chain.py                 # retrieval + generation pipeline
│   └── __main__.py              # CLI
├── pyproject.toml
└── requirements.txt
```

## Getting started

**Requirements:** Python 3.10+. A GPU with about 16 GB of VRAM is recommended for Zephyr-7B in fp16, or about 6 GB with `--4bit`.
It runs on CPU, but slowly.

```bash
git clone <this-repo-url> && cd genai_chatbot
python -m venv .venv && source .venv/bin/activate
pip install -e .            # add [nlp] for Phase 1 analysis, [quant] for 4-bit loading
```

**1. Build the vector index** (downloads the dataset and saves the index to `artifacts/faiss_index/`):

```bash
python -m mental_health_rag build-index            # full dataset
python -m mental_health_rag build-index --limit 500 # quick test
```

**2. Ask a question:**

```bash
python -m mental_health_rag ask "How can I manage anxiety?"
python -m mental_health_rag ask "How does stress impact mental health?" --strategy all --sources
```

**3. Or chat interactively:**

```bash
python -m mental_health_rag chat --strategy tot
```

### Use it from Python

```python
from mental_health_rag.chain import MentalHealthRAG
from mental_health_rag.llm import load_llm
from mental_health_rag.vectorstore import load_index

rag = MentalHealthRAG(load_index(), load_llm())
answer = rag.ask("How can I sleep better when I'm stressed?", strategy="got")
print(answer.text)
```

### Configuration

Any setting in `config.py` can be overridden with an environment variable, for example:

```bash
LLM_MODEL=mistralai/Mistral-7B-Instruct-v0.3 TOP_K=6 python -m mental_health_rag ask "..."
```

| Variable | Default |
|---|---|
| `LLM_MODEL` | `HuggingFaceH4/zephyr-7b-beta` |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` |
| `INDEX_DIR` | `artifacts/faiss_index` |
| `TOP_K` | `4` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `500` / `50` |
| `MAX_NEW_TOKENS`, `TEMPERATURE`, `TOP_P` | `512`, `0.7`, `0.95` |

## Improvements over the original notebook

- The model, dataset and index load once instead of 3–4 times in duplicated cells. The index is saved to disk and reused.
- Retrieval now uses raw text instead of lowercased, stopword-stripped text, which keeps the embeddings and the LLM context meaningful.
- Fixed crashes: `Document` was used before it was imported, and `source_documents` was read but never returned.
- Uses the model's chat template, returns only the generated answer (no echoed prompt), and enables sampling so `temperature` works.
- Masked mean pooling and batching for BERT embeddings, and batched spaCy tagging.
- Moved to current LangChain packages (`langchain-huggingface`, LCEL) from deprecated imports.
- Added a safety-oriented system prompt with crisis-line guidance.
