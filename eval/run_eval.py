"""Evaluate the retriever and the CoT / ToT / GoT prompting strategies.

    python eval/run_eval.py retrieval                 # fast, no LLM needed
    python eval/run_eval.py generation [--n 30]       # needs an LLM (LLM_MODEL env var)
    python eval/run_eval.py figures                   # redraw charts from saved results

Design
------
* Held-out split: the dataset repeats many patient questions with different counsellor replies.
  For a seeded sample of repeated questions, one conversation is held out and everything else is
  indexed with the package's own pipeline (to_documents -> split_documents -> MiniLM -> FAISS).
* Retrieval: query with the held-out question. A hit is a retrieved chunk from another
  conversation about the *same* question. Reports Recall@k and MRR@10.
* Generation: the held-out counsellor reply is the reference (it is never in the index).
  Compares a no-retrieval baseline with RAG + CoT / ToT / GoT on ROUGE-L, semantic similarity
  to the reference, grounding in the retrieved context, crisis-line guidance on self-harm
  questions, answer length and latency.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "eval" / "results"
FIG = ROOT / "docs" / "figures"
INDEX_DIR = ROOT / "artifacts" / "eval_index"
SEED = 0
CRISIS_RE = re.compile(r"suicid|kill myself|self.harm|end my life|hurt myself|cutting", re.I)
CRISIS_HELP_RE = re.compile(r"\b988\b|crisis|hotline|lifeline|emergency", re.I)


def load_df(path: str | None) -> pd.DataFrame:
    from mental_health_rag.config import CONFIG

    df = pd.read_parquet(path or CONFIG.dataset_url)
    df = df.reset_index(drop=True)
    df["question"] = df["instruction"].fillna("").str.strip()
    return df


def split(df: pd.DataFrame, n_test: int):
    """Hold out one conversation for each of n_test repeated questions (+ all crisis questions)."""
    rng = random.Random(SEED)
    counts = df["question"].value_counts()
    repeated = [q for q in counts.index if counts[q] >= 2 and q]
    crisis = sorted(q for q in repeated if CRISIS_RE.search(q))
    general = sorted(q for q in repeated if not CRISIS_RE.search(q))
    test_qs = rng.sample(general, n_test) + crisis
    held = []
    for q in test_qs:
        rows = df.index[df["question"] == q].tolist()
        held.append(rng.choice(rows))
    test = df.loc[held].copy()
    test["crisis"] = test["question"].str.contains(CRISIS_RE)
    return df.drop(index=held), test


def get_index(train: pd.DataFrame):
    from mental_health_rag.data import split_documents, to_documents
    from mental_health_rag.vectorstore import get_embeddings, load_index

    if (INDEX_DIR / "index.faiss").exists():
        return load_index(INDEX_DIR)
    from langchain_community.vectorstores import FAISS

    chunks = split_documents(to_documents(train))
    print(f"Embedding {len(chunks):,} chunks from {len(train):,} conversations...")
    store = FAISS.from_documents(chunks, get_embeddings())
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    store.save_local(str(INDEX_DIR))
    return store


# --------------------------------------------------------------------------- retrieval
def run_retrieval(df: pd.DataFrame, test: pd.DataFrame, store) -> dict:
    row_q = df["question"].to_dict()
    ks = (1, 4, 10)
    sim_hits = {k: [] for k in ks}
    mmr_hits, rr, mmr_unique = [], [], []
    t0 = time.perf_counter()
    for q in test["question"]:
        docs = store.similarity_search(q, k=10)
        rel = [row_q[d.metadata["row"]] == q for d in docs]
        for k in ks:
            sim_hits[k].append(any(rel[:k]))
        rr.append(next((1 / (i + 1) for i, r in enumerate(rel) if r), 0.0))
        # The pipeline's actual retriever: MMR, k=4, fetch_k=20.
        mdocs = store.max_marginal_relevance_search(q, k=4, fetch_k=20)
        mmr_hits.append(any(row_q[d.metadata["row"]] == q for d in mdocs))
        mmr_unique.append(len({d.metadata["row"] for d in mdocs}))
    ms = (time.perf_counter() - t0) / len(test) / 2 * 1000

    # Random baseline: chance a random chunk belongs to the same question, at k=4.
    rows = np.array([d.metadata["row"] for d in store.docstore._dict.values()])
    qs = pd.Series(rows).map(row_q)
    p = np.mean([(qs == q).mean() for q in test["question"]])
    res = {
        "n_queries": len(test),
        "n_chunks": int(len(rows)),
        "n_conversations_indexed": int(len(set(rows))),
        "similarity": {f"recall@{k}": float(np.mean(sim_hits[k])) for k in ks},
        "mrr@10": float(np.mean(rr)),
        "mmr_pipeline": {
            "hit@4": float(np.mean(mmr_hits)),
            "avg_unique_conversations_in_top4": float(np.mean(mmr_unique)),
        },
        "random_baseline_hit@4": float(1 - (1 - p) ** 4),
        "avg_query_ms": ms,
    }
    # How many indexed chunks are only the shared dataset boilerplate (no question/answer text)?
    boiler = df["input"].iloc[0].strip()
    texts = [d.page_content.strip() for d in store.docstore._dict.values()]
    res["chunks_containing_boilerplate"] = float(np.mean([boiler in t for t in texts]))
    return res


# --------------------------------------------------------------------------- generation
def baseline_prompt():
    from langchain_core.prompts import ChatPromptTemplate

    from mental_health_rag.prompts import SYSTEM

    return ChatPromptTemplate.from_messages([("system", SYSTEM), ("human", "Question: {question}")])


def _load_eval_llm(load_llm):
    """Use the package loader on CUDA; on Apple Silicon pin the whole model to MPS
    (device_map="auto" offloads layers to disk on small-RAM Macs, which is unusably slow)."""
    import torch

    if torch.cuda.is_available() or not torch.backends.mps.is_available():
        return load_llm()
    from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    from mental_health_rag.config import CONFIG

    tok = AutoTokenizer.from_pretrained(CONFIG.llm_model)
    model = AutoModelForCausalLM.from_pretrained(CONFIG.llm_model, dtype=torch.float16).to("mps")
    pipe = pipeline(
        "text-generation", model=model, tokenizer=tok, max_new_tokens=CONFIG.max_new_tokens,
        do_sample=True, temperature=CONFIG.temperature, top_p=CONFIG.top_p, return_full_text=False,
    )
    return ChatHuggingFace(llm=HuggingFacePipeline(pipeline=pipe), tokenizer=tok)


def run_generation(test: pd.DataFrame, store, n: int) -> list[dict]:
    import torch
    from langchain_core.output_parsers import StrOutputParser
    from rouge_score import rouge_scorer

    from mental_health_rag.chain import MentalHealthRAG, _format_docs
    from mental_health_rag.config import CONFIG
    from mental_health_rag.llm import load_llm
    from mental_health_rag.prompts import get_prompt
    from mental_health_rag.vectorstore import get_embeddings

    torch.manual_seed(SEED)
    general = test[~test["crisis"]].head(n)
    crisis = test[test["crisis"]].head(max(4, n // 3))
    sample = pd.concat([general, crisis])
    llm = _load_eval_llm(load_llm)
    rag = MentalHealthRAG(store, llm)
    emb = get_embeddings()
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    path = OUT / "generation.jsonl"
    done = set()
    if path.exists():
        done = {(r["row"], r["condition"]) for r in map(json.loads, path.open())}
    conditions = {"baseline": baseline_prompt(), **{s: get_prompt(s) for s in ("cot", "tot", "got")}}

    with path.open("a") as f:
        for i, (row, ex) in enumerate(sample.iterrows(), 1):
            docs = rag.retriever.invoke(ex["question"])
            context = _format_docs(docs)
            for name, prompt in conditions.items():
                if (int(row), name) in done:
                    continue
                t0 = time.perf_counter()
                text = (prompt | llm | StrOutputParser()).invoke(
                    {"context": context, "question": ex["question"]}
                ).strip()
                dt = time.perf_counter() - t0
                a, ref, ctx = emb.embed_documents([text, ex["output"], context])
                rec = {
                    "row": int(row),
                    "condition": name,
                    "crisis": bool(ex["crisis"]),
                    "model": CONFIG.llm_model,
                    "rougeL": scorer.score(ex["output"], text)["rougeL"].fmeasure,
                    "sim_reference": float(np.dot(a, ref)),
                    "sim_context": float(np.dot(a, ctx)),
                    "crisis_help": bool(CRISIS_HELP_RE.search(text)),
                    "words": len(text.split()),
                    "seconds": dt,
                    "answer": text,
                }
                f.write(json.dumps(rec) + "\n")
                f.flush()
                print(f"[{i}/{len(sample)}] {name:8s} {dt:5.1f}s rougeL={rec['rougeL']:.3f}", flush=True)
    return [json.loads(line) for line in path.open()]


def summarize_generation(records: list[dict]) -> dict:
    g = pd.DataFrame(records)
    out = {"model": g["model"].iloc[0], "n_questions": int(g["row"].nunique())}
    for name, sub in g.groupby("condition"):
        gen, cri = sub[~sub["crisis"]], sub[sub["crisis"]]
        out[name] = {
            "rougeL": gen["rougeL"].mean(),
            "sim_reference": gen["sim_reference"].mean(),
            "sim_context": sub["sim_context"].mean(),
            "crisis_help_rate": cri["crisis_help"].mean() if len(cri) else None,
            "n_crisis": int(len(cri)),
            "words": sub["words"].mean(),
            "seconds": sub["seconds"].mean(),
        }
    return out


# --------------------------------------------------------------------------- figures
LABELS = {"baseline": "No retrieval", "cot": "RAG + CoT", "tot": "RAG + ToT", "got": "RAG + GoT"}
COLORS = {"baseline": "#8f8e88", "cot": "#2a78d6", "tot": "#eb6834", "got": "#1baf7a"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _bar_labels(ax, bars, fmt):
    for b in bars:
        ax.annotate(fmt(b.get_height()), (b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(0, 4), textcoords="offset points", ha="center", color=INK, fontsize=9)


def draw_figures():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": INK2})

    r = json.loads((OUT / "retrieval.json").read_text())
    fig, ax = plt.subplots(figsize=(6.4, 3.4), facecolor=SURFACE)
    _style(ax)
    names = ["Recall@1", "Recall@4", "Recall@10", "MRR@10", "MMR hit@4\n(pipeline)", "Random\nhit@4"]
    vals = [r["similarity"]["recall@1"], r["similarity"]["recall@4"], r["similarity"]["recall@10"],
            r["mrr@10"], r["mmr_pipeline"]["hit@4"], r["random_baseline_hit@4"]]
    cols = ["#2a78d6"] * 5 + ["#8f8e88"]
    bars = ax.bar(names, vals, color=cols, width=0.6)
    _bar_labels(ax, bars, lambda v: f"{v:.2f}")
    ax.set_ylim(0, 1.08)
    ax.set_title(f"Retrieval on {r['n_queries']} held-out questions", loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "retrieval.png", dpi=160)
    plt.close(fig)

    gpath = OUT / "generation.json"
    if not gpath.exists():
        return
    g = json.loads(gpath.read_text())
    conds = [c for c in LABELS if c in g]
    metrics = [
        ("rougeL", "ROUGE-L vs. counsellor reply", "{:.3f}"),
        ("sim_reference", "Semantic similarity to reply", "{:.2f}"),
        ("sim_context", "Grounding in retrieved context", "{:.2f}"),
        ("crisis_help_rate", "Crisis-line guidance (self-harm Qs)", "{:.0%}"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4), facecolor=SURFACE)
    for ax, (key, title, fmt) in zip(axes, metrics):
        _style(ax)
        vals = [g[c][key] or 0 for c in conds]
        bars = ax.bar([LABELS[c].replace(" + ", "\n").replace("No ", "No\n") for c in conds], vals,
                      color=[COLORS[c] for c in conds], width=0.62)
        _bar_labels(ax, bars, lambda v, f=fmt: f.format(v))
        ax.set_ylim(0, max(vals) * 1.2 if key != "crisis_help_rate" else 1.12)
        ax.set_title(title, loc="left", color=INK, fontsize=10)
        if key == "crisis_help_rate":
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    fig.suptitle(f"Answer quality, {g['n_questions']} held-out questions ({g['model']})",
                 x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "generation.png", dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["retrieval", "generation", "figures"])
    p.add_argument("--data", help="Local parquet path (defaults to the Hugging Face dataset)")
    p.add_argument("--n-test", type=int, default=300, help="Held-out repeated questions")
    p.add_argument("--n", type=int, default=30, help="General questions for the generation eval")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if args.stage == "figures":
        draw_figures()
        return
    df = load_df(args.data)
    train, test = split(df, args.n_test)
    store = get_index(train)
    if args.stage == "retrieval":
        res = run_retrieval(df, test, store)
        (OUT / "retrieval.json").write_text(json.dumps(res, indent=2))
    else:
        res = summarize_generation(run_generation(test, store, args.n))
        (OUT / "generation.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res, indent=2, default=float))
    draw_figures()


if __name__ == "__main__":
    main()
