"""Retrieval-augmented generation chain built with LangChain Expression Language (LCEL)."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from .config import CONFIG
from .prompts import get_prompt


@dataclass
class Answer:
    strategy: str
    text: str
    sources: list[Document]


def _format_docs(docs: list[Document]) -> str:
    return "\n\n".join(f"[{i}] {d.page_content}" for i, d in enumerate(docs, 1))


class MentalHealthRAG:
    def __init__(self, vectorstore, llm, top_k: int = CONFIG.top_k):
        # MMR keeps the retrieved passages relevant but not near-duplicates of each other.
        self.retriever = vectorstore.as_retriever(
            search_type="mmr", search_kwargs={"k": top_k, "fetch_k": top_k * 5}
        )
        self.llm = llm

    def ask(self, question: str, strategy: str = "cot") -> Answer:
        docs = self.retriever.invoke(question)
        chain = get_prompt(strategy) | self.llm | StrOutputParser()
        text = chain.invoke({"context": _format_docs(docs), "question": question})
        return Answer(strategy=strategy, text=text.strip(), sources=docs)
