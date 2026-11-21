"""Command-line interface.

    python -m mental_health_rag build-index [--limit N]
    python -m mental_health_rag ask "How can I manage anxiety?" [--strategy cot|tot|got|all] [--sources]
    python -m mental_health_rag chat [--strategy cot]
"""

from __future__ import annotations

import argparse

from .prompts import STRATEGIES


def _print_answer(answer, show_sources: bool) -> None:
    print(f"\n=== {STRATEGIES[answer.strategy]} ===\n")
    print(answer.text)
    if show_sources:
        print("\n--- Sources ---")
        for i, doc in enumerate(answer.sources, 1):
            snippet = doc.page_content.replace("\n", " ")
            print(f"[{i}] {snippet[:200]}{'...' if len(snippet) > 200 else ''}")


def _make_rag(args):
    from .chain import MentalHealthRAG
    from .llm import load_llm
    from .vectorstore import load_index

    return MentalHealthRAG(load_index(), load_llm(load_in_4bit=args.four_bit))


def main() -> None:
    parser = argparse.ArgumentParser(prog="mental_health_rag", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build-index", help="Download the dataset and build the FAISS index")
    build.add_argument("--limit", type=int, help="Only index the first N conversations (for quick tests)")

    for name in ("ask", "chat"):
        p = sub.add_parser(name)
        if name == "ask":
            p.add_argument("question")
        p.add_argument("--strategy", choices=[*STRATEGIES, "all"], default="cot")
        p.add_argument("--sources", action="store_true", help="Show the retrieved passages")
        p.add_argument("--4bit", dest="four_bit", action="store_true", help="Load the LLM in 4-bit (needs bitsandbytes + CUDA)")

    args = parser.parse_args()

    if args.command == "build-index":
        from .vectorstore import build_index

        build_index(limit=args.limit)
        return

    rag = _make_rag(args)
    strategies = list(STRATEGIES) if args.strategy == "all" else [args.strategy]

    def respond(question: str) -> None:
        for s in strategies:
            _print_answer(rag.ask(question, s), args.sources)

    if args.command == "ask":
        respond(args.question)
    else:
        print("Type a question (Ctrl+D or 'exit' to quit).")
        while True:
            try:
                q = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if q.lower() in {"exit", "quit"}:
                break
            if q:
                respond(q)


if __name__ == "__main__":
    main()
