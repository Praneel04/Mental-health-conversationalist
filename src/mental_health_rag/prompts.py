"""Reasoning-strategy prompts: Chain of Thought, Tree of Thoughts and Graph of Thoughts."""

from langchain_core.prompts import ChatPromptTemplate

SYSTEM = (
    "You are a supportive, non-judgmental mental-health information assistant. "
    "Ground your answer in the provided context; if the context does not cover the question, say so "
    "and give general, well-established guidance. You are not a therapist and must not diagnose. "
    "If the user mentions self-harm, suicide or being in danger, encourage them to contact local "
    "emergency services or a crisis line (e.g. 988 in the US) right away."
)

_INSTRUCTIONS = {
    "cot": (
        "Reason step by step (Chain of Thought):\n"
        "1. Identify the key details in the context.\n"
        "2. Work through the reasoning one step at a time.\n"
        "3. Give a clear, well-structured final answer."
    ),
    "tot": (
        "Explore several reasoning paths before answering (Tree of Thoughts):\n"
        "1. Branch 1: one possible explanation, analysed.\n"
        "2. Branch 2: an alternative viewpoint, analysed.\n"
        "3. Weigh the branches and answer using the strongest path."
    ),
    "got": (
        "Connect ideas across the context (Graph of Thoughts):\n"
        "1. Identify the key ideas in the retrieved passages.\n"
        "2. Find the connections between them.\n"
        "3. Integrate these insights into one cohesive answer."
    ),
}

STRATEGIES = {
    "cot": "Chain of Thought",
    "tot": "Tree of Thoughts",
    "got": "Graph of Thoughts",
}


def get_prompt(strategy: str) -> ChatPromptTemplate:
    if strategy not in _INSTRUCTIONS:
        raise ValueError(f"Unknown strategy {strategy!r}; choose from {sorted(_INSTRUCTIONS)}")
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM),
            (
                "human",
                "Context:\n{context}\n\n" + _INSTRUCTIONS[strategy] + "\n\nQuestion: {question}",
            ),
        ]
    )
