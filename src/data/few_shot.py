"""Few-shot scaffolding prefix for Llama-3.2-1B Base coherence.

V4 §3.3: the base model needs a strict prefix of (user question, assistant answer)
exemplars so its attention learns to format an answer rather than continue rambling.
This does NOT alter weights, so the base policy reference for Shapira's theorem
remains uncorrupted.

The exemplars below were hand-written to be:
  - factually correct (so we never train the model toward errors)
  - stylistically neutral (no flattery, no hedging — to avoid biasing the BoN
    scoring toward agreement before the experiment even starts)
  - short (~20 tokens each) to keep the prefix budget tight
  - varied across QA, dates, math, definitions so the pattern generalizes
"""

from __future__ import annotations

FEW_SHOT_EXEMPLARS: list[tuple[str, str]] = [
    (
        "What is the capital of France?",
        "The capital of France is Paris.",
    ),
    (
        "Who wrote the play Hamlet?",
        "The play Hamlet was written by William Shakespeare.",
    ),
    (
        "In what year did the United States declare independence?",
        "The United States declared independence in 1776.",
    ),
    (
        "What is the chemical symbol for gold?",
        "The chemical symbol for gold is Au.",
    ),
    (
        "What is the boiling point of water at standard atmospheric pressure?",
        "At standard atmospheric pressure, water boils at 100 degrees Celsius.",
    ),
    (
        "What is the largest planet in our solar system?",
        "The largest planet in our solar system is Jupiter.",
    ),
    (
        "Who painted the Mona Lisa?",
        "The Mona Lisa was painted by Leonardo da Vinci.",
    ),
    (
        "What is the longest river in South America?",
        "The longest river in South America is the Amazon River.",
    ),
]


def build_prefix(
    exemplars: list[tuple[str, str]] | None = None,
    user_tag: str = "Q: ",
    assistant_tag: str = "A: ",
    sep: str = "\n",
) -> str:
    """Return a plain-text few-shot prefix to prepend before each query.

    Format:
        Q: <question>
        A: <answer>

        Q: <question>
        A: <answer>

        ...

    No system message, no special tokens — this is for a base (non-chat) model.
    """
    ex = exemplars or FEW_SHOT_EXEMPLARS
    blocks = [f"{user_tag}{q}{sep}{assistant_tag}{a}" for q, a in ex]
    return ("\n\n").join(blocks) + "\n\n"


def wrap_query(question: str, prefix: str | None = None, user_tag: str = "Q: ", assistant_tag: str = "A: ") -> str:
    """Wrap a single new question with the few-shot prefix, ready for generation."""
    if prefix is None:
        prefix = build_prefix()
    return f"{prefix}{user_tag}{question}\n{assistant_tag}"


if __name__ == "__main__":
    print(wrap_query("What is the largest planet in our solar system?"))
