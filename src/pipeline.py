"""
src/pipeline.py
---------------
Thin orchestration layer that wires together:
  retrieval.retrieve  →  retrieval.format_context  →  generation.generate_answer

Both app.py and evaluate.py import this single function so the logic
lives in exactly one place.
"""

import time
from typing import Dict, List, Tuple

from src.retrieval  import retrieve, format_context
from src.generation import generate_answer


def run_pipeline(
    question: str,
    top_k: int = 5,
) -> Tuple[str, List[Dict], float]:
    """
    End-to-end RAG pipeline for one question.

    Args:
        question: the user's natural-language question
        top_k:    number of articles to retrieve (default 5)

    Returns:
        (answer, hits, latency_seconds)
        - answer:          the LLM-generated response string
        - hits:            list of retrieved article dicts (from retrieval.retrieve)
        - latency_seconds: wall-clock time for the full pipeline call
    """
    t0 = time.perf_counter()

    # 1. Retrieve relevant articles
    hits = retrieve(question, top_k=top_k)

    # 2. Format context for the prompt
    context = format_context(hits)

    # 3. Generate answer
    answer = generate_answer(question, context)

    latency = time.perf_counter() - t0
    return answer, hits, latency
