"""
evaluate.py
-----------
Runs the full RAG pipeline on 10 standard French labor law questions and
saves results to evaluation_results.csv.

Metrics captured per question:
  - retrieved_articles: comma-separated article numbers
  - answer:             LLM-generated response
  - latency_s:          wall-clock seconds for the full pipeline
  - has_citation:       True if the answer mentions at least one article number

Run:
    python evaluate.py
"""

import re
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv(Path(__file__).resolve().parent / ".env")

from src.ingest   import ingest
from src.pipeline import run_pipeline

# ── Evaluation questions ──────────────────────────────────────────────────────
QUESTIONS = [
    "How many days of paid annual leave is an employee entitled to?",
    "What are the legal grounds for dismissing an employee in France?",
    "What is the maximum number of working hours per week in France?",
    "What is the minimum wage (SMIC) and how is it calculated?",
    "What is the maximum duration of a trial period (période d'essai)?",
    "What are the rights of employees during parental leave?",
    "How is overtime work compensated in France?",
    "What legal protections exist against workplace harassment (harcèlement moral)?",
    "What is the minimum notice period an employer must give before termination?",
    "What are the rules regarding sick leave and compensation during illness?",
]

# Article citation pattern — e.g. "Article L1234-1" or "L1234-1"
CITATION_RE = re.compile(r"\b[LRD]\d{3,4}-\d+")


def has_citation(answer: str) -> bool:
    """Return True if the answer contains at least one article number citation."""
    return bool(CITATION_RE.search(answer))


def main():
    # Ensure vector store is ready
    print("[evaluate] Checking vector store …")
    ingest()

    rows = []
    for question in tqdm(QUESTIONS, desc="Evaluating"):
        answer, hits, latency = run_pipeline(question)

        article_numbers = ", ".join(h["article_number"] for h in hits)

        rows.append(
            {
                "question":           question,
                "retrieved_articles": article_numbers,
                "answer":             answer,
                "latency_s":          round(latency, 2),
                "has_citation":       has_citation(answer),
            }
        )

    df = pd.DataFrame(rows)

    output_path = Path(__file__).resolve().parent / "evaluation_results.csv"
    df.to_csv(output_path, index=False)
    print(f"\n[evaluate] Results saved to {output_path}")

    # Quick summary
    citation_rate = df["has_citation"].mean() * 100
    avg_latency   = df["latency_s"].mean()
    print(f"[evaluate] Citation rate: {citation_rate:.0f}%  |  Avg latency: {avg_latency:.1f}s")
    print(df[["question", "retrieved_articles", "latency_s", "has_citation"]].to_string(index=False))


if __name__ == "__main__":
    main()
