"""
src/compare_embeddings.py
-------------------------
Compares two embedding models on French legal retrieval:
  - paraphrase-multilingual-MiniLM-L12-v2  (multilingual, current model)
  - all-MiniLM-L6-v2                        (English-only, baseline)

Strategy: pull a representative sample of articles directly from ChromaDB,
encode them with both models in memory, then compute cosine similarity
against each test query. This avoids re-ingesting the full PDF.

Output: comparison_results.csv
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import chromadb
from tqdm import tqdm

ROOT    = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "db"
OUT_CSV = ROOT / "comparison_results.csv"

MODELS = {
    "multilingual-MiniLM-L12-v2": "paraphrase-multilingual-MiniLM-L12-v2",
    "all-MiniLM-L6-v2":           "sentence-transformers/all-MiniLM-L6-v2",
}

TEST_QUESTIONS = [
    "How many days of paid annual leave is an employee entitled to?",
    "What are the legal conditions for dismissing an employee?",
    "What is the maximum number of working hours per week?",
    "What are the rules for overtime compensation?",
    "What are the employee's rights during a trial period?",
]

SAMPLE_SIZE = 500   # articles to compare against (representative, fast)
TOP_K       = 5


def cosine_top_k(query_vec: np.ndarray, corpus_vecs: np.ndarray,
                 ids: list, k: int) -> list[tuple[str, float]]:
    """Return top-k (article_id, score) sorted by descending cosine similarity."""
    sims = cosine_similarity(query_vec.reshape(1, -1), corpus_vecs)[0]
    top_idx = np.argsort(sims)[::-1][:k]
    return [(ids[i], float(sims[i])) for i in top_idx]


def main():
    # ── 1. Load sample articles from ChromaDB ─────────────────────────────────
    print(f"[compare] Loading {SAMPLE_SIZE} articles from ChromaDB …")
    client     = chromadb.PersistentClient(path=str(DB_PATH))
    collection = client.get_collection("french_labor_code")

    # Get a spread of articles: first SAMPLE_SIZE by offset
    result = collection.get(
        limit=SAMPLE_SIZE,
        include=["documents", "metadatas"],
    )
    texts = result["documents"]
    ids   = [m["article_number"] for m in result["metadatas"]]
    print(f"[compare] Got {len(texts)} articles.")

    # ── 2. Encode corpus + queries with both models ───────────────────────────
    records = []

    for model_key, model_name in MODELS.items():
        print(f"\n[compare] Loading model: {model_name}")
        model = SentenceTransformer(model_name)

        # Encode corpus
        t0 = time.perf_counter()
        corpus_vecs = model.encode(texts, show_progress_bar=True,
                                   batch_size=64, normalize_embeddings=True)
        corpus_time = time.perf_counter() - t0
        print(f"  Corpus encoded in {corpus_time:.1f}s  "
              f"| dim={corpus_vecs.shape[1]}")

        # Encode + retrieve for each question
        for question in TEST_QUESTIONS:
            t_q = time.perf_counter()
            q_vec = model.encode([question], normalize_embeddings=True)[0]
            hits  = cosine_top_k(q_vec, corpus_vecs, ids, TOP_K)
            latency = time.perf_counter() - t_q

            article_ids = [h[0] for h in hits]
            top_score   = hits[0][1] if hits else 0.0
            avg_score   = float(np.mean([h[1] for h in hits]))

            records.append({
                "model":            model_key,
                "question":         question,
                "top_articles":     ", ".join(article_ids),
                "top_score":        round(top_score, 4),
                "avg_top5_score":   round(avg_score, 4),
                "latency_s":        round(latency, 4),
                "n_retrieved":      len(hits),
            })

    # ── 3. Build DataFrame ─────────────────────────────────────────────────────
    df = pd.DataFrame(records)
    df.to_csv(OUT_CSV, index=False)
    print(f"\n[compare] Results saved → {OUT_CSV}")

    # ── 4. Summary ────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)

    summary = (
        df.groupby("model")
          .agg(
              avg_top_score   = ("top_score",       "mean"),
              avg_avg_score   = ("avg_top5_score",  "mean"),
              avg_latency_s   = ("latency_s",       "mean"),
          )
          .round(4)
    )
    print(summary.to_string())

    # Winner by avg_top_score
    best = summary["avg_top_score"].idxmax()
    print(f"\n→ Best model for French legal retrieval: [{best}]")
    print(
        "  (paraphrase-multilingual is expected to outperform all-MiniLM-L6-v2\n"
        "   because the corpus is in French and all-MiniLM-L6-v2 is English-only.)"
    )

    print("\n" + "="*70)
    print("PER-QUESTION DETAIL")
    print("="*70)
    pivot = df.pivot_table(
        index="question",
        columns="model",
        values=["top_score", "avg_top5_score"],
        aggfunc="first",
    ).round(4)
    print(pivot.to_string())


if __name__ == "__main__":
    main()
