"""
src/ingest.py
-------------
Ingests the French Labor Code PDF into ChromaDB.

Pipeline:
  PDF (pypdf) → raw text → regex article-level chunks
  → Sentence-Transformer embeddings → ChromaDB persistent store

Run once:  python -m src.ingest
Re-runs are safe: if the collection is already populated the function
returns immediately without re-encoding anything.

Article format in this PDF
--------------------------
Each article appears on its own line as:
    L. 1222-2  Ordonnance 2007-329 2007-03-12 JORF 13 mars 2007  -  Conseil Constit. …
    <metadata lines: Legif. / Plan / Jp.Judi. / Jp.Admin. / Juricaf>
    <actual legal text>

The regex therefore anchors at the START of a line (re.MULTILINE) and
matches  ^[LRD]\\. \\d{3,4}-\\d+  followed by at least one space.
"""

import re
import os
from pathlib import Path
from typing import List, Dict

import pypdf
from sentence_transformers import SentenceTransformer
import chromadb
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent
PDF_PATH = ROOT / "data" / "Code_du_travail.pdf"
DB_PATH  = ROOT / "db"

# ── Constants ─────────────────────────────────────────────────────────────────
COLLECTION_NAME = "french_labor_code"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# Matches article headers at the START of a line, e.g.:
#   "L. 1222-2  Ordonnance …"  or  "R. 3142-50  Décret …"
# Captures: group(1) = type letter, group(2) = numeric part
ARTICLE_HEADER = re.compile(
    r"^([LRD])\.\s+(\d{3,4}-\d+(?:-\d+)*)\s",
    re.MULTILINE,
)

# Lines that are PDF navigation artefacts — strip them from article bodies
JUNK_LINES = re.compile(
    r"^\s*(?:Legif\.|Plan|Jp\.Judi\.|Jp\.Admin\.|Juricaf)\s*$",
    re.MULTILINE,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_raw_text(pdf_path: Path) -> str:
    """
    Read all 2782 pages with pypdf and return a single concatenated string.
    pypdf is significantly faster than PyPDFLoader for large PDFs.
    """
    print(f"[ingest] Reading PDF: {pdf_path.name}")
    reader = pypdf.PdfReader(str(pdf_path))
    pages  = reader.pages
    print(f"[ingest] {len(pages)} pages found — extracting text …")
    parts = []
    for page in tqdm(pages, desc="  pages", unit="pg"):
        parts.append(page.extract_text() or "")
    return "\n".join(parts)


def clean_body(text: str) -> str:
    """Remove PDF navigation artefact lines from article body."""
    return JUNK_LINES.sub("", text).strip()


def split_into_articles(raw_text: str) -> List[Dict]:
    """
    Split raw_text into article-level chunks.

    Returns a list of dicts:
        {
            "article_number": "L1222-2",   # normalised (no period-space)
            "article_type":   "L",
            "raw_text":       str,
            "char_count":     int,
        }
    """
    matches = list(ARTICLE_HEADER.finditer(raw_text))
    if not matches:
        raise ValueError(
            "No articles detected — check ARTICLE_HEADER regex or PDF quality."
        )
    print(f"[ingest] {len(matches)} article headers detected.")

    articles: List[Dict] = []
    seen:     Dict[str, int] = {}   # article_number → index in articles list

    for i, m in enumerate(matches):
        # Article text runs from this match to the start of the next (or EOF)
        end  = matches[i + 1].start() if i + 1 < len(matches) else len(raw_text)
        body = clean_body(raw_text[m.start():end])

        # Normalise article number: "L. 1222-2" → "L1222-2"
        article_number = f"{m.group(1)}{m.group(2)}"

        # Skip very short entries (TOC stubs)
        if len(body) < 80:
            continue

        if article_number in seen:
            # Keep the longest occurrence (body text > TOC entry)
            idx = seen[article_number]
            if len(body) > articles[idx]["char_count"]:
                articles[idx] = _make_chunk(article_number, body)
        else:
            seen[article_number] = len(articles)
            articles.append(_make_chunk(article_number, body))

    print(f"[ingest] {len(articles)} unique articles after deduplication.")
    return articles


def _make_chunk(article_number: str, text: str) -> Dict:
    return {
        "article_number": article_number,
        "article_type":   article_number[0],
        "raw_text":       text,
        "char_count":     len(text),
    }


# ── Main ingestion function ────────────────────────────────────────────────────

def ingest(force: bool = False) -> chromadb.Collection:
    """
    Full ingestion pipeline.  Returns the populated ChromaDB collection.

    Args:
        force: drop the existing collection and re-ingest from scratch.
    """
    client   = chromadb.PersistentClient(path=str(DB_PATH))
    existing = [c.name for c in client.list_collections()]

    if COLLECTION_NAME in existing and not force:
        collection = client.get_collection(COLLECTION_NAME)
        count = collection.count()
        if count > 0:
            print(
                f"[ingest] Collection '{COLLECTION_NAME}' already has {count} docs. "
                "Skipping re-ingestion.  Pass force=True to re-index."
            )
            return collection

    if COLLECTION_NAME in existing:
        print("[ingest] Dropping existing collection …")
        client.delete_collection(COLLECTION_NAME)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # ── 1. Parse PDF ──────────────────────────────────────────────────────────
    raw_text = load_raw_text(PDF_PATH)
    articles = split_into_articles(raw_text)

    # ── 2. Load embedding model ───────────────────────────────────────────────
    print(f"[ingest] Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # ── 3. Encode + upsert in batches ─────────────────────────────────────────
    BATCH_SIZE = 64
    print(f"[ingest] Encoding and storing {len(articles)} articles …")

    for i in tqdm(range(0, len(articles), BATCH_SIZE), desc="  batches"):
        batch = articles[i : i + BATCH_SIZE]

        texts     = [a["raw_text"]       for a in batch]
        ids       = [a["article_number"] for a in batch]
        metadatas = [
            {
                "article_number": a["article_number"],
                "article_type":   a["article_type"],
                "char_count":     a["char_count"],
            }
            for a in batch
        ]

        embeddings = model.encode(texts, show_progress_bar=False).tolist()

        collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    print(f"[ingest] Done — {collection.count()} articles in ChromaDB.")
    return collection


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ingest(force=True)   # force=True so CLI always re-indexes from scratch
