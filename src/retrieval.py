"""
src/retrieval.py
----------------
Encodes a user query and retrieves the top-k most relevant articles
from ChromaDB using cosine similarity.

Designed to be imported by pipeline.py — no side-effects on import.
"""

from pathlib import Path
from typing import List, Dict

from sentence_transformers import SentenceTransformer
import chromadb

# ── Paths / constants (must match ingest.py) ──────────────────────────────────
ROOT            = Path(__file__).resolve().parent.parent
DB_PATH         = ROOT / "db"
COLLECTION_NAME = "french_labor_code"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# ── Module-level singletons (loaded once per process) ─────────────────────────
_model:      SentenceTransformer | None = None
_collection: chromadb.Collection | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _get_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(DB_PATH))
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


# ── Public API ────────────────────────────────────────────────────────────────

def retrieve(query: str, top_k: int = 5) -> List[Dict]:
    """
    Encode *query* and return the top_k most relevant articles.

    Strategy: because the user asks in English but the articles are in French,
    we issue two embeddings — the original query AND a direct French translation
    using key French legal terms appended.  We merge and deduplicate the two
    result sets, keeping the best (lowest) distance per article.

    Returns a list of dicts (one per result), sorted by distance ascending:
        {
            "article_number": str,
            "article_type":   str,
            "char_count":     int,
            "text":           str,
            "distance":       float,
        }
    """
    model      = _get_model()
    collection = _get_collection()

    # Three query variants maximise recall across English and French queries:
    #   1. Original query as typed (English or French)
    #   2. Bilingual hybrid: original + French legal terms appended
    #   3. Pure French: domain keywords only (for English input) OR the
    #      original query again (for French input — it IS already in the
    #      right embedding space, so no penalty for reusing it)
    fr_hints = _to_french_hints(query)
    pure_fr  = _pure_french(query) or query   # fallback to original if already French
    query_variants = [query, fr_hints, pure_fr]

    # Encode all variants at once (efficient batched encoding)
    embeddings = model.encode(query_variants).tolist()

    # Fetch top_k * 4 per variant
    fetch_k = top_k * 4

    # ── Reciprocal Rank Fusion (RRF) merge ────────────────────────────────────
    # Simple min-distance merge fails when the English query consistently pulls
    # irrelevant articles that score better than the correct French ones.
    # RRF weights each article by its rank position across all variants:
    #   RRF_score(d) = Σ  1 / (rank_in_variant + K)
    # A high RRF score means the article ranked well in MULTIPLE variants,
    # making it robust to any single noisy variant.
    RRF_K = 60   # standard constant; dampens rank-1 dominance
    # pure-FR variant (index 2) gets 2× weight — it lives in the same embedding
    # space as the corpus and is the most reliable signal for French legal text.
    variant_weights = [1.0, 1.0, 2.0]
    rrf_scores: Dict[str, float] = {}
    texts_meta: Dict[str, tuple] = {}

    for emb, weight in zip(embeddings, variant_weights):
        results = collection.query(
            query_embeddings=[emb],
            n_results=fetch_k,
            include=["documents", "metadatas", "distances"],
        )
        for rank, (doc, meta, dist) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )):
            art_id = meta.get("article_number", "unknown")
            rrf_scores[art_id] = rrf_scores.get(art_id, 0.0) + weight / (rank + RRF_K)
            if art_id not in texts_meta:
                texts_meta[art_id] = (doc, meta, dist)

    # Sort by descending RRF score and return top_k
    sorted_ids = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)[:top_k]
    hits = []
    for art_id in sorted_ids:
        doc, meta, dist = texts_meta[art_id]
        hits.append(
            {
                "article_number": meta.get("article_number", "unknown"),
                "article_type":   meta.get("article_type",   "?"),
                "char_count":     meta.get("char_count",      0),
                "text":           doc,
                "distance":       dist,
            }
        )
    return hits


# ── French keyword expansion map ──────────────────────────────────────────────
# Maps key English legal phrases to their French equivalents so that the second
# query variant is in the same language as the document corpus.

# Each entry: (english_trigger, french_expansion, pure_fr).
# - english_trigger : substring matched against lowercased query
# - french_expansion: appended in the bilingual hybrid variant
# - pure_fr         : True = also included in the pure-French-only variant
#                     False = too generic; bilingual only (dilutes pure-FR embedding)
_EN_TO_FR: List[tuple] = [
    # Leave
    ("paid leave",      "congé payé droits congés",                       True),
    ("annual leave",    "congé annuel payé période",                       True),
    ("parental leave",  "congé parental éducation droits salarié ancienneté protection", True),
    ("sick leave",      "arrêt maladie indemnité journalière maladie L1226", True),
    ("maternity",       "congé maternité",                                 True),
    ("paternity",       "congé paternité",                                 True),
    # Dismissal / termination
    ("dismiss",         "licenciement motif personnel cause réelle sérieuse", True),
    ("grounds for",     "licenciement cause réelle sérieuse justifiée",      True),
    ("fire ",           "licenciement rupture contrat travail",               True),
    ("terminat",        "rupture contrat travail",                            True),
    ("redundan",        "licenciement économique suppression emploi",         True),
    ("layoff",          "licenciement économique",                            True),
    # Hours / overtime
    ("overtime",        "heures supplémentaires majoration taux 25 50",       True),
    ("working hour",    "durée légale travail 35 heures hebdomadaire",        True),
    ("work hour",       "durée légale travail hebdomadaire",                  True),
    ("working time",    "durée légale travail",                               True),
    ("maximum hour",    "durée maximale hebdomadaire quotidienne",            True),
    # Trial period
    ("trial period",    "période essai renouvellement durée maximale",        True),
    ("probation",       "période essai",                                      True),
    # Pay / wages
    ("minimum wage",    "salaire minimum SMIC calcul",                        True),
    ("wage",            "salaire rémunération",                               True),
    ("compensation",    "majoration indemnité",                               False),  # generic — bilingual only
    # Other domain-specific (pure-FR = True)
    ("harassment",      "harcèlement moral",                               True),
    ("notice period",   "préavis durée délai minimum licenciement",         True),
    ("notice ",         "préavis durée délai rupture contrat",             True),
    ("collective barg", "convention collective accord",                    True),
    ("union",           "syndicat représentant",                           True),
    ("resignation",     "démission",                                       True),
    # Generic — bilingual hybrid only (pure_fr = False)
    ("contract",        "contrat de travail",                              False),
    ("employer",        "employeur",                                       False),
    ("employee",        "salarié",                                         False),
    ("refuse",          "refus",                                           False),
    ("grant",           "accorder droit",                                  False),
    ("rights",          "droits",                                          False),
    ("conditions",      "conditions modalités",                            False),
]


def _to_french_hints(query: str) -> str:
    """Return query + French legal terms appended (bilingual hybrid)."""
    q_lower = query.lower()
    extras = [fr for en, fr, _ in _EN_TO_FR if en in q_lower]
    return (query + " " + " ".join(extras)) if extras else query


def _pure_french(query: str) -> str:
    """
    Return only domain-specific French keywords (pure_fr=True entries).
    Generic terms are excluded so they don't dilute the embedding.
    Returns empty string if the query is already in French or no terms match.
    """
    q_lower = query.lower()
    extras = [fr for en, fr, keep in _EN_TO_FR if keep and en in q_lower]
    return " ".join(extras)


def format_context(hits: List[Dict]) -> str:
    """
    Format retrieved articles into a string ready to be injected into
    the LLM prompt.

    Each article is preceded by a clear header so the LLM can cite it.
    """
    parts = []
    for h in hits:
        parts.append(
            f"--- Article {h['article_number']} ---\n{h['text'].strip()}"
        )
    return "\n\n".join(parts)
