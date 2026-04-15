"""
src/generation.py
-----------------
Wraps the HuggingFace InferenceClient to generate answers.

The prompt template instructs Mistral-7B to:
  - Cite article numbers explicitly
  - Answer in English
  - Be factually accurate and concise
  - Admit when context is insufficient

Environment variable required:  HF_TOKEN  (loaded via python-dotenv)
"""

import os
import re
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load .env from project root (works whether called as module or script)
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# ── Constants ─────────────────────────────────────────────────────────────────
# Mistral-7B-Instruct-v0.3 is no longer available on the HF free-tier router.
# Qwen2.5-7B-Instruct is an equivalent open-weight instruction model that works.
# Swap back to Mistral if you upgrade to a paid HF plan.
HF_MODEL    = "Qwen/Qwen2.5-7B-Instruct"
MAX_TOKENS  = 512
TEMPERATURE = 0.1

# ── Language detection ────────────────────────────────────────────────────────
# Two tiers:
#   STRONG — French legal/domain words that never appear in English text.
#            A single match is enough to classify as French.
#   WEAK   — Common French function words shared with other Romance languages.
#            Two matches required to avoid false positives on short inputs.
_FR_STRONG = re.compile(
    r"\b(salarié|employeur|congé|licenciement|préavis|heures?\s+supplémentaires?"
    r"|durée\s+du\s+travail|contrat\s+de\s+travail|période\s+d.essai"
    r"|harcèlement|convention\s+collective|salaire\s+minimum|ancienneté"
    r"|règles?|quelles?|combien|est-ce|comment|quand|pourquoi)\b",
    re.IGNORECASE,
)
_FR_WEAK = re.compile(
    r"\b(le|la|les|un|une|des|du|de|je|tu|il|elle|nous|vous|ils|elles"
    r"|est|sont|avec|pour|dans|sur|par|que|qui|aux|au|en|et|ou)\b",
    re.IGNORECASE,
)

def detect_language(text: str) -> str:
    """Return 'fr' if the text looks French, 'en' otherwise."""
    if _FR_STRONG.search(text):
        return "fr"
    return "fr" if len(_FR_WEAK.findall(text)) >= 2 else "en"


# ── Prompt templates ──────────────────────────────────────────────────────────
_PROMPT_EN = """\
You are a French labor law expert. Answer the question below using ONLY the articles provided.
Rules:
- Be concise and direct — 2 to 4 sentences maximum.
- Start your answer immediately with the legal fact (no preamble).
- Always cite at least one article number inline, e.g. "Under Article L1234-1, …".
- If the articles are only partially relevant, cite what they do say and note what is missing.
- Only reply "The provided articles do not cover this point." if NONE of the articles relate to the topic at all.
- Answer in English.

Articles:
{context}

Question: {question}

Answer:"""

_PROMPT_FR = """\
Tu es un expert en droit du travail français. Réponds à la question ci-dessous en utilisant UNIQUEMENT les articles fournis.
Règles :
- Sois concis et direct — 2 à 4 phrases maximum.
- Commence immédiatement par le fait juridique (sans introduction).
- Cite les numéros d'articles directement, ex. : « Selon l'article L1234-1, … ».
- Cite toujours au moins un numéro d'article, ex. : « Selon l'article L1234-1, … ».
- Si les articles sont partiellement pertinents, cite ce qu'ils disent et précise ce qui manque.
- Réponds « Les articles fournis ne couvrent pas ce point. » uniquement si AUCUN des articles n'est lié au sujet.
- Réponds en français.

Articles :
{context}

Question : {question}

Réponse :"""

# ── Module-level singleton ─────────────────────────────────────────────────────
_client: InferenceClient | None = None


def _get_client() -> InferenceClient:
    global _client
    if _client is None:
        token = os.getenv("HF_TOKEN")
        if not token:
            raise EnvironmentError(
                "HF_TOKEN not found. Add it to your .env file or export it in the shell."
            )
        _client = InferenceClient(model=HF_MODEL, token=token)
    return _client


# ── Public API ────────────────────────────────────────────────────────────────

def generate_answer(question: str, context: str) -> str:
    """
    Build the prompt, call Mistral-7B via the HF Inference API, and return
    the generated answer string.

    Args:
        question: the user's natural-language question
        context:  the pre-formatted retrieved articles (from retrieval.format_context)

    Returns:
        The model's answer as a plain string.

    Raises:
        EnvironmentError: if HF_TOKEN is missing
        huggingface_hub.errors.HfHubHTTPError: if the API call fails
    """
    client = _get_client()

    template = _PROMPT_FR if detect_language(question) == "fr" else _PROMPT_EN
    prompt = template.format(context=context, question=question)

    # Use chat_completion — the current HF InferenceClient routes Mistral via
    # conversational providers (novita, together, etc.) which require this API.
    response = client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )

    return response.choices[0].message.content.strip()
