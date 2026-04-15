"""
app.py
------
Gradio web interface for the French Labor Code RAG assistant.
Modern dark/light design — navy + gold palette.
"""

from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

from src.ingest   import ingest
from src.pipeline import run_pipeline

print("[app] Checking vector store …")
ingest()


# ── Custom CSS ────────────────────────────────────────────────────────────────
CSS = """
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif; }

/* ════════════════════════════════════════
   DARK MODE  (default — html.dark)
   ════════════════════════════════════════ */
html.dark body,
html.dark .gradio-container {
    background: linear-gradient(160deg, #070f1e 0%, #0d1b35 60%, #0a1525 100%) !important;
    color: #dde3f0 !important;
}

html.dark .block,
html.dark .form,
html.dark .gap,
html.dark label.block {
    background: rgba(15, 28, 55, 0.85) !important;
    border: 1px solid rgba(180, 145, 60, 0.18) !important;
    border-radius: 14px !important;
    box-shadow: 0 4px 32px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.04) !important;
}

html.dark textarea,
html.dark input[type="text"] {
    background: rgba(8, 16, 35, 0.9) !important;
    border: 1px solid rgba(180, 145, 60, 0.25) !important;
    color: #dde3f0 !important;
    border-radius: 10px !important;
}

html.dark textarea:focus,
html.dark input[type="text"]:focus {
    border-color: rgba(201, 168, 76, 0.6) !important;
    box-shadow: 0 0 0 3px rgba(201, 168, 76, 0.12) !important;
}

/* ════════════════════════════════════════
   LIGHT MODE  (html:not(.dark))
   ════════════════════════════════════════ */
html:not(.dark) body,
html:not(.dark) .gradio-container {
    background: linear-gradient(160deg, #eef2f9 0%, #f5f7fb 60%, #eef2f9 100%) !important;
    color: #0d1b35 !important;
}

html:not(.dark) .block,
html:not(.dark) .form,
html:not(.dark) .gap,
html:not(.dark) label.block {
    background: #ffffff !important;
    border: 1px solid rgba(14, 42, 90, 0.10) !important;
    border-radius: 14px !important;
    box-shadow: 0 2px 16px rgba(14, 42, 90, 0.08), 0 1px 4px rgba(14, 42, 90, 0.06) !important;
}

html:not(.dark) textarea,
html:not(.dark) input[type="text"] {
    background: #f8fafd !important;
    border: 1px solid rgba(14, 42, 90, 0.15) !important;
    color: #0d1b35 !important;
    border-radius: 10px !important;
}

html:not(.dark) textarea:focus,
html:not(.dark) input[type="text"]:focus {
    border-color: #8b6914 !important;
    box-shadow: 0 0 0 3px rgba(139, 105, 20, 0.10) !important;
}

/* ════════════════════════════════════════
   HEADER
   ════════════════════════════════════════ */
.app-header {
    text-align: center;
    padding: 2rem 1rem 1.5rem;
    margin-bottom: 0.5rem;
}

.app-header .badge {
    display: inline-block;
    font-size: 3rem;
    margin-bottom: 0.6rem;
    filter: drop-shadow(0 2px 8px rgba(201,168,76,0.35));
}

html.dark .app-header h1 {
    font-size: 2rem !important;
    font-weight: 700 !important;
    background: linear-gradient(90deg, #c9a84c, #e8d48a, #c9a84c) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    letter-spacing: -0.5px;
    margin: 0.2rem 0 0.4rem !important;
}

html:not(.dark) .app-header h1 {
    font-size: 2rem !important;
    font-weight: 700 !important;
    color: #0d2257 !important;
    margin: 0.2rem 0 0.4rem !important;
}

html.dark .app-header p {
    color: #8a9ab8 !important;
    font-size: 0.95rem !important;
    font-weight: 400 !important;
}

html:not(.dark) .app-header p {
    color: #4a5878 !important;
    font-size: 0.95rem !important;
}

.app-header .divider {
    width: 60px;
    height: 2px;
    background: linear-gradient(90deg, transparent, #c9a84c, transparent);
    margin: 0.8rem auto 0;
    border-radius: 2px;
}

/* ════════════════════════════════════════
   LABELS
   ════════════════════════════════════════ */
html.dark label span,
html.dark .label-wrap span {
    color: #c9a84c !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}

html:not(.dark) label span,
html:not(.dark) .label-wrap span {
    color: #6b4f10 !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}

/* ════════════════════════════════════════
   BUTTONS
   ════════════════════════════════════════ */
button.primary {
    background: linear-gradient(135deg, #b8922a 0%, #c9a84c 50%, #b8922a 100%) !important;
    border: none !important;
    color: #0a1525 !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em !important;
    border-radius: 10px !important;
    box-shadow: 0 2px 12px rgba(185, 146, 42, 0.35) !important;
    transition: all 0.2s ease !important;
}

button.primary:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(185, 146, 42, 0.5) !important;
}

html.dark button.secondary {
    background: rgba(20, 35, 65, 0.9) !important;
    border: 1px solid rgba(180, 145, 60, 0.3) !important;
    color: #b8a070 !important;
    font-weight: 500 !important;
    border-radius: 10px !important;
    transition: all 0.2s ease !important;
}

html:not(.dark) button.secondary {
    background: #f0f4fa !important;
    border: 1px solid rgba(14, 42, 90, 0.15) !important;
    color: #3a4f78 !important;
    font-weight: 500 !important;
    border-radius: 10px !important;
    transition: all 0.2s ease !important;
}

html.dark button.secondary:hover {
    border-color: rgba(201, 168, 76, 0.5) !important;
    color: #c9a84c !important;
}

/* ════════════════════════════════════════
   ACCORDION (sources)
   ════════════════════════════════════════ */
html.dark .accordion {
    background: rgba(4, 10, 24, 0.95) !important;
}

html.dark .accordion > .label-wrap {
    color: #b8a070 !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
}

html:not(.dark) .accordion > .label-wrap {
    color: #6b4f10 !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
}

/* ════════════════════════════════════════
   MARKDOWN OUTPUT
   ════════════════════════════════════════ */
html.dark .prose p, html.dark .prose li {
    color: #b0bccc !important;
    font-size: 0.88rem !important;
    line-height: 1.7 !important;
}

html.dark .prose strong {
    color: #c9a84c !important;
}

html:not(.dark) .prose strong {
    color: #7a5510 !important;
}

html:not(.dark) .prose p, html:not(.dark) .prose li {
    color: #1e2a45 !important;
    font-size: 0.88rem !important;
    line-height: 1.7 !important;
}

/* Gradio 6.x markdown uses span.md, not .prose */
html.dark span.md p, html.dark span.md li, html.dark span.md em {
    color: #b0bccc !important;
}
html:not(.dark) span.md p, html:not(.dark) span.md li, html:not(.dark) span.md em {
    color: #1e2a45 !important;
}
html.dark span.md strong { color: #c9a84c !important; }
html:not(.dark) span.md strong { color: #7a5510 !important; }

/* ════════════════════════════════════════
   SCROLLBAR (dark)
   ════════════════════════════════════════ */
html.dark ::-webkit-scrollbar { width: 6px; }
html.dark ::-webkit-scrollbar-track { background: #0a1525; }
html.dark ::-webkit-scrollbar-thumb {
    background: rgba(201, 168, 76, 0.3);
    border-radius: 3px;
}
html.dark ::-webkit-scrollbar-thumb:hover {
    background: rgba(201, 168, 76, 0.5);
}

/* ════════════════════════════════════════
   FOOTER — keep settings gear, hide only the "Built with Gradio" text
   ════════════════════════════════════════ */
footer .svelte-1rjryqp { display: none !important; }
"""

# Default to dark mode on first load
JS_DARK_DEFAULT = """
() => {
    if (!localStorage.getItem('theme-set')) {
        document.documentElement.classList.add('dark');
        localStorage.setItem('theme-set', 'dark');
    }
}
"""


# ── Gradio callback ───────────────────────────────────────────────────────────

def answer_question(question: str):
    if not question.strip():
        return "Please enter a question.", ""

    answer, hits, latency = run_pipeline(question)

    source_lines = []
    for h in hits:
        preview = h["text"][:220].replace("\n", " ").strip()
        source_lines.append(
            f"**Article {h['article_number']}** &nbsp;·&nbsp; score : {h['distance']:.3f}\n\n"
            f"*{preview}…*"
        )
    sources_md = "\n\n---\n\n".join(source_lines)
    sources_md += f"\n\n---\n⏱ *Latency: {latency:.1f}s*"

    return answer, sources_md


# ── Interface ─────────────────────────────────────────────────────────────────

theme = gr.themes.Base(
    primary_hue=gr.themes.colors.yellow,
    neutral_hue=gr.themes.colors.slate,
    radius_size=gr.themes.sizes.radius_lg,
    font=gr.themes.GoogleFont("Inter"),
)

with gr.Blocks(
    title="Code du Travail — Assistant IA",
    theme=theme,
    css=CSS,
    js=JS_DARK_DEFAULT,
) as demo:

    # ── Header ────────────────────────────────────────────────────────────────
    gr.Markdown(
        """
        <div class="app-header">
          <div class="badge">⚖️</div>
          <h1>French Labor Code Assistant</h1>
          <p>Ask your question in English or French — relevant articles are cited automatically.</p>
          <div class="divider"></div>
        </div>
        """
    )

    # ── Input ─────────────────────────────────────────────────────────────────
    question_box = gr.Textbox(
        label="Your question",
        placeholder="e.g. How many days of paid leave is an employee entitled to?",
        lines=3,
    )

    with gr.Row():
        submit_btn = gr.Button("⟶  Submit", variant="primary", scale=3)
        clear_btn  = gr.Button("✕  Clear",  variant="secondary", scale=1)

    # ── Output ────────────────────────────────────────────────────────────────
    answer_box = gr.Textbox(
        label="Answer",
        lines=10,
        interactive=False,
    )

    with gr.Accordion("📄  Source articles", open=False):
        gr.HTML("""<style>
            .sources-wrap { background: #03080f; border-radius: 10px; padding: 12px; }
            .sources-wrap p, .sources-wrap li, .sources-wrap em { color: #c8d4e8 !important; font-size: 0.88rem; line-height: 1.7; }
            .sources-wrap strong { color: #c9a84c !important; }
        </style>""")
        sources_box = gr.Markdown(elem_classes=["sources-wrap"])

    # ── Events ────────────────────────────────────────────────────────────────
    submit_btn.click(
        fn=answer_question,
        inputs=[question_box],
        outputs=[answer_box, sources_box],
    )
    clear_btn.click(
        fn=lambda: ("", ""),
        inputs=[],
        outputs=[question_box, answer_box],
    )
    question_box.submit(
        fn=answer_question,
        inputs=[question_box],
        outputs=[answer_box, sources_box],
    )


if __name__ == "__main__":
    demo.launch(share=False)
