"""Gradio demo for docvault — Ingest PDFs and query them with source citations."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure the src layout is on the path when running app.py directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import gradio as gr

from docvault.config import Config
from docvault.ingest.chunking import ChunkingStrategy
from docvault.pipeline import Pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared state — one Pipeline instance per server process
# ---------------------------------------------------------------------------

_config = Config()
_pipeline = Pipeline(config=_config)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _index_stats_md() -> str:
    """Return a Markdown string summarising the current index state."""
    try:
        stats = _pipeline.stats()
        docs = stats["total_docs"]
        chunks = stats["total_chunks"]
        if chunks == 0:
            return "_No documents indexed yet._"
        doc_list = ", ".join(f"`{d}`" for d in stats["doc_names"][:5])
        if docs > 5:
            doc_list += f" … and {docs - 5} more"
        return (
            f"**{docs} document(s)** · **{chunks} chunk(s)** in index\n\n"
            f"{doc_list}"
        )
    except Exception as exc:
        return f"_Could not read index: {exc}_"


def _citation_cards_html(cited_sources: list) -> str:
    """Render citation cards as an HTML string."""
    if not cited_sources:
        return "<p style='color:#94a3b8;font-style:italic;'>No citations found in response.</p>"

    cards: list[str] = []
    for src in cited_sources:
        # Escape HTML in chunk text
        chunk_escaped = (
            src.chunk_text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        cards.append(
            f"""
<div style="border:1px solid #e2e8f0;border-radius:8px;margin:8px 0;overflow:hidden;">
  <details>
    <summary style="padding:12px 16px;cursor:pointer;background:#f8fafc;
                    font-weight:600;list-style:none;display:flex;
                    justify-content:space-between;align-items:center;">
      <span>[{src.citation_number}] &nbsp;{src.doc_name} &mdash; page {src.page_number}</span>
      <span style="color:#64748b;font-size:0.82em;font-weight:400;">
        relevance&nbsp;{src.score:.3f}
      </span>
    </summary>
    <div style="padding:12px 16px;font-size:0.88em;color:#475569;
                background:#ffffff;border-top:1px solid #e2e8f0;
                white-space:pre-wrap;line-height:1.6;">
{chunk_escaped}
    </div>
  </details>
</div>"""
        )
    return "\n".join(cards)


# ---------------------------------------------------------------------------
# Ingest callback
# ---------------------------------------------------------------------------


def run_ingest(
    files: list | None,
    strategy: str,
    progress: gr.Progress = gr.Progress(track_tqdm=False),
) -> tuple[str, str]:
    """Handle the Build Index button click.

    Args:
        files: Uploaded file objects from ``gr.File``.
        strategy: Selected chunking strategy label.
        progress: Gradio progress tracker.

    Returns:
        Tuple of ``(status_message, updated_stats_markdown)``.
    """
    if not files:
        return "⚠️ Please upload at least one PDF before building the index.", _index_stats_md()

    strategy_map = {
        "Recursive (recommended)": ChunkingStrategy.RECURSIVE,
        "Fixed-size": ChunkingStrategy.FIXED,
        "Semantic": ChunkingStrategy.SEMANTIC,
    }
    chosen = strategy_map.get(strategy, ChunkingStrategy.RECURSIVE)

    pdf_paths = [Path(f.name) for f in files]

    progress(0.05, desc="Starting ingest…")
    try:
        progress(0.15, desc="Extracting text from PDFs…")
        summary = _pipeline.ingest(pdf_paths, strategy=chosen, show_progress=False)
        progress(1.0, desc="Done!")
    except Exception as exc:
        logger.exception("Ingest failed")
        return f"❌ Ingest failed: {exc}", _index_stats_md()

    docs = summary["docs_processed"]
    chunks = summary["total_chunks"]
    hits = summary["cache_hits"]
    misses = summary["cache_misses"]
    total = summary["index_total"]

    status = (
        f"✅ Indexed **{docs}** document(s) → **{chunks}** chunk(s)  \n"
        f"Embeddings: {misses} new, {hits} from cache  \n"
        f"Index total: **{total}** vector(s)"
    )
    return status, _index_stats_md()


# ---------------------------------------------------------------------------
# Query callback
# ---------------------------------------------------------------------------


def run_query(
    question: str,
    top_k: int,
    use_rerank: bool,
    progress: gr.Progress = gr.Progress(track_tqdm=False),
) -> tuple[str, str]:
    """Handle the Ask button click.

    Args:
        question: User question string.
        top_k: Number of chunks to retrieve.
        use_rerank: Whether to apply cross-encoder reranking.
        progress: Gradio progress tracker.

    Returns:
        Tuple of ``(answer_markdown, citations_html)``.
    """
    if not question.strip():
        return "⚠️ Please enter a question.", ""

    progress(0.1, desc="Retrieving relevant chunks…")
    try:
        response = _pipeline.query(question, top_k=int(top_k), rerank=use_rerank)
        progress(1.0, desc="Done!")
    except RuntimeError as exc:
        return f"❌ {exc}", ""
    except Exception as exc:
        logger.exception("Query failed")
        return f"❌ Query failed: {exc}", ""

    answer_md = response.answer
    citations_html = _citation_cards_html(response.cited_sources)
    return answer_md, citations_html


# ---------------------------------------------------------------------------
# Gradio layout
# ---------------------------------------------------------------------------


def build_app() -> gr.Blocks:
    """Construct and return the Gradio Blocks application."""

    with gr.Blocks(
        title="docvault",
        theme=gr.themes.Default(
            primary_hue="slate",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter"),
        ),
        css="""
        .tab-nav button { font-size: 1rem; font-weight: 600; }
        footer { display: none !important; }
        """,
    ) as app:

        gr.Markdown(
            """
# docvault
**Retrieval-Augmented Generation over your PDF documents**
"""
        )

        # ── Shared index status bar ────────────────────────────────────────
        index_status = gr.Markdown(value=_index_stats_md(), label="Index status")

        with gr.Tabs():

            # ── Tab 1: Ingest ──────────────────────────────────────────────
            with gr.TabItem("📥 Ingest"):
                gr.Markdown("Upload one or more PDFs, choose a chunking strategy, then click **Build Index**.")

                with gr.Row():
                    with gr.Column(scale=3):
                        file_upload = gr.File(
                            label="PDF files",
                            file_count="multiple",
                            file_types=[".pdf"],
                        )
                    with gr.Column(scale=1):
                        strategy_dd = gr.Dropdown(
                            choices=[
                                "Recursive (recommended)",
                                "Fixed-size",
                                "Semantic",
                            ],
                            value="Recursive (recommended)",
                            label="Chunking strategy",
                        )
                        ingest_btn = gr.Button("Build Index", variant="primary")

                ingest_status = gr.Markdown(label="Status")

                ingest_btn.click(
                    fn=run_ingest,
                    inputs=[file_upload, strategy_dd],
                    outputs=[ingest_status, index_status],
                )

            # ── Tab 2: Query ───────────────────────────────────────────────
            with gr.TabItem("🔍 Query"):
                gr.Markdown("Ask a question. Answers are grounded in your indexed documents.")

                with gr.Row():
                    with gr.Column(scale=5):
                        question_box = gr.Textbox(
                            placeholder="e.g. What are the key findings of the report?",
                            label="Question",
                            lines=2,
                        )
                    with gr.Column(scale=1):
                        top_k_slider = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=_config.top_k,
                            step=1,
                            label="Top-k chunks",
                        )
                        rerank_check = gr.Checkbox(
                            value=_config.rerank,
                            label="Rerank results",
                        )
                        ask_btn = gr.Button("Ask", variant="primary")

                answer_box = gr.Markdown(label="Answer")

                gr.Markdown("### Sources")
                citations_box = gr.HTML()

                ask_btn.click(
                    fn=run_query,
                    inputs=[question_box, top_k_slider, rerank_check],
                    outputs=[answer_box, citations_box],
                )

                # Allow submitting with Enter in the question box
                question_box.submit(
                    fn=run_query,
                    inputs=[question_box, top_k_slider, rerank_check],
                    outputs=[answer_box, citations_box],
                )

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
