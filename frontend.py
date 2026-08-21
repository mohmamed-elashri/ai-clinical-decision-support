"""Gradio frontend for AI Clinical Decision Support Lite.

The frontend talks to the existing Flask API and deliberately keeps the RAG
pipeline behind that HTTP boundary.
"""

import html
import logging
import os

import gradio as gr
import requests
from dotenv import load_dotenv

load_dotenv()

LOGGER = logging.getLogger("clinical_frontend")
FLASK_API_URL = os.environ.get("FLASK_API_URL", "http://127.0.0.1:7860").rstrip("/")
FLASK_API_KEY = os.environ.get("FLASK_API_KEY", os.environ.get("API_AUTH_TOKEN", ""))
GRADIO_PORT = int(os.environ.get("GRADIO_PORT", os.environ.get("PORT", 7860)))
GRADIO_SERVER_NAME = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
GRADIO_SHARE = os.environ.get("GRADIO_SHARE", "0") == "1"

EXAMPLE_QUESTIONS = [
    "What is the recommended first-line treatment for hypertension?",
    "When should combination therapy be considered?",
    "What blood pressure target is recommended for adults?",
    "What are the recommendations for patients with diabetes?",
]

METHOD_LABELS = {
    "llm": "NVIDIA NIM",
    "extractive_fallback": "Extractive fallback",
    "none": "No generation",
}


def _request_headers() -> dict:
    return {"X-API-Key": FLASK_API_KEY} if FLASK_API_KEY else {}


def _call_ask(question: str, k: int, use_reranker: bool) -> dict:
    """Call the existing API and convert transport failures to safe UI data."""
    try:
        response = requests.post(
            f"{FLASK_API_URL}/api/ask",
            json={"question": question, "k": k, "use_reranker": use_reranker},
            headers=_request_headers(),
            timeout=20,
        )
        payload = response.json()
        if response.status_code == 400:
            return {"_http_error": payload.get("error", "Please check the question and try again.")}
        if not response.ok:
            LOGGER.error("Backend request failed with status %s: %s", response.status_code, payload)
            return {"_http_error": "The evidence engine is temporarily unavailable."}
        return payload
    except requests.exceptions.ConnectionError:
        LOGGER.error("Could not connect to backend at %s", FLASK_API_URL)
        return {"_http_error": "The evidence engine is offline. Please start the clinical API and try again."}
    except (requests.exceptions.RequestException, ValueError) as exc:
        LOGGER.error("Frontend request failed: %s", exc)
        return {"_http_error": "The evidence engine could not complete the request."}


def _safe_score(value) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _display_document(value) -> str:
    document = str(value or "Unknown source")
    return document.replace(".pdf", "").replace("_", " ").title()


def _status_badge(label: str, tone: str = "idle") -> str:
    return f'<span class="status-badge status-{tone}"><span class="status-dot"></span>{html.escape(label)}</span>'


def _format_citations(citations: list) -> str:
    if not citations:
        return '<p class="muted-note">No citation metadata was returned for this response.</p>'

    cards = []
    for citation in citations:
        document = html.escape(_display_document(citation.get("document_name")))
        page = html.escape(str(citation.get("page_number", "Not provided")))
        chunk_id = citation.get("chunk_id")
        chunk_html = (
            f'<span class="source-detail">Chunk {html.escape(str(chunk_id))}</span>'
            if chunk_id
            else ""
        )
        cards.append(
            f"""
            <article class="source-card">
              <div class="source-icon" aria-hidden="true">DOC</div>
              <div class="source-content">
                <strong>{document}</strong>
                <div class="source-meta"><span>Page {page}</span>{chunk_html}</div>
              </div>
            </article>
            """
        )
    return '<div class="source-list">' + "".join(cards) + "</div>"


def _format_chunks(chunks: list) -> str:
    if not chunks:
        return '<p class="muted-note">No evidence passages were returned.</p>'

    cards = []
    for index, chunk in enumerate(chunks, start=1):
        score = _safe_score(chunk.get("score", chunk.get("dense_score")))
        confidence = html.escape(str(chunk.get("confidence", "not labelled")).title())
        document = html.escape(_display_document(chunk.get("document_name")))
        page = html.escape(str(chunk.get("page_number", "Not provided")))
        chunk_id = html.escape(str(chunk.get("chunk_id", "Not provided")))
        passage = html.escape(str(chunk.get("text", "")).strip())
        cards.append(
            f"""
            <details class="evidence-card">
              <summary>
                <span class="evidence-title"><span class="evidence-index">{index:02d}</span>{document}</span>
                <span class="evidence-score">{score:.3f}</span>
              </summary>
              <div class="evidence-body">
                <div class="evidence-meta"><span>Page {page}</span><span>Chunk {chunk_id}</span><span>{confidence}</span></div>
                <blockquote>{passage or "No passage text returned."}</blockquote>
              </div>
            </details>
            """
        )
    return '<div class="evidence-list">' + "".join(cards) + "</div>"


def _build_answer_html(data: dict) -> str:
    """Render the response card from fields returned by the backend."""
    if data.get("status") == "abstain":
        reason = html.escape(str(data.get("reason") or "The indexed guidelines do not provide enough evidence."))
        return f"""
        <div class="answer-state answer-state-warning">
          <div class="answer-state-label">Outside indexed guideline scope</div>
          <p>{reason}</p>
          <span class="answer-state-note">No clinical recommendation was generated.</span>
        </div>
        """

    answer = str(data.get("answer") or "").strip()
    if not answer:
        return '<div class="answer-state"><div class="answer-state-label">No answer generated</div><p>The evidence engine did not return a clinical response.</p></div>'
    return answer


def _build_metrics_html(data: dict) -> str:
    status = "Grounded" if data.get("status") == "grounded" else "Abstained"
    tone = "good" if status == "Grounded" else "warning"
    confidence = str(data.get("confidence", "insufficient")).title()
    method = METHOD_LABELS.get(data.get("generation_method", "none"), "Unknown")
    chunks = data.get("retrieved_chunks") or []
    score = _safe_score(data.get("top_score"))
    return f"""
    <div class="metrics-grid" aria-label="Response metadata">
      <div class="metric-card"><span>Decision</span><strong class="metric-{tone}">{status}</strong></div>
      <div class="metric-card"><span>Confidence</span><strong>{html.escape(confidence)}</strong></div>
      <div class="metric-card"><span>Evidence chunks</span><strong>{len(chunks)}</strong></div>
      <div class="metric-card"><span>Top relevance</span><strong>{score:.3f}</strong></div>
      <div class="metric-card metric-wide"><span>Generation path</span><strong>{html.escape(method)}</strong></div>
    </div>
    """


def _build_transparency_html(data: dict) -> str:
    chunks = data.get("retrieved_chunks") or []
    sources = {str(chunk.get("document_name")) for chunk in chunks if chunk.get("document_name")}
    source_label = ", ".join(sorted(_display_document(source) for source in sources)) or "None returned"
    return f"""
    <div class="transparency-grid">
      <div><span>Retrieved chunks</span><strong>{len(chunks)}</strong></div>
      <div><span>Source documents</span><strong>{html.escape(source_label)}</strong></div>
      <div><span>Retrieval mode</span><strong>API response</strong></div>
    </div>
    """


def _build_error_markdown(message: str) -> str:
    return f"### Evidence engine unavailable\n\n{html.escape(message)}\n\nPlease try again or contact the system administrator if the problem persists."


def respond(question: str, history: list, k: int, use_reranker: bool):
    """Yield response-panel state without changing the backend contract."""
    del history
    empty_evidence = '<p class="muted-note">Evidence will appear here after a question.</p>'
    empty_metrics = '<p class="muted-note">Response metadata will appear after submission.</p>'
    if not question or not question.strip():
        yield "", "### Enter a clinical question\n\nAsk about hypertension management in the indexed WHO guidelines.", empty_evidence, empty_metrics, _status_badge("Ready", "idle")
        return

    yield "", "", '<div class="loading-state"><span class="loader"></span>Retrieving evidence and generating a grounded response...</div>', empty_metrics, _status_badge("Working", "busy")
    data = _call_ask(question.strip(), k=k, use_reranker=use_reranker)
    if "_http_error" in data:
        yield "", _build_error_markdown(data["_http_error"]), empty_evidence, empty_metrics, _status_badge("Unavailable", "error")
        return

    chunks = data.get("retrieved_chunks") or []
    citations = data.get("citations") or []
    citation_html = (
        '<div class="evidence-subsection"><div class="section-eyebrow">Citations</div>'
        + _format_citations(citations)
        + "</div>"
    )
    yield (
        "",
        _build_answer_html(data),
        citation_html + '<div class="evidence-subsection"><div class="section-eyebrow">Retrieved passages</div>' + _format_chunks(chunks) + "</div>",
        _build_metrics_html(data) + _build_transparency_html(data),
        _status_badge("Grounded" if data.get("status") == "grounded" else "Review", "good" if data.get("status") == "grounded" else "warning"),
    )


def check_health() -> str:
    """Return a compact readiness indicator from the existing health endpoint."""
    try:
        response = requests.get(f"{FLASK_API_URL}/api/health", timeout=5)
        payload = response.json()
        if response.ok and payload.get("ready"):
            chunks = html.escape(str(payload.get("chunks_indexed", "?")))
            return _status_badge(f"Evidence engine online · {chunks} chunks", "good")
        LOGGER.error("Backend health check returned %s: %s", response.status_code, payload)
        return _status_badge("Engine needs attention", "warning")
    except (requests.exceptions.RequestException, ValueError) as exc:
        LOGGER.error("Health check failed: %s", exc)
        return _status_badge("Backend offline", "error")


CUSTOM_CSS = """
:root {
  --navy: #102a43; --teal: #087f8c; --teal-soft: #e7f5f3; --ink: #182536;
  --muted: #63758a; --line: #dbe5ec; --canvas: #f3f7fa; --paper: #ffffff;
  --warning: #9a6618; --danger: #b3433c;
}
* { box-sizing: border-box; }
body, .gradio-container { background: radial-gradient(circle at 100% 0, #e7f5f3 0, var(--canvas) 34rem) !important; color: var(--ink) !important; font-family: "DM Sans", "Avenir Next", sans-serif !important; }
.gradio-container { max-width: 1240px !important; padding: 30px 24px 44px !important; }
.app-header { background: var(--navy); border-radius: 14px; padding: 28px 30px; box-shadow: 0 14px 30px rgba(16,42,67,.16); }
.brand-mark { color: #9ed9d7; font-size: .72rem; font-weight: 800; letter-spacing: .14em; text-transform: uppercase; }
.brand-title { color: white; font-size: 1.7rem; font-weight: 800; margin: 8px 0 5px; }
.brand-subtitle { color: #d6e5ed; font-size: .96rem; line-height: 1.55; margin: 0; max-width: 620px; }
.status-badge { align-items: center; border: 1px solid rgba(255,255,255,.2); border-radius: 999px; display: inline-flex; font-size: .76rem; font-weight: 800; gap: 8px; padding: 8px 12px; white-space: nowrap; }
.status-dot { background: currentColor; border-radius: 50%; box-shadow: 0 0 0 3px rgba(255,255,255,.08); height: 7px; width: 7px; }
.status-good { background: rgba(185,239,220,.12); color: #b9efdc; } .status-idle { background: rgba(255,255,255,.1); color: #d6e5ed; } .status-busy { background: rgba(255,211,138,.12); color: #ffd38a; } .status-warning { background: #fff8e8; border-color: #f0d59c; color: var(--warning); } .status-error { background: rgba(255,196,189,.12); color: #ffc4bd; }
.trust-strip { display: flex; flex-wrap: wrap; gap: 9px; margin: 18px 0; }
.trust-item { background: var(--paper); border: 1px solid var(--line); border-radius: 999px; color: var(--muted); font-size: .76rem; font-weight: 700; padding: 7px 12px; }
.trust-item strong { color: var(--teal); margin-right: 5px; }
.notice { background: #fffaf0; border: 1px solid #f1dfb9; border-left: 4px solid #d89a35; border-radius: 10px; color: #6b501f; font-size: .82rem; line-height: 1.55; margin-bottom: 18px; padding: 13px 16px; }
.workspace, .response-card { background: var(--paper); border: 1px solid var(--line); border-radius: 14px; box-shadow: 0 7px 20px rgba(29,54,75,.06); padding: 22px; }
.section-eyebrow { color: var(--teal); font-size: .7rem; font-weight: 800; letter-spacing: .12em; text-transform: uppercase; }
.section-title { color: var(--ink); font-size: 1.2rem; font-weight: 800; margin: 5px 0; }
.section-copy { color: var(--muted); font-size: .84rem; line-height: 1.55; margin: 0 0 16px; }
.question-box textarea { background: #fbfdfe !important; border: 1px solid #c7d5e1 !important; border-radius: 10px !important; color: var(--ink) !important; font-size: .98rem !important; line-height: 1.55 !important; }
.question-box textarea:focus { border-color: var(--teal) !important; box-shadow: 0 0 0 3px rgba(8,127,140,.15) !important; }
.primary-button { background: var(--teal) !important; border: 0 !important; border-radius: 9px !important; color: white !important; font-weight: 800 !important; min-height: 46px !important; }
.primary-button:hover { background: #066b76 !important; }
.secondary-button, .example-button { background: #f8fbfc !important; border: 1px solid var(--line) !important; border-radius: 9px !important; color: #426176 !important; }
.secondary-button:hover, .example-button:hover { background: var(--teal-soft) !important; border-color: #8ac6c6 !important; }
.example-button { font-size: .78rem !important; text-align: left !important; }
.response-panel { background: #fff !important; border: 0 !important; color: var(--ink) !important; min-height: 250px; padding: 0 !important; }
.response-panel h1, .response-panel h2, .response-panel h3 { color: var(--navy); } .response-panel p, .response-panel li { line-height: 1.7; }
.gradio-container .prose, .gradio-container .prose p, .gradio-container .prose li, .gradio-container .md, .gradio-container .md p, .gradio-container .md li { color: var(--ink) !important; opacity: 1 !important; }
.app-header .prose, .app-header .prose p, .app-header .brand-subtitle { color: #d6e5ed !important; opacity: 1 !important; }
.app-header .brand-mark { color: #9ed9d7 !important; }
.app-header .brand-title { color: #ffffff !important; }
.response-panel .prose, .response-panel .md { color: var(--ink) !important; }
.accordion-panel .prose, .accordion-panel .md, .accordion-panel .prose p, .accordion-panel .md p { color: #425466 !important; opacity: 1 !important; }
.gradio-container .block, .gradio-container .form, .gradio-container .panel, .gradio-container .wrap { background: transparent !important; color: var(--ink) !important; }
.question-box, .question-box > div, .question-box .wrap, .question-box .label-wrap { background: var(--paper) !important; color: var(--navy) !important; }
.question-box label, .question-box label span, .question-box .label-wrap { color: var(--navy) !important; font-weight: 800 !important; opacity: 1 !important; }
.workspace label, .workspace label span, .workspace .head label, .workspace .head label span { color: var(--navy) !important; opacity: 1 !important; }
.workspace input[type="number"] { color: var(--navy) !important; background: #ffffff !important; }
.workspace input[type="range"] { accent-color: var(--teal) !important; }
.workspace .md, .workspace .md p, .workspace .chatbot { color: var(--muted) !important; opacity: 1 !important; }
.workspace .svelte-j9uq24, .workspace .svelte-j9uq24 p { color: var(--muted) !important; opacity: 1 !important; }
.accordion-panel, .accordion-panel > div, .accordion-panel .wrap, .accordion-panel .label-wrap, .gradio-container .accordion { background: var(--paper) !important; border-color: var(--line) !important; color: var(--navy) !important; }
.accordion-panel .label-wrap { font-weight: 800 !important; }
.notice, .notice strong { color: #6b501f !important; }
.response-card { align-self: flex-start !important; height: fit-content !important; }
.response-card .status-idle { background: #f3f7fa; border-color: var(--line); color: var(--muted); }
.response-card .status-good { background: var(--teal-soft); border-color: #b7dfda; color: var(--teal); }
.answer-state { border: 1px solid var(--line); border-radius: 10px; color: var(--muted); padding: 18px; } .answer-state-warning { background: #fffaf0; border-color: #f0d59c; }
.answer-state-label { color: var(--navy); font-size: .95rem; font-weight: 800; } .answer-state p { margin: 7px 0; }
.answer-state-note, .muted-note { color: var(--muted); font-size: .82rem; }
.metrics-grid, .transparency-grid { display: grid; gap: 8px; grid-template-columns: repeat(4, 1fr); margin-top: 16px; }
.metric-card, .transparency-grid > div { background: #fbfdfe; border: 1px solid var(--line); border-radius: 9px; min-width: 0; padding: 10px 11px; }
.metric-card span, .transparency-grid span { color: var(--muted); display: block; font-size: .68rem; font-weight: 700; margin-bottom: 4px; }
.metric-card strong, .transparency-grid strong { color: var(--navy); display: block; font-size: .82rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.metric-wide { grid-column: span 2; } .metric-good { color: var(--teal) !important; } .metric-warning { color: var(--warning) !important; }
.loading-state { align-items: center; background: var(--teal-soft); border-radius: 9px; color: var(--teal); display: flex; font-size: .88rem; gap: 9px; padding: 15px; }
.loader { animation: pulse 1.2s infinite; background: var(--teal); border-radius: 50%; height: 8px; width: 8px; } @keyframes pulse { 50% { opacity: .3; transform: scale(.75); } }
.source-list, .evidence-list { display: flex; flex-direction: column; gap: 9px; } .source-card, .evidence-card { background: #fbfdfe; border: 1px solid var(--line); border-radius: 10px; }
.source-card { align-items: center; display: flex; gap: 12px; padding: 13px 14px; } .source-icon { color: var(--teal); font-size: .65rem; font-weight: 900; letter-spacing: .05em; }
.source-content { min-width: 0; } .source-content strong { color: var(--navy) !important; font-size: .84rem; } .source-meta, .source-meta span, .evidence-meta, .evidence-meta span { color: var(--muted) !important; display: flex; flex-wrap: wrap; font-size: .72rem; gap: 12px; margin-top: 4px; opacity: 1 !important; }
.evidence-card summary { align-items: center; cursor: pointer; display: flex; justify-content: space-between; list-style: none; padding: 13px 14px; } .evidence-card summary::-webkit-details-marker { display: none; }
.evidence-title { align-items: center; color: var(--navy); display: flex; font-size: .83rem; font-weight: 800; gap: 9px; min-width: 0; } .evidence-index { color: var(--teal); font-size: .7rem; }
.evidence-score { background: var(--teal-soft); border-radius: 5px; color: var(--teal); font-size: .72rem; font-weight: 800; padding: 4px 6px; }
.evidence-body { border-top: 1px solid var(--line); padding: 12px 14px 14px; } .evidence-body blockquote { border-left: 3px solid #9ed9d7; color: #425466; font-size: .82rem; line-height: 1.65; margin: 10px 0 0; padding-left: 12px; white-space: pre-wrap; }
.accordion-panel { margin-top: 16px; } footer { display: none !important; }
@media (max-width: 760px) { .gradio-container { padding: 16px 12px 28px !important; } .app-header { padding: 21px; } .brand-title { font-size: 1.4rem; } .header-status { margin-top: 16px; } .metrics-grid, .transparency-grid { grid-template-columns: repeat(2, 1fr); } .metric-wide { grid-column: span 2; } .evidence-card summary { align-items: flex-start; gap: 10px; } }
"""


with gr.Blocks(
    title="AI Clinical Decision Support Lite",
    theme=gr.themes.Base(primary_hue="teal", neutral_hue="slate", font=[gr.themes.GoogleFont("DM Sans"), "Avenir Next", "sans-serif"]),
    css=CUSTOM_CSS,
) as demo:
    with gr.Row(elem_classes=["app-header"], equal_height=False):
        with gr.Column(scale=8):
            gr.HTML('<div class="brand-mark">Clinical evidence engine</div><div class="brand-title">AI Clinical Decision Support Lite</div><p class="brand-subtitle">Evidence-grounded hypertension decision support powered by retrieval-augmented generation and WHO clinical guidelines.</p>')
        with gr.Column(scale=3, elem_classes=["header-status"]):
            status_box = gr.HTML(value=_status_badge("Checking engine", "idle"))
            health_btn = gr.Button("Check status", elem_classes=["secondary-button"], size="sm")

    gr.HTML('<div class="trust-strip" aria-label="System capabilities"><span class="trust-item"><strong>✓</strong> WHO guidelines</span><span class="trust-item"><strong>✓</strong> RAG retrieval</span><span class="trust-item"><strong>✓</strong> ChromaDB index</span><span class="trust-item"><strong>✓</strong> Evidence citations</span></div><div class="notice"><strong>Clinical review required.</strong> This assistant provides evidence-grounded decision support from indexed WHO hypertension guidelines. It is not a substitute for professional judgment or direct patient-care review.</div>')

    with gr.Row(equal_height=False):
        with gr.Column(scale=4, elem_classes=["workspace"]):
            gr.HTML('<div class="section-eyebrow">Clinical workspace</div><div class="section-title">Ask a focused question</div><p class="section-copy">Ask within the indexed hypertension guideline scope for a traceable answer with supporting evidence.</p>')
            question_box = gr.Textbox(label="Clinical question", placeholder="Ask a clinical question about hypertension management...", lines=5, max_lines=7, elem_classes=["question-box"])
            with gr.Row():
                send_btn = gr.Button("Ask Clinical AI", variant="primary", elem_classes=["primary-button"], scale=3)
                clear_btn = gr.Button("Clear", elem_classes=["secondary-button"], scale=1)
            gr.HTML('<div class="section-eyebrow" style="margin-top:22px">Suggested questions</div>')
            example_buttons = [gr.Button(question, elem_classes=["example-button"], size="sm") for question in EXAMPLE_QUESTIONS]
            with gr.Accordion("Retrieval settings", open=False):
                k_slider = gr.Slider(minimum=1, maximum=10, step=1, value=3, label="Evidence chunks")
                reranker_toggle = gr.Checkbox(value=False, label="Use dense reranker", info="Useful for hard paraphrase queries; adds latency.")

        with gr.Column(scale=6, elem_classes=["response-card"]):
            gr.HTML('<div class="section-eyebrow">AI response</div><div class="section-title">Evidence-grounded guidance</div>')
            response_panel = gr.Markdown("### Ready for a clinical question\n\nYour grounded response will appear here.", elem_classes=["response-panel"])
            status_panel = gr.HTML(value=_status_badge("Ready", "idle"))
            metrics_panel = gr.HTML('<p class="muted-note">Response metadata will appear after submission.</p>')

    with gr.Accordion("Evidence and source passages", open=False, elem_classes=["accordion-panel"]):
        gr.Markdown("Source cards use citation metadata returned by the clinical API. Expand a passage to inspect its retrieved text and relevance score.")
        evidence_panel = gr.HTML('<p class="muted-note">Evidence will appear here after a question.</p>')

    def _submit(question, k, use_reranker):
        yield from respond(question, [], k, use_reranker)

    outputs = [question_box, response_panel, evidence_panel, metrics_panel, status_panel]
    send_btn.click(fn=_submit, inputs=[question_box, k_slider, reranker_toggle], outputs=outputs, api_name=False)
    question_box.submit(fn=_submit, inputs=[question_box, k_slider, reranker_toggle], outputs=outputs, api_name=False)
    clear_btn.click(fn=lambda: ("", "### Ready for a clinical question\n\nYour grounded response will appear here.", '<p class="muted-note">Evidence will appear here after a question.</p>', '<p class="muted-note">Response metadata will appear after submission.</p>', _status_badge("Ready", "idle")), outputs=outputs, api_name=False)
    for button, question in zip(example_buttons, EXAMPLE_QUESTIONS):
        button.click(fn=lambda value=question: value, outputs=[question_box], api_name=False)
    health_btn.click(fn=check_health, inputs=[], outputs=[status_box], api_name=False)
    demo.load(fn=check_health, inputs=[], outputs=[status_box], api_name=False)


if __name__ == "__main__":
    print(f"Starting Gradio frontend on http://127.0.0.1:{GRADIO_PORT}")
    print(f"Calling Flask backend at: {FLASK_API_URL}")
    demo.launch(server_name=GRADIO_SERVER_NAME, server_port=GRADIO_PORT, share=GRADIO_SHARE)
