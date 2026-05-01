"""
R-RAG: Reasoning-Aware Retrieval-Augmented Generation System
ChatGPT-style interface with multi-hop reasoning, citations, and reasoning trace.
"""

import streamlit as st
from backend.ingest import process_uploaded_file
from backend.embed import generate_embeddings
from backend.store import create_index, add_embeddings
from backend.reasoning import reasoning_pipeline
from backend import llm

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="R-RAG | Reasoning-Aware AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Google Fonts ─────────────────────────────────────────────────────────────
st.markdown(
    '<link href="https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,200;0,300;0,400;0,500;0,600;1,400&display=swap" rel="stylesheet">',
    unsafe_allow_html=True
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>

/* ════════════════════════════════════════════════════════════════════════
   GLOBAL RESET
   ════════════════════════════════════════════════════════════════════════ */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    -webkit-font-smoothing: antialiased;
}

.stApp {
    background-color: #0a0a0a !important;
    color: #e2e2e2;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.93rem;
    line-height: 1.7;
}

/* Kill ALL default Streamlit padding/margin on main containers */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stStatusWidget"] { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }
.stDeployButton { display: none !important; }
iframe[title="streamlit_analytics"] { display: none !important; }

/* Hide bottom deploy/status bar */
.stAppDeployButton { display: none !important; }
div[data-testid="stBottom"] { display: none !important; }
div[data-testid="InputInstructions"] { display: none !important; }

.main, [data-testid="stMain"] {
    padding: 0 !important;
    margin: 0 !important;
}

.block-container, [data-testid="stMainBlockContainer"] {
    padding: 0 !important;
    margin: 0 !important;
    max-width: 100% !important;
}

/* ════════════════════════════════════════════════════════════════════════
   SIDEBAR
   ════════════════════════════════════════════════════════════════════════ */
section[data-testid="stSidebar"] {
    background: #111111 !important;
    border-right: 1px solid #1e1e1e;
    width: 17rem !important;
    min-width: 17rem !important;
    transform: none !important;
}
section[data-testid="stSidebar"] > div {
    width: 17rem !important;
    padding: 1.4rem 1rem !important;
}

/* Hide collapse button */
button[data-testid="baseButton-headerNoPadding"],
[data-testid="collapsedControl"] {
    display: none !important;
}

/* Sidebar title */
.sidebar-title {
    color: #f9fafb;
    font-weight: 300;
    font-size: 1.1rem;
    letter-spacing: 0.02em;
    margin-bottom: 1rem;
    display: block;
}

/* Sidebar section headings (h3) */
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #4b5563 !important;
    font-size: 0.65rem !important;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-weight: 500 !important;
    margin-top: 1.2rem !important;
    margin-bottom: 0.3rem !important;
}

/* Sidebar radio labels */
section[data-testid="stSidebar"] .stRadio label span {
    color: #9ca3af !important;
    font-size: 0.85rem !important;
}

/* Sidebar password input */
section[data-testid="stSidebar"] .stTextInput input {
    background: #1a1a1a !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 6px !important;
    color: #e2e2e2 !important;
    font-size: 0.83rem !important;
    font-family: 'Inter', sans-serif !important;
    box-shadow: none !important;
}

/* Sidebar Build KB button */
section[data-testid="stSidebar"] .stButton button {
    background: #6366f1 !important;
    border: none !important;
    border-radius: 6px !important;
    font-size: 0.83rem !important;
    font-weight: 500 !important;
    color: #fff !important;
    box-shadow: none !important;
    width: 100%;
}
section[data-testid="stSidebar"] .stButton button:hover {
    background: #4f52c8 !important;
}

/* ════════════════════════════════════════════════════════════════════════
   CHAT AREA  — simple block, no flex, no min-height
   ════════════════════════════════════════════════════════════════════════ */
.chat-wrapper {
    margin-left: 17rem;
    padding: 1.2rem 0 140px 0;
}
.chat-column {
    max-width: 760px;
    margin: 0 auto;
    padding: 0 3.5rem;
}

/* ─── Role labels ──────────────────────────────────────────────────────── */
.role-label {
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    color: #4b5563;
    font-weight: 500;
    text-transform: uppercase;
    margin-bottom: 5px;
}

/* ─── User bubble ──────────────────────────────────────────────────────── */
.user-msg-wrap { margin-bottom: 1.4rem; }
.user-bubble {
    background: #161616;
    border: 1px solid #222;
    border-radius: 10px;
    padding: 12px 16px;
    color: #d1d5db;
    font-size: 0.93rem;
    line-height: 1.7;
}

/* ─── Assistant message ────────────────────────────────────────────────── */
.assistant-msg-wrap { margin-bottom: 1.4rem; }

/* ─── Answer block ─────────────────────────────────────────────────────── */
.answer-block {
    background: #0f172a;
    border: 1px solid #1e2a45;
    border-left: 3px solid #3b82f6;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 8px;
}
.answer-label {
    display: block;
    color: #60a5fa;
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-weight: 600;
    margin-bottom: 8px;
}
.answer-text {
    color: #e2e8f0;
    font-size: 0.93rem;
    line-height: 1.75;
}

/* ─── Low-confidence banner ────────────────────────────────────────────── */
.low-conf-banner {
    background: #1c1200;
    border: 1px solid #92400e;
    border-radius: 8px;
    padding: 8px 14px;
    margin-bottom: 10px;
    font-size: 0.83rem;
    color: #d97706;
}

/* ════════════════════════════════════════════════════════════════════════
   EXPANDERS  — targeted selectors only, no broad > div > div
   ════════════════════════════════════════════════════════════════════════ */
div[data-testid="stExpander"] {
    background: #141414 !important;
    border: 1px solid #1e1e1e !important;
    border-radius: 8px !important;
    margin-bottom: 6px;
    overflow: hidden;
}

/* Summary bar */
div[data-testid="stExpander"] summary {
    background: #141414 !important;
    padding: 10px 14px !important;
}
div[data-testid="stExpander"] summary:hover {
    background: #1a1a1a !important;
}

/* Label text in summary */
div[data-testid="stExpander"] summary p {
    color: #9ca3af !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    margin: 0 !important;
    font-family: 'Inter', sans-serif !important;
}

/* Chevron icon */
div[data-testid="stExpander"] summary svg {
    color: #6b7280 !important;
    fill: #6b7280 !important;
    flex-shrink: 0;
}

/* Content pane — use stExpanderDetails, not the broad > div > div */
div[data-testid="stExpanderDetails"] {
    background: #141414 !important;
    padding: 8px 14px 12px !important;
}

/* Text inside expander content */
div[data-testid="stExpanderDetails"] p,
div[data-testid="stExpanderDetails"] li,
div[data-testid="stExpanderDetails"] em,
div[data-testid="stExpanderDetails"] span {
    color: #d1d5db !important;
    font-size: 0.88rem !important;
    line-height: 1.65 !important;
}

/* Code spans inside expanders */
div[data-testid="stExpanderDetails"] code {
    color: #a5b4fc !important;
    background: #1e1e2e !important;
    padding: 1px 5px;
    border-radius: 4px;
    font-size: 0.82rem !important;
}

/* ─── Reasoning trace steps ────────────────────────────────────────────── */
.trace-step {
    background: #1a1a1a;
    border-left: 2px solid #374151;
    padding: 8px 14px;
    margin-bottom: 8px;
    border-radius: 0 6px 6px 0;
}
.trace-step-title {
    font-size: 0.73rem;
    font-weight: 500;
    color: #9ca3af;
    margin-bottom: 3px;
    letter-spacing: 0.03em;
}
.trace-step-detail {
    font-size: 0.82rem;
    color: #6b7280;
    line-height: 1.55;
}

/* ─── Caption text (used in sources) ──────────────────────────────────── */
[data-testid="stCaptionContainer"] p {
    color: #6b7280 !important;
    font-size: 0.78rem !important;
    font-family: 'Inter', sans-serif !important;
    line-height: 1.5 !important;
}

/* ════════════════════════════════════════════════════════════════════════
   FIXED INPUT BAR  — no gradient, solid bg, no black band
   ════════════════════════════════════════════════════════════════════════ */
.input-area {
    position: fixed;
    bottom: 0;
    left: 17rem;
    right: 0;
    background: #0a0a0a;
    border-top: 1px solid #1c1c1c;
    padding: 14px 0 20px 0;
    z-index: 999;
}
.input-inner {
    max-width: 760px;
    margin: 0 auto;
    padding: 0 3.5rem;
    display: flex;
    gap: 8px;
    align-items: center;
}

/* Chat text input — target within input-inner context */
.input-inner .stTextInput,
.input-inner [data-testid="stTextInput"] {
    flex: 1;
}
.input-inner .stTextInput input,
.input-inner [data-testid="stTextInput"] input {
    height: 44px !important;
    background: #161616 !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 8px !important;
    color: #e2e2e2 !important;
    font-size: 0.9rem !important;
    font-family: 'Inter', sans-serif !important;
    padding: 0 14px !important;
    box-shadow: none !important;
}
.input-inner .stTextInput input:focus,
.input-inner [data-testid="stTextInput"] input:focus {
    border-color: #3b82f6 !important;
    box-shadow: none !important;
    outline: none !important;
}
.input-inner .stTextInput input::placeholder {
    color: #4b5563 !important;
}

/* All text inputs globally (fallback) */
div[data-testid="column"]:first-child .stTextInput input {
    height: 44px !important;
    background: #161616 !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 8px !important;
    color: #e2e2e2 !important;
    font-size: 0.9rem !important;
    font-family: 'Inter', sans-serif !important;
    padding: 0 14px !important;
    box-shadow: none !important;
}
div[data-testid="column"]:first-child .stTextInput input:focus {
    border-color: #3b82f6 !important;
    box-shadow: none !important;
}
div[data-testid="column"]:first-child .stTextInput input::placeholder {
    color: #4b5563 !important;
}

/* Send button — BLUE */
div[data-testid="column"]:last-child .stButton button {
    background: #2563eb !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    height: 44px !important;
    min-width: 72px;
    box-shadow: none !important;
    font-family: 'Inter', sans-serif !important;
}
div[data-testid="column"]:last-child .stButton button:hover {
    background: #1d4ed8 !important;
}

/* ─── Welcome screen ───────────────────────────────────────────────────── */
.welcome {
    text-align: center;
    padding: 48px 20px 32px;
}
.welcome h1 {
    font-weight: 200;
    font-size: 2rem;
    color: #f9fafb;
    letter-spacing: -0.02em;
    margin-bottom: 10px;
    line-height: 1.2;
}
.welcome p {
    color: #4b5563;
    font-size: 0.88rem;
    margin: 0;
}

</style>
""", unsafe_allow_html=True)

# ─── Session State Init ────────────────────────────────────────────────────────
for key, default in [
    ("chat_history", []),
    ("chunks", None),
    ("index", None),
    ("kb_built", False),
    ("kb_doc_names", []),
    ("uploaded_filenames", []),
    ("raw_docs", []),
    ("pending_question", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<span class="sidebar-title">R-RAG</span>', unsafe_allow_html=True)

    st.markdown("### LLM Mode")
    mode = st.radio(
        "Select backend",
        ["Online (Groq)", "Offline (Ollama)"],
        label_visibility="collapsed"
    )
    if "Online" in mode:
        llm.MODE = "online"
        st.markdown('<span style="color:#10b981;font-size:0.78rem;font-weight:500;">Groq · llama3-8b</span>', unsafe_allow_html=True)
    else:
        llm.MODE = "offline"
        st.markdown('<span style="color:#10b981;font-size:0.78rem;font-weight:500;">Ollama · phi3:mini</span>', unsafe_allow_html=True)

    st.markdown("### Groq API Key")
    api_key_input = st.text_input(
        "Groq API Key",
        value=llm.GROQ_API_KEY,
        type="password",
        label_visibility="collapsed",
        placeholder="gsk_..."
    )
    if api_key_input:
        llm.GROQ_API_KEY = api_key_input

    st.markdown("### Upload Documents")
    uploaded_files = st.file_uploader(
        "PDF, TXT, DOCX",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:
        current_names = [f.name for f in uploaded_files]
        if current_names != st.session_state.uploaded_filenames:
            st.session_state.uploaded_filenames = current_names
            st.session_state.raw_docs = uploaded_files
            st.session_state.kb_built = False
            st.success(f"{len(uploaded_files)} file(s) ready")

    st.markdown("### Knowledge Base")
    if st.button("Build Knowledge Base", use_container_width=True):
        if not st.session_state.raw_docs:
            st.error("Upload documents first.")
        else:
            with st.spinner("Processing documents..."):
                all_chunks = []
                processed_names = []
                for file in st.session_state.raw_docs:
                    try:
                        chunks = process_uploaded_file(file, file.name)
                        all_chunks.extend(chunks)
                        processed_names.append(file.name)
                    except Exception as e:
                        st.error(f"Error: {file.name} — {e}")

                if all_chunks:
                    texts = [c["text"] for c in all_chunks]
                    embeddings = generate_embeddings(texts)
                    index = create_index(embeddings.shape[1])
                    add_embeddings(index, embeddings)
                    st.session_state.chunks = all_chunks
                    st.session_state.index = index
                    st.session_state.kb_built = True
                    st.session_state.kb_doc_names = processed_names
                else:
                    st.error("No text extracted. Check your documents.")

    if st.session_state.kb_built:
        st.markdown('<span style="color:#3b82f6;font-size:0.78rem;font-weight:500;">Knowledge base ready</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span style="color:#4b5563;font-size:0.78rem;">No knowledge base loaded</span>', unsafe_allow_html=True)

# ─── Phase 2: Process pending question ────────────────────────────────────────
if st.session_state.pending_question:
    q = st.session_state.pending_question
    st.session_state.pending_question = None

    if not st.session_state.kb_built:
        st.error("Please upload documents and build the knowledge base first.")
    elif not llm.GROQ_API_KEY and llm.MODE == "online":
        st.error("Please enter your Groq API key in the sidebar.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": q})

        with st.spinner("Reasoning over documents..."):
            result = reasoning_pipeline(
                question=q,
                index=st.session_state.index,
                documents=st.session_state.chunks
            )

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": result["answer"],
            "answer_parsed": result.get("answer_parsed", {}),
            "sub_queries": result.get("sub_queries", []),
            "reasoning_trace": result.get("reasoning_trace", []),
            "sources": result.get("sources", [])
        })
        st.rerun()

# ─── Chat area ────────────────────────────────────────────────────────────────
st.markdown('<div class="chat-wrapper"><div class="chat-column">', unsafe_allow_html=True)

if not st.session_state.chat_history:
    st.markdown("""
<div class="welcome">
    <h1>R-RAG</h1>
    <p>Upload documents and ask questions.</p>
</div>""", unsafe_allow_html=True)

for msg in st.session_state.chat_history:
    role = msg["role"]

    if role == "user":
        st.markdown(f"""
<div class="user-msg-wrap">
  <div class="role-label">You</div>
  <div class="user-bubble">{msg["content"]}</div>
</div>""", unsafe_allow_html=True)

    elif role == "assistant":
        st.markdown('<div class="assistant-msg-wrap">', unsafe_allow_html=True)
        st.markdown('<div class="role-label">Assistant</div>', unsafe_allow_html=True)

        parsed          = msg.get("answer_parsed", {})
        answer_text     = parsed.get("answer_text", "")
        evidence_points = parsed.get("evidence_points", [])
        reasoning_text  = parsed.get("reasoning_text", "")
        low_confidence  = parsed.get("low_confidence", False)

        if not answer_text:
            answer_text = msg.get("content", "")

        if low_confidence:
            st.markdown('<div class="low-conf-banner">Low confidence — retrieved content may not fully address this question.</div>', unsafe_allow_html=True)

        if answer_text:
            st.markdown(f"""
<div class="answer-block">
  <div class="answer-label">Answer</div>
  <div class="answer-text">{answer_text}</div>
</div>""", unsafe_allow_html=True)

        if evidence_points:
            with st.expander("Evidence used"):
                for pt in evidence_points:
                    st.markdown(f"- {pt}")

        if reasoning_text:
            with st.expander("Reasoning"):
                st.markdown(f"*{reasoning_text}*")

        if msg.get("sub_queries") and len(msg["sub_queries"]) > 1:
            with st.expander(f"Retrieval steps ({len(msg['sub_queries'])})"):
                for i, sq in enumerate(msg["sub_queries"], 1):
                    st.markdown(f"`{i}.` {sq}")

        if msg.get("reasoning_trace"):
            with st.expander("Reasoning Trace"):
                for step in msg["reasoning_trace"]:
                    st.markdown(f"""
<div class="trace-step">
  <div class="trace-step-title">{step['step']}</div>
  <div class="trace-step-detail">{step['detail']}</div>
</div>""", unsafe_allow_html=True)

        if msg.get("sources"):
            with st.expander(f"Sources ({len(msg['sources'])})"):
                for i, src in enumerate(msg["sources"], 1):
                    score_pct = int(src.get("score", 0) * 100)
                    st.markdown(
                        f'<span style="color:#6b7280;font-size:0.78rem;font-family:monospace;">'
                        f'{src["filename"]} · p.{src["page"]} · {score_pct}%</span>',
                        unsafe_allow_html=True
                    )
                    st.caption(src["text"][:350] + ("..." if len(src["text"]) > 350 else ""))
                    if i < len(msg["sources"]):
                        st.markdown("<hr style='border:none;border-top:1px solid #1e1e1e;margin:8px 0'>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div></div>", unsafe_allow_html=True)

# ─── Fixed input bar ──────────────────────────────────────────────────────────
st.markdown('<div class="input-area"><div class="input-inner">', unsafe_allow_html=True)

col1, col2 = st.columns([6, 1])
with col1:
    question = st.text_input(
        "question",
        placeholder="Ask a question..." if st.session_state.kb_built else "Build knowledge base first...",
        label_visibility="collapsed",
        key="question_input"
    )
with col2:
    send = st.button("Send", use_container_width=True, key="send_btn")

st.markdown("</div></div>", unsafe_allow_html=True)

# ─── Phase 1: store and rerun ─────────────────────────────────────────────────
if send and question.strip():
    st.session_state.pending_question = question.strip()
    st.rerun()