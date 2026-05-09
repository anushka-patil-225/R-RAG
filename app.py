"""
TraceAI: Reasoning-Aware Retrieval-Augmented Generation System
Modern AI chat interface — violet/indigo accent, Gemini-inspired design.
"""

import streamlit as st
from backend.ingest import process_uploaded_file
from backend.embed import generate_embeddings
from backend.store import create_index, add_embeddings
from backend.reasoning import reasoning_pipeline
from backend import llm

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TraceAI · Reasoning RAG",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Google Fonts ─────────────────────────────────────────────────────────────
st.markdown(
    '<link href="https://fonts.googleapis.com/css2?family=Google+Sans:wght@300;400;500;600&family=Google+Sans+Text:wght@400;500&display=swap" rel="stylesheet">',
    unsafe_allow_html=True
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>

/* ════════════════════════════════════════════════════════════════════════
   GLOBAL RESET & BASE
   ════════════════════════════════════════════════════════════════════════ */
html, body, [class*="css"] {
    font-family: 'Google Sans Text', 'Google Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    -webkit-font-smoothing: antialiased;
}

.stApp {
    background-color: #0c0f1d !important;
    color: #e3e3e3;
    font-family: 'Google Sans Text', sans-serif !important;
    font-size: 0.94rem;
    line-height: 1.72;
}

/* Hide all default Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stStatusWidget"] { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }
.stDeployButton { display: none !important; }
.stAppDeployButton { display: none !important; }
div[data-testid="stBottom"] { display: none !important; }
div[data-testid="InputInstructions"] { display: none !important; }
iframe[title="streamlit_analytics"] { display: none !important; }

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
    background: #0a0d1a !important;
    border-right: 1px solid rgba(255,255,255,0.05) !important;
    width: 17rem !important;
    min-width: 17rem !important;
    transform: none !important;
}
section[data-testid="stSidebar"] > div {
    width: 17rem !important;
    padding: 1.6rem 1.2rem !important;
}

/* Hide collapse button */
button[data-testid="baseButton-headerNoPadding"],
[data-testid="collapsedControl"] {
    display: none !important;
}

/* ─── Sidebar brand ─────────────────────────────────────────────────── */
.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 2rem;
    padding-bottom: 1.4rem;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}
.sidebar-brand-icon {
    width: 32px;
    height: 32px;
    background: linear-gradient(135deg, #5b5ef4, #8b5cf6);
    border-radius: 9px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.9rem;
    flex-shrink: 0;
    box-shadow: 0 2px 12px rgba(91,94,244,0.35);
}
.sidebar-brand-name {
    color: #f0f4ff;
    font-weight: 500;
    font-size: 1.12rem;
    letter-spacing: -0.01em;
    font-family: 'Google Sans', sans-serif;
}

/* ─── Sidebar section labels ─────────────────────────────────────────── */
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #3a4560 !important;
    font-size: 0.62rem !important;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    font-weight: 600 !important;
    margin-top: 1.5rem !important;
    margin-bottom: 0.5rem !important;
    font-family: 'Google Sans', sans-serif !important;
}

/* ─── Sidebar radio ──────────────────────────────────────────────────── */
section[data-testid="stSidebar"] .stRadio label span {
    color: #8a9ab8 !important;
    font-size: 0.84rem !important;
    transition: color 0.15s ease;
}
section[data-testid="stSidebar"] .stRadio label:hover span {
    color: #b8c4de !important;
}
section[data-testid="stSidebar"] .stRadio [data-testid="stRadio"] {
    gap: 4px;
}

/* ─── Sidebar text input ─────────────────────────────────────────────── */
section[data-testid="stSidebar"] .stTextInput input {
    background: #10162a !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 10px !important;
    color: #dde4f4 !important;
    font-size: 0.84rem !important;
    font-family: 'Google Sans Text', sans-serif !important;
    box-shadow: none !important;
    padding: 9px 13px !important;
    transition: border-color 0.18s ease, box-shadow 0.18s ease;
}
section[data-testid="stSidebar"] .stTextInput input:focus {
    border-color: rgba(99,102,241,0.5) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.1) !important;
}
section[data-testid="stSidebar"] .stTextInput input::placeholder {
    color: #2e3d5c !important;
}

/* ─── Sidebar Build KB button ────────────────────────────────────────── */
section[data-testid="stSidebar"] .stButton button {
    background: linear-gradient(135deg, #5b5ef4, #7c3aed) !important;
    border: none !important;
    border-radius: 10px !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    color: #fff !important;
    box-shadow: 0 2px 14px rgba(91,94,244,0.28) !important;
    width: 100%;
    padding: 10px 0 !important;
    transition: opacity 0.18s ease, box-shadow 0.18s ease;
    font-family: 'Google Sans', sans-serif !important;
}
section[data-testid="stSidebar"] .stButton button:hover {
    opacity: 0.9;
    box-shadow: 0 4px 20px rgba(91,94,244,0.4) !important;
}

/* ─── Sidebar divider ────────────────────────────────────────────────── */
.sidebar-sep {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.05);
    margin: 1.2rem 0;
}

/* ─── KB status indicator ────────────────────────────────────────────── */
.kb-status {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 0.7rem;
    font-size: 0.8rem;
    font-weight: 500;
}
.kb-dot-ready {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #22c55e;
    flex-shrink: 0;
    box-shadow: 0 0 8px rgba(34,197,94,0.55);
    animation: pulse-green 2.5s infinite;
}
@keyframes pulse-green {
    0%, 100% { box-shadow: 0 0 6px rgba(34,197,94,0.55); }
    50% { box-shadow: 0 0 12px rgba(34,197,94,0.8); }
}
.kb-dot-idle {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #2a3450;
    flex-shrink: 0;
}

/* ════════════════════════════════════════════════════════════════════════
   MAIN CHAT AREA
   ════════════════════════════════════════════════════════════════════════ */
.chat-wrapper {
    margin-left: 17rem;
    padding: 0.8rem 0 150px 0;
}
.chat-column {
    max-width: 720px;
    margin: 0 auto;
    padding: 0 2rem;
}

/* ─── Role labels ────────────────────────────────────────────────────── */
.role-label {
    font-size: 0.62rem;
    letter-spacing: 0.15em;
    color: #3a4560;
    font-weight: 600;
    text-transform: uppercase;
    margin-bottom: 6px;
    font-family: 'Google Sans', sans-serif;
}

/* ─── User bubble ────────────────────────────────────────────────────── */
.user-msg-wrap {
    margin-bottom: 2rem;
    display: flex;
    flex-direction: column;
    align-items: flex-end;
}
.user-bubble {
    background: #161d34;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 18px 18px 4px 18px;
    padding: 12px 18px;
    color: #cdd5e8;
    font-size: 0.94rem;
    line-height: 1.72;
    max-width: 72%;
    box-shadow: 0 2px 12px rgba(0,0,0,0.25);
    transition: box-shadow 0.2s ease;
}
.user-bubble:hover {
    box-shadow: 0 4px 18px rgba(0,0,0,0.35);
}

/* ─── Assistant message ──────────────────────────────────────────────── */
.assistant-msg-wrap {
    margin-bottom: 2.2rem;
}
.assistant-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 12px;
}
.assistant-icon {
    width: 24px;
    height: 24px;
    background: linear-gradient(135deg, #5b5ef4, #8b5cf6);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.6rem;
    flex-shrink: 0;
    box-shadow: 0 0 10px rgba(91,94,244,0.4);
}

/* ─── Answer block ───────────────────────────────────────────────────── */
.answer-block {
    background: linear-gradient(160deg, #111829 0%, #0f1526 100%);
    border: 1px solid rgba(255,255,255,0.05);
    border-left: 2px solid rgba(91,94,244,0.6);
    border-radius: 14px;
    padding: 18px 22px;
    margin-bottom: 12px;
    box-shadow: 0 2px 16px rgba(0,0,0,0.2), inset 0 1px 0 rgba(255,255,255,0.03);
    transition: border-left-color 0.2s ease;
}
.answer-block:hover {
    border-left-color: rgba(91,94,244,0.9);
}
.answer-label {
    display: block;
    color: #7c7ff5;
    font-size: 0.6rem;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    font-weight: 600;
    margin-bottom: 10px;
    font-family: 'Google Sans', sans-serif;
}
.answer-text {
    color: #dde4f4;
    font-size: 0.94rem;
    line-height: 1.8;
}

/* ─── Low-confidence banner ──────────────────────────────────────────── */
.low-conf-banner {
    background: rgba(28,16,0,0.7);
    border: 1px solid rgba(120,53,15,0.5);
    border-radius: 10px;
    padding: 10px 16px;
    margin-bottom: 12px;
    font-size: 0.82rem;
    color: #f59e0b;
    backdrop-filter: blur(4px);
}

/* ════════════════════════════════════════════════════════════════════════
   EXPANDERS
   ════════════════════════════════════════════════════════════════════════ */
div[data-testid="stExpander"] {
    background: #0f1626 !important;
    border: 1px solid rgba(255,255,255,0.05) !important;
    border-radius: 12px !important;
    margin-bottom: 8px;
    overflow: hidden;
    transition: border-color 0.18s ease;
}
div[data-testid="stExpander"]:hover {
    border-color: rgba(255,255,255,0.09) !important;
}
div[data-testid="stExpander"] summary {
    background: #0f1626 !important;
    padding: 11px 16px !important;
    transition: background 0.18s ease;
}
div[data-testid="stExpander"] summary:hover {
    background: #141c36 !important;
}
div[data-testid="stExpander"] summary p {
    color: #5c6d8c !important;
    font-size: 0.83rem !important;
    font-weight: 500 !important;
    margin: 0 !important;
    font-family: 'Google Sans Text', sans-serif !important;
    transition: color 0.18s ease;
}
div[data-testid="stExpander"] summary:hover p {
    color: #7a8fad !important;
}
div[data-testid="stExpander"] summary svg {
    color: #3a4560 !important;
    fill: #3a4560 !important;
    flex-shrink: 0;
}
div[data-testid="stExpanderDetails"] {
    background: #0f1626 !important;
    padding: 10px 16px 16px !important;
}
div[data-testid="stExpanderDetails"] p,
div[data-testid="stExpanderDetails"] li,
div[data-testid="stExpanderDetails"] em,
div[data-testid="stExpanderDetails"] span {
    color: #8a9ab8 !important;
    font-size: 0.88rem !important;
    line-height: 1.68 !important;
}
div[data-testid="stExpanderDetails"] code {
    color: #a5b4fc !important;
    background: rgba(30,30,63,0.8) !important;
    padding: 2px 6px;
    border-radius: 5px;
    font-size: 0.82rem !important;
}

/* ─── Reasoning trace steps ──────────────────────────────────────────── */
.trace-step {
    background: rgba(17,24,39,0.7);
    border-left: 2px solid rgba(49,46,129,0.6);
    padding: 9px 15px;
    margin-bottom: 9px;
    border-radius: 0 9px 9px 0;
    transition: border-left-color 0.18s ease;
}
.trace-step:hover {
    border-left-color: rgba(99,102,241,0.7);
}
.trace-step-title {
    font-size: 0.72rem;
    font-weight: 600;
    color: #7c7ff5;
    margin-bottom: 4px;
    letter-spacing: 0.03em;
    font-family: 'Google Sans', sans-serif;
}
.trace-step-detail {
    font-size: 0.83rem;
    color: #5c6d8c;
    line-height: 1.58;
}

/* ─── Caption (sources) ──────────────────────────────────────────────── */
[data-testid="stCaptionContainer"] p {
    color: #3d4f6b !important;
    font-size: 0.77rem !important;
    font-family: 'Google Sans Text', sans-serif !important;
    line-height: 1.52 !important;
}

/* ════════════════════════════════════════════════════════════════════════
   LOADING STATE
   ════════════════════════════════════════════════════════════════════════ */
.reasoning-loading {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 14px 18px;
    background: #0f1626;
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 12px;
    margin-bottom: 1.8rem;
    color: #7c7ff5;
    font-size: 0.85rem;
    font-weight: 500;
}
.dot-pulse {
    display: flex;
    gap: 4px;
    align-items: center;
}
.dot-pulse span {
    width: 5px;
    height: 5px;
    border-radius: 50%;
    background: #5b5ef4;
    animation: dotPulse 1.4s infinite;
    display: inline-block;
}
.dot-pulse span:nth-child(2) { animation-delay: 0.2s; }
.dot-pulse span:nth-child(3) { animation-delay: 0.4s; }
@keyframes dotPulse {
    0%, 80%, 100% { opacity: 0.2; transform: scale(0.8); }
    40% { opacity: 1; transform: scale(1.1); }
}

/* ════════════════════════════════════════════════════════════════════════
   FIXED INPUT BAR
   ════════════════════════════════════════════════════════════════════════ */
.input-area {
    position: fixed;
    bottom: 0;
    left: 17rem;
    right: 0;
    background: linear-gradient(180deg, transparent 0%, #0c0f1d 24px, #0c0f1d 100%);
    padding: 14px 0 22px 0;
    z-index: 999;
}
.input-inner {
    max-width: 720px;
    margin: 0 auto;
    padding: 0 2rem;
    display: flex;
    gap: 8px;
    align-items: center;
}

/* Input field */
.input-inner .stTextInput input,
.input-inner [data-testid="stTextInput"] input,
div[data-testid="column"]:first-child .stTextInput input {
    height: 50px !important;
    background: #111929 !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 16px !important;
    color: #dde4f4 !important;
    font-size: 0.92rem !important;
    font-family: 'Google Sans Text', sans-serif !important;
    padding: 0 18px !important;
    box-shadow: 0 2px 16px rgba(0,0,0,0.2) !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}
.input-inner .stTextInput input:focus,
.input-inner [data-testid="stTextInput"] input:focus,
div[data-testid="column"]:first-child .stTextInput input:focus {
    border-color: rgba(91,94,244,0.5) !important;
    box-shadow: 0 0 0 3px rgba(91,94,244,0.1), 0 2px 16px rgba(0,0,0,0.2) !important;
    outline: none !important;
}
.input-inner .stTextInput input::placeholder,
div[data-testid="column"]:first-child .stTextInput input::placeholder {
    color: #2c3a55 !important;
}

/* Send button */
div[data-testid="column"]:last-child .stButton button {
    background: linear-gradient(135deg, #5b5ef4, #7c3aed) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 14px !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    height: 50px !important;
    min-width: 76px;
    box-shadow: 0 2px 14px rgba(91,94,244,0.3) !important;
    font-family: 'Google Sans', sans-serif !important;
    transition: opacity 0.18s ease, box-shadow 0.18s ease, transform 0.1s ease !important;
}
div[data-testid="column"]:last-child .stButton button:hover {
    opacity: 0.9;
    box-shadow: 0 4px 22px rgba(91,94,244,0.45) !important;
    transform: translateY(-1px);
}
div[data-testid="column"]:last-child .stButton button:active {
    transform: translateY(0px);
}

/* ════════════════════════════════════════════════════════════════════════
   WELCOME SCREEN
   ════════════════════════════════════════════════════════════════════════ */
.welcome {
    text-align: center;
    padding: 42px 20px 28px;
}
.welcome-glow {
    width: 44px;
    height: 44px;
    background: linear-gradient(135deg, #5b5ef4, #8b5cf6);
    border-radius: 13px;
    margin: 0 auto 18px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
    box-shadow: 0 0 28px rgba(91,94,244,0.3), 0 0 60px rgba(91,94,244,0.12);
}
.welcome h1 {
    font-weight: 300;
    font-size: 1.75rem;
    color: #e8eeff;
    letter-spacing: -0.025em;
    margin-bottom: 8px;
    line-height: 1.25;
    font-family: 'Google Sans', sans-serif;
}
.welcome p {
    color: #3a4560;
    font-size: 0.9rem;
    margin: 0;
    line-height: 1.6;
}

/* ════════════════════════════════════════════════════════════════════════
   MISC STREAMLIT OVERRIDES
   ════════════════════════════════════════════════════════════════════════ */
/* Spinner */
[data-testid="stSpinner"] p {
    color: #5c6d8c !important;
    font-size: 0.85rem !important;
    font-family: 'Google Sans Text', sans-serif !important;
}

/* File uploader */
section[data-testid="stSidebar"] [data-testid="stFileUploader"] {
    background: transparent !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
    background: #10162a !important;
    border: 1px dashed rgba(255,255,255,0.07) !important;
    border-radius: 10px !important;
    padding: 10px !important;
    transition: border-color 0.18s ease;
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"]:hover {
    border-color: rgba(91,94,244,0.35) !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] p,
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] span {
    color: #3a4560 !important;
    font-size: 0.78rem !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] svg {
    color: #3a4560 !important;
}

/* Success / error messages */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    font-size: 0.83rem !important;
    font-family: 'Google Sans Text', sans-serif !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.07); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.12); }

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
    st.markdown("""
<div class="sidebar-brand">
  <div class="sidebar-brand-icon">✦</div>
  <span class="sidebar-brand-name">TraceAI</span>
</div>
""", unsafe_allow_html=True)

    st.markdown("### LLM Mode")
    mode = st.radio(
        "Select backend",
        ["Online (Groq)", "Offline (Ollama)"],
        label_visibility="collapsed"
    )
    if "Online" in mode:
        llm.MODE = "online"
        st.markdown('<span style="color:#22c55e;font-size:0.77rem;font-weight:500;">● Groq · llama3-8b</span>', unsafe_allow_html=True)
    else:
        llm.MODE = "offline"
        st.markdown('<span style="color:#22c55e;font-size:0.77rem;font-weight:500;">● Ollama · phi3:mini</span>', unsafe_allow_html=True)

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
        st.markdown(
            '<div class="kb-status">'
            '<div class="kb-dot-ready"></div>'
            '<span style="color:#22c55e;">Knowledge base ready</span>'
            '</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="kb-status">'
            '<div class="kb-dot-idle"></div>'
            '<span style="color:#3a4560;">No knowledge base loaded</span>'
            '</div>',
            unsafe_allow_html=True
        )

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
  <div class="welcome-glow">✦</div>
  <h1>How can I help you?</h1>
  <p>Upload your documents and ask anything.</p>
</div>
""", unsafe_allow_html=True)

for msg in st.session_state.chat_history:
    role = msg["role"]

    if role == "user":
        st.markdown(f"""
<div class="user-msg-wrap">
  <div class="role-label">You</div>
  <div class="user-bubble">{msg["content"]}</div>
</div>""", unsafe_allow_html=True)

    elif role == "assistant":
        st.markdown("""
<div class="assistant-msg-wrap">
  <div class="assistant-header">
    <div class="assistant-icon">✦</div>
    <div class="role-label" style="margin-bottom:0;">TraceAI</div>
  </div>
""", unsafe_allow_html=True)

        parsed          = msg.get("answer_parsed", {})
        answer_text     = parsed.get("answer_text", "")
        evidence_points = parsed.get("evidence_points", [])
        reasoning_text  = parsed.get("reasoning_text", "")
        low_confidence  = parsed.get("low_confidence", False)

        if not answer_text:
            answer_text = msg.get("content", "")

        if low_confidence:
            st.markdown(
                '<div class="low-conf-banner">⚠ Low confidence — retrieved content may not fully address this question.</div>',
                unsafe_allow_html=True
            )

        if answer_text:
            st.markdown(f"""
<div class="answer-block">
  <span class="answer-label">Answer</span>
  <div class="answer-text">{answer_text}</div>
</div>""", unsafe_allow_html=True)

        if reasoning_text:
            with st.expander("💡 Reasoning"):
                st.markdown(f"*{reasoning_text}*")

        if msg.get("sources"):
            with st.expander(f"📄 Sources ({len(msg['sources'])})"):
                for i, src in enumerate(msg["sources"], 1):
                    score_pct = int(src.get("score", 0) * 100)
                    st.markdown(
                        f'<span style="color:#4b5563;font-size:0.77rem;font-family:monospace;">'
                        f'{src["filename"]} · p.{src["page"]} · {score_pct}%</span>',
                        unsafe_allow_html=True
                    )
                    st.caption(src["text"][:350] + ("..." if len(src["text"]) > 350 else ""))
                    if i < len(msg["sources"]):
                        st.markdown("<hr style='border:none;border-top:1px solid rgba(255,255,255,0.04);margin:10px 0'>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div></div>", unsafe_allow_html=True)

# ─── Fixed input bar ──────────────────────────────────────────────────────────
st.markdown('<div class="input-area"><div class="input-inner">', unsafe_allow_html=True)

col1, col2 = st.columns([6, 1])
with col1:
    question = st.text_input(
        "question",
        placeholder="Ask a question about your documents..." if st.session_state.kb_built else "Build knowledge base first...",
        label_visibility="collapsed",
        key="question_input"
    )
with col2:
    send = st.button("Send ↑", use_container_width=True, key="send_btn")

st.markdown("</div></div>", unsafe_allow_html=True)

# ─── Phase 1: store and rerun ─────────────────────────────────────────────────
if send and question.strip():
    st.session_state.pending_question = question.strip()
    st.rerun()