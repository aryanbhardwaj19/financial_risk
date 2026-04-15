"""
Streamlit front-end for the Multi-Agent Financial Risk Analysis System.

Run with::

    streamlit run app.py

Layout
------
- **Sidebar**: file uploader, query input, Run Analysis button, JSON export.
- **Main area**: 4-tab output (Risk Report · Financial Metrics · Sources · Agent Trace).

Session state is used to cache the FAISS vector store between queries so
documents are only embedded once per upload batch.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

import pandas as pd
import streamlit as st

# ── Ensure project root is importable ───────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import settings
from ingestion.pdf_loader import load_pdf
from ingestion.csv_loader import load_csv
from rag.chunker import DocumentChunker
from rag.embedder import DocumentEmbedder
from graph.workflow import run_analysis
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Page configuration ──────────────────────────────────────────────
st.set_page_config(
    page_title="FinRisk · AI Financial Risk Analysis",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Instrument+Serif:ital@0;1&display=swap');

    :root {
        --bg: #0c0f1a;
        --bg-alt: #111528;
        --surface: rgba(255,255,255,0.04);
        --surface-hover: rgba(255,255,255,0.07);
        --surface-border: rgba(255,255,255,0.07);
        --border: rgba(255,255,255,0.08);
        --text: #eee;
        --text-secondary: #9ca3af;
        --text-muted: #6b7280;
        --accent: #818cf8;
        --accent-2: #a78bfa;
        --accent-3: #c084fc;
        --gradient: linear-gradient(135deg, #818cf8, #a78bfa, #c084fc);
        --success: #34d399;
        --warning: #fbbf24;
        --danger: #f87171;
    }

    /* ── Global ─────────────────────────────────────────────────── */
    .stApp {
        font-family: 'Inter', sans-serif !important;
        background: var(--bg) !important;
        color: var(--text) !important;
    }
    .stApp > header { background: transparent !important; }

    .main .block-container {
        padding: 1rem 2rem 2rem !important;
        max-width: 100% !important;
    }

    .stApp, .stApp p, .stApp span, .stApp div, .stApp li, .stApp label,
    .stApp input, .stApp textarea, .stApp button {
        font-family: 'Inter', sans-serif !important;
    }

    h1 {
        font-weight: 700 !important;
        font-size: 1.5rem !important;
        letter-spacing: -0.03em !important;
        color: #fff !important;
        background: none !important;
        -webkit-text-fill-color: #fff !important;
    }
    h2 {
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        color: #fff !important;
    }
    h3 {
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        color: #fff !important;
    }
    p, li {
        color: var(--text-secondary) !important;
        line-height: 1.65 !important;
        font-size: 0.88rem !important;
    }

    /* ── Sidebar ────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: #0a0d17 !important;
        border-right: 1px solid var(--border) !important;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2 {
        font-size: 0.72rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.12em !important;
        font-weight: 600 !important;
        color: var(--text-muted) !important;
        -webkit-text-fill-color: var(--text-muted) !important;
    }

    /* ── File uploader ──────────────────────────────────────────── */
    [data-testid="stFileUploader"] {
        background: var(--surface) !important;
        border: 1px dashed rgba(129,140,248,0.25) !important;
        border-radius: 10px !important;
        transition: all 0.25s ease !important;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: var(--accent) !important;
        background: rgba(129,140,248,0.05) !important;
    }
    [data-testid="stFileUploader"] label { color: var(--text-secondary) !important; font-size: 0.82rem !important; }
    [data-testid="stFileUploader"] small { color: var(--text-muted) !important; }

    /* ── Buttons ────────────────────────────────────────────────── */
    .stButton > button {
        background: var(--gradient) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        padding: 0.55rem 1.2rem !important;
        transition: all 0.25s ease !important;
        box-shadow: 0 4px 20px rgba(129,140,248,0.25) !important;
    }
    .stButton > button p,
    .stButton > button span,
    .stButton > button div {
        color: #fff !important;
        -webkit-text-fill-color: #fff !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 25px rgba(129,140,248,0.35) !important;
    }
    .stButton > button:disabled {
        background: rgba(255,255,255,0.06) !important;
        color: var(--text-muted) !important;
        box-shadow: none !important;
    }
    .stButton > button:disabled p,
    .stButton > button:disabled span,
    .stButton > button:disabled div {
        color: var(--text-muted) !important;
        -webkit-text-fill-color: var(--text-muted) !important;
    }
    .stDownloadButton > button {
        background: rgba(52,211,153,0.1) !important;
        color: var(--success) !important;
        border: 1px solid rgba(52,211,153,0.2) !important;
        box-shadow: none !important;
    }
    .stDownloadButton > button p,
    .stDownloadButton > button span {
        color: var(--success) !important;
        -webkit-text-fill-color: var(--success) !important;
    }
    .stDownloadButton > button:hover {
        background: rgba(52,211,153,0.2) !important;
    }

    /* ── Text area ──────────────────────────────────────────────── */
    .stTextArea textarea {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        color: var(--text) !important;
        font-size: 0.88rem !important;
        transition: border-color 0.2s ease !important;
    }
    .stTextArea textarea:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px rgba(129,140,248,0.15) !important;
    }
    .stTextArea textarea::placeholder { color: var(--text-muted) !important; }

    /* ── Tabs ───────────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--surface) !important;
        border-radius: 10px !important;
        padding: 4px !important;
        border: 1px solid var(--border) !important;
        gap: 4px !important;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: var(--text-muted) !important;
        border-radius: 7px !important;
        font-weight: 500 !important;
        font-size: 0.82rem !important;
        padding: 0.45rem 0.9rem !important;
        border: none !important;
        transition: all 0.2s ease !important;
    }
    .stTabs [data-baseweb="tab"] p,
    .stTabs [data-baseweb="tab"] span {
        color: var(--text-muted) !important;
        -webkit-text-fill-color: var(--text-muted) !important;
    }
    .stTabs [aria-selected="true"] {
        background: var(--gradient) !important;
        color: #fff !important;
        font-weight: 600 !important;
    }
    .stTabs [aria-selected="true"] p,
    .stTabs [aria-selected="true"] span {
        color: #fff !important;
        -webkit-text-fill-color: #fff !important;
    }
    .stTabs [data-baseweb="tab-highlight"],
    .stTabs [data-baseweb="tab-border"] { display: none !important; }

    /* ── DataFrames ─────────────────────────────────────────────── */
    [data-testid="stDataFrame"] {
        border-radius: 10px !important;
        border: 1px solid var(--border) !important;
        overflow: hidden !important;
    }

    /* ── Expanders ──────────────────────────────────────────────── */
    details {
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        background: var(--surface) !important;
    }
    details summary { font-weight: 500 !important; color: var(--text) !important; font-size: 0.85rem !important; }

    /* ── Alerts ─────────────────────────────────────────────────── */
    .stAlert, [data-testid="stAlert"] { border-radius: 8px !important; border: none !important; }

    /* ── Divider ────────────────────────────────────────────────── */
    hr { border-color: var(--border) !important; margin: 1.25rem 0 !important; }

    /* ── Metrics ────────────────────────────────────────────────── */
    [data-testid="stMetric"] {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        padding: 0.85rem 1rem !important;
    }
    [data-testid="stMetricValue"] { color: var(--accent) !important; font-weight: 700 !important; }
    [data-testid="stMetricLabel"] { color: var(--text-muted) !important; font-size: 0.72rem !important; text-transform: uppercase !important; letter-spacing: 0.05em !important; }

    /* ── Caption ────────────────────────────────────────────────── */
    .stCaption, [data-testid="stCaption"] { color: var(--text-muted) !important; font-size: 0.72rem !important; }

    /* ── Scrollbar ──────────────────────────────────────────────── */
    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(129,140,248,0.2); border-radius: 3px; }

    /* ═══════════════════════════════════════════════════════════════
       RISK BADGES
       ═══════════════════════════════════════════════════════════════ */
    .risk-badge {
        display: inline-block;
        padding: 0.4rem 1.2rem;
        border-radius: 50px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 1.25rem;
    }
    .risk-LOW      { background: rgba(52,211,153,0.12); color: #34d399; border: 1px solid rgba(52,211,153,0.25); }
    .risk-MEDIUM   { background: rgba(251,191,36,0.12); color: #fbbf24; border: 1px solid rgba(251,191,36,0.25); }
    .risk-HIGH     { background: rgba(248,113,113,0.12); color: #f87171; border: 1px solid rgba(248,113,113,0.25); }
    .risk-CRITICAL { background: rgba(248,113,113,0.2); color: #fca5a5; border: 1px solid rgba(248,113,113,0.3); }
    .risk-UNKNOWN  { background: var(--surface); color: var(--text-muted); border: 1px solid var(--border); }

    /* ═══════════════════════════════════════════════════════════════
       AGENT TRACE
       ═══════════════════════════════════════════════════════════════ */
    .trace-step {
        display: flex; align-items: center; gap: 0.65rem;
        padding: 0.5rem 0.65rem; border-radius: 6px;
        font-size: 0.85rem; color: var(--text);
        transition: background 0.15s ease; margin-bottom: 2px;
    }
    .trace-step:hover { background: var(--surface); }
    .trace-dot {
        width: 8px; height: 8px; border-radius: 50%;
        background: var(--gradient); flex-shrink: 0;
    }

    /* ═══════════════════════════════════════════════════════════════
       SECTION HEADER
       ═══════════════════════════════════════════════════════════════ */
    .sh { display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.85rem; padding-bottom: 0.55rem; border-bottom: 1px solid var(--border); }
    .sh-icon { font-size: 0.95rem; }
    .sh-title { font-size: 0.9rem; font-weight: 600; color: #fff; margin: 0; }

    /* ═══════════════════════════════════════════════════════════════
       HERO LANDING
       ═══════════════════════════════════════════════════════════════ */

    /* Background glow effects */
    .bg-glow {
        position: fixed; top: 0; left: 0; right: 0; bottom: 0;
        pointer-events: none; z-index: 0; overflow: hidden;
    }
    .bg-glow .orb {
        position: absolute; border-radius: 50%; filter: blur(100px); opacity: 0.12;
    }
    .bg-glow .orb-1 { width: 600px; height: 600px; background: #818cf8; top: -200px; right: -100px; animation: orbFloat 20s ease-in-out infinite; }
    .bg-glow .orb-2 { width: 500px; height: 500px; background: #c084fc; bottom: -150px; left: -100px; animation: orbFloat 25s ease-in-out infinite reverse; }
    .bg-glow .orb-3 { width: 400px; height: 400px; background: #818cf8; top: 40%; left: 50%; animation: orbFloat 18s ease-in-out infinite 5s; }
    @keyframes orbFloat {
        0%, 100% { transform: translate(0, 0); }
        33% { transform: translate(30px, -30px); }
        66% { transform: translate(-20px, 20px); }
    }

    .hero-wrapper {
        position: relative; z-index: 1;
    }

    /* Top bar */
    .top-bar {
        display: flex; justify-content: space-between; align-items: center;
        padding: 0.5rem 0 1.5rem;
    }
    .top-logo {
        display: flex; align-items: center; gap: 0.5rem;
        font-size: 1.1rem; font-weight: 700; color: #fff;
    }
    .top-logo span { background: var(--gradient); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 1.3rem; }
    .top-badge {
        font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.1em;
        color: var(--accent); background: rgba(129,140,248,0.1);
        padding: 0.3rem 0.7rem; border-radius: 50px;
        border: 1px solid rgba(129,140,248,0.2); font-weight: 600;
    }

    /* Hero text */
    .hero-center {
        text-align: center; padding: 2rem 0 2.5rem;
    }
    .hero-title {
        font-family: 'Instrument Serif', Georgia, serif !important;
        font-size: 3.2rem !important; font-weight: 400 !important; font-style: italic;
        color: #fff !important; -webkit-text-fill-color: #fff !important;
        background: none !important;
        margin-bottom: 0.6rem !important; line-height: 1.1 !important;
    }
    .hero-sub {
        color: var(--text-muted); font-size: 1rem; max-width: 550px;
        margin: 0 auto; line-height: 1.65;
    }

    /* Stats row */
    .stats-row {
        display: flex; justify-content: center; gap: 3rem;
        padding: 1.5rem 0 2.5rem;
    }
    .stat-item { text-align: center; }
    .stat-value {
        font-size: 1.6rem; font-weight: 800; letter-spacing: -0.03em;
        background: var(--gradient); -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .stat-label { font-size: 0.7rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.1em; margin-top: 0.2rem; }

    /* Steps row */
    .steps-row {
        display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem;
        margin-bottom: 1.5rem;
    }
    .step-card {
        background: var(--surface); border: 1px solid var(--surface-border);
        border-radius: 12px; padding: 1.5rem; text-align: left;
        transition: all 0.25s ease; position: relative; overflow: hidden;
    }
    .step-card::after {
        content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
        background: var(--gradient); opacity: 0; transition: opacity 0.25s ease;
    }
    .step-card:hover { background: var(--surface-hover); border-color: rgba(129,140,248,0.15); transform: translateY(-2px); }
    .step-card:hover::after { opacity: 1; }
    .step-num {
        display: inline-flex; align-items: center; justify-content: center;
        width: 30px; height: 30px; border-radius: 8px;
        background: var(--gradient); color: #fff; font-weight: 700;
        font-size: 0.78rem; margin-bottom: 0.85rem;
    }
    .step-title { color: #fff; font-weight: 600; font-size: 0.92rem; margin-bottom: 0.3rem; }
    .step-desc { color: var(--text-muted); font-size: 0.8rem; line-height: 1.5; }

    /* Pipeline + Features grid */
    .bottom-grid {
        display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;
    }

    .card-block {
        background: var(--surface); border: 1px solid var(--surface-border);
        border-radius: 12px; padding: 1.25rem; overflow: hidden;
    }
    .card-header {
        font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.1em;
        color: var(--text-muted); font-weight: 600; margin-bottom: 1rem;
        padding-bottom: 0.6rem; border-bottom: 1px solid var(--border);
    }

    /* Pipeline items */
    .pipe-list { display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; }
    .pipe-item {
        display: flex; align-items: center; gap: 0.55rem;
        padding: 0.55rem 0.65rem; border-radius: 8px; font-size: 0.82rem;
        color: var(--text-secondary); transition: background 0.15s ease;
    }
    .pipe-item:hover { background: rgba(255,255,255,0.03); }
    .pipe-icon {
        width: 28px; height: 28px; border-radius: 6px;
        display: flex; align-items: center; justify-content: center;
        font-size: 0.85rem; flex-shrink: 0;
    }
    .pipe-icon.p1 { background: rgba(129,140,248,0.12); }
    .pipe-icon.p2 { background: rgba(52,211,153,0.12); }
    .pipe-icon.p3 { background: rgba(251,191,36,0.12); }
    .pipe-icon.p4 { background: rgba(248,113,113,0.12); }
    .pipe-icon.p5 { background: rgba(167,139,250,0.12); }
    .pipe-icon.p6 { background: rgba(192,132,252,0.12); }
    .pipe-name { font-weight: 600; color: #fff; font-size: 0.82rem; }
    .pipe-desc { color: var(--text-muted); font-size: 0.72rem; }

    /* Feature items */
    .feat-list { display: flex; flex-direction: column; gap: 0.65rem; }
    .feat-item {
        display: flex; align-items: flex-start; gap: 0.65rem;
        padding: 0.55rem 0.65rem; border-radius: 8px;
        transition: background 0.15s ease;
    }
    .feat-item:hover { background: rgba(255,255,255,0.03); }
    .feat-dot {
        width: 6px; height: 6px; border-radius: 50%; margin-top: 6px;
        flex-shrink: 0;
    }
    .feat-dot.d1 { background: var(--accent); }
    .feat-dot.d2 { background: var(--accent-2); }
    .feat-dot.d3 { background: var(--accent-3); }
    .feat-dot.d4 { background: var(--success); }
    .feat-name { font-weight: 600; color: #fff; font-size: 0.82rem; }
    .feat-desc { color: var(--text-muted); font-size: 0.75rem; line-height: 1.45; }

    /* Supported formats bar */
    .formats-bar {
        display: flex; justify-content: center; gap: 0.5rem;
        padding: 1.25rem 0 0.5rem; flex-wrap: wrap;
    }
    .format-tag {
        font-size: 0.68rem; font-weight: 600; text-transform: uppercase;
        letter-spacing: 0.06em; color: var(--text-muted);
        background: var(--surface); border: 1px solid var(--border);
        padding: 0.3rem 0.65rem; border-radius: 5px;
    }

    /* ── Sidebar extras ────────────────────────────────────────── */
    .sidebar-brand {
        display: flex; align-items: center; gap: 0.55rem;
        padding: 0.25rem 0 1rem; margin-bottom: 0.5rem;
        border-bottom: 1px solid var(--border);
    }
    .brand-logo { font-size: 1.3rem; background: var(--gradient); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .brand-text { font-size: 0.92rem; font-weight: 700; color: #fff; }
    .brand-sub { font-size: 0.62rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.1em; }

    .upload-status {
        display: flex; align-items: center; gap: 0.4rem;
        padding: 0.5rem 0.65rem; border-radius: 6px;
        font-size: 0.78rem; font-weight: 500; margin: 0.5rem 0;
    }
    .upload-status.success { background: rgba(52,211,153,0.1); color: var(--success); border: 1px solid rgba(52,211,153,0.2); }
    .upload-status.error { background: rgba(248,113,113,0.1); color: var(--danger); border: 1px solid rgba(248,113,113,0.2); }
    .upload-status.warning { background: rgba(251,191,36,0.1); color: var(--warning); border: 1px solid rgba(251,191,36,0.2); }

    .settings-card {
        background: var(--surface); border: 1px solid var(--border);
        border-radius: 8px; padding: 0.65rem 0.75rem;
        font-size: 0.72rem; color: var(--text-muted); line-height: 1.9;
    }
    .settings-card code {
        background: rgba(129,140,248,0.1); color: var(--accent);
        padding: 0.1rem 0.3rem; border-radius: 3px; font-size: 0.68rem;
    }
    .settings-label {
        font-weight: 600; color: var(--text-secondary);
        font-size: 0.65rem; text-transform: uppercase;
        letter-spacing: 0.08em; margin-bottom: 0.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Session state defaults ──────────────────────────────────────────
def _init_session():
    defaults = {
        "store_built": False,
        "ingested_docs": [],
        "report": None,
        "uploaded_names": [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_session()


# ═════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div><span class="brand-logo">✦</span></div>
            <div>
                <div class="brand-text">FinRisk</div>
                <div class="brand-sub">AI Risk Analysis</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.header("Documents")

    uploaded_files = st.file_uploader(
        "Upload PDFs or CSVs",
        type=["pdf", "csv", "tsv", "xls", "xlsx"],
        accept_multiple_files=True,
        help="Financial reports, balance sheets, income statements, etc.",
    )

    # ── Ingest + embed on new uploads ───────────────────────────────
    if uploaded_files:
        current_names = sorted([f.name for f in uploaded_files])

        if current_names != st.session_state["uploaded_names"]:
            data_dir = _PROJECT_ROOT / "data"
            data_dir.mkdir(exist_ok=True)

            with st.spinner("Ingesting documents …"):
                all_docs: list = []

                for uf in uploaded_files:
                    dest = data_dir / uf.name
                    dest.write_bytes(uf.getvalue())

                    ext = Path(uf.name).suffix.lower()
                    try:
                        if ext == ".pdf":
                            pages = load_pdf(str(dest))
                            all_docs.extend(pages)
                        else:
                            result = load_csv(str(dest))
                            all_docs.append(
                                {
                                    "text": result["summary"],
                                    "source": result["source"],
                                    "page": 0,
                                    "tables": [],
                                }
                            )
                    except Exception as exc:
                        st.error(f"Failed to load **{uf.name}**: {exc}")
                        logger.exception("Ingestion error for %s", uf.name)

                st.session_state["ingested_docs"] = all_docs

            if all_docs:
                with st.spinner("Chunking & embedding …"):
                    try:
                        chunker = DocumentChunker()
                        chunks = chunker.chunk_documents(all_docs)

                        if not chunks:
                            st.markdown(
                                '<div class="upload-status warning">'
                                "⚠ No extractable text found. This usually happens "
                                "with scanned/image-based PDFs. Try a digitally-born "
                                "PDF or CSV/Excel file."
                                "</div>",
                                unsafe_allow_html=True,
                            )
                            logger.warning(
                                "Chunker returned 0 chunks for %d doc(s)", len(all_docs)
                            )
                        else:
                            embedder = DocumentEmbedder()
                            store = embedder.embed_and_store(
                                chunks, settings.VECTOR_DB_PATH
                            )

                            if store is not None:
                                st.session_state["store_built"] = True
                                st.session_state["uploaded_names"] = current_names
                                st.markdown(
                                    f'<div class="upload-status success">'
                                    f"✓ Indexed {len(chunks)} chunks from "
                                    f"{len(uploaded_files)} file(s)"
                                    f"</div>",
                                    unsafe_allow_html=True,
                                )
                            else:
                                st.markdown(
                                    '<div class="upload-status warning">'
                                    "⚠ Embedding returned no results."
                                    "</div>",
                                    unsafe_allow_html=True,
                                )
                    except Exception as exc:
                        st.markdown(
                            f'<div class="upload-status error">'
                            f"✕ Embedding failed: {exc}"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                        logger.exception("Embedding error")
            else:
                st.markdown(
                    '<div class="upload-status warning">'
                    "⚠ No content extracted from the uploaded files."
                    "</div>",
                    unsafe_allow_html=True,
                )

    st.divider()

    st.header("Query")
    query = st.text_area(
        "Enter your financial risk question:",
        height=100,
        placeholder=(
            "e.g., Assess the credit risk of ACME Corp based on their "
            "2024 annual report."
        ),
    )

    run_btn = st.button(
        "Run Analysis →",
        type="primary",
        use_container_width=True,
        disabled=not st.session_state["store_built"],
    )

    if not st.session_state["store_built"]:
        st.caption("Upload documents to enable analysis.")

    st.divider()

    st.markdown(
        f"""
        <div class="settings-card">
            <div class="settings-label">Configuration</div>
            Model <code>{settings.LLM_MODEL}</code><br>
            Embeddings <code>{settings.EMBEDDING_MODEL}</code><br>
            Chunk <code>{settings.CHUNK_SIZE}</code> · Overlap <code>{settings.CHUNK_OVERLAP}</code>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state["report"]:
        st.markdown("<br>", unsafe_allow_html=True)
        report_json = json.dumps(st.session_state["report"], indent=2, default=str)
        st.download_button(
            label="↓ Export Report JSON",
            data=report_json,
            file_name="risk_report.json",
            mime="application/json",
            use_container_width=True,
        )


# ═════════════════════════════════════════════════════════════════════
# MAIN: RUN ANALYSIS
# ═════════════════════════════════════════════════════════════════════
if run_btn:
    if not query.strip():
        st.warning("Please enter a question before running the analysis.")
        st.stop()

    with st.spinner("Analyzing …"):
        try:
            report = run_analysis(query=query.strip())
            st.session_state["report"] = report
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            logger.exception("run_analysis error")
            st.stop()


# ═════════════════════════════════════════════════════════════════════
# MAIN: DISPLAY
# ═════════════════════════════════════════════════════════════════════
report = st.session_state.get("report")

if report:
    st.title("Financial Risk Report")

    tab_report, tab_metrics, tab_sources, tab_trace = st.tabs(
        ["Risk Report", "Financial Metrics", "Sources", "Agent Trace"]
    )

    # ── TAB 1: Risk Report ──────────────────────────────────────────
    with tab_report:
        risk_level = report.get("risk_level", "UNKNOWN")
        badge = report.get("risk_badge", risk_level)
        st.markdown(
            f'<div class="risk-badge risk-{risk_level}">{badge}</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sh"><span class="sh-icon">◈</span><span class="sh-title">Executive Summary</span></div>', unsafe_allow_html=True)
        st.markdown(report.get("summary", "*No summary available.*"))
        st.divider()

        anomalies = report.get("anomalies", [])
        if anomalies:
            st.markdown('<div class="sh"><span class="sh-icon">△</span><span class="sh-title">Detected Anomalies</span></div>', unsafe_allow_html=True)
            rows = [{"Metric": a.get("metric",""), "Description": a.get("description",""), "Severity": a.get("severity","info").upper()} for a in anomalies]
            df_a = pd.DataFrame(rows)
            def _cs(v):
                return {"CRITICAL":"color:#f87171;font-weight:700","WARNING":"color:#fbbf24;font-weight:600","INFO":"color:#60a5fa"}.get(v,"")
            st.dataframe(df_a.style.applymap(_cs, subset=["Severity"]), use_container_width=True, hide_index=True)
        else:
            st.info("No anomalies detected.")
        st.divider()

        recs = report.get("recommendations", [])
        if recs:
            st.markdown('<div class="sh"><span class="sh-icon">◇</span><span class="sh-title">Recommendations</span></div>', unsafe_allow_html=True)
            for i, rec in enumerate(recs, 1):
                st.markdown(f"**{i}.** {rec}")
        st.divider()

        st.markdown('<div class="sh"><span class="sh-icon">◎</span><span class="sh-title">Risk Justification</span></div>', unsafe_allow_html=True)
        st.markdown(report.get("justification", "*No justification.*"))
        st.divider()

        verified = report.get("verification_status", False)
        v_notes = report.get("verification_notes", [])
        if verified:
            st.success("✓ **Verified** — All key metrics traced to source documents.")
        else:
            st.warning("△ **Partially Verified** — Some metrics could not be confirmed.")
        if v_notes:
            with st.expander("Verification details"):
                for note in v_notes:
                    st.markdown(f"- {note}")

    # ── TAB 2: Financial Metrics ────────────────────────────────────
    with tab_metrics:
        key_metrics = report.get("key_metrics", [])
        if key_metrics:
            st.markdown('<div class="sh"><span class="sh-icon">◈</span><span class="sh-title">Key Financial Ratios</span></div>', unsafe_allow_html=True)
            cols = st.columns(min(len(key_metrics), 4))
            for idx, m in enumerate(key_metrics[:4]):
                with cols[idx]:
                    s = m.get("status","").upper()
                    dc = "inverse" if s == "CRITICAL" else ("off" if s == "WARNING" else "normal")
                    st.metric(label=m.get("name",""), value=m.get("value","N/A"), delta=s or None, delta_color=dc)
            st.divider()

            rows = [{"Ratio":m.get("name",""),"Value":m.get("value"),"Status":m.get("status","").upper(),"Threshold":m.get("threshold","")} for m in key_metrics]
            df_m = pd.DataFrame(rows)
            def _cst(v):
                return {"SAFE":"background-color:rgba(52,211,153,0.1);color:#34d399;font-weight:600","WARNING":"background-color:rgba(251,191,36,0.1);color:#fbbf24;font-weight:600","CRITICAL":"background-color:rgba(248,113,113,0.1);color:#f87171;font-weight:700"}.get(v,"")
            st.dataframe(df_m.style.applymap(_cst, subset=["Status"]), use_container_width=True, hide_index=True)
            st.divider()

            st.markdown('<div class="sh"><span class="sh-icon">▊</span><span class="sh-title">Ratio Values</span></div>', unsafe_allow_html=True)
            cd = pd.DataFrame({"Ratio":[m.get("name","") for m in key_metrics],"Value":[m.get("value",0) or 0 for m in key_metrics]}).set_index("Ratio")
            st.bar_chart(cd)
        else:
            st.info("No financial metrics computed.")

    # ── TAB 3: Sources ──────────────────────────────────────────────
    with tab_sources:
        sources = report.get("sources", [])
        if sources:
            st.markdown(f'<div class="sh"><span class="sh-icon">◉</span><span class="sh-title">Retrieved Sources ({len(sources)})</span></div>', unsafe_allow_html=True)
            for i, src in enumerate(sources, 1):
                pg = src.get("page","?")
                sf = Path(src.get("source","unknown")).name
                with st.expander(f"Source {i} — {sf} · p.{pg}"):
                    st.markdown(f"**File:** `{sf}` · **Page:** {pg}")
                    st.divider()
                    st.markdown(src.get("excerpt",""))
        else:
            st.info("No source chunks retrieved.")

    # ── TAB 4: Agent Trace ──────────────────────────────────────────
    with tab_trace:
        trace = report.get("agent_trace", [])
        if trace:
            st.markdown('<div class="sh"><span class="sh-icon">◈</span><span class="sh-title">Execution Timeline</span></div>', unsafe_allow_html=True)
            rc = trace.count("retriever")
            if rc > 1:
                st.info(f"↻ **{rc-1} re-retrieval(s)** during analysis (total: {rc}).")
            icons = {"planner":"◇","retriever":"◎","analyst":"▊","evaluator":"△","verifier":"◈","reporter":"◉"}
            for idx, a in enumerate(trace, 1):
                st.markdown(f'<div class="trace-step"><span class="trace-dot"></span><strong>Step {idx}</strong>&nbsp;&nbsp;{icons.get(a,"·")} {a}</div>', unsafe_allow_html=True)
            st.divider()
            st.caption(" → ".join(trace))
        else:
            st.info("No agent trace available.")

else:
    # ═════════════════════════════════════════════════════════════════
    # LANDING PAGE — FULL SCREEN
    # ═════════════════════════════════════════════════════════════════

    # Background orbs
    st.markdown(
        """
        <div class="bg-glow">
            <div class="orb orb-1"></div>
            <div class="orb orb-2"></div>
            <div class="orb orb-3"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Hero wrapper
    st.markdown('<div class="hero-wrapper">', unsafe_allow_html=True)

    # Top bar
    st.markdown(
        """
        <div class="top-bar">
            <div class="top-logo"><span>✦</span> FinRisk</div>
            <div class="top-badge">Multi-Agent AI</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Hero center
    st.markdown(
        """
        <div class="hero-center">
            <div class="hero-title">Intelligent Financial<br>Risk Analysis</div>
            <div class="hero-sub">
                Upload your financial documents and let our AI agent pipeline
                analyze risks, detect anomalies, and deliver verified assessments
                with actionable recommendations.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Stats
    st.markdown(
        """
        <div class="stats-row">
            <div class="stat-item">
                <div class="stat-value">6</div>
                <div class="stat-label">AI Agents</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">5+</div>
                <div class="stat-label">File Formats</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">RAG</div>
                <div class="stat-label">Powered</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">✓</div>
                <div class="stat-label">Source Verified</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 3-step cards
    st.markdown(
        """
        <div class="steps-row">
            <div class="step-card">
                <div class="step-num">1</div>
                <div class="step-title">Upload Documents</div>
                <div class="step-desc">Drag & drop PDFs, CSVs, or Excel files containing financial reports, balance sheets, or income statements.</div>
            </div>
            <div class="step-card">
                <div class="step-num">2</div>
                <div class="step-title">Ask a Question</div>
                <div class="step-desc">Enter a specific risk analysis question — credit risk, liquidity concerns, solvency ratios, trend analysis.</div>
            </div>
            <div class="step-card">
                <div class="step-num">3</div>
                <div class="step-title">Get AI Report</div>
                <div class="step-desc">Receive a comprehensive risk report with anomaly detection, financial ratios, and source-verified recommendations.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Bottom grid: Pipeline + Features
    st.markdown(
        """
        <div class="bottom-grid">
            <div class="card-block">
                <div class="card-header">Agent Pipeline</div>
                <div class="pipe-list">
                    <div class="pipe-item">
                        <div class="pipe-icon p1">◇</div>
                        <div><div class="pipe-name">Planner</div><div class="pipe-desc">Designs analysis strategy</div></div>
                    </div>
                    <div class="pipe-item">
                        <div class="pipe-icon p2">◎</div>
                        <div><div class="pipe-name">Retriever</div><div class="pipe-desc">Fetches relevant sections</div></div>
                    </div>
                    <div class="pipe-item">
                        <div class="pipe-icon p3">▊</div>
                        <div><div class="pipe-name">Analyst</div><div class="pipe-desc">Computes financial metrics</div></div>
                    </div>
                    <div class="pipe-item">
                        <div class="pipe-icon p4">△</div>
                        <div><div class="pipe-name">Evaluator</div><div class="pipe-desc">Assesses risk levels</div></div>
                    </div>
                    <div class="pipe-item">
                        <div class="pipe-icon p5">◈</div>
                        <div><div class="pipe-name">Verifier</div><div class="pipe-desc">Cross-checks sources</div></div>
                    </div>
                    <div class="pipe-item">
                        <div class="pipe-icon p6">◉</div>
                        <div><div class="pipe-name">Reporter</div><div class="pipe-desc">Generates final report</div></div>
                    </div>
                </div>
            </div>
            <div class="card-block">
                <div class="card-header">Features</div>
                <div class="feat-list">
                    <div class="feat-item">
                        <div class="feat-dot d1"></div>
                        <div><div class="feat-name">Smart Document Parsing</div><div class="feat-desc">Extracts text and tables from PDFs with automatic fallback to PyMuPDF for complex layouts.</div></div>
                    </div>
                    <div class="feat-item">
                        <div class="feat-dot d2"></div>
                        <div><div class="feat-name">RAG-Powered Analysis</div><div class="feat-desc">Retrieval-augmented generation ensures answers are grounded in your actual documents.</div></div>
                    </div>
                    <div class="feat-item">
                        <div class="feat-dot d3"></div>
                        <div><div class="feat-name">Anomaly Detection</div><div class="feat-desc">Automatically identifies unusual patterns, outliers, and red flags in financial data.</div></div>
                    </div>
                    <div class="feat-item">
                        <div class="feat-dot d4"></div>
                        <div><div class="feat-name">Source Verification</div><div class="feat-desc">Every finding is traced back to the original document with page-level citations.</div></div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Supported formats
    st.markdown(
        """
        <div class="formats-bar">
            <div class="format-tag">PDF</div>
            <div class="format-tag">CSV</div>
            <div class="format-tag">TSV</div>
            <div class="format-tag">XLS</div>
            <div class="format-tag">XLSX</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('</div>', unsafe_allow_html=True)
