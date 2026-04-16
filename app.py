"""
Streamlit front-end for FinRisk — AI Financial Document Q&A.

Run with::

    streamlit run app.py

Uses a LangGraph multi-agent pipeline (Retriever → Analyzer →
Reporter) to answer questions about uploaded financial documents.
"""

from __future__ import annotations

import sys
from pathlib import Path

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
from rag.qa import ask_question
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Page configuration ──────────────────────────────────────────────
st.set_page_config(
    page_title="FinRisk · AI Financial Intelligence",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═════════════════════════════════════════════════════════════════════
# DESIGN SYSTEM — Premium Fintech Dark Theme
# ═════════════════════════════════════════════════════════════════════
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Instrument+Serif:ital@0;1&display=swap');

    /* ═══════════════════ TOKEN SYSTEM ═══════════════════ */
    :root {
        --bg-primary:   #0D1117;
        --bg-surface:   #161B22;
        --bg-elevated:  #1C2128;
        --bg-overlay:   rgba(22,27,34,0.85);
        --border:       rgba(139,148,158,0.12);
        --border-subtle:rgba(139,148,158,0.08);
        --border-accent:rgba(99,102,241,0.25);

        --text-primary: #E6EDF3;
        --text-body:    #C9D1D9;
        --text-muted:   #8B949E;
        --text-faint:   #484F58;

        --accent:       #6366F1;
        --accent-light: #818CF8;
        --accent-glow:  rgba(99,102,241,0.15);
        --teal:         #10B981;
        --teal-glow:    rgba(16,185,129,0.12);
        --amber:        #F59E0B;
        --amber-glow:   rgba(245,158,11,0.12);
        --purple:       #A78BFA;
        --purple-glow:  rgba(167,139,250,0.12);
        --red:          #F87171;
        --red-glow:     rgba(248,113,113,0.12);

        --gradient-brand: linear-gradient(135deg, #6366F1 0%, #818CF8 50%, #A78BFA 100%);
        --gradient-subtle:linear-gradient(180deg, rgba(99,102,241,0.06) 0%, transparent 100%);

        --radius-sm: 6px;
        --radius-md: 10px;
        --radius-lg: 14px;
        --radius-xl: 20px;
        --radius-pill: 50px;

        --shadow-sm: 0 1px 3px rgba(0,0,0,0.3);
        --shadow-md: 0 4px 16px rgba(0,0,0,0.4);
        --shadow-lg: 0 8px 32px rgba(0,0,0,0.5);
        --shadow-glow: 0 0 30px rgba(99,102,241,0.15);

        --transition: 180ms cubic-bezier(0.4,0,0.2,1);
    }

    /* ═══════════════════ GLOBAL ═══════════════════ */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background: var(--bg-primary) !important;
        color: var(--text-body) !important;
    }
    .stApp > header { background: transparent !important; }

    .main .block-container {
        padding: 0 2.5rem 2rem !important;
        max-width: 100% !important;
    }

    .stApp, .stApp p, .stApp span, .stApp div, .stApp li, .stApp label,
    .stApp input, .stApp textarea, .stApp button {
        font-family: 'Inter', -apple-system, sans-serif !important;
    }

    /* ── Typography Scale ───────────────────────────────── */
    h1 {
        font-weight: 700 !important; font-size: 1.5rem !important;
        letter-spacing: -0.03em !important; color: var(--text-primary) !important;
        background: none !important; -webkit-text-fill-color: var(--text-primary) !important;
    }
    h2 { font-weight: 600 !important; font-size: 1.15rem !important; color: var(--text-primary) !important; }
    h3 { font-weight: 600 !important; font-size: 0.95rem !important; color: var(--text-primary) !important; }
    p, li {
        color: var(--text-body) !important; line-height: 1.7 !important;
        font-size: 0.875rem !important;
    }

    /* ═══════════════════ SIDEBAR ═══════════════════ */
    [data-testid="stSidebar"] {
        background: #0B0F19 !important;
        border-right: 1px solid var(--border) !important;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2 {
        font-size: 0.68rem !important; text-transform: uppercase !important;
        letter-spacing: 0.12em !important; font-weight: 600 !important;
        color: var(--text-muted) !important;
        -webkit-text-fill-color: var(--text-muted) !important;
    }

    /* ── File uploader ──────────────────────────────────── */
    [data-testid="stFileUploader"] {
        background: rgba(99,102,241,0.04) !important;
        border: 1.5px dashed var(--border-accent) !important;
        border-radius: var(--radius-lg) !important;
        transition: all var(--transition) !important;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: var(--accent) !important;
        background: var(--accent-glow) !important;
        box-shadow: var(--shadow-glow) !important;
    }
    [data-testid="stFileUploader"] label {
        color: var(--text-body) !important; font-size: 0.8rem !important;
    }
    [data-testid="stFileUploader"] small { color: var(--text-muted) !important; }

    /* ── Buttons ─────────────────────────────────────────── */
    .stButton > button {
        background: var(--gradient-brand) !important; color: #fff !important;
        border: none !important; border-radius: var(--radius-md) !important;
        font-weight: 600 !important; font-size: 0.85rem !important;
        padding: 0.6rem 1.4rem !important;
        transition: all var(--transition) !important;
        box-shadow: 0 2px 12px rgba(99,102,241,0.3) !important;
    }
    .stButton > button p, .stButton > button span, .stButton > button div {
        color: #fff !important; -webkit-text-fill-color: #fff !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 24px rgba(99,102,241,0.4) !important;
    }
    .stButton > button:disabled {
        background: var(--bg-elevated) !important;
        color: var(--text-faint) !important; box-shadow: none !important;
    }
    .stButton > button:disabled p, .stButton > button:disabled span {
        color: var(--text-faint) !important; -webkit-text-fill-color: var(--text-faint) !important;
    }

    /* ── Text area ──────────────────────────────────────── */
    .stTextArea textarea {
        background: var(--bg-surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-md) !important;
        color: var(--text-primary) !important;
        font-size: 0.875rem !important;
        transition: all var(--transition) !important;
    }
    .stTextArea textarea:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px var(--accent-glow) !important;
    }
    .stTextArea textarea::placeholder { color: var(--text-faint) !important; }

    /* ── Expanders ──────────────────────────────────────── */
    details {
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-md) !important;
        background: var(--bg-surface) !important;
    }
    details summary {
        font-weight: 500 !important; color: var(--text-primary) !important;
        font-size: 0.82rem !important;
    }

    .stAlert, [data-testid="stAlert"] { border-radius: var(--radius-md) !important; border: none !important; }
    hr { border-color: var(--border-subtle) !important; margin: 1rem 0 !important; }
    .stCaption, [data-testid="stCaption"] { color: var(--text-muted) !important; font-size: 0.72rem !important; }

    /* ── Scrollbar ──────────────────────────────────────── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--text-faint); }

    /* ═══════════════════ SIDEBAR BRAND ═══════════════════ */
    .sidebar-brand {
        display: flex; align-items: center; gap: 0.65rem;
        padding: 0.3rem 0 1.1rem; margin-bottom: 0.6rem;
        border-bottom: 1px solid var(--border);
    }
    .brand-mark {
        width: 34px; height: 34px; border-radius: 8px;
        background: var(--gradient-brand);
        display: flex; align-items: center; justify-content: center;
        font-size: 1rem; color: #fff; font-weight: 800;
        box-shadow: 0 2px 10px rgba(99,102,241,0.3);
    }
    .brand-text { font-size: 0.95rem; font-weight: 700; color: var(--text-primary); letter-spacing: -0.02em; }
    .brand-tagline { font-size: 0.6rem; color: var(--text-muted); letter-spacing: 0.04em; margin-top: 1px; }

    /* ── Sidebar sections ──────────────────────────────── */
    .sb-section-title {
        font-size: 0.62rem; text-transform: uppercase; letter-spacing: 0.12em;
        color: var(--text-faint); font-weight: 700; margin: 1rem 0 0.5rem;
        display: flex; align-items: center; gap: 0.4rem;
    }
    .sb-section-title svg { width: 12px; height: 12px; opacity: 0.6; }

    /* ── Config chips ──────────────────────────────────── */
    .config-grid {
        display: grid; grid-template-columns: 1fr 1fr; gap: 0.4rem;
    }
    .config-chip {
        background: var(--bg-surface); border: 1px solid var(--border-subtle);
        border-radius: var(--radius-sm); padding: 0.4rem 0.55rem;
        transition: all var(--transition);
    }
    .config-chip:hover { border-color: var(--border-accent); background: var(--accent-glow); }
    .config-chip .chip-label {
        font-size: 0.58rem; text-transform: uppercase; letter-spacing: 0.08em;
        color: var(--text-faint); font-weight: 600; margin-bottom: 2px;
    }
    .config-chip .chip-value {
        font-size: 0.72rem; color: var(--accent-light); font-weight: 600;
        font-family: 'SF Mono', 'Consolas', monospace;
    }

    /* ── Upload status ─────────────────────────────────── */
    .upload-status {
        display: flex; align-items: center; gap: 0.45rem;
        padding: 0.55rem 0.7rem; border-radius: var(--radius-md);
        font-size: 0.78rem; font-weight: 500; margin: 0.5rem 0;
    }
    .upload-status.success { background: var(--teal-glow); color: var(--teal); border: 1px solid rgba(16,185,129,0.2); }
    .upload-status.error  { background: var(--red-glow); color: var(--red); border: 1px solid rgba(248,113,113,0.2); }
    .upload-status.warning { background: var(--amber-glow); color: var(--amber); border: 1px solid rgba(245,158,11,0.2); }

    /* ═══════════════════ HERO SECTION ═══════════════════ */
    .hero-bg {
        position: fixed; top: 0; left: 0; right: 0; bottom: 0;
        pointer-events: none; z-index: 0; overflow: hidden;
        background:
            radial-gradient(ellipse 800px 600px at 70% 10%, rgba(99,102,241,0.08), transparent),
            radial-gradient(ellipse 600px 500px at 20% 80%, rgba(167,139,250,0.06), transparent),
            radial-gradient(ellipse 500px 400px at 50% 50%, rgba(16,185,129,0.04), transparent);
    }
    .hero-bg::before {
        content: ''; position: absolute; inset: 0;
        background: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.03'/%3E%3C/svg%3E");
        background-size: 200px; opacity: 0.4;
    }

    .hero-wrapper { position: relative; z-index: 1; }

    /* Hero top bar */
    .hero-topbar {
        display: flex; justify-content: space-between; align-items: center;
        padding: 0 0 1.25rem;
    }
    .hero-logo {
        display: flex; align-items: center; gap: 0.6rem;
    }
    .hero-logo-mark {
        width: 36px; height: 36px; border-radius: 9px;
        background: var(--gradient-brand);
        display: flex; align-items: center; justify-content: center;
        font-weight: 800; color: #fff; font-size: 0.95rem;
        box-shadow: 0 2px 12px rgba(99,102,241,0.35);
    }
    .hero-logo-text { font-size: 1.1rem; font-weight: 700; color: var(--text-primary); letter-spacing: -0.02em; }
    .hero-tag {
        font-size: 0.62rem; text-transform: uppercase; letter-spacing: 0.1em;
        color: var(--accent-light); background: var(--accent-glow);
        padding: 0.3rem 0.75rem; border-radius: var(--radius-pill);
        border: 1px solid var(--border-accent); font-weight: 600;
    }

    /* Hero center */
    .hero-center { text-align: center; padding: 1.5rem 0 1rem; }
    .hero-eyebrow {
        font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.15em;
        color: var(--accent-light); font-weight: 600; margin-bottom: 0.75rem;
    }
    .hero-title {
        font-family: 'Instrument Serif', Georgia, serif !important;
        font-size: 3rem !important; font-weight: 400 !important; font-style: italic;
        color: var(--text-primary) !important; -webkit-text-fill-color: var(--text-primary) !important;
        background: none !important; margin-bottom: 0.85rem !important;
        line-height: 1.1 !important; letter-spacing: -0.02em !important;
    }
    .hero-sub {
        color: var(--text-muted); font-size: 1rem; max-width: 520px;
        margin: 0 auto; line-height: 1.7;
    }
    .hero-trust {
        display: flex; justify-content: center; gap: 1.5rem;
        margin-top: 1.25rem; flex-wrap: wrap;
    }
    .hero-trust span {
        font-size: 0.7rem; color: var(--text-faint); font-weight: 500;
        display: flex; align-items: center; gap: 0.3rem;
    }
    .hero-trust .dot { width: 4px; height: 4px; border-radius: 50%; background: var(--accent); }

    /* ── Stat cards ──────────────────────────────────────── */
    .stat-grid {
        display: grid; grid-template-columns: repeat(4,1fr); gap: 0.85rem;
        padding: 2rem 0 2.5rem; max-width: 700px; margin: 0 auto;
    }
    .stat-card {
        background: var(--bg-surface); border: 1px solid var(--border);
        border-radius: var(--radius-lg); padding: 1rem 0.85rem;
        text-align: center; transition: all var(--transition);
    }
    .stat-card:hover {
        border-color: var(--border-accent); transform: translateY(-2px);
        box-shadow: var(--shadow-glow);
    }
    .stat-icon { font-size: 1.3rem; margin-bottom: 0.4rem; }
    .stat-val {
        font-size: 1.35rem; font-weight: 800; letter-spacing: -0.03em;
        background: var(--gradient-brand); -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stat-desc {
        font-size: 0.65rem; color: var(--text-muted); margin-top: 0.2rem;
        line-height: 1.4; text-transform: uppercase; letter-spacing: 0.06em;
    }

    /* ── How it works cards ──────────────────────────────── */
    .steps-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 0.85rem; margin-bottom: 1.5rem; }
    .step-card {
        background: var(--bg-surface); border: 1px solid var(--border);
        border-left: 3px solid var(--accent);
        border-radius: var(--radius-lg); padding: 1.35rem;
        transition: all var(--transition); position: relative; overflow: hidden;
    }
    .step-card:hover {
        border-color: var(--border-accent);
        background: var(--bg-elevated); transform: translateY(-2px);
        box-shadow: var(--shadow-md);
    }
    .step-badge {
        display: inline-flex; align-items: center; justify-content: center;
        width: 28px; height: 28px; border-radius: 7px;
        background: var(--accent-glow); border: 1px solid var(--border-accent);
        color: var(--accent-light); font-weight: 700; font-size: 0.75rem;
        margin-bottom: 0.75rem;
    }
    .step-icon {
        width: 32px; height: 32px; border-radius: 8px; margin-bottom: 0.75rem;
        display: flex; align-items: center; justify-content: center;
        font-size: 1rem;
    }
    .step-icon.si-1 { background: var(--accent-glow); }
    .step-icon.si-2 { background: var(--teal-glow); }
    .step-icon.si-3 { background: var(--amber-glow); }
    .step-title { color: var(--text-primary); font-weight: 600; font-size: 0.88rem; margin-bottom: 0.3rem; }
    .step-desc { color: var(--text-muted); font-size: 0.78rem; line-height: 1.55; }

    /* ── Agent pipeline ─────────────────────────────────── */
    .pipeline-section {
        background: var(--bg-surface); border: 1px solid var(--border);
        border-radius: var(--radius-xl); padding: 1.5rem;
        margin-bottom: 1.5rem; position: relative; overflow: hidden;
    }
    .pipeline-section::before {
        content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
        background: var(--gradient-brand);
    }
    .pipeline-header {
        font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.12em;
        color: var(--text-faint); font-weight: 700; margin-bottom: 1.25rem;
        text-align: center;
    }
    .pipeline-row {
        display: flex; justify-content: center; align-items: stretch;
        gap: 0.5rem; flex-wrap: wrap;
    }
    .agent-card {
        flex: 1; max-width: 200px; padding: 0.85rem; border-radius: var(--radius-md);
        text-align: center; transition: all var(--transition);
        position: relative; overflow: hidden;
    }
    .agent-card::before {
        content: ''; position: absolute; inset: 0; border-radius: var(--radius-md);
        padding: 1px; background: var(--gradient-brand); mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
        mask-composite: exclude; -webkit-mask-composite: xor; opacity: 0;
        transition: opacity var(--transition);
    }
    .agent-card:hover::before { opacity: 1; }
    .agent-card.ag-retriever { background: rgba(16,185,129,0.06); border: 1px solid rgba(16,185,129,0.12); }
    .agent-card.ag-analyzer  { background: rgba(245,158,11,0.06); border: 1px solid rgba(245,158,11,0.12); }
    .agent-card.ag-reporter  { background: rgba(167,139,250,0.06); border: 1px solid rgba(167,139,250,0.12); }
    .agent-card:hover { transform: translateY(-2px); box-shadow: var(--shadow-sm); }
    .agent-icon { font-size: 1.2rem; margin-bottom: 0.35rem; }
    .agent-name { font-size: 0.78rem; font-weight: 700; color: var(--text-primary); margin-bottom: 0.2rem; }
    .agent-role { font-size: 0.67rem; color: var(--text-muted); line-height: 1.45; }
    .agent-name.an-retriever { color: var(--teal); }
    .agent-name.an-analyzer  { color: var(--amber); }
    .agent-name.an-reporter  { color: var(--purple); }
    .pipeline-arrow {
        display: flex; align-items: center; color: var(--text-faint);
        font-size: 1.1rem; padding: 0 0.15rem;
    }

    /* ── File format chips ──────────────────────────────── */
    .format-bar {
        display: flex; justify-content: center; gap: 0.5rem;
        padding: 0.5rem 0 0; flex-wrap: wrap;
    }
    .fmt-chip {
        display: inline-flex; align-items: center; gap: 0.3rem;
        padding: 0.3rem 0.7rem; border-radius: var(--radius-sm);
        font-size: 0.68rem; font-weight: 600; letter-spacing: 0.05em;
        transition: all var(--transition);
    }
    .fmt-chip:hover { transform: translateY(-1px); }
    .fmt-chip.fc-pdf  { background: rgba(248,113,113,0.08); color: #F87171; border: 1px solid rgba(248,113,113,0.15); }
    .fmt-chip.fc-csv  { background: var(--teal-glow); color: var(--teal); border: 1px solid rgba(16,185,129,0.15); }
    .fmt-chip.fc-xls  { background: var(--accent-glow); color: var(--accent-light); border: 1px solid var(--border-accent); }
    .fmt-chip .fmt-icon { font-size: 0.8rem; }

    /* ═══════════════════ Q&A INTERFACE ═══════════════════ */
    .qa-topbar {
        display: flex; justify-content: space-between; align-items: center;
        padding: 0 0 0.75rem; margin-bottom: 0.75rem;
        border-bottom: 1px solid var(--border-subtle);
    }
    .qa-logo {
        display: flex; align-items: center; gap: 0.55rem;
    }
    .qa-logo-mark {
        width: 30px; height: 30px; border-radius: 7px;
        background: var(--gradient-brand);
        display: flex; align-items: center; justify-content: center;
        font-weight: 800; color: #fff; font-size: 0.8rem;
    }
    .qa-logo-text { font-size: 0.95rem; font-weight: 700; color: var(--text-primary); }
    .qa-badge {
        font-size: 0.6rem; text-transform: uppercase; letter-spacing: 0.08em;
        color: var(--teal); background: var(--teal-glow);
        padding: 0.25rem 0.6rem; border-radius: var(--radius-pill);
        border: 1px solid rgba(16,185,129,0.2); font-weight: 600;
    }

    /* ── Chat bubbles ──────────────────────────────────── */
    .q-bubble {
        background: var(--accent-glow); border: 1px solid var(--border-accent);
        border-radius: var(--radius-lg); padding: 0.8rem 1.15rem;
        margin-bottom: 0.65rem; display: flex; align-items: flex-start; gap: 0.55rem;
    }
    .q-bubble .q-icon {
        width: 24px; height: 24px; border-radius: 6px;
        background: var(--gradient-brand); display: flex; align-items: center;
        justify-content: center; font-size: 0.7rem; color: #fff;
        flex-shrink: 0; margin-top: 1px;
    }
    .q-bubble .q-text {
        color: var(--accent-light) !important; font-size: 0.88rem !important;
        font-weight: 500; line-height: 1.55 !important;
    }

    .a-card {
        background: var(--bg-surface); border: 1px solid var(--border);
        border-radius: var(--radius-lg); padding: 1.4rem 1.6rem; margin-bottom: 0.5rem;
        border-left: 3px solid var(--accent);
    }
    .a-card p, .a-card li {
        color: var(--text-body) !important; font-size: 0.88rem !important;
        line-height: 1.8 !important;
    }
    .a-card strong, .a-card b { color: var(--text-primary) !important; }
    .a-card h1, .a-card h2, .a-card h3 {
        font-size: 0.95rem !important; margin-top: 0.8rem !important;
        color: var(--text-primary) !important;
    }
    .a-card code {
        background: var(--bg-elevated); padding: 0.1rem 0.35rem;
        border-radius: 4px; font-size: 0.8rem; color: var(--accent-light);
    }

    /* ── Agent trace ──────────────────────────────────── */
    .trace-bar {
        display: flex; align-items: center; gap: 0.3rem;
        margin-bottom: 0.5rem; flex-wrap: wrap;
    }
    .trace-label {
        font-size: 0.62rem; text-transform: uppercase; letter-spacing: 0.08em;
        color: var(--text-faint); font-weight: 700; margin-right: 0.2rem;
    }
    .trace-pill {
        display: inline-flex; align-items: center; gap: 0.25rem;
        padding: 0.2rem 0.55rem; border-radius: var(--radius-pill);
        font-size: 0.68rem; font-weight: 600; text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .trace-pill.tp-retriever { background: var(--teal-glow); color: var(--teal); border: 1px solid rgba(16,185,129,0.18); }
    .trace-pill.tp-analyzer  { background: var(--amber-glow); color: var(--amber); border: 1px solid rgba(245,158,11,0.18); }
    .trace-pill.tp-reporter  { background: var(--purple-glow); color: var(--purple); border: 1px solid rgba(167,139,250,0.18); }
    .trace-arrow { color: var(--text-faint); font-size: 0.65rem; }

    /* ── Source cards ──────────────────────────────────── */
    .src-card {
        background: var(--bg-elevated); border: 1px solid var(--border-subtle);
        border-radius: var(--radius-md); padding: 0.6rem 0.8rem; margin-bottom: 0.35rem;
        transition: all var(--transition);
    }
    .src-card:hover { border-color: var(--border-accent); }
    .src-card .src-header {
        font-size: 0.76rem; font-weight: 600; color: var(--text-primary);
        margin-bottom: 0.2rem; display: flex; align-items: center; gap: 0.35rem;
    }
    .src-card .src-excerpt {
        font-size: 0.72rem; color: var(--text-muted); line-height: 1.5;
        font-style: italic; opacity: 0.85;
    }

    /* ═══════════════════ FOOTER ═══════════════════ */
    .app-footer {
        text-align: center; padding: 2rem 0 1rem;
        margin-top: 3rem; border-top: 1px solid var(--border);
    }
    .footer-mission {
        font-size: 0.82rem; color: var(--text-muted); max-width: 480px;
        margin: 0 auto 1.25rem; line-height: 1.65;
    }
    .footer-credit {
        font-size: 0.72rem; color: var(--text-faint); margin-bottom: 0.15rem;
    }
    .footer-name {
        background: var(--gradient-brand); -webkit-background-clip: text;
        -webkit-text-fill-color: transparent; font-weight: 700;
    }
    .footer-legal {
        font-size: 0.65rem; color: var(--text-faint);
        letter-spacing: 0.02em;
    }
    .footer-links {
        display: flex; justify-content: center; gap: 1rem;
        margin-top: 0.85rem;
    }
    .footer-links a {
        font-size: 0.7rem; color: var(--text-muted); text-decoration: none;
        transition: color var(--transition); font-weight: 500;
    }
    .footer-links a:hover { color: var(--accent-light); }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Session state defaults ──────────────────────────────────────────
def _init_session():
    defaults = {
        "store_built": False,
        "uploaded_names": [],
        "chat_history": [],
        "query_counter": 0,
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
            <div class="brand-mark">◆</div>
            <div>
                <div class="brand-text">FinRisk</div>
                <div class="brand-tagline">Instant answers from your financial documents</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sb-section-title">'
        '<svg viewBox="0 0 16 16" fill="currentColor"><path d="M3.5 3.75a.25.25 0 01.25-.25h8.5a.25.25 0 01.25.25v8.5a.25.25 0 01-.25.25h-8.5a.25.25 0 01-.25-.25v-8.5z"/></svg>'
        'Documents</div>',
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        "Upload financial documents",
        type=["pdf", "csv", "tsv", "xls", "xlsx"],
        accept_multiple_files=True,
        help="Annual reports, balance sheets, income statements, SEC filings",
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
                            all_docs.append({
                                "text": result["summary"],
                                "source": result["source"],
                                "page": 0,
                                "tables": [],
                            })
                    except Exception as exc:
                        st.error(f"Failed to load **{uf.name}**: {exc}")
                        logger.exception("Ingestion error for %s", uf.name)

            if all_docs:
                with st.spinner("Chunking & embedding …"):
                    try:
                        chunker = DocumentChunker()
                        chunks = chunker.chunk_documents(all_docs)

                        if not chunks:
                            st.markdown(
                                '<div class="upload-status warning">'
                                "⚠ No extractable text found."
                                "</div>",
                                unsafe_allow_html=True,
                            )
                        else:
                            embedder = DocumentEmbedder()
                            store = embedder.embed_and_store(chunks, settings.VECTOR_DB_PATH)

                            if store is not None:
                                st.session_state["store_built"] = True
                                st.session_state["uploaded_names"] = current_names
                                st.session_state["chat_history"] = []
                                st.markdown(
                                    f'<div class="upload-status success">'
                                    f"✓ Indexed {len(chunks)} chunks from "
                                    f"{len(uploaded_files)} file(s)"
                                    f"</div>",
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

    st.markdown(
        '<div class="sb-section-title" style="margin-top: 1.25rem;">'
        '<svg viewBox="0 0 16 16" fill="currentColor"><path d="M8 12.5a4.5 4.5 0 100-9 4.5 4.5 0 000 9zM8 15A7 7 0 118 1a7 7 0 010 14z"/></svg>'
        'System Configuration</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="config-grid">
            <div class="config-chip">
                <div class="chip-label">Provider</div>
                <div class="chip-value">{settings.LLM_PROVIDER}</div>
            </div>
            <div class="config-chip">
                <div class="chip-label">Model</div>
                <div class="chip-value">{settings.LLM_MODEL.split('/')[-1][:18]}</div>
            </div>
            <div class="config-chip">
                <div class="chip-label">Embeddings</div>
                <div class="chip-value">{settings.EMBEDDING_MODEL[:16]}</div>
            </div>
            <div class="config-chip">
                <div class="chip-label">Chunk / Overlap</div>
                <div class="chip-value">{settings.CHUNK_SIZE} / {settings.CHUNK_OVERLAP}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════════════════
# FOOTER HELPER
# ═════════════════════════════════════════════════════════════════════

def _render_footer():
    st.markdown(
        """
        <div class="app-footer">
            <div class="footer-mission">
                FinRisk makes financial document analysis instant, accurate,
                and explainable — powered by multi-agent AI and RAG.
            </div>
            <div class="footer-credit">
                Developed by <span class="footer-name">Aryan Bhardwaj</span>
            </div>
            <div class="footer-legal">© 2026. All rights reserved.</div>
            <div class="footer-links">
                <a href="#">GitHub</a>
                <a href="#">Documentation</a>
                <a href="#">Contact</a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════════════════
# MAIN AREA
# ═════════════════════════════════════════════════════════════════════

if st.session_state["store_built"]:
    # ── Q&A Top bar ─────────────────────────────────────────────────
    st.markdown(
        """
        <div class="qa-topbar">
            <div class="qa-logo">
                <div class="qa-logo-mark">◆</div>
                <div class="qa-logo-text">FinRisk</div>
            </div>
            <div class="qa-badge">● Connected · Ready</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.title("Ask About Your Documents")

    # ── Display chat history ────────────────────────────────────────
    for entry in st.session_state["chat_history"]:
        if entry["role"] == "user":
            st.markdown(
                f'<div class="q-bubble">'
                f'<div class="q-icon">?</div>'
                f'<span class="q-text">{entry["content"]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            # Agent trace
            trace = entry.get("agent_trace", [])
            if trace:
                trace_html = ""
                for i, step in enumerate(trace):
                    cls = step if step in ("retriever", "analyzer", "reporter") else "analyzer"
                    icons = {"retriever": "⊕", "analyzer": "◈", "reporter": "▣"}
                    icon = icons.get(step, "⚙")
                    trace_html += f'<span class="trace-pill tp-{cls}">{icon} {step}</span>'
                    if i < len(trace) - 1:
                        trace_html += '<span class="trace-arrow">→</span>'
                st.markdown(
                    f'<div class="trace-bar">'
                    f'<span class="trace-label">Agents:</span>'
                    f'{trace_html}</div>',
                    unsafe_allow_html=True,
                )

            # Answer
            st.markdown(
                f'<div class="a-card">{entry["content"]}</div>',
                unsafe_allow_html=True,
            )

            # Sources
            sources = entry.get("sources", [])
            if sources:
                with st.expander(f"📑 Sources — {len(sources)} document section(s)"):
                    for i, src in enumerate(sources, 1):
                        st.markdown(
                            f"""
                            <div class="src-card">
                                <div class="src-header">▸ {src['source']} · Page {src['page']}</div>
                                <div class="src-excerpt">"{src['excerpt']}"</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

    # ── Question input ──────────────────────────────────────────────
    st.markdown("---")
    query = st.text_area(
        "Ask a question about your financial documents:",
        height=80,
        placeholder=(
            "e.g., What is the company's total revenue? "
            "Is the debt ratio concerning? "
            "Summarize the key financial highlights."
        ),
        key=f"query_input_{st.session_state['query_counter']}",
    )

    col1, col2, _ = st.columns([1, 1, 3])
    with col1:
        ask_btn = st.button("Ask →", type="primary", use_container_width=True)
    with col2:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state["chat_history"] = []
            st.rerun()

    # ── Process question ────────────────────────────────────────────
    if ask_btn and query.strip():
        st.session_state["chat_history"].append({
            "role": "user",
            "content": query.strip(),
            "sources": [],
            "agent_trace": [],
        })

        with st.spinner("⊕ Retrieving → ◈ Analyzing → ▣ Reporting …"):
            try:
                result = ask_question(query.strip())
                st.session_state["chat_history"].append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                    "agent_trace": result.get("agent_trace", []),
                })
            except Exception as exc:
                st.session_state["chat_history"].append({
                    "role": "assistant",
                    "content": f"⚠ Error generating answer: {exc}",
                    "sources": [],
                    "agent_trace": [],
                })
                logger.exception("Q&A error")

        st.session_state["query_counter"] += 1
        st.rerun()

    elif ask_btn:
        st.warning("Please enter a question.")

    _render_footer()

else:
    # ═════════════════════════════════════════════════════════════════
    # LANDING PAGE
    # ═════════════════════════════════════════════════════════════════
    st.markdown(
        '<div class="hero-bg"></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="hero-wrapper">', unsafe_allow_html=True)

    # ── Top bar ─────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="hero-topbar">
            <div class="hero-logo">
                <div class="hero-logo-mark">◆</div>
                <span class="hero-logo-text">FinRisk</span>
            </div>
            <div class="hero-tag">LangGraph · LangChain · RAG</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Hero center ─────────────────────────────────────────────────
    st.markdown(
        """
        <div class="hero-center">
            <div class="hero-eyebrow">Multi-Agent Financial Intelligence</div>
            <div class="hero-title">Turn dense financial reports<br>into instant insight.</div>
            <div class="hero-sub">
                Upload annual reports, balance sheets, or SEC filings.
                Our LangGraph agent pipeline retrieves, analyzes, and formats
                accurate answers with source citations.
            </div>
            <div class="hero-trust">
                <span><span class="dot"></span> Annual Reports</span>
                <span><span class="dot"></span> Balance Sheets</span>
                <span><span class="dot"></span> Income Statements</span>
                <span><span class="dot"></span> SEC Filings</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Stat cards ──────────────────────────────────────────────────
    st.markdown(
        """
        <div class="stat-grid">
            <div class="stat-card">
                <div class="stat-icon">⊕</div>
                <div class="stat-val">3</div>
                <div class="stat-desc">AI Agents</div>
            </div>
            <div class="stat-card">
                <div class="stat-icon">◈</div>
                <div class="stat-val">5+</div>
                <div class="stat-desc">File Formats</div>
            </div>
            <div class="stat-card">
                <div class="stat-icon">⬡</div>
                <div class="stat-val">RAG</div>
                <div class="stat-desc">Retrieval</div>
            </div>
            <div class="stat-card">
                <div class="stat-icon">▣</div>
                <div class="stat-val">✓</div>
                <div class="stat-desc">Source Cited</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── How it works ────────────────────────────────────────────────
    st.markdown(
        """
        <div class="steps-grid">
            <div class="step-card">
                <div class="step-icon si-1">↑</div>
                <div class="step-title">Upload Documents</div>
                <div class="step-desc">
                    Drag & drop PDFs, CSVs, or Excel files. Financial reports are
                    chunked, embedded, and indexed locally.
                </div>
            </div>
            <div class="step-card">
                <div class="step-icon si-2">◎</div>
                <div class="step-title">Ask Any Question</div>
                <div class="step-desc">
                    Revenue trends, debt ratios, risk factors — ask in natural
                    language and let the agents find answers.
                </div>
            </div>
            <div class="step-card">
                <div class="step-icon si-3">✦</div>
                <div class="step-title">Get Cited Answers</div>
                <div class="step-desc">
                    Receive structured, professional answers grounded in your
                    documents with exact source citations.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Agent pipeline ──────────────────────────────────────────────
    st.markdown(
        """
        <div class="pipeline-section">
            <div class="pipeline-header">LangGraph Agent Pipeline</div>
            <div class="pipeline-row">
                <div class="agent-card ag-retriever">
                    <div class="agent-icon">⊕</div>
                    <div class="agent-name an-retriever">Retriever</div>
                    <div class="agent-role">Scans and retrieves relevant document sections using FAISS vector search</div>
                </div>
                <div class="pipeline-arrow">→</div>
                <div class="agent-card ag-analyzer">
                    <div class="agent-icon">◈</div>
                    <div class="agent-name an-analyzer">Analyzer</div>
                    <div class="agent-role">Identifies key metrics, patterns, and answers using LLM reasoning</div>
                </div>
                <div class="pipeline-arrow">→</div>
                <div class="agent-card ag-reporter">
                    <div class="agent-icon">▣</div>
                    <div class="agent-name an-reporter">Reporter</div>
                    <div class="agent-role">Formats a professional, cited answer with clear structure</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── File formats ────────────────────────────────────────────────
    st.markdown(
        """
        <div class="format-bar">
            <div class="fmt-chip fc-pdf"><span class="fmt-icon">▤</span> PDF</div>
            <div class="fmt-chip fc-csv"><span class="fmt-icon">▦</span> CSV</div>
            <div class="fmt-chip fc-csv"><span class="fmt-icon">▦</span> TSV</div>
            <div class="fmt-chip fc-xls"><span class="fmt-icon">▧</span> XLS</div>
            <div class="fmt-chip fc-xls"><span class="fmt-icon">▧</span> XLSX</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('</div>', unsafe_allow_html=True)

    _render_footer()
