# 📊 Multi-Agent Financial Risk Analysis System

A production-grade, multi-agent financial risk analysis pipeline built with **LangChain**, **LangGraph**, and **FAISS**. The system ingests financial documents (PDFs and CSVs), builds a retrieval-augmented generation (RAG) pipeline, and orchestrates six specialised AI agents to produce a comprehensive credit-risk report — complete with computed financial ratios, anomaly detection, source verification, and actionable recommendations.

> Built with Python 3.11+ · No external APIs required for embeddings (runs sentence-transformers locally) · OpenAI GPT-4o for agent reasoning

---

## 🏗️ Architecture

```
                           ┌─────────────────────────────────────────────────────────┐
                           │                   LangGraph Workflow                    │
                           │                                                         │
  ┌──────────┐             │   ┌──────────┐    ┌───────────┐    ┌──────────┐        │
  │  Upload   │────────────┼──▶│ Planner  │───▶│ Retriever │───▶│ Analyst  │        │
  │ PDF / CSV │            │   └──────────┘    └─────┬─────┘    └────┬─────┘        │
  └──────────┘             │        🗺️           🔍  │  ▲            │  📊           │
                           │                         │  │            ▼               │
  ┌──────────┐             │                    (re-retrieve    ┌──────────┐         │
  │   FAISS  │◀────────────┤                     if missing)   │Evaluator │         │
  │  Vector  │─────────────┤                         │  │       └────┬─────┘         │
  │  Store   │             │                         ▼  │            │  ⚖️            │
  └──────────┘             │                    ┌────────┴──┐        ▼               │
                           │                    │ Verifier  │   ┌──────────┐         │
                           │                    └─────┬─────┘   │ Reporter │         │
                           │                      🔎  │         └────┬─────┘         │
                           │                          │              │  📝           │
                           │                          │  (loop if    ▼               │
                           │                          │  unverified) ┌────────┐      │
                           │                          └─────────────▶│  END   │      │
                           │                                         └────────┘      │
                           └─────────────────────────────────────────────────────────┘
```

### Agent Responsibilities

| # | Agent | Role | Uses LLM? |
|---|-------|------|-----------|
| 1 | **Planner** | Decomposes the user query into steps, focus areas, and ratio requirements | ✅ GPT-4o |
| 2 | **Retriever** | Fetches relevant chunks from FAISS using hybrid (vector + keyword) search | ❌ |
| 3 | **Analyst** | Extracts numeric data via regex, computes financial ratios, detects anomalies | ❌ |
| 4 | **Evaluator** | Classifies overall risk level with deterministic escalation rules | ✅ GPT-4o |
| 5 | **Verifier** | Cross-checks extracted numbers against source documents (no hallucination) | ❌ |
| 6 | **Reporter** | Compiles the executive risk report with summary and recommendations | ✅ GPT-4o |

---

## 🛠️ Tech Stack

| Tool | Purpose | Version |
|------|---------|---------|
| LangChain | LLM orchestration & prompt engineering | 0.3.14 |
| LangGraph | Multi-agent state graph with conditional edges | 0.2.60 |
| LangChain-OpenAI | GPT-4o integration | 0.3.0 |
| LangChain-Community | HuggingFace embeddings & FAISS wrappers | 0.3.14 |
| FAISS (CPU) | Vector similarity search | 1.9.0 |
| sentence-transformers | Local embedding model (all-MiniLM-L6-v2) | 3.3.1 |
| pdfplumber | PDF text & table extraction (primary) | 0.11.4 |
| PyMuPDF | PDF extraction fallback | 1.25.1 |
| pandas | Tabular data handling | 2.2.3 |
| NumPy | Numerical computation | 1.26.4 |
| Streamlit | Interactive web UI | 1.41.1 |
| tiktoken | Token counting | 0.8.0 |
| OpenAI SDK | API client | 1.58.1 |
| python-dotenv | Environment variable management | 1.0.1 |
| pytest | Test framework | 8.3.4 |
| reportlab | Synthetic PDF generation (tests) | 4.2.5 |
| MLflow *(optional)* | Experiment tracking & artifact logging | 2.19.0 |

---

## 🚀 Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/your-org/financial-risk-analysis.git
cd financial-risk-analysis/project
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt

# If you want MLflow tracking (optional):
pip install mlflow==2.19.0
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in your values:

```env
OPENAI_API_KEY=sk-your-key-here
EMBEDDING_MODEL=all-MiniLM-L6-v2
LLM_MODEL=gpt-4o
CHUNK_SIZE=512
CHUNK_OVERLAP=64
VECTOR_DB_PATH=./vector_store
LOG_LEVEL=INFO

# Optional: MLflow tracking
# MLFLOW_TRACKING_URI=http://localhost:5000
```

---

## ▶️ How to Run

```bash
streamlit run app.py
```

Then:
1. **Upload** financial documents (PDFs or CSVs) in the sidebar
2. **Enter** a risk analysis question (e.g. *"Assess ACME Corp's credit risk based on their 2024 annual report"*)
3. **Click** "Run Analysis" and wait for the agents to complete
4. **Explore** results across the 4 tabs: Risk Report · Financial Metrics · Sources · Agent Trace
5. **Export** the report as JSON via the sidebar download button

---

## 🧪 How to Run Tests

```bash
pytest tests/ -v
```

Tests use a **synthetic PDF** generated by reportlab with engineered numbers (D/E ≈ 2.8, CR ≈ 0.9, declining net income) and **mocked LLM calls** — no OpenAI API key required.

```
tests/
├── __init__.py
├── conftest.py          # Auto-patches ChatOpenAI with deterministic responses
└── test_pipeline.py     # 24 tests across 4 classes
    ├── TestPDFIngestion      (5 tests)
    ├── TestRAGPipeline       (5 tests)
    ├── TestFinancialRatios   (16 tests — incl. edge cases)
    └── TestFullPipeline      (2 tests — full end-to-end)
```

---

## 📋 Sample Output

The `run_analysis()` function returns a `final_report` dict:

```json
{
  "summary": "ACME Corporation exhibits elevated credit risk driven by high leverage (D/E 2.8), weak liquidity (CR 0.9), and declining profitability over three consecutive quarters.",
  "risk_level": "HIGH",
  "risk_badge": "🔴 HIGH",
  "key_metrics": [
    { "name": "debt_to_equity", "value": 2.8, "status": "critical", "threshold": "> 2.0" },
    { "name": "current_ratio", "value": 0.9, "status": "critical", "threshold": "< 1.0" },
    { "name": "profit_margin", "value": 0.056, "status": "critical", "threshold": "< 5%" },
    { "name": "quick_ratio", "value": 0.7, "status": "warning", "threshold": "0.7 – 1.0" }
  ],
  "anomalies": [
    { "metric": "debt_to_equity", "description": "D/E is 2.8 (critical threshold: > 2.0)", "severity": "critical" },
    { "metric": "net_income", "description": "net_income declined for 2 consecutive periods: [120, 85, 42]", "severity": "warning" }
  ],
  "justification": "• Debt-to-equity of 2.8 far exceeds the 2.0 critical threshold\n• Current ratio of 0.9 indicates inability to cover short-term obligations\n• Net income declined 65% from Q1 to Q3",
  "recommendations": [
    "Restructure long-term debt to reduce leverage below 2.0x",
    "Improve working capital by accelerating receivables collection",
    "Review cost structure to reverse the declining profit margin trend",
    "Establish a cash reserve buffer of at least 3 months of operating expenses"
  ],
  "sources": [
    { "page": 1, "source": "acme_corp_annual_2024.pdf", "excerpt": "Total Debt: 1,400 | Total Equity: 500 …" }
  ],
  "verification_status": true,
  "verification_notes": [],
  "agent_trace": ["planner", "retriever", "analyst", "evaluator", "verifier", "reporter"]
}
```

---

## 📁 Project Structure

```
project/
├── data/                          # Uploaded financial documents
├── ingestion/
│   ├── __init__.py
│   ├── pdf_loader.py              # pdfplumber + PyMuPDF fallback
│   └── csv_loader.py              # pandas loader with auto-type detection
├── rag/
│   ├── __init__.py
│   ├── chunker.py                 # Table-preserving recursive chunker
│   ├── embedder.py                # HuggingFace embeddings → FAISS
│   └── retriever.py               # MMR, scored, and hybrid retrieval
├── agents/
│   ├── __init__.py
│   ├── planner.py                 # Query decomposition (LLM)
│   ├── retriever_agent.py         # Hybrid document retrieval
│   ├── analyst.py                 # Ratio computation & anomaly detection
│   ├── evaluator.py               # Risk classification with escalation rules
│   ├── verifier.py                # Source cross-checking (no LLM)
│   └── reporter.py                # Executive report generation (LLM)
├── graph/
│   ├── __init__.py
│   ├── state.py                   # AgentState TypedDict with reducers
│   └── workflow.py                # LangGraph DAG + conditional loops
├── utils/
│   ├── __init__.py
│   ├── financial_ratios.py        # FinancialRatioEngine (5 ratios + interpret)
│   ├── logger.py                  # Centralised Python logger
│   └── mlflow_tracker.py          # Optional MLflow experiment tracking
├── tests/
│   ├── __init__.py
│   ├── conftest.py                # Mock fixtures (no API key needed)
│   └── test_pipeline.py           # 24 tests across 4 test classes
├── app.py                         # Streamlit UI (4-tab layout)
├── config.py                      # Settings from .env (frozen dataclass)
├── requirements.txt               # Pinned dependencies
├── .env.example                   # Environment variable template
└── README.md                      # This file
```

---

## 🔍 How RAG Works in This Project

### Document Ingestion
- **PDFs**: `pdfplumber` extracts text and tables (with PyMuPDF as fallback for scanned documents). Tables are converted to Markdown strings.
- **CSVs**: `pandas` loads the data, auto-detects column types (numeric, categorical, datetime), and generates a natural-language statistical summary.

### Chunking Strategy
The `DocumentChunker` uses LangChain's `RecursiveCharacterTextSplitter` with **financial-document-aware** customisations:

1. **Table preservation**: Markdown table blocks (detected by `|` column patterns) are **never split** — they become standalone chunks regardless of size.
2. **Sentence-safe splitting**: Custom separators (`\n\n` → `\n` → `. ` → `; ` → `, ` → ` `) ensure splits happen at sentence boundaries. No bare `.` separator is used, protecting decimal numbers like `1,234.56`.
3. **Metadata propagation**: Every chunk carries `source`, `page`, `chunk_index`, `char_count`, and `is_table` metadata for traceability.

### Embedding & Storage
- **Model**: `all-MiniLM-L6-v2` (384-dim, runs locally — no API cost)
- **Store**: FAISS `IndexFlatL2` persisted to disk via LangChain's FAISS wrapper
- **Normalisation**: Embeddings are L2-normalised for consistent similarity scores

### Retrieval Strategies
| Strategy | Method | Use Case |
|----------|--------|----------|
| **MMR** | `retrieve(query, k)` | Diverse results — reduces redundant chunks |
| **Scored** | `retrieve_with_scores(query, k)` | When you need confidence scores |
| **Hybrid** | `hybrid_retrieve(query, keywords, k)` | Domain-term precision — vector fetch + keyword re-ranking |

---

## 🔧 How to Extend

### Add a New Agent

1. Create `agents/my_agent.py`:

    ```python
    from graph.state import AgentState
    from utils.logger import get_logger

    logger = get_logger(__name__)

    def my_agent(state: AgentState) -> dict:
        logger.info("▶ my_agent ENTER")
        # ... your logic ...
        logger.info("◀ my_agent EXIT")
        return {
            "some_field": result,
            "agent_trace": ["my_agent"],
        }
    ```

2. Add the field to `AgentState` in `graph/state.py` if needed.

3. Register the node in `graph/workflow.py`:

    ```python
    workflow.add_node("my_agent", my_agent)
    workflow.add_edge("evaluator", "my_agent")   # wire it in
    workflow.add_edge("my_agent", "verifier")
    ```

### Swap the Embedding Model

1. Update `.env`:

    ```env
    EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
    ```

2. **Delete the existing FAISS index** (the dimensions will differ):

    ```bash
    rm -rf vector_store/
    ```

3. Re-upload documents and rebuild the index via the Streamlit UI.

> Any model on [HuggingFace sentence-transformers](https://huggingface.co/models?library=sentence-transformers) is supported.

### Swap the LLM

Update `.env`:

```env
LLM_MODEL=gpt-4o-mini    # cheaper & faster
```

No code changes required — all agents read from `settings.LLM_MODEL`.

### Enable MLflow Tracking

1. Install MLflow: `pip install mlflow==2.19.0`
2. Start the tracking server: `mlflow ui --port 5000`
3. Add to `.env`:

    ```env
    MLFLOW_TRACKING_URI=http://localhost:5000
    ```

4. Every `run_analysis()` call will now log parameters, metrics, and the full report as an artifact.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
