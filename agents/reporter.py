"""
Reporter agent — compiles the full state into a structured executive
risk report.

Reads every field of the ``AgentState`` and produces a single
``final_report`` dict with a fixed schema that the Streamlit UI (or
any downstream consumer) can render directly.

Node signature
--------------
    reporter_agent(state: AgentState) -> dict   # partial state update
"""

from __future__ import annotations

from typing import Any, Dict, List

from langchain.schema import Document
from langchain_core.messages import HumanMessage, SystemMessage

from config import settings
from utils.llm_factory import get_llm
from graph.state import AgentState
from utils.logger import get_logger

logger = get_logger(__name__)

_RISK_BADGES: Dict[str, str] = {
    "LOW": "🟢 LOW",
    "MEDIUM": "🟡 MEDIUM",
    "HIGH": "🔴 HIGH",
    "CRITICAL": "⛔ CRITICAL",
}

_SYSTEM_PROMPT = """\
You are a senior financial report writer.  Given a risk level,
key metrics, anomalies, and justification, produce:
1. A 3-5 sentence executive summary.
2. A list of 3-5 actionable recommendations prioritised by urgency.

Return **only** valid JSON:
{
  "summary": "…",
  "recommendations": ["…", "…"]
}
"""


def _build_key_metrics(ratios: dict) -> List[dict]:
    """Convert the analyst's ratio interpretations into the report
    schema ``[{"name", "value", "status", "threshold"}, …]``."""
    key_metrics: List[dict] = []
    for name, interp in ratios.items():
        key_metrics.append(
            {
                "name": name,
                "value": interp.get("value"),
                "status": interp.get("status", "unknown"),
                "threshold": interp.get("threshold", ""),
            }
        )
    return key_metrics


def _build_anomalies(anomalies: List[dict]) -> List[dict]:
    """Normalise anomaly dicts to the report schema."""
    return [
        {
            "metric": a.get("metric", "unknown"),
            "description": a.get("description", ""),
            "severity": a.get("severity", "unknown"),
        }
        for a in anomalies
    ]


def _build_sources(chunks: List[Document], max_sources: int = 10) -> List[dict]:
    """Extract source citations from retrieved chunks."""
    sources: List[dict] = []
    seen = set()

    for doc in chunks[:max_sources]:
        meta = doc.metadata
        key = (meta.get("source", ""), meta.get("page", ""))
        if key in seen:
            continue
        seen.add(key)

        excerpt = doc.page_content[:200].replace("\n", " ").strip()
        if len(doc.page_content) > 200:
            excerpt += " …"

        sources.append(
            {
                "page": meta.get("page", "?"),
                "source": meta.get("source", "?"),
                "excerpt": excerpt,
            }
        )
    return sources


def reporter_agent(state: AgentState) -> Dict[str, Any]:
    """Compile the final structured risk report.

    Parameters
    ----------
    state :
        The complete ``AgentState`` after all prior agents have run.

    Returns
    -------
    dict
        Partial state update with ``final_report`` and ``agent_trace``.
    """
    logger.info("▶ reporter_agent ENTER")

    risk_level: str = state.get("risk_level", "MEDIUM")
    justification: str = state.get("risk_justification", "")
    metrics: dict = state.get("financial_metrics", {})
    anomalies: List[dict] = state.get("anomalies", [])
    chunks: List[Document] = state.get("retrieved_chunks", [])
    verified: bool = state.get("verified", False)
    v_notes: List[str] = state.get("verification_notes", [])
    agent_trace: List[str] = state.get("agent_trace", [])

    # ── LLM-generated summary + recommendations ────────────────────
    llm = get_llm(temperature=0.3)

    context = (
        f"Risk level: {risk_level}\n"
        f"Justification: {justification}\n"
        f"Key ratios: {metrics.get('ratios', {})}\n"
        f"Anomalies: {anomalies}\n"
        f"Verified: {verified}\n"
    )

    import json as _json

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=context),
    ]

    response = llm.invoke(messages)
    raw = response.content.strip()

    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
    if raw.endswith("```"):
        raw = raw.rsplit("```", 1)[0]
    raw = raw.strip()

    try:
        llm_output = _json.loads(raw)
    except _json.JSONDecodeError:
        logger.error("Reporter LLM returned invalid JSON:\n%s", raw)
        llm_output = {
            "summary": "Unable to generate summary — see raw data below.",
            "recommendations": ["Review the analysis manually."],
        }

    # ── Assemble final report ───────────────────────────────────────
    final_report: Dict[str, Any] = {
        "summary": llm_output.get("summary", ""),
        "risk_level": risk_level,
        "risk_badge": _RISK_BADGES.get(risk_level, f"❓ {risk_level}"),
        "key_metrics": _build_key_metrics(metrics.get("ratios", {})),
        "anomalies": _build_anomalies(anomalies),
        "justification": justification,
        "recommendations": llm_output.get("recommendations", []),
        "sources": _build_sources(chunks),
        "verification_status": verified,
        "verification_notes": v_notes,
        "agent_trace": agent_trace + ["reporter"],
    }

    logger.info(
        "◀ reporter_agent EXIT  |  risk=%s  metrics=%d  anomalies=%d  "
        "sources=%d  verified=%s",
        risk_level,
        len(final_report["key_metrics"]),
        len(final_report["anomalies"]),
        len(final_report["sources"]),
        verified,
    )

    return {
        "final_report": final_report,
        "agent_trace": ["reporter"],
    }
