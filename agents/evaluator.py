"""
Evaluator agent — classifies overall credit risk from computed metrics
and detected anomalies.

Uses an LLM call with deterministic escalation rules applied *after*
the model responds to ensure policy compliance:
- Any ``CRITICAL`` ratio → at minimum ``HIGH``.
- 2+ ``WARNING`` ratios → at minimum ``MEDIUM``.
- Declining trend on profit/cash flow → escalate one level.

Node signature
--------------
    evaluator_agent(state: AgentState) -> dict   # partial state update
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import settings
from graph.state import AgentState
from utils.logger import get_logger

logger = get_logger(__name__)

_RISK_LEVELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

_SYSTEM_PROMPT = """\
You are a credit-risk evaluation engine.

Given a set of financial metrics (with computed ratios and their
interpretive statuses) and a list of detected anomalies, classify the
overall credit risk.

Return **only** valid JSON with these keys:

{
  "risk_level": "LOW" | "MEDIUM" | "HIGH" | "CRITICAL",
  "justification": "3 bullet-point explanation",
  "key_drivers": ["driver1", "driver2", …]
}

Rules you must follow:
- If ANY ratio has status "critical", the risk level must be at least HIGH.
- If 2 or more ratios have status "warning", the risk level must be at
  least MEDIUM.
- If there is a declining trend anomaly on profit or cash flow, escalate
  the risk level by one notch.
- Be specific and cite the metric values in your justification.
- Output ONLY the JSON object.
"""


def _escalate(current: str, minimum: str) -> str:
    """Return whichever risk level is higher on the severity scale."""
    cur_idx = _RISK_LEVELS.index(current) if current in _RISK_LEVELS else 0
    min_idx = _RISK_LEVELS.index(minimum) if minimum in _RISK_LEVELS else 0
    return _RISK_LEVELS[max(cur_idx, min_idx)]


def _bump_one(level: str) -> str:
    """Escalate a risk level by exactly one notch (capped at CRITICAL)."""
    idx = _RISK_LEVELS.index(level) if level in _RISK_LEVELS else 0
    return _RISK_LEVELS[min(idx + 1, len(_RISK_LEVELS) - 1)]


def evaluator_agent(state: AgentState) -> Dict[str, Any]:
    """Classify overall credit risk from metrics and anomalies.

    Parameters
    ----------
    state :
        Must contain ``financial_metrics`` (with ``"ratios"`` sub-dict)
        and ``anomalies``.

    Returns
    -------
    dict
        Partial state update with ``risk_level``,
        ``risk_justification``, and ``agent_trace``.
    """
    metrics: dict = state.get("financial_metrics", {})
    anomalies: List[dict] = state.get("anomalies", [])

    logger.info(
        "▶ evaluator_agent ENTER  |  ratios=%d  anomalies=%d",
        len(metrics.get("ratios", {})),
        len(anomalies),
    )

    # ── Build prompt payload ────────────────────────────────────────
    payload = {
        "ratios": metrics.get("ratios", {}),
        "raw_extractions": metrics.get("raw_extractions", {}),
        "anomalies": anomalies,
    }
    payload_str = json.dumps(payload, indent=2, default=str)

    llm = ChatOpenAI(
        model=settings.LLM_MODEL,
        api_key=settings.OPENAI_API_KEY,
        temperature=0.0,
    )

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=f"Financial data:\n{payload_str}"),
    ]

    response = llm.invoke(messages)
    raw = response.content.strip()

    # Strip code fences.
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
    if raw.endswith("```"):
        raw = raw.rsplit("```", 1)[0]
    raw = raw.strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        logger.error("Evaluator returned invalid JSON:\n%s", raw)
        result = {
            "risk_level": "MEDIUM",
            "justification": "Unable to parse evaluation — defaulting to MEDIUM.",
            "key_drivers": ["evaluation_parse_error"],
        }

    risk_level: str = result.get("risk_level", "MEDIUM").upper()
    if risk_level not in _RISK_LEVELS:
        risk_level = "MEDIUM"

    justification: str = result.get("justification", "")
    key_drivers: List[str] = result.get("key_drivers", [])

    # ── Deterministic escalation rules ──────────────────────────────
    ratios = metrics.get("ratios", {})

    # Rule 1: any CRITICAL ratio → at minimum HIGH.
    has_critical = any(
        r.get("status") == "critical" for r in ratios.values()
    )
    if has_critical:
        risk_level = _escalate(risk_level, "HIGH")
        logger.info("Escalation rule: CRITICAL ratio detected → minimum HIGH")

    # Rule 2: 2+ WARNING ratios → at minimum MEDIUM.
    warning_count = sum(
        1 for r in ratios.values() if r.get("status") == "warning"
    )
    if warning_count >= 2:
        risk_level = _escalate(risk_level, "MEDIUM")
        logger.info(
            "Escalation rule: %d WARNING ratios → minimum MEDIUM", warning_count
        )

    # Rule 3: declining trend on profit/cash → bump one level.
    profit_cash_keywords = {"net_income", "revenue", "gross_income", "cash", "ebit"}
    has_declining_trend = any(
        a.get("severity") == "warning"
        and a.get("metric", "") in profit_cash_keywords
        for a in anomalies
    )
    if has_declining_trend:
        old = risk_level
        risk_level = _bump_one(risk_level)
        logger.info(
            "Escalation rule: declining profit/cash trend → %s → %s",
            old,
            risk_level,
        )

    # ── Compose justification ───────────────────────────────────────
    full_justification = justification
    if key_drivers:
        full_justification += "\n\nKey drivers: " + ", ".join(key_drivers)

    logger.info(
        "◀ evaluator_agent EXIT  |  risk_level=%s", risk_level
    )

    return {
        "risk_level": risk_level,
        "risk_justification": full_justification,
        "agent_trace": ["evaluator"],
    }
