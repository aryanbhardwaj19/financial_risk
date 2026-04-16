"""
Planner agent — decomposes a financial-risk query into an ordered plan.

Receives ``state.query`` and produces a structured JSON plan that
guides all downstream agents.  The plan is stored inside
``state.financial_metrics["plan"]``.

Node signature
--------------
    planner_agent(state: AgentState) -> dict   # partial state update
"""

from __future__ import annotations

import json
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage

from config import settings
from graph.state import AgentState
from utils.logger import get_logger
from utils.llm_factory import get_llm

logger = get_logger(__name__)

_SYSTEM_PROMPT = """\
You are a senior financial analyst specialising in risk assessment.

Given a user's financial analysis query, decompose it into a clear,
ordered execution plan.  Return **only** valid JSON with these keys:

{
  "steps": [
    "Step 1: …",
    "Step 2: …",
    …                          // 3-5 steps, imperative voice
  ],
  "focus_areas": [
    "liquidity",               // domain areas to retrieve docs for
    "leverage",
    …
  ],
  "requires_ratios": true      // whether financial ratios should be computed
}

Rules:
- Keep each step actionable and specific to financial data.
- ``focus_areas`` should be 2-5 concise domain keywords that a retriever
  can use for targeted document search.
- ``requires_ratios`` is true when the query involves quantitative
  assessment (debt, profitability, coverage, etc.).
- Output ONLY the JSON object — no markdown fences, no commentary.
"""


def planner_agent(state: AgentState) -> Dict[str, Any]:
    """Decompose the user query into an analysis plan.

    Parameters
    ----------
    state :
        Current workflow state.  Must contain ``query``.

    Returns
    -------
    dict
        Partial state update with ``financial_metrics`` (containing the
        plan) and ``agent_trace``.
    """
    query: str = state["query"]
    logger.info("▶ planner_agent ENTER  |  query='%.120s…'", query)

    llm = get_llm(temperature=0.0)

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=query),
    ]

    response = llm.invoke(messages)
    raw_text: str = response.content.strip()

    # ── Parse LLM output ────────────────────────────────────────────
    # Strip markdown code-fence wrappers if the model added them.
    if raw_text.startswith("```"):
        raw_text = raw_text.split("\n", 1)[1]  # drop opening fence
    if raw_text.endswith("```"):
        raw_text = raw_text.rsplit("```", 1)[0]
    raw_text = raw_text.strip()

    try:
        plan = json.loads(raw_text)
    except json.JSONDecodeError:
        logger.error("Planner returned invalid JSON:\n%s", raw_text)
        plan = {
            "steps": ["Analyse the query manually"],
            "focus_areas": ["general financial analysis"],
            "requires_ratios": True,
        }

    # Validate expected keys.
    plan.setdefault("steps", [])
    plan.setdefault("focus_areas", [])
    plan.setdefault("requires_ratios", True)

    # ── Merge into state ────────────────────────────────────────────
    metrics = dict(state.get("financial_metrics", {}))
    metrics["plan"] = plan

    logger.info(
        "◀ planner_agent EXIT   |  steps=%d  focus_areas=%s  requires_ratios=%s",
        len(plan["steps"]),
        plan["focus_areas"],
        plan["requires_ratios"],
    )

    return {
        "financial_metrics": metrics,
        "agent_trace": ["planner"],
    }
