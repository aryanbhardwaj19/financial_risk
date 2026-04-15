"""
LangGraph workflow — wires the six agents into a stateful, checkpointed
graph with conditional re-retrieval and verification loops.

Graph topology
--------------
::

    START → planner → retriever ──┐
                         ▲        │
                         │   (missing_data && iter < 3?)
                         │        │
                         └── yes ─┘
                              no ──→ analyst → evaluator → verifier ──┐
                                                             ▲        │
                                                             │  (¬verified && iter < 2?)
                                                             │        │
                                                             └── yes ─┘
                                                                  no ──→ reporter → END

Checkpointing
-------------
Uses ``MemorySaver`` so conversation state persists across turns within
the same process.

Public API
----------
    from graph.workflow import run_analysis

    report = run_analysis(
        query="Assess ACME Corp credit risk",
        documents=loaded_docs,
    )
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from langchain.schema import Document
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from graph.state import AgentState, initial_state
from agents.planner import planner_agent
from agents.retriever_agent import retriever_agent
from agents.analyst import analyst_agent
from agents.evaluator import evaluator_agent
from agents.verifier import verifier_agent
from agents.reporter import reporter_agent
from utils.logger import get_logger
from utils.mlflow_tracker import RiskAnalysisTracker

logger = get_logger(__name__)

# ── Query broadening suffix (used on re-entry) ─────────────────────
_BROADENED_SUFFIX = (
    " financial statements overview balance sheet cash flow"
)


# ── Wrapper nodes ───────────────────────────────────────────────────
# The retriever node needs special pre-processing logic on re-entry,
# so we wrap it rather than registering the raw agent function.


def _retriever_node(state: AgentState) -> Dict[str, Any]:
    """Retriever wrapper that widens the query on re-entry and bumps
    the iteration counter.

    On the **first** call (``iteration_count == 0``) the original query
    is used as-is.  On subsequent calls the query is broadened by
    appending domain keywords so the vector search casts a wider net.
    """
    iteration: int = state.get("iteration_count", 0)
    original_query: str = state["query"]

    if iteration > 0:
        widened = original_query + _BROADENED_SUFFIX
        logger.info(
            "Retriever re-entry (iter=%d) — broadened query: '%.120s…'",
            iteration,
            widened,
        )
        # Temporarily override query for this retrieval pass.
        state_copy = dict(state)
        state_copy["query"] = widened
        result = retriever_agent(state_copy)
    else:
        result = retriever_agent(state)

    # Always increment iteration_count on every retriever pass.
    result["iteration_count"] = iteration + 1
    return result


# ── Conditional edge functions ──────────────────────────────────────


def _after_retriever(state: AgentState) -> str:
    """Decide whether to re-retrieve or proceed to the analyst.

    Re-retrieves if:
    - ``missing_data`` is ``True``  **and**
    - ``iteration_count < 3``
    """
    missing: bool = state.get("missing_data", False)
    iteration: int = state.get("iteration_count", 0)

    if missing and iteration < 3:
        logger.info(
            "Conditional: missing_data=True, iter=%d < 3 → re-retrieve",
            iteration,
        )
        return "retriever"

    logger.info(
        "Conditional: missing_data=%s, iter=%d → analyst",
        missing,
        iteration,
    )
    return "analyst"


def _after_verifier(state: AgentState) -> str:
    """Decide whether to loop back for more evidence or proceed to the
    reporter.

    Loops back if:
    - ``verified`` is ``False``  **and**
    - ``iteration_count < 2``
    """
    verified: bool = state.get("verified", False)
    iteration: int = state.get("iteration_count", 0)

    if not verified and iteration < 2:
        logger.info(
            "Conditional: verified=False, iter=%d < 2 → retriever (re-evidence)",
            iteration,
        )
        return "retriever"

    logger.info(
        "Conditional: verified=%s, iter=%d → reporter",
        verified,
        iteration,
    )
    return "reporter"


# ── Graph assembly ──────────────────────────────────────────────────


def build_graph() -> StateGraph:
    """Construct the full LangGraph workflow.

    Returns
    -------
    CompiledGraph
        A compiled, checkpointed graph ready for ``.invoke()``.
    """
    workflow = StateGraph(AgentState)

    # ── Register nodes ──────────────────────────────────────────────
    workflow.add_node("planner", planner_agent)
    workflow.add_node("retriever", _retriever_node)
    workflow.add_node("analyst", analyst_agent)
    workflow.add_node("evaluator", evaluator_agent)
    workflow.add_node("verifier", verifier_agent)
    workflow.add_node("reporter", reporter_agent)

    # ── Edges ───────────────────────────────────────────────────────
    # START → planner → retriever
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "retriever")

    # retriever → conditional: re-retrieve or analyst
    workflow.add_conditional_edges(
        "retriever",
        _after_retriever,
        {
            "retriever": "retriever",
            "analyst": "analyst",
        },
    )

    # analyst → evaluator → verifier
    workflow.add_edge("analyst", "evaluator")
    workflow.add_edge("evaluator", "verifier")

    # verifier → conditional: loop back to retriever or reporter
    workflow.add_conditional_edges(
        "verifier",
        _after_verifier,
        {
            "retriever": "retriever",
            "reporter": "reporter",
        },
    )

    # reporter → END
    workflow.add_edge("reporter", END)

    # ── Compile with checkpointing ──────────────────────────────────
    memory = MemorySaver()
    compiled = workflow.compile(checkpointer=memory)

    logger.info("LangGraph workflow compiled with MemorySaver checkpointing")
    return compiled


# Module-level singleton so the graph is compiled once per process.
_graph = None


def _get_graph():
    """Lazily build and cache the compiled graph."""
    global _graph  # noqa: PLW0603
    if _graph is None:
        _graph = build_graph()
    return _graph


# ── Public API ──────────────────────────────────────────────────────


def run_analysis(
    query: str,
    documents: Optional[List[Document]] = None,
    thread_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute the full multi-agent financial risk analysis pipeline.

    Parameters
    ----------
    query :
        The user's financial-risk question.
    documents :
        Pre-loaded ``Document`` objects to attach to the initial state
        (e.g. from the ingestion layer).  These are informational; the
        retriever pulls from the FAISS index independently.
    thread_id :
        Optional conversation thread ID for the ``MemorySaver``
        checkpointer.  If ``None`` a new UUID is generated.

    Returns
    -------
    dict
        The ``final_report`` dict produced by the reporter agent.
        Returns an error dict if the pipeline fails.
    """
    graph = _get_graph()

    # Build clean initial state.
    state = initial_state(query)
    if documents:
        state["documents"] = documents

    # Checkpointer config — each thread_id gets its own memory lane.
    config = {
        "configurable": {
            "thread_id": thread_id or str(uuid.uuid4()),
        }
    }

    # ── Optional MLflow tracking ─────────────────────────────────────
    tracker = RiskAnalysisTracker()
    tracker.start_run(query)

    logger.info(
        "═══════════════════════════════════════════════════════\n"
        "  WORKFLOW START  |  query='%.120s…'\n"
        "═══════════════════════════════════════════════════════",
        query,
    )

    try:
        final_state = graph.invoke(state, config=config)
    except Exception as exc:
        logger.exception("Workflow failed with exception: %s", exc)
        tracker.end_run()
        return {
            "summary": "Analysis failed due to an internal error.",
            "risk_level": "UNKNOWN",
            "risk_badge": "❓ UNKNOWN",
            "key_metrics": [],
            "anomalies": [],
            "justification": str(exc),
            "recommendations": ["Investigate the error and retry."],
            "sources": [],
            "verification_status": False,
            "verification_notes": [f"Pipeline error: {exc}"],
            "agent_trace": [],
        }

    # ── Log the full agent trace ────────────────────────────────────
    trace = final_state.get("agent_trace", [])
    logger.info(
        "═══════════════════════════════════════════════════════\n"
        "  WORKFLOW COMPLETE\n"
        "  Agent trace: %s\n"
        "  Risk level:  %s\n"
        "  Verified:    %s\n"
        "═══════════════════════════════════════════════════════",
        " → ".join(trace),
        final_state.get("risk_level", "?"),
        final_state.get("verified", "?"),
    )

    report = final_state.get("final_report", {})

    # ── Log to MLflow ───────────────────────────────────────────────
    tracker.log_metrics(final_state.get("financial_metrics", {}))
    tracker.log_report(report)
    tracker.end_run()

    return report
