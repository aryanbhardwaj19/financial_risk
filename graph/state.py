"""
Shared agent state for the LangGraph multi-agent financial risk system.

Every node in the workflow reads from and writes to a single
``AgentState`` TypedDict.  This makes inter-agent data flow explicit,
inspectable, and serialisable.

LangGraph's ``Annotated`` + reducer pattern is used for append-only
fields (``agent_trace``, ``verification_notes``, ``messages``) so that
each node can *add* items without overwriting previous entries.

Usage
-----
    from graph.state import AgentState, initial_state

    state = initial_state("What is ACME Corp's credit risk exposure?")
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, Optional, Sequence

from langchain.schema import Document
from langchain_core.messages import AnyMessage, BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


# ── Reducer helpers ─────────────────────────────────────────────────
# ``operator.add`` concatenates lists so each node can return new
# items without clobbering what earlier nodes wrote.


class AgentState(TypedDict, total=False):
    """Mutable state passed through every node in the LangGraph
    workflow.

    Fields are grouped by pipeline stage.  Append-only fields use
    ``Annotated[list, operator.add]`` so nodes simply return the *new*
    elements and LangGraph merges them automatically.

    Attributes
    ----------

    **Input**

    query : str
        The user's original financial-risk question.
    documents : List[Document]
        Raw loaded documents (from the ingestion layer) before
        chunking / embedding.

    **Message history**

    messages : Annotated[list[AnyMessage], add_messages]
        Full LLM message history.  Uses LangGraph's built-in
        ``add_messages`` reducer which handles de-duplication by
        message ID.

    **RAG**

    retrieved_chunks : List[Document]
        Chunks returned by the retriever for the current query.
    retrieval_scores : List[float]
        Corresponding similarity scores (L2 distance — lower is
        better), one per chunk in ``retrieved_chunks``.

    **Analysis**

    financial_metrics : dict
        All computed ratios / numbers produced by the analyst, keyed
        by ratio name.  Example::

            {"debt_to_equity": 1.23, "current_ratio": 2.05, …}

    anomalies : List[dict]
        Detected anomalies, each a dict with at least ``"description"``
        and ``"severity"`` keys.

    **Risk**

    risk_level : str
        Overall risk classification — one of
        ``"LOW"``, ``"MEDIUM"``, ``"HIGH"``, ``"CRITICAL"``.
    risk_justification : str
        Plain-English explanation of the risk rating.

    **Verification**

    verified : bool
        Whether the verifier agent accepted the analyst's findings.
    verification_notes : Annotated[List[str], operator.add]
        Append-only list of issues or observations raised by the
        verifier.

    **Report**

    final_report : dict
        Structured report output ready for rendering / serialisation.
        Typically contains keys like ``"executive_summary"``,
        ``"findings"``, ``"recommendations"``, etc.

    **Control flow**

    agent_trace : Annotated[List[str], operator.add]
        Ordered, append-only trace of agent names that have executed.
        Useful for debugging and for the reporter to cite which agents
        contributed.
    error : Optional[str]
        If any node encounters an unrecoverable error it writes a
        message here.  Downstream nodes can check this field and
        short-circuit.
    missing_data : bool
        Set to ``True`` by the analyst when it determines that the
        retrieved context is insufficient.  A conditional edge can use
        this flag to loop back to the retriever for a second pass.
    iteration_count : int
        Monotonically increasing counter incremented each time the
        workflow loops through the analyse → evaluate cycle.  Used
        to enforce a maximum-iteration guard and prevent infinite
        loops.
    """

    # ── Input ───────────────────────────────────────────────────────
    query: str
    documents: List[Document]

    # ── Message history ─────────────────────────────────────────────
    messages: Annotated[Sequence[AnyMessage], add_messages]

    # ── RAG ─────────────────────────────────────────────────────────
    retrieved_chunks: List[Document]
    retrieval_scores: List[float]

    # ── Analysis ────────────────────────────────────────────────────
    financial_metrics: Dict[str, Any]
    anomalies: List[Dict[str, Any]]

    # ── Risk ────────────────────────────────────────────────────────
    risk_level: str
    risk_justification: str

    # ── Verification ────────────────────────────────────────────────
    verified: bool
    verification_notes: Annotated[List[str], operator.add]

    # ── Report ──────────────────────────────────────────────────────
    final_report: Dict[str, Any]

    # ── Control flow ────────────────────────────────────────────────
    agent_trace: Annotated[List[str], operator.add]
    error: Optional[str]
    missing_data: bool
    iteration_count: int


# ── Factory ─────────────────────────────────────────────────────────


def initial_state(query: str) -> AgentState:
    """Return a clean starting state for a new workflow run.

    All collection fields are initialised to empty containers, booleans
    to ``False``, counters to ``0``, and nullable fields to ``None``.

    Parameters
    ----------
    query :
        The user's financial-risk question that will drive the entire
        pipeline.

    Returns
    -------
    AgentState
        A fully populated state dict ready to be passed to
        ``workflow.invoke()``.
    """
    return AgentState(
        # Input
        query=query,
        documents=[],

        # Message history
        messages=[],

        # RAG
        retrieved_chunks=[],
        retrieval_scores=[],

        # Analysis
        financial_metrics={},
        anomalies=[],

        # Risk
        risk_level="",
        risk_justification="",

        # Verification
        verified=False,
        verification_notes=[],

        # Report
        final_report={},

        # Control flow
        agent_trace=[],
        error=None,
        missing_data=False,
        iteration_count=0,
    )
