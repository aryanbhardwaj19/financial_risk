"""
Retriever agent — fetches relevant financial-document chunks.

Uses :class:`rag.retriever.FinancialRetriever` with hybrid search
(vector similarity + keyword re-ranking) driven by the planner's
``focus_areas``.

Sets ``state.missing_data = True`` when fewer than 3 chunks are returned
so the workflow can decide to retry or route differently.

Node signature
--------------
    retriever_agent(state: AgentState) -> dict   # partial state update
"""

from __future__ import annotations

from typing import Any, Dict, List

from langchain.schema import Document

from config import settings
from graph.state import AgentState
from rag.embedder import DocumentEmbedder
from rag.retriever import FinancialRetriever
from utils.logger import get_logger

logger = get_logger(__name__)

# Minimum number of chunks we consider "sufficient".
_MIN_CHUNKS = 3


def retriever_agent(state: AgentState) -> Dict[str, Any]:
    """Retrieve chunks relevant to the user query and planner focus areas.

    Parameters
    ----------
    state :
        Must contain ``query`` and ``financial_metrics["plan"]["focus_areas"]``.

    Returns
    -------
    dict
        Partial state update with ``retrieved_chunks``,
        ``retrieval_scores``, ``missing_data``, and ``agent_trace``.
    """
    query: str = state["query"]
    plan: dict = state.get("financial_metrics", {}).get("plan", {})
    focus_areas: List[str] = plan.get("focus_areas", [])

    logger.info(
        "▶ retriever_agent ENTER  |  query='%.80s…'  focus_areas=%s",
        query,
        focus_areas,
    )

    # ── Load vector store ───────────────────────────────────────────
    embedder = DocumentEmbedder()
    store = embedder.load_store(settings.VECTOR_DB_PATH)

    if store is None:
        logger.error("FAISS store not found — cannot retrieve")
        return {
            "retrieved_chunks": [],
            "retrieval_scores": [],
            "missing_data": True,
            "agent_trace": ["retriever"],
            "error": (
                "Vector store not found at "
                f"'{settings.VECTOR_DB_PATH}'. Build the index first."
            ),
        }

    retriever = FinancialRetriever(store)

    # ── Hybrid retrieval ────────────────────────────────────────────
    # Use focus_areas as keyword boost terms.
    keywords = focus_areas if focus_areas else []

    if keywords:
        chunks: List[Document] = retriever.hybrid_retrieve(
            query=query,
            keywords=keywords,
            k=5,
            vector_fetch_k=20,
        )
    else:
        chunks = retriever.retrieve(query=query, k=5)

    # ── Scored retrieval for score metadata ──────────────────────────
    scored_results = retriever.retrieve_with_scores(query=query, k=5)
    scores: List[float] = [score for _, score in scored_results]

    # If hybrid returned results, align scores by matching content.
    # Otherwise fall back to scored_results ordering.
    if chunks and scored_results:
        score_map: Dict[str, float] = {}
        for doc, score in scored_results:
            # Use first 200 chars of page_content as a key.
            key = doc.page_content[:200]
            score_map[key] = score

        aligned_scores: List[float] = []
        for doc in chunks:
            key = doc.page_content[:200]
            aligned_scores.append(score_map.get(key, -1.0))
        scores = aligned_scores

    # ── Missing data flag ───────────────────────────────────────────
    missing = len(chunks) < _MIN_CHUNKS

    if missing:
        logger.warning(
            "Only %d chunk(s) retrieved (threshold=%d) — setting missing_data=True",
            len(chunks),
            _MIN_CHUNKS,
        )
    else:
        logger.info("Retrieved %d chunk(s)", len(chunks))

    for i, doc in enumerate(chunks):
        meta = doc.metadata
        logger.debug(
            "  chunk %d | source=%s | page=%s | chars=%s",
            i,
            meta.get("source", "?"),
            meta.get("page", "?"),
            meta.get("char_count", len(doc.page_content)),
        )

    logger.info("◀ retriever_agent EXIT  |  chunks=%d  missing=%s", len(chunks), missing)

    return {
        "retrieved_chunks": chunks,
        "retrieval_scores": scores,
        "missing_data": missing,
        "agent_trace": ["retriever"],
    }
