"""
Simple LangGraph workflow for financial document Q&A.

Demonstrates LangChain + LangGraph agent orchestration with a
3-node pipeline:

    ┌──────────┐     ┌──────────┐     ┌──────────┐
    │ Retriever│ ──▶ │ Analyzer │ ──▶ │ Reporter │
    └──────────┘     └──────────┘     └──────────┘

- **Retriever**: Fetches relevant chunks from the FAISS vector store.
- **Analyzer**: Uses the LLM to analyze the retrieved context.
- **Reporter**: Formats the final answer with structured output.

Usage
-----
    from graph.workflow import run_qa_pipeline
    result = run_qa_pipeline(query, store)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

from langchain.schema import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

from rag.retriever import FinancialRetriever
from utils.llm_factory import get_llm
from utils.logger import get_logger

logger = get_logger(__name__)


# ── Graph State ─────────────────────────────────────────────────────

class QAState(TypedDict, total=False):
    """State object passed between nodes in the LangGraph."""
    query: str
    context: str
    chunk_count: int
    analysis: str
    sources: List[dict]
    final_answer: str
    agent_trace: List[str]


# ── Node definitions ───────────────────────────────────────────────

_ANALYZER_SYSTEM = """\
You are a senior financial analyst. Analyze the provided document
excerpts and answer the user's question.

Rules:
- Base your answer strictly on the context provided.
- If the context does not contain enough information, say so clearly.
- Use specific numbers, ratios, and data from the context.
- Be concise but thorough.
- Structure your answer with bullet points or sections when helpful.
"""

_REPORTER_SYSTEM = """\
You are a report formatter. Take the analysis provided and produce
a clean, well-structured final answer. Keep the same content but
ensure it reads professionally. Use markdown formatting (bold,
bullet points, headers) for clarity.

Output ONLY the formatted answer — no preamble.
"""


def _make_retriever_agent(store):
    """Create a retriever agent node with the FAISS store bound via closure."""
    _RETRIEVER_K = 8

    def retriever_agent(state: QAState) -> Dict[str, Any]:
        """Retrieve relevant chunks from the vector store."""
        query = state["query"]
        logger.info("▶ retriever_agent | query='%.80s…'", query)

        retriever = FinancialRetriever(store)
        chunks = retriever.retrieve(query, k=_RETRIEVER_K)

        # Build context string from chunks
        context = "\n\n".join(
            f"[Section {i}]\n{doc.page_content}"
            for i, doc in enumerate(chunks, 1)
        )

        # Build source citations
        sources = []
        seen = set()
        for doc in chunks:
            meta = doc.metadata
            source_key = (meta.get("source", ""), meta.get("page", ""))
            if source_key not in seen:
                seen.add(source_key)
                excerpt = doc.page_content[:200].replace("\n", " ").strip()
                if len(doc.page_content) > 200:
                    excerpt += " …"
                sources.append({
                    "source": Path(meta.get("source", "unknown")).name,
                    "page": str(meta.get("page", "?")),
                    "excerpt": excerpt,
                })

        logger.info("◀ retriever_agent | chunks=%d, sources=%d", len(chunks), len(sources))

        return {
            "context": context,
            "chunk_count": len(chunks),
            "sources": sources,
            "agent_trace": ["retriever"],
        }

    return retriever_agent


def analyzer_agent(state: QAState) -> Dict[str, Any]:
    """Analyze retrieved context using the LLM."""
    query = state["query"]
    context = state.get("context", "")
    chunk_count = state.get("chunk_count", 0)

    logger.info("▶ analyzer_agent | context_length=%d", len(context))

    if not context:
        return {
            "analysis": "No relevant information was found in the uploaded documents for this question.",
            "agent_trace": state.get("agent_trace", []) + ["analyzer"],
        }

    llm = get_llm(temperature=0.2)
    messages = [
        SystemMessage(content=_ANALYZER_SYSTEM),
        HumanMessage(content=(
            f"Document context:\n\n{context}\n\n---\n\n"
            f"Question: {query}"
        )),
    ]

    response = llm.invoke(messages)
    analysis = response.content.strip()

    logger.info("◀ analyzer_agent | answer_length=%d", len(analysis))

    return {
        "analysis": analysis,
        "agent_trace": state.get("agent_trace", []) + ["analyzer"],
    }


def reporter_agent(state: QAState) -> Dict[str, Any]:
    """Format the analysis into a polished final answer."""
    analysis = state.get("analysis", "")
    chunk_count = state.get("chunk_count", 0)

    logger.info("▶ reporter_agent | analysis_length=%d", len(analysis))

    if not analysis or "no relevant information" in analysis.lower():
        return {
            "final_answer": analysis or "Unable to generate a response.",
            "agent_trace": state.get("agent_trace", []) + ["reporter"],
        }

    llm = get_llm(temperature=0.1)
    messages = [
        SystemMessage(content=_REPORTER_SYSTEM),
        HumanMessage(content=(
            f"Analysis to format (based on {chunk_count} document sections):\n\n{analysis}"
        )),
    ]

    response = llm.invoke(messages)
    final_answer = response.content.strip()

    logger.info("◀ reporter_agent | final_length=%d", len(final_answer))

    return {
        "final_answer": final_answer,
        "agent_trace": state.get("agent_trace", []) + ["reporter"],
    }


# ── Build & Run ─────────────────────────────────────────────────────

def run_qa_pipeline(
    query: str,
    store,
) -> Dict[str, Any]:
    """Run the full Q&A pipeline.

    Parameters
    ----------
    query :
        The user's question.
    store :
        A loaded FAISS vector store.

    Returns
    -------
    dict
        ``{"answer": str, "sources": list, "agent_trace": list}``
    """
    logger.info("═══ Starting Q&A pipeline for: '%.80s…' ═══", query)

    # Build graph with store bound via closure
    workflow = StateGraph(QAState)

    workflow.add_node("retriever", _make_retriever_agent(store))
    workflow.add_node("analyzer", analyzer_agent)
    workflow.add_node("reporter", reporter_agent)

    workflow.set_entry_point("retriever")
    workflow.add_edge("retriever", "analyzer")
    workflow.add_edge("analyzer", "reporter")
    workflow.add_edge("reporter", END)

    graph = workflow.compile()

    # Run
    result = graph.invoke({
        "query": query,
        "context": "",
        "chunk_count": 0,
        "analysis": "",
        "sources": [],
        "final_answer": "",
        "agent_trace": [],
    })

    logger.info(
        "═══ Pipeline complete | trace=%s ═══",
        result.get("agent_trace", []),
    )

    return {
        "answer": result.get("final_answer", "No answer generated."),
        "sources": result.get("sources", []),
        "agent_trace": result.get("agent_trace", []),
    }
