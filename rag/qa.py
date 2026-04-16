"""
RAG Q&A — thin wrapper that loads the vector store and invokes
the LangGraph pipeline.

Usage
-----
    from rag.qa import ask_question
    result = ask_question("Is this company good to invest in?")
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from config import settings
from rag.embedder import DocumentEmbedder
from graph.workflow import run_qa_pipeline
from utils.logger import get_logger

logger = get_logger(__name__)


def ask_question(
    query: str,
    store_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Load the vector store and run the LangGraph Q&A pipeline.

    Parameters
    ----------
    query :
        The user's financial question.
    store_path :
        Path to the FAISS vector store directory.

    Returns
    -------
    dict
        ``{"answer": str, "sources": list, "agent_trace": list}``
    """
    store_path = store_path or settings.VECTOR_DB_PATH

    # Load the persisted vector store
    embedder = DocumentEmbedder()
    store = embedder.load_store(store_path)

    if store is None:
        return {
            "answer": "No documents have been indexed yet. Please upload documents first.",
            "sources": [],
            "agent_trace": [],
        }

    # Run the LangGraph pipeline
    return run_qa_pipeline(query, store)
