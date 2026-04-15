"""
Financial document retriever with vector, scored, and hybrid search.

Wraps a LangChain ``FAISS`` vector store and provides three retrieval
strategies:

1. **MMR search** (``retrieve``) — maximises relevance *and* diversity.
2. **Scored search** (``retrieve_with_scores``) — returns documents
   alongside their L2 similarity scores.
3. **Hybrid search** (``hybrid_retrieve``) — vector retrieval followed
   by keyword-based re-ranking for precision on domain terminology.

Usage
-----
    from rag.embedder import DocumentEmbedder
    from rag.retriever import FinancialRetriever

    store    = DocumentEmbedder().load_store()
    retriever = FinancialRetriever(store)
    docs     = retriever.retrieve("What is the company's debt-to-equity?")
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from langchain.schema import Document
from langchain_community.vectorstores import FAISS

from utils.logger import get_logger

logger = get_logger(__name__)


class FinancialRetriever:
    """Retriever tailored for financial-document RAG pipelines.

    Parameters
    ----------
    store : FAISS
        A populated LangChain FAISS vector store (created or loaded
        via :class:`rag.embedder.DocumentEmbedder`).
    """

    def __init__(self, store: FAISS) -> None:
        if store is None:
            raise ValueError(
                "FinancialRetriever requires a non-None FAISS store. "
                "Make sure the index has been built and loaded."
            )
        self._store = store
        logger.info(
            "FinancialRetriever initialised with %d vectors",
            self._store.index.ntotal,
        )

    # ── 1. MMR search ───────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        k: int = 5,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
    ) -> List[Document]:
        """Retrieve the top-*k* documents using Maximal Marginal Relevance.

        MMR balances **relevance** to the query with **diversity** among
        results, reducing redundant chunks.

        Parameters
        ----------
        query :
            Natural-language search string.
        k :
            Number of documents to return.
        fetch_k :
            Number of candidates to fetch before MMR re-ranking
            (must be ≥ *k*).
        lambda_mult :
            Diversity factor (0 = max diversity, 1 = max relevance).

        Returns
        -------
        List[Document]
            Ranked list of LangChain ``Document`` objects.
        """
        logger.info("MMR retrieve: query='%.100s…', k=%d", query, k)

        docs = self._store.max_marginal_relevance_search(
            query,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
        )

        self._log_results(docs, method="MMR")
        return docs

    # ── 2. Scored search ────────────────────────────────────────────

    def retrieve_with_scores(
        self,
        query: str,
        k: int = 5,
    ) -> List[Tuple[Document, float]]:
        """Retrieve documents with their L2 similarity scores.

        Lower scores indicate higher similarity.

        Parameters
        ----------
        query :
            Natural-language search string.
        k :
            Number of results to return.

        Returns
        -------
        List[Tuple[Document, float]]
            ``[(doc, score), ...]`` sorted by ascending score (best first).
        """
        logger.info("Scored retrieve: query='%.100s…', k=%d", query, k)

        results: List[Tuple[Document, float]] = (
            self._store.similarity_search_with_score(query, k=k)
        )

        for doc, score in results:
            meta = doc.metadata
            logger.debug(
                "  score=%.4f | source=%s | page=%s | chunk=%s | chars=%s",
                score,
                meta.get("source", "?"),
                meta.get("page", "?"),
                meta.get("chunk_index", "?"),
                meta.get("char_count", "?"),
            )

        logger.info("Scored retrieve returned %d result(s)", len(results))
        return results

    # ── 3. Hybrid search ────────────────────────────────────────────

    def hybrid_retrieve(
        self,
        query: str,
        keywords: List[str],
        k: int = 5,
        vector_fetch_k: int = 20,
    ) -> List[Document]:
        """Two-stage retrieval: vector similarity → keyword re-ranking.

        1. Fetch *vector_fetch_k* candidates via plain similarity search.
        2. Score each candidate by how many of the supplied *keywords*
           appear in its ``page_content`` (case-insensitive).
        3. Sort by ``(keyword_hits DESC, original_rank ASC)`` and
           return the top *k*.

        This is useful when the user's query contains specific financial
        terms (e.g. ``"EBITDA"``, ``"Tier 1 capital"``) that must appear
        in the retrieved context.

        Parameters
        ----------
        query :
            Natural-language search string.
        keywords :
            Domain-specific terms that retrieved chunks should contain.
        k :
            Number of final results to return.
        vector_fetch_k :
            Number of initial vector candidates to fetch for re-ranking.

        Returns
        -------
        List[Document]
            Re-ranked list of ``Document`` objects.
        """
        logger.info(
            "Hybrid retrieve: query='%.80s…', keywords=%s, k=%d",
            query,
            keywords,
            k,
        )

        # Stage 1 — broad vector fetch.
        candidates: List[Document] = self._store.similarity_search(
            query, k=vector_fetch_k
        )

        if not candidates:
            logger.warning("Hybrid retrieve: vector stage returned 0 candidates")
            return []

        # Stage 2 — keyword re-ranking.
        lower_keywords = [kw.lower() for kw in keywords]

        scored: List[Tuple[int, int, Document]] = []
        for rank, doc in enumerate(candidates):
            content_lower = doc.page_content.lower()
            hits = sum(1 for kw in lower_keywords if kw in content_lower)
            # Negate hits so higher hit-count sorts first with ascending sort.
            scored.append((-hits, rank, doc))

        scored.sort(key=lambda t: (t[0], t[1]))
        top_docs = [doc for _, _, doc in scored[:k]]

        # Log re-ranked results.
        for doc in top_docs:
            content_lower = doc.page_content.lower()
            matched = [kw for kw in keywords if kw.lower() in content_lower]
            meta = doc.metadata
            logger.debug(
                "  hybrid | source=%s | page=%s | chunk=%s | kw_matches=%s",
                meta.get("source", "?"),
                meta.get("page", "?"),
                meta.get("chunk_index", "?"),
                matched,
            )

        logger.info(
            "Hybrid retrieve: %d candidates → %d re-ranked result(s)",
            len(candidates),
            len(top_docs),
        )
        return top_docs

    # ── Logging helper ──────────────────────────────────────────────

    @staticmethod
    def _log_results(docs: List[Document], method: str) -> None:
        """Log metadata for each retrieved document."""
        for doc in docs:
            meta = doc.metadata
            logger.debug(
                "  %s | source=%s | page=%s | chunk=%s | chars=%s",
                method,
                meta.get("source", "?"),
                meta.get("page", "?"),
                meta.get("chunk_index", "?"),
                meta.get("char_count", "?"),
            )
        logger.info("%s retrieve returned %d result(s)", method, len(docs))
