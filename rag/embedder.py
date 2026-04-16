"""
Document embedder — builds and manages a FAISS vector store backed by
local HuggingFace sentence-transformer embeddings.

Uses ``all-MiniLM-L6-v2`` by default (~80 MB, runs on CPU).  No API
keys or network access required for embedding — fully offline after
the model is downloaded once.

The embedding model is configured via ``settings.EMBEDDING_MODEL``.
The FAISS index is persisted to disk so it survives process restarts.

Usage
-----
    from rag.embedder import DocumentEmbedder

    embedder = DocumentEmbedder()
    store = embedder.embed_and_store(chunks, "vector_store")
    # later …
    store = embedder.load_store("vector_store")
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class DocumentEmbedder:
    """Wraps HuggingFace sentence-transformer embeddings + FAISS for
    persist-and-reload workflows.

    Parameters
    ----------
    model_name : str, optional
        HuggingFace model identifier.  Defaults to
        ``settings.EMBEDDING_MODEL`` (``all-MiniLM-L6-v2``).
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        **kwargs,
    ) -> None:
        self.model_name = model_name or settings.EMBEDDING_MODEL

        logger.info(
            "Initialising HuggingFace embeddings: model=%s",
            self.model_name,
        )

        self._embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    # ── Public API ──────────────────────────────────────────────────

    def embed_and_store(
        self,
        chunks: List[Document],
        persist_path: Optional[str] = None,
    ) -> Optional[FAISS]:
        """Embed *chunks* and build a FAISS vector store.

        The store is persisted to *persist_path* (defaults to
        ``settings.VECTOR_DB_PATH``).  If the directory does not
        exist it will be created.

        Parameters
        ----------
        chunks :
            LangChain ``Document`` objects produced by the chunker.
        persist_path :
            Directory where the FAISS index + docstore are saved.

        Returns
        -------
        FAISS or None
            The constructed (and persisted) vector store, or ``None``
            if *chunks* is empty.
        """
        persist_path = persist_path or settings.VECTOR_DB_PATH
        store_dir = Path(persist_path)

        if not chunks:
            logger.warning("embed_and_store called with 0 chunks — skipping")
            return None

        logger.info(
            "Embedding %d chunk(s) with '%s' …",
            len(chunks),
            self.model_name,
        )

        store = FAISS.from_documents(chunks, self._embeddings)

        # Persist to disk.
        store_dir.mkdir(parents=True, exist_ok=True)
        store.save_local(str(store_dir))

        logger.info(
            "FAISS store saved to %s  (%d vectors)",
            store_dir,
            store.index.ntotal,
        )
        return store

    def load_store(
        self,
        persist_path: Optional[str] = None,
    ) -> Optional[FAISS]:
        """Load a previously persisted FAISS store from disk.

        Parameters
        ----------
        persist_path :
            Directory containing the FAISS index files.  Defaults to
            ``settings.VECTOR_DB_PATH``.

        Returns
        -------
        FAISS or None
            The loaded vector store, or ``None`` if the store does not
            exist (a warning is logged instead of raising).
        """
        persist_path = persist_path or settings.VECTOR_DB_PATH
        store_dir = Path(persist_path)
        index_file = store_dir / "index.faiss"

        if not index_file.exists():
            logger.warning(
                "No FAISS index found at %s — "
                "run embed_and_store first to create one.",
                store_dir,
            )
            return None

        logger.info("Loading FAISS store from %s …", store_dir)

        store = FAISS.load_local(
            str(store_dir),
            self._embeddings,
            allow_dangerous_deserialization=True,
        )

        logger.info(
            "FAISS store loaded: %d vectors", store.index.ntotal
        )
        return store
