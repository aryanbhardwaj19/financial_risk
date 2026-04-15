"""
Production-grade document chunker for financial documents.

Splits ingested documents into overlapping chunks using LangChain's
``RecursiveCharacterTextSplitter`` with financial-document-aware logic:

- Markdown table blocks (detected by ``|`` column patterns) are **never**
  split and are emitted as standalone chunks regardless of size.
- Sentence and number boundaries are respected — splits only happen at
  paragraph, sentence, or whitespace boundaries, never mid-sentence or
  mid-number.
- Every chunk carries provenance metadata: ``source``, ``page``,
  ``chunk_index``, and ``char_count``.

Usage
-----
    from rag.chunker import DocumentChunker

    chunker = DocumentChunker()
    lc_docs = chunker.chunk_documents(raw_docs)
"""

from __future__ import annotations

import re
from typing import List, Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Regex for detecting Markdown table blocks ───────────────────────
# Matches lines that start with optional whitespace then a pipe ``|``.
_TABLE_LINE_RE = re.compile(r"^\s*\|", re.MULTILINE)
# A "table block" is two or more consecutive table-like lines.
_TABLE_BLOCK_RE = re.compile(
    r"((?:^\s*\|.*$\n?){2,})",
    re.MULTILINE,
)

# Separators tuned for financial documents.  We *avoid* splitting
# inside numbers (no bare ``.`` separator) and prefer sentence
# boundaries.
_FINANCIAL_SEPARATORS = [
    "\n\n",          # paragraph break
    "\n",            # line break
    ".\n",           # sentence ending at line break
    ". ",            # sentence boundary
    "; ",            # clause boundary
    ", ",            # list-item boundary
    " ",             # word boundary (last resort for text)
]


class DocumentChunker:
    """Chunks financial documents while preserving tables and sentence
    integrity.

    Parameters
    ----------
    chunk_size : int, optional
        Maximum characters per chunk.  Defaults to ``settings.CHUNK_SIZE``.
    chunk_overlap : int, optional
        Character overlap between consecutive chunks.  Defaults to
        ``settings.CHUNK_OVERLAP``.
    """

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> None:
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=_FINANCIAL_SEPARATORS,
            keep_separator=True,
        )

        logger.info(
            "DocumentChunker initialised (size=%d, overlap=%d)",
            self.chunk_size,
            self.chunk_overlap,
        )

    # ── Internal helpers ────────────────────────────────────────────

    @staticmethod
    def _extract_tables_and_text(text: str) -> List[dict]:
        """Split *text* into an ordered sequence of segments, each
        tagged as either ``"table"`` or ``"text"``.

        Table blocks are identified by the ``_TABLE_BLOCK_RE`` regex
        and are never subjected to further splitting.

        Parameters
        ----------
        text :
            Raw page text (may contain interleaved prose and tables).

        Returns
        -------
        List[dict]
            ``[{"type": "table"|"text", "content": str}, ...]``
        """
        segments: List[dict] = []
        last_end = 0

        for match in _TABLE_BLOCK_RE.finditer(text):
            start, end = match.span()

            # Any text before this table block.
            if start > last_end:
                pre = text[last_end:start].strip()
                if pre:
                    segments.append({"type": "text", "content": pre})

            segments.append({"type": "table", "content": match.group(0).strip()})
            last_end = end

        # Trailing text after the last table (or everything if no tables).
        if last_end < len(text):
            tail = text[last_end:].strip()
            if tail:
                segments.append({"type": "text", "content": tail})

        return segments

    # ── Public API ──────────────────────────────────────────────────

    def chunk_documents(self, docs: List[dict]) -> List[Document]:
        """Chunk a list of ingested document dicts into LangChain
        ``Document`` objects.

        Parameters
        ----------
        docs :
            Each dict must contain at least a ``"text"`` key.
            Optional keys — ``"source"``, ``"page"``, ``"tables"``
            (list of markdown strings) — are propagated as metadata.

        Returns
        -------
        List[Document]
            Flat list of ``Document`` instances with ``page_content``
            and ``metadata`` (``source``, ``page``, ``chunk_index``,
            ``char_count``).
        """
        all_chunks: List[Document] = []
        global_index = 0

        for doc in docs:
            text: str = doc.get("text", "")
            source: str = doc.get("source", "unknown")
            page: int = doc.get("page", doc.get("row", 0))
            inline_tables: List[str] = doc.get("tables", [])

            base_meta = {"source": source, "page": page}

            # ── 1. Handle explicitly extracted tables ───────────────
            for tbl_md in inline_tables:
                tbl_md = tbl_md.strip()
                if not tbl_md:
                    continue
                all_chunks.append(
                    Document(
                        page_content=tbl_md,
                        metadata={
                            **base_meta,
                            "chunk_index": global_index,
                            "char_count": len(tbl_md),
                            "is_table": True,
                        },
                    )
                )
                global_index += 1

            # ── 2. Segment remaining text into tables vs prose ──────
            if not text.strip():
                continue

            segments = self._extract_tables_and_text(text)

            for seg in segments:
                if seg["type"] == "table":
                    # Preserve table as a single chunk.
                    content = seg["content"]
                    all_chunks.append(
                        Document(
                            page_content=content,
                            metadata={
                                **base_meta,
                                "chunk_index": global_index,
                                "char_count": len(content),
                                "is_table": True,
                            },
                        )
                    )
                    global_index += 1
                else:
                    # Split prose respecting sentence/number boundaries.
                    pieces = self._splitter.split_text(seg["content"])
                    for piece in pieces:
                        piece = piece.strip()
                        if not piece:
                            continue
                        all_chunks.append(
                            Document(
                                page_content=piece,
                                metadata={
                                    **base_meta,
                                    "chunk_index": global_index,
                                    "char_count": len(piece),
                                    "is_table": False,
                                },
                            )
                        )
                        global_index += 1

        table_chunks = sum(1 for c in all_chunks if c.metadata.get("is_table"))
        text_chunks = len(all_chunks) - table_chunks

        logger.info(
            "Chunked %d document(s) → %d chunk(s) "
            "(%d text, %d table)  [size=%d, overlap=%d]",
            len(docs),
            len(all_chunks),
            text_chunks,
            table_chunks,
            self.chunk_size,
            self.chunk_overlap,
        )
        return all_chunks
