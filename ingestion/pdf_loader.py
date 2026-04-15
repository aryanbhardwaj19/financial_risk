"""
PDF document loader for the financial risk analysis pipeline.

Extraction strategy
-------------------
1. **Primary – pdfplumber**:  Best for digitally-born PDFs with embedded
   tables (10-K filings, annual reports).  Tables are extracted as
   ``pandas.DataFrame`` objects and serialised to Markdown strings.
2. **Fallback – PyMuPDF (fitz)**:  Handles scanned / image-heavy PDFs
   where pdfplumber yields no text.

Return format
-------------
Each page produces a dict::

    {
        "page":   int,        # 1-indexed page number
        "text":   str,        # raw extracted text
        "tables": [str, ...], # each table as a Markdown string
        "source": str,        # absolute file path
        "title":  str,        # inferred document title
    }

Usage
-----
    from ingestion.pdf_loader import load_pdf, load_pdfs_from_directory

    pages = load_pdf("data/annual_report.pdf")
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

import pandas as pd
import pdfplumber
import fitz  # PyMuPDF

from utils.logger import get_logger

logger = get_logger(__name__)


# ── Helpers ─────────────────────────────────────────────────────────


def _table_to_markdown(table_data: list[list]) -> Optional[str]:
    """Convert a pdfplumber table (list-of-lists) to a Markdown string.

    Returns ``None`` if the table is empty or contains only whitespace.

    Parameters
    ----------
    table_data:
        Raw table output from ``pdfplumber.Page.extract_tables()``.
        The first row is treated as the header.
    """
    if not table_data or len(table_data) < 2:
        return None

    # Clean cells: replace None / whitespace-only with empty string.
    cleaned: list[list[str]] = []
    for row in table_data:
        cleaned.append([str(cell).strip() if cell else "" for cell in row])

    # Build a DataFrame so we can use its built-in Markdown renderer.
    header = cleaned[0]
    body = cleaned[1:]

    # Guard against fully-empty tables.
    if all(cell == "" for row in body for cell in row):
        return None

    df = pd.DataFrame(body, columns=header)
    return df.to_markdown(index=False)


def _infer_title(first_page_text: str, file_path: Path) -> str:
    """Best-effort document title extraction.

    Heuristics (in priority order):
    1. First non-empty line that looks like a heading (≤120 chars,
       no obvious boilerplate like page numbers).
    2. The filename stem as a fallback.

    Parameters
    ----------
    first_page_text:
        The raw text extracted from the first page of the PDF.
    file_path:
        Path object of the source file (used for fallback).

    Returns
    -------
    str
        Inferred title string.
    """
    for line in first_page_text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Skip lines that are just numbers / dates / short noise.
        if len(line) < 5 or re.fullmatch(r"[\d\s/\-\.]+", line):
            continue
        if len(line) <= 120:
            return line

    # Fallback: prettified filename.
    return file_path.stem.replace("_", " ").replace("-", " ").title()


# ── Primary loader (pdfplumber) ─────────────────────────────────────


def _extract_with_pdfplumber(file_path: Path) -> List[dict]:
    """Extract text and tables from every page using pdfplumber.

    Parameters
    ----------
    file_path:
        Resolved ``Path`` to the PDF.

    Returns
    -------
    List[dict]
        One dict per page.  Pages with neither text nor tables are
        still included (with empty values) so page indexing stays
        consistent.
    """
    pages: List[dict] = []

    with pdfplumber.open(file_path) as pdf:
        for idx, page in enumerate(pdf.pages, start=1):
            # ── Text ────────────────────────────────────────────────
            raw_text = page.extract_text() or ""

            # ── Tables ──────────────────────────────────────────────
            raw_tables = page.extract_tables() or []
            md_tables: List[str] = []
            for tbl in raw_tables:
                md = _table_to_markdown(tbl)
                if md:
                    md_tables.append(md)

            pages.append(
                {
                    "page": idx,
                    "text": raw_text,
                    "tables": md_tables,
                    "source": str(file_path),
                }
            )

    return pages


# ── Fallback loader (PyMuPDF) ───────────────────────────────────────


def _extract_with_pymupdf(file_path: Path) -> List[dict]:
    """Fallback extraction via PyMuPDF when pdfplumber yields nothing.

    PyMuPDF does not offer structured table extraction, so the
    ``tables`` key will always be an empty list.

    Parameters
    ----------
    file_path:
        Resolved ``Path`` to the PDF.

    Returns
    -------
    List[dict]
        One dict per page.
    """
    pages: List[dict] = []
    doc = fitz.open(str(file_path))

    for idx, page in enumerate(doc, start=1):
        raw_text = page.get_text() or ""
        pages.append(
            {
                "page": idx,
                "text": raw_text,
                "tables": [],
                "source": str(file_path),
            }
        )

    doc.close()
    return pages


# ── Public API ──────────────────────────────────────────────────────


def load_pdf(file_path: str | Path) -> List[dict]:
    """Load a PDF and return per-page extraction results.

    Each returned dict contains::

        {
            "page":   int,
            "text":   str,
            "tables": [str, ...],
            "source": str,
            "title":  str,
        }

    Parameters
    ----------
    file_path:
        Path to a ``.pdf`` file on disk.

    Returns
    -------
    List[dict]
        One entry per page.

    Raises
    ------
    FileNotFoundError
        If *file_path* does not exist.
    """
    file_path = Path(file_path).resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    logger.info("Loading PDF: %s", file_path)

    # ── Try pdfplumber first ────────────────────────────────────────
    pages: List[dict] = []
    try:
        pages = _extract_with_pdfplumber(file_path)
    except Exception as exc:
        logger.warning(
            "pdfplumber failed on %s (%s) – falling back to PyMuPDF",
            file_path.name,
            exc,
        )

    # Check whether pdfplumber actually extracted any text.
    has_text = any(p["text"].strip() for p in pages)

    if not has_text:
        logger.info("pdfplumber yielded no text – using PyMuPDF fallback for %s", file_path.name)
        pages = _extract_with_pymupdf(file_path)

    # ── Infer document title from the first page ────────────────────
    first_page_text = pages[0]["text"] if pages else ""
    title = _infer_title(first_page_text, file_path)

    for page_dict in pages:
        page_dict["title"] = title

    text_pages = sum(1 for p in pages if p["text"].strip())
    table_count = sum(len(p["tables"]) for p in pages)
    logger.info(
        "Extracted %d page(s) (%d with text, %d table(s)) from %s",
        len(pages),
        text_pages,
        table_count,
        file_path.name,
    )
    return pages


def load_pdfs_from_directory(directory: str | Path) -> List[dict]:
    """Batch-load every ``*.pdf`` found under *directory*.

    Parameters
    ----------
    directory:
        Path to a folder containing PDF files.

    Returns
    -------
    List[dict]
        Aggregated list of page dicts from all PDFs.
    """
    directory = Path(directory).resolve()
    all_pages: List[dict] = []

    pdf_files = sorted(directory.glob("*.pdf"))
    if not pdf_files:
        logger.warning("No PDF files found in %s", directory)
        return all_pages

    for pdf_file in pdf_files:
        all_pages.extend(load_pdf(pdf_file))

    logger.info(
        "Loaded %d total page(s) from %d PDF(s) in %s",
        len(all_pages),
        len(pdf_files),
        directory,
    )
    return all_pages
