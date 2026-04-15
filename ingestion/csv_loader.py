"""
CSV / tabular document loader for the financial risk analysis pipeline.

Reads a CSV file with ``pandas``, auto-detects column types (numeric,
categorical, datetime), and produces a rich natural-language summary of
the dataset – shape, dtypes, descriptive stats, and null counts – that
downstream agents can consume directly as context.

Return format
-------------
::

    {
        "summary":   str,            # NL description of the dataset
        "dataframe": pd.DataFrame,   # the loaded data
        "source":    str,            # absolute file path
    }

Usage
-----
    from ingestion.csv_loader import load_csv, load_csvs_from_directory

    result = load_csv("data/financials.csv")
    print(result["summary"])
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)

# File extensions this loader can handle.
_SUPPORTED_EXTENSIONS = {".csv", ".tsv", ".xls", ".xlsx"}


# ── Column-type detection ───────────────────────────────────────────


def _classify_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Classify every column into numeric, categorical, or datetime.

    Parameters
    ----------
    df:
        The loaded DataFrame.

    Returns
    -------
    dict
        ``{"numeric": [...], "categorical": [...], "datetime": [...]}``
    """
    numeric_cols: List[str] = []
    categorical_cols: List[str] = []
    datetime_cols: List[str] = []

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            datetime_cols.append(col)
        else:
            # Attempt to parse as datetime before falling back to categorical.
            try:
                pd.to_datetime(df[col], infer_datetime_format=True, errors="raise")
                datetime_cols.append(col)
            except (ValueError, TypeError):
                categorical_cols.append(col)

    return {
        "numeric": numeric_cols,
        "categorical": categorical_cols,
        "datetime": datetime_cols,
    }


# ── Summary generation ──────────────────────────────────────────────


def _build_summary(
    df: pd.DataFrame,
    col_types: Dict[str, List[str]],
    source: str,
) -> str:
    """Generate a natural-language summary of the DataFrame.

    The summary includes:
    - Dataset shape and source file.
    - Column names grouped by detected type.
    - Descriptive statistics for numeric columns
      (mean, min, max, std, median).
    - Null/missing-value counts per column.

    Parameters
    ----------
    df:
        The loaded DataFrame.
    col_types:
        Output of :func:`_classify_columns`.
    source:
        Absolute path string of the source file.

    Returns
    -------
    str
        Multi-line human-readable summary.
    """
    lines: List[str] = []

    # ── Header ──────────────────────────────────────────────────────
    rows, cols = df.shape
    lines.append(f"Dataset: {Path(source).name}")
    lines.append(f"Shape: {rows:,} rows × {cols} columns")
    lines.append("")

    # ── Column type breakdown ───────────────────────────────────────
    lines.append("Column types:")
    if col_types["numeric"]:
        lines.append(f"  Numeric ({len(col_types['numeric'])}): {', '.join(col_types['numeric'])}")
    if col_types["categorical"]:
        lines.append(f"  Categorical ({len(col_types['categorical'])}): {', '.join(col_types['categorical'])}")
    if col_types["datetime"]:
        lines.append(f"  Datetime ({len(col_types['datetime'])}): {', '.join(col_types['datetime'])}")
    lines.append("")

    # ── Numeric statistics ──────────────────────────────────────────
    if col_types["numeric"]:
        lines.append("Numeric column statistics:")
        for col in col_types["numeric"]:
            series = df[col].dropna()
            if series.empty:
                lines.append(f"  {col}: all values are null")
                continue
            lines.append(
                f"  {col}: "
                f"mean={series.mean():.4f}, "
                f"median={series.median():.4f}, "
                f"min={series.min():.4f}, "
                f"max={series.max():.4f}, "
                f"std={series.std():.4f}"
            )
        lines.append("")

    # ── Categorical value counts (top 5) ────────────────────────────
    if col_types["categorical"]:
        lines.append("Categorical column top values:")
        for col in col_types["categorical"]:
            n_unique = df[col].nunique()
            top_vals = df[col].value_counts().head(5)
            top_str = ", ".join(f"{v} ({c})" for v, c in top_vals.items())
            lines.append(f"  {col}: {n_unique} unique — top: {top_str}")
        lines.append("")

    # ── Datetime ranges ─────────────────────────────────────────────
    if col_types["datetime"]:
        lines.append("Datetime column ranges:")
        for col in col_types["datetime"]:
            try:
                dt_series = pd.to_datetime(df[col], errors="coerce").dropna()
                if not dt_series.empty:
                    lines.append(f"  {col}: {dt_series.min()} → {dt_series.max()}")
                else:
                    lines.append(f"  {col}: unable to parse dates")
            except Exception:
                lines.append(f"  {col}: unable to parse dates")
        lines.append("")

    # ── Null counts ─────────────────────────────────────────────────
    null_counts = df.isnull().sum()
    has_nulls = null_counts[null_counts > 0]
    if not has_nulls.empty:
        lines.append("Missing values:")
        for col, count in has_nulls.items():
            pct = count / rows * 100
            lines.append(f"  {col}: {count:,} ({pct:.1f}%)")
    else:
        lines.append("Missing values: none")

    return "\n".join(lines)


# ── Public API ──────────────────────────────────────────────────────


def load_csv(
    file_path: str | Path,
    sheet_name: Optional[str | int] = 0,
) -> Dict:
    """Load a CSV/TSV/Excel file and return a summary + DataFrame.

    Parameters
    ----------
    file_path:
        Path to the tabular data file.
    sheet_name:
        For Excel files, which sheet to read (default: first sheet).

    Returns
    -------
    dict
        ``{"summary": str, "dataframe": pd.DataFrame, "source": str}``

    Raises
    ------
    FileNotFoundError
        If *file_path* does not exist.
    ValueError
        If the file extension is not supported.
    """
    file_path = Path(file_path).resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = file_path.suffix.lower()
    if ext not in _SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Expected one of {_SUPPORTED_EXTENSIONS}."
        )

    logger.info("Loading tabular file: %s", file_path)

    # ── Read into DataFrame ─────────────────────────────────────────
    if ext in {".xls", ".xlsx"}:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
    elif ext == ".tsv":
        df = pd.read_csv(file_path, sep="\t")
    else:
        df = pd.read_csv(file_path)

    # ── Classify & summarise ────────────────────────────────────────
    col_types = _classify_columns(df)
    summary = _build_summary(df, col_types, str(file_path))

    logger.info(
        "Loaded %d row(s) × %d column(s) from %s",
        len(df),
        len(df.columns),
        file_path.name,
    )

    return {
        "summary": summary,
        "dataframe": df,
        "source": str(file_path),
    }


def load_csvs_from_directory(
    directory: str | Path,
) -> List[Dict]:
    """Batch-load every supported tabular file under *directory*.

    Parameters
    ----------
    directory:
        Path to a folder containing CSV/Excel files.

    Returns
    -------
    List[dict]
        One result dict per file.
    """
    directory = Path(directory).resolve()
    results: List[Dict] = []

    found_files: List[Path] = []
    for ext in _SUPPORTED_EXTENSIONS:
        found_files.extend(directory.glob(f"*{ext}"))
    found_files.sort()

    if not found_files:
        logger.warning("No tabular files found in %s", directory)
        return results

    for f in found_files:
        try:
            results.append(load_csv(f))
        except Exception as exc:
            logger.error("Failed to load %s: %s", f.name, exc)

    logger.info(
        "Loaded %d tabular file(s) from %s",
        len(results),
        directory,
    )
    return results
