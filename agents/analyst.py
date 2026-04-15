"""
Analyst agent — extracts financial data, computes ratios, and detects
anomalies from retrieved document chunks.

Pipeline
--------
1. Regex-extract all numeric values and their surrounding labels from
   chunk text.
2. Map extracted values to canonical financial metric names.
3. Compute standard ratios via :class:`utils.financial_ratios.FinancialRatioEngine`.
4. Interpret each ratio and flag anything in ``"critical"`` status.
5. Detect trend anomalies: if ≥ 3 data points exist for the same metric
   and the last 2+ are declining, flag a trend anomaly.

Node signature
--------------
    analyst_agent(state: AgentState) -> dict   # partial state update
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from langchain.schema import Document

from graph.state import AgentState
from utils.financial_ratios import FinancialRatioEngine
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Regex for extracting labelled numbers ───────────────────────────
# Matches patterns like "Total Debt: 1,234.56" or "revenue of $5.2M".
_LABELLED_NUMBER_RE = re.compile(
    r"(?P<label>[A-Za-z][A-Za-z\s/\-&]{2,40})"  # 1) text label
    r"[\s:=\$]*"                                   # 2) separator
    r"(?P<sign>[−\-]?)"                            # 3) optional sign
    r"(?P<number>[\d,]+\.?\d*)"                    # 4) the number
    r"(?P<suffix>[BMKbmk%]?)",                     # 5) optional suffix
)

# Canonical mapping: lowercased keyword fragments → standard names.
_LABEL_MAP: Dict[str, str] = {
    "total debt": "total_debt",
    "long-term debt": "total_debt",
    "long term debt": "total_debt",
    "short-term debt": "short_term_debt",
    "total equity": "total_equity",
    "shareholder equity": "total_equity",
    "shareholders equity": "total_equity",
    "stockholder equity": "total_equity",
    "total assets": "total_assets",
    "current assets": "current_assets",
    "current liabilities": "current_liabilities",
    "total liabilities": "total_liabilities",
    "net income": "net_income",
    "net profit": "net_income",
    "revenue": "revenue",
    "total revenue": "revenue",
    "net sales": "revenue",
    "sales": "revenue",
    "gross income": "gross_income",
    "gross profit": "gross_income",
    "cash and equivalents": "cash",
    "cash & equivalents": "cash",
    "cash and cash equivalents": "cash",
    "cash": "cash",
    "short-term investments": "short_term_investments",
    "short term investments": "short_term_investments",
    "marketable securities": "short_term_investments",
    "accounts receivable": "receivables",
    "receivables": "receivables",
    "ebit": "ebit",
    "operating income": "ebit",
    "interest expense": "interest_expense",
    "cost of goods sold": "cogs",
    "cogs": "cogs",
    "inventory": "inventory",
    "inventories": "inventory",
    "profit margin": "profit_margin",
    "debt to equity": "debt_to_equity",
}


# ── Helpers ─────────────────────────────────────────────────────────


def _parse_number(raw: str, suffix: str) -> float:
    """Convert a raw string number (with commas and optional suffix) to float.

    Parameters
    ----------
    raw : str
        E.g. ``"1,234.56"``.
    suffix : str
        One of ``"B"``, ``"M"``, ``"K"``, ``"%"`` or empty.

    Returns
    -------
    float
    """
    value = float(raw.replace(",", ""))
    suffix = suffix.upper()
    if suffix == "B":
        value *= 1_000_000_000
    elif suffix == "M":
        value *= 1_000_000
    elif suffix == "K":
        value *= 1_000
    elif suffix == "%":
        value /= 100.0
    return value


def _best_label(raw_label: str) -> Optional[str]:
    """Map a raw extracted label to a canonical metric name.

    Returns ``None`` if no mapping matches.
    """
    cleaned = raw_label.strip().lower()
    # Try exact match first, then substring containment.
    if cleaned in _LABEL_MAP:
        return _LABEL_MAP[cleaned]
    for key, canonical in _LABEL_MAP.items():
        if key in cleaned:
            return canonical
    return None


def _extract_metrics(chunks: List[Document]) -> Dict[str, List[float]]:
    """Regex-extract all labelled numeric values from chunk text.

    Returns
    -------
    dict
        ``{canonical_name: [value, …]}`` — multiple values per metric
        are preserved (they may represent different periods).
    """
    metrics: Dict[str, List[float]] = {}

    for doc in chunks:
        text = doc.page_content
        for match in _LABELLED_NUMBER_RE.finditer(text):
            raw_label = match.group("label")
            canonical = _best_label(raw_label)
            if canonical is None:
                continue

            sign = match.group("sign")
            number_str = match.group("number")
            suffix = match.group("suffix")

            try:
                value = _parse_number(number_str, suffix)
                if sign in ("−", "-"):
                    value = -value
            except ValueError:
                continue

            metrics.setdefault(canonical, []).append(value)

    return metrics


def _compute_ratios(
    raw: Dict[str, List[float]],
) -> Tuple[Dict[str, Any], List[dict]]:
    """Compute standard financial ratios and detect anomalies.

    Parameters
    ----------
    raw :
        ``{metric_name: [values]}`` from ``_extract_metrics``.

    Returns
    -------
    (ratios_dict, anomalies_list)
        ``ratios_dict`` maps ratio names to their interpretation dict.
        ``anomalies_list`` is a list of anomaly dicts.
    """
    engine = FinancialRatioEngine()
    ratios: Dict[str, Any] = {}
    anomalies: List[dict] = []

    # Helper to grab the latest (last) value for a metric.
    def _latest(name: str) -> Optional[float]:
        vals = raw.get(name)
        return vals[-1] if vals else None

    # ── Compute each ratio if inputs are available ──────────────────
    ratio_inputs = {
        "debt_to_equity": lambda: engine.debt_to_equity(
            _latest("total_debt"), _latest("total_equity")
        ),
        "current_ratio": lambda: engine.current_ratio(
            _latest("current_assets"), _latest("current_liabilities")
        ),
        "profit_margin": lambda: engine.profit_margin(
            _latest("net_income"), _latest("revenue")
        ),
        "debt_to_income": lambda: engine.debt_to_income(
            _latest("total_debt"), _latest("gross_income")
        ),
        "quick_ratio": lambda: engine.quick_ratio(
            _latest("cash") or 0.0,
            _latest("short_term_investments") or 0.0,
            _latest("receivables") or 0.0,
            _latest("current_liabilities") or 0.0,
        ),
    }

    for ratio_name, compute_fn in ratio_inputs.items():
        try:
            value = compute_fn()
        except (TypeError, ValueError):
            value = None

        if value is None:
            continue

        interpretation = engine.interpret_ratio(ratio_name, value)
        ratios[ratio_name] = interpretation

        # Flag critical ratios as anomalies.
        if interpretation["status"] == "critical":
            anomalies.append(
                {
                    "metric": ratio_name,
                    "description": (
                        f"{ratio_name} is {value:.4f} "
                        f"(critical threshold: {interpretation['threshold']})"
                    ),
                    "severity": "critical",
                }
            )

    return ratios, anomalies


def _detect_trend_anomalies(
    raw: Dict[str, List[float]],
) -> List[dict]:
    """Flag declining trends when ≥ 3 data points exist for a metric
    and the last 2+ are consecutively declining.

    Parameters
    ----------
    raw :
        ``{metric_name: [values]}`` from ``_extract_metrics``.

    Returns
    -------
    list[dict]
        Trend anomaly dicts.
    """
    anomalies: List[dict] = []
    # Metrics where a decline is concerning.
    watch_metrics = {
        "net_income", "revenue", "gross_income", "ebit",
        "cash", "current_assets", "total_equity",
    }

    for metric, values in raw.items():
        if metric not in watch_metrics:
            continue
        if len(values) < 3:
            continue

        # Count consecutive declines from the end.
        consecutive_declines = 0
        for i in range(len(values) - 1, 0, -1):
            if values[i] < values[i - 1]:
                consecutive_declines += 1
            else:
                break

        if consecutive_declines >= 2:
            anomalies.append(
                {
                    "metric": metric,
                    "description": (
                        f"{metric} has declined for {consecutive_declines} "
                        f"consecutive period(s): "
                        f"{[round(v, 2) for v in values]}"
                    ),
                    "severity": "warning",
                }
            )

    return anomalies


# ── Node function ───────────────────────────────────────────────────


def analyst_agent(state: AgentState) -> Dict[str, Any]:
    """Extract financial data, compute ratios, and detect anomalies.

    Parameters
    ----------
    state :
        Must contain ``retrieved_chunks``.

    Returns
    -------
    dict
        Partial state update with enriched ``financial_metrics`` and
        ``anomalies``.
    """
    chunks: List[Document] = state.get("retrieved_chunks", [])
    logger.info("▶ analyst_agent ENTER  |  chunks=%d", len(chunks))

    # ── 1. Extract raw numeric metrics ──────────────────────────────
    raw_metrics = _extract_metrics(chunks)
    logger.info(
        "Extracted %d unique metric(s): %s",
        len(raw_metrics),
        list(raw_metrics.keys()),
    )

    # ── 2. Compute ratios & flag critical ones ──────────────────────
    ratios, ratio_anomalies = _compute_ratios(raw_metrics)
    logger.info("Computed %d ratio(s)", len(ratios))

    # ── 3. Detect trend anomalies ───────────────────────────────────
    trend_anomalies = _detect_trend_anomalies(raw_metrics)
    logger.info("Detected %d trend anomaly(ies)", len(trend_anomalies))

    # ── 4. Merge into state ─────────────────────────────────────────
    all_anomalies = ratio_anomalies + trend_anomalies

    # Preserve existing metrics (like the plan) and add new data.
    metrics = dict(state.get("financial_metrics", {}))
    metrics["raw_extractions"] = {
        k: [round(v, 4) for v in vals] for k, vals in raw_metrics.items()
    }
    metrics["ratios"] = ratios

    logger.info(
        "◀ analyst_agent EXIT   |  ratios=%d  anomalies=%d",
        len(ratios),
        len(all_anomalies),
    )

    return {
        "financial_metrics": metrics,
        "anomalies": all_anomalies,
        "agent_trace": ["analyst"],
    }
