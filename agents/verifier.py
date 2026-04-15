"""
Verifier agent — cross-checks analyst-extracted numbers against raw
source chunks.

For every metric in ``state.financial_metrics["raw_extractions"]``, the
verifier searches the retrieved chunks for the exact number.  If the
number appears in at least one chunk, the metric is marked *verified*;
otherwise it is added to ``verification_notes``.

If > 30 % of metrics are unverified, ``state.verified`` is set to
``False``.  The verifier **never halluccinates** — it only uses text
that exists in ``state.retrieved_chunks``.

Node signature
--------------
    verifier_agent(state: AgentState) -> dict   # partial state update
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from langchain.schema import Document

from graph.state import AgentState
from utils.logger import get_logger

logger = get_logger(__name__)

# Threshold: if more than this fraction of metrics are unverified,
# the overall verification fails.
_FAIL_THRESHOLD = 0.30


def _number_to_patterns(value: float) -> List[str]:
    """Generate plausible string representations of *value* that might
    appear in financial documents.

    Returns
    -------
    List[str]
        A set of regex-safe literal strings to search for.
    """
    patterns: List[str] = []
    abs_val = abs(value)

    # ── Integer form ────────────────────────────────────────────────
    if abs_val == int(abs_val):
        int_val = int(abs_val)
        # Plain: 1234567
        patterns.append(str(int_val))
        # Comma-separated: 1,234,567
        patterns.append(f"{int_val:,}")
    else:
        # Decimal forms.
        # 1234.56
        patterns.append(f"{abs_val:.2f}")
        patterns.append(f"{abs_val:.4f}")
        # With commas: 1,234.56
        int_part = int(abs_val)
        dec_part = abs_val - int_part
        patterns.append(f"{int_part:,}" + f"{dec_part:.2f}"[1:])

    # ── Suffix forms (M, B, K) ──────────────────────────────────────
    if abs_val >= 1_000_000_000:
        b = abs_val / 1_000_000_000
        patterns.append(f"{b:.1f}B")
        patterns.append(f"{b:.2f}B")
        patterns.append(f"{b:.1f} billion")
    if abs_val >= 1_000_000:
        m = abs_val / 1_000_000
        patterns.append(f"{m:.1f}M")
        patterns.append(f"{m:.2f}M")
        patterns.append(f"{m:.1f} million")
    if abs_val >= 1_000:
        k = abs_val / 1_000
        patterns.append(f"{k:.1f}K")
        patterns.append(f"{k:.1f}k")

    # ── Percentage form ─────────────────────────────────────────────
    if abs_val < 1.0:
        pct = abs_val * 100
        patterns.append(f"{pct:.1f}%")
        patterns.append(f"{pct:.2f}%")

    return patterns


def _find_in_chunks(
    patterns: List[str],
    chunks: List[Document],
) -> bool:
    """Return ``True`` if any of *patterns* is found in any chunk."""
    for pat in patterns:
        escaped = re.escape(pat)
        for doc in chunks:
            if re.search(escaped, doc.page_content, re.IGNORECASE):
                return True
    return False


def verifier_agent(state: AgentState) -> Dict[str, Any]:
    """Cross-check extracted metrics against raw chunk text.

    Parameters
    ----------
    state :
        Must contain ``financial_metrics["raw_extractions"]`` and
        ``retrieved_chunks``.

    Returns
    -------
    dict
        Partial state update with ``verified``, ``verification_notes``,
        and ``agent_trace``.
    """
    metrics: dict = state.get("financial_metrics", {})
    raw_extractions: Dict[str, List[float]] = metrics.get("raw_extractions", {})
    chunks: List[Document] = state.get("retrieved_chunks", [])
    risk_level: str = state.get("risk_level", "")

    total_metrics = sum(len(vals) for vals in raw_extractions.values())
    logger.info(
        "▶ verifier_agent ENTER  |  metrics_to_verify=%d  chunks=%d  risk=%s",
        total_metrics,
        len(chunks),
        risk_level,
    )

    if total_metrics == 0:
        logger.warning("No raw extractions to verify")
        return {
            "verified": False,
            "verification_notes": [
                "No financial metrics were extracted — nothing to verify."
            ],
            "agent_trace": ["verifier"],
        }

    verified_count = 0
    unverified_count = 0
    notes: List[str] = []

    for metric_name, values in raw_extractions.items():
        for value in values:
            patterns = _number_to_patterns(value)
            found = _find_in_chunks(patterns, chunks)

            if found:
                verified_count += 1
                logger.debug("  ✓ %s = %s — verified", metric_name, value)
            else:
                unverified_count += 1
                note = (
                    f"{metric_name} = {value} — "
                    f"NOT found in any retrieved chunk"
                )
                notes.append(note)
                logger.warning("  ✗ %s", note)

    # ── Overall verdict ─────────────────────────────────────────────
    if total_metrics > 0:
        unverified_ratio = unverified_count / total_metrics
    else:
        unverified_ratio = 1.0

    overall_verified = unverified_ratio <= _FAIL_THRESHOLD

    if not overall_verified:
        notes.insert(
            0,
            f"Verification FAILED: {unverified_count}/{total_metrics} metrics "
            f"({unverified_ratio:.0%}) could not be traced back to source documents.",
        )

    logger.info(
        "◀ verifier_agent EXIT  |  verified=%d  unverified=%d  "
        "ratio=%.0f%%  pass=%s",
        verified_count,
        unverified_count,
        unverified_ratio * 100,
        overall_verified,
    )

    return {
        "verified": overall_verified,
        "verification_notes": notes,
        "agent_trace": ["verifier"],
    }
