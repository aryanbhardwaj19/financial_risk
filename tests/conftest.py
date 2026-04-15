"""
Shared pytest fixtures and mock configuration.

Patches ``ChatOpenAI`` across all agent modules so that tests run
without real OpenAI API calls.  Every LLM invocation returns a
deterministic fake response appropriate for its agent.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── Ensure the project root is importable ───────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── Set env vars BEFORE importing any project module ────────────────
os.environ["OPENAI_API_KEY"] = "sk-test-mock-key-for-pytest-00000000"
os.environ["EMBEDDING_MODEL"] = "all-MiniLM-L6-v2"
os.environ["LLM_MODEL"] = "gpt-4o"
os.environ["CHUNK_SIZE"] = "512"
os.environ["CHUNK_OVERLAP"] = "64"
os.environ["VECTOR_DB_PATH"] = str(_PROJECT_ROOT / "test_vector_store")
os.environ["LOG_LEVEL"] = "DEBUG"


# ── Deterministic fake LLM responses ───────────────────────────────

_PLANNER_RESPONSE = json.dumps(
    {
        "steps": [
            "Step 1: Retrieve balance sheet data",
            "Step 2: Compute key financial ratios",
            "Step 3: Assess credit risk based on ratios and trends",
        ],
        "focus_areas": ["debt", "equity", "liquidity", "profitability"],
        "requires_ratios": True,
    }
)

_EVALUATOR_RESPONSE = json.dumps(
    {
        "risk_level": "HIGH",
        "justification": (
            "• Debt-to-equity ratio of 2.8 exceeds the critical threshold of 2.0.\n"
            "• Current ratio of 0.9 is below the critical threshold of 1.0.\n"
            "• Net income has declined for 2 consecutive quarters."
        ),
        "key_drivers": [
            "Excessive leverage",
            "Insufficient liquidity",
            "Declining profitability",
        ],
    }
)

_REPORTER_RESPONSE = json.dumps(
    {
        "summary": (
            "The company exhibits elevated credit risk driven by high leverage "
            "(D/E 2.8), weak liquidity (CR 0.9), and a declining profit trend. "
            "Immediate action is recommended to restructure debt and improve "
            "working capital management."
        ),
        "recommendations": [
            "Restructure long-term debt to reduce leverage below 2.0x",
            "Improve working capital by accelerating receivables collection",
            "Review cost structure to reverse the declining profit margin trend",
            "Establish a cash reserve buffer of at least 3 months of operating expenses",
        ],
    }
)


def _make_fake_response(content: str) -> MagicMock:
    """Create a mock LLM response object with ``.content``."""
    resp = MagicMock()
    resp.content = content
    return resp


def _route_llm_response(*args, **kwargs) -> MagicMock:
    """Inspect the messages passed to ``ChatOpenAI.invoke()`` and return
    the appropriate deterministic response based on system prompt keywords."""
    messages = args[0] if args else kwargs.get("input", [])

    # Combine all message content to detect which agent is calling.
    combined = ""
    for msg in messages:
        if hasattr(msg, "content"):
            combined += msg.content + " "

    if "decompose" in combined.lower() or "break it into" in combined.lower():
        return _make_fake_response(_PLANNER_RESPONSE)
    elif "classify" in combined.lower() or "credit risk" in combined.lower():
        return _make_fake_response(_EVALUATOR_RESPONSE)
    elif "executive summary" in combined.lower() or "report writer" in combined.lower():
        return _make_fake_response(_REPORTER_RESPONSE)
    else:
        # Generic fallback.
        return _make_fake_response(_PLANNER_RESPONSE)


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def mock_llm():
    """Patch ChatOpenAI globally so no real API calls are made.

    The mock routes responses based on prompt content so each agent
    gets a contextually appropriate reply.
    """
    with patch("agents.planner.ChatOpenAI") as mock_planner, \
         patch("agents.evaluator.ChatOpenAI") as mock_evaluator, \
         patch("agents.reporter.ChatOpenAI") as mock_reporter:

        for mock_cls in (mock_planner, mock_evaluator, mock_reporter):
            instance = MagicMock()
            instance.invoke.side_effect = _route_llm_response
            mock_cls.return_value = instance

        yield {
            "planner": mock_planner,
            "evaluator": mock_evaluator,
            "reporter": mock_reporter,
        }


@pytest.fixture
def tmp_vector_store(tmp_path):
    """Override the vector store path to a temporary directory."""
    store_path = str(tmp_path / "faiss_store")
    os.environ["VECTOR_DB_PATH"] = store_path
    yield store_path
    os.environ["VECTOR_DB_PATH"] = str(_PROJECT_ROOT / "test_vector_store")
