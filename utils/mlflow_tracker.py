"""
Optional MLflow experiment tracking for the risk analysis pipeline.

Logs parameters, financial-ratio metrics, risk classifications, and the
full report JSON as an MLflow artifact.  Activates **only** when
``MLFLOW_TRACKING_URI`` is set in the environment — otherwise all
methods are silent no-ops.

Usage
-----
    from utils.mlflow_tracker import RiskAnalysisTracker

    tracker = RiskAnalysisTracker()
    if tracker.enabled:
        tracker.start_run(query="Assess ACME credit risk")
        tracker.log_metrics(state["financial_metrics"])
        tracker.log_report(state["final_report"])
        tracker.end_run()
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Conditional import ──────────────────────────────────────────────
# MLflow is an optional dependency.  If it is not installed or no
# tracking URI is configured, the tracker gracefully degrades.

_mlflow = None
_TRACKING_URI: Optional[str] = os.getenv("MLFLOW_TRACKING_URI")

if _TRACKING_URI:
    try:
        import mlflow as _mlflow

        _mlflow.set_tracking_uri(_TRACKING_URI)
        logger.info("MLflow tracking enabled: %s", _TRACKING_URI)
    except ImportError:
        logger.warning(
            "MLFLOW_TRACKING_URI is set but mlflow is not installed. "
            "Run: pip install mlflow==2.19.0"
        )
        _mlflow = None


class RiskAnalysisTracker:
    """Thin wrapper around MLflow for financial-risk experiment tracking.

    All public methods are safe to call even when MLflow is unavailable
    — they silently return without error.

    Attributes
    ----------
    enabled : bool
        ``True`` if MLflow is installed **and** ``MLFLOW_TRACKING_URI``
        is set.
    """

    EXPERIMENT_NAME = "financial_risk_analysis"

    def __init__(self) -> None:
        self.enabled: bool = _mlflow is not None and _TRACKING_URI is not None
        self._run_active: bool = False

        if self.enabled:
            _mlflow.set_experiment(self.EXPERIMENT_NAME)
            logger.info(
                "RiskAnalysisTracker initialised (experiment='%s')",
                self.EXPERIMENT_NAME,
            )

    # ── start_run ───────────────────────────────────────────────────

    def start_run(self, query: str) -> None:
        """Start a new MLflow run and log initial parameters.

        Parameters
        ----------
        query :
            The user's financial-risk question.
        """
        if not self.enabled:
            return

        timestamp = datetime.now(timezone.utc).isoformat()
        run_name = f"risk_{timestamp[:19].replace(':', '-')}"

        _mlflow.start_run(run_name=run_name)
        self._run_active = True

        _mlflow.log_param("query", query[:250])  # MLflow param limit
        _mlflow.log_param("timestamp", timestamp)
        _mlflow.log_param("embedding_model", settings.EMBEDDING_MODEL)
        _mlflow.log_param("llm_model", settings.LLM_MODEL)
        _mlflow.log_param("chunk_size", settings.CHUNK_SIZE)
        _mlflow.log_param("chunk_overlap", settings.CHUNK_OVERLAP)

        logger.info("MLflow run started: %s", run_name)

    # ── log_metrics ─────────────────────────────────────────────────

    def log_metrics(self, financial_metrics: Dict[str, Any]) -> None:
        """Log each computed financial ratio as an MLflow metric.

        Parameters
        ----------
        financial_metrics :
            The ``state["financial_metrics"]`` dict.  Expected to
            contain a ``"ratios"`` sub-dict where each value has a
            ``"value"`` key.
        """
        if not self.enabled or not self._run_active:
            return

        ratios: dict = financial_metrics.get("ratios", {})
        logged = 0

        for ratio_name, interpretation in ratios.items():
            value = interpretation.get("value")
            if value is not None and isinstance(value, (int, float)):
                _mlflow.log_metric(f"ratio_{ratio_name}", float(value))
                logged += 1

        # Log anomaly count.
        raw = financial_metrics.get("raw_extractions", {})
        _mlflow.log_metric("extracted_metrics_count", sum(len(v) for v in raw.values()))
        _mlflow.log_metric("ratios_computed", logged)

        logger.info("Logged %d ratio metric(s) to MLflow", logged)

    # ── log_report ──────────────────────────────────────────────────

    def log_report(self, final_report: Dict[str, Any]) -> None:
        """Log the final report: risk level as a param, verification as
        a metric, and the full JSON as an artifact.

        Parameters
        ----------
        final_report :
            The ``state["final_report"]`` dict.
        """
        if not self.enabled or not self._run_active:
            return

        # ── Params ──────────────────────────────────────────────────
        risk_level = final_report.get("risk_level", "UNKNOWN")
        _mlflow.log_param("risk_level", risk_level)

        risk_badge = final_report.get("risk_badge", "")
        _mlflow.log_param("risk_badge", risk_badge)

        # ── Metrics ─────────────────────────────────────────────────
        verified = final_report.get("verification_status", False)
        _mlflow.log_metric("verification_status", 1.0 if verified else 0.0)

        anomaly_count = len(final_report.get("anomalies", []))
        _mlflow.log_metric("anomaly_count", anomaly_count)

        source_count = len(final_report.get("sources", []))
        _mlflow.log_metric("source_count", source_count)

        agent_steps = len(final_report.get("agent_trace", []))
        _mlflow.log_metric("agent_steps", agent_steps)

        # ── Artifact: full report JSON ──────────────────────────────
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            prefix="risk_report_",
            delete=False,
            encoding="utf-8",
        ) as f:
            json.dump(final_report, f, indent=2, default=str)
            artifact_path = f.name

        _mlflow.log_artifact(artifact_path, artifact_path="reports")

        # Clean up temp file.
        try:
            os.unlink(artifact_path)
        except OSError:
            pass

        logger.info(
            "Logged report to MLflow: risk=%s, verified=%s, anomalies=%d",
            risk_level,
            verified,
            anomaly_count,
        )

    # ── end_run ─────────────────────────────────────────────────────

    def end_run(self) -> None:
        """End the current MLflow run."""
        if not self.enabled or not self._run_active:
            return

        _mlflow.end_run()
        self._run_active = False
        logger.info("MLflow run ended")
