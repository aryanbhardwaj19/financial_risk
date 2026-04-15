"""
Financial ratio computation engine.

Provides a ``FinancialRatioEngine`` class whose methods compute standard
financial ratios with input validation, rounding, and interpretive
thresholds.  Designed to be used by analysis agents in the multi-agent
pipeline.

Usage
-----
    from utils.financial_ratios import FinancialRatioEngine

    engine = FinancialRatioEngine()
    ratio  = engine.current_ratio(500_000, 250_000)
    interp = engine.interpret_ratio("current_ratio", ratio)
    print(interp)
    # {"value": 2.0, "status": "safe", "threshold": ">= 1.5"}
"""

from __future__ import annotations

from typing import Dict, Optional

from utils.logger import get_logger

logger = get_logger(__name__)

# ── Industry-standard thresholds ────────────────────────────────────
# Each entry maps a ratio name to ``(safe_min, warning_min)`` bounds or
# ``(safe_max, warning_max)`` bounds depending on the ratio's semantics.
#
# Structure:  ratio_name → {
#     "safe":     (operator, threshold),
#     "warning":  (operator, threshold),
#     "critical": (operator, threshold),   ← implicit "else"
#     "label":    human-readable threshold description,
# }

_THRESHOLDS: Dict[str, dict] = {
    "debt_to_equity": {
        "safe_max": 1.0,
        "warning_max": 2.0,
        "direction": "lower_is_better",
        "safe_label": "< 1.0",
        "warning_label": "1.0 – 2.0",
        "critical_label": "> 2.0",
    },
    "current_ratio": {
        "safe_min": 1.5,
        "warning_min": 1.0,
        "direction": "higher_is_better",
        "safe_label": ">= 1.5",
        "warning_label": "1.0 – 1.5",
        "critical_label": "< 1.0",
    },
    "profit_margin": {
        "safe_min": 0.10,
        "warning_min": 0.05,
        "direction": "higher_is_better",
        "safe_label": ">= 10%",
        "warning_label": "5% – 10%",
        "critical_label": "< 5%",
    },
    "debt_to_income": {
        "safe_max": 0.36,
        "warning_max": 0.43,
        "direction": "lower_is_better",
        "safe_label": "<= 36%",
        "warning_label": "36% – 43%",
        "critical_label": "> 43%",
    },
    "quick_ratio": {
        "safe_min": 1.0,
        "warning_min": 0.7,
        "direction": "higher_is_better",
        "safe_label": ">= 1.0",
        "warning_label": "0.7 – 1.0",
        "critical_label": "< 0.7",
    },
}


class FinancialRatioEngine:
    """Stateless engine that computes and interprets financial ratios.

    Every ratio method validates its inputs, logs a warning on invalid
    data (e.g. division by zero), and returns ``None`` in that case.
    Valid results are rounded to 4 decimal places.
    """

    # ── Ratio methods ───────────────────────────────────────────────

    @staticmethod
    def debt_to_equity(
        total_debt: float,
        total_equity: float,
    ) -> Optional[float]:
        """Compute the Debt-to-Equity ratio.

        .. math:: D/E = \\frac{\\text{Total Debt}}{\\text{Total Equity}}

        A lower ratio indicates less financial leverage and lower risk.

        Parameters
        ----------
        total_debt:
            Sum of short-term and long-term debt.
        total_equity:
            Total shareholder equity.

        Returns
        -------
        float or None
            The ratio rounded to 4 dp, or ``None`` if *total_equity*
            is zero.
        """
        if total_equity == 0:
            logger.warning(
                "debt_to_equity: total_equity is 0 — cannot compute ratio"
            )
            return None
        return round(total_debt / total_equity, 4)

    @staticmethod
    def current_ratio(
        current_assets: float,
        current_liabilities: float,
    ) -> Optional[float]:
        """Compute the Current Ratio.

        .. math:: CR = \\frac{\\text{Current Assets}}{\\text{Current Liabilities}}

        Measures short-term liquidity; a value above 1.5 is generally
        considered healthy.

        Parameters
        ----------
        current_assets:
            Total current assets.
        current_liabilities:
            Total current liabilities.

        Returns
        -------
        float or None
            The ratio rounded to 4 dp, or ``None`` if
            *current_liabilities* is zero.
        """
        if current_liabilities == 0:
            logger.warning(
                "current_ratio: current_liabilities is 0 — cannot compute ratio"
            )
            return None
        return round(current_assets / current_liabilities, 4)

    @staticmethod
    def profit_margin(
        net_income: float,
        revenue: float,
    ) -> Optional[float]:
        """Compute the Net Profit Margin.

        .. math:: PM = \\frac{\\text{Net Income}}{\\text{Revenue}}

        Expressed as a decimal (0.10 = 10%).

        Parameters
        ----------
        net_income:
            Net income after taxes.
        revenue:
            Total revenue / net sales.

        Returns
        -------
        float or None
            The ratio rounded to 4 dp, or ``None`` if *revenue* is
            zero.
        """
        if revenue == 0:
            logger.warning(
                "profit_margin: revenue is 0 — cannot compute ratio"
            )
            return None
        return round(net_income / revenue, 4)

    @staticmethod
    def debt_to_income(
        total_debt: float,
        gross_income: float,
    ) -> Optional[float]:
        """Compute the Debt-to-Income ratio.

        .. math:: DTI = \\frac{\\text{Total Debt}}{\\text{Gross Income}}

        Commonly used in credit risk assessment.  Values above 0.43
        are generally considered high-risk.

        Parameters
        ----------
        total_debt:
            Total outstanding debt obligations.
        gross_income:
            Annual gross income.

        Returns
        -------
        float or None
            The ratio rounded to 4 dp, or ``None`` if *gross_income*
            is zero.
        """
        if gross_income == 0:
            logger.warning(
                "debt_to_income: gross_income is 0 — cannot compute ratio"
            )
            return None
        return round(total_debt / gross_income, 4)

    @staticmethod
    def quick_ratio(
        cash: float,
        short_term_investments: float,
        receivables: float,
        current_liabilities: float,
    ) -> Optional[float]:
        """Compute the Quick (Acid-Test) Ratio.

        .. math::

            QR = \\frac{\\text{Cash} + \\text{Short-term Investments}
                        + \\text{Receivables}}
                       {\\text{Current Liabilities}}

        A stricter liquidity measure than the current ratio because it
        excludes inventories and prepaid expenses.

        Parameters
        ----------
        cash:
            Cash and cash equivalents.
        short_term_investments:
            Marketable securities / short-term investments.
        receivables:
            Accounts receivable (net).
        current_liabilities:
            Total current liabilities.

        Returns
        -------
        float or None
            The ratio rounded to 4 dp, or ``None`` if
            *current_liabilities* is zero.
        """
        if current_liabilities == 0:
            logger.warning(
                "quick_ratio: current_liabilities is 0 — cannot compute ratio"
            )
            return None
        numerator = cash + short_term_investments + receivables
        return round(numerator / current_liabilities, 4)

    # ── Interpretation ──────────────────────────────────────────────

    @staticmethod
    def interpret_ratio(
        ratio_name: str,
        value: Optional[float],
    ) -> Dict[str, object]:
        """Interpret a computed ratio against industry-standard thresholds.

        Parameters
        ----------
        ratio_name:
            One of ``"debt_to_equity"``, ``"current_ratio"``,
            ``"profit_margin"``, ``"debt_to_income"``, ``"quick_ratio"``.
        value:
            The computed ratio value.  If ``None``, the returned status
            will be ``"error"``.

        Returns
        -------
        dict
            ::

                {
                    "value":     float | None,
                    "status":    "safe" | "warning" | "critical" | "error",
                    "threshold": str,   # human-readable threshold description
                }

        Raises
        ------
        ValueError
            If *ratio_name* is not recognised.
        """
        if ratio_name not in _THRESHOLDS:
            raise ValueError(
                f"Unknown ratio '{ratio_name}'. "
                f"Known ratios: {list(_THRESHOLDS.keys())}"
            )

        if value is None:
            return {
                "value": None,
                "status": "error",
                "threshold": "unable to compute (invalid inputs)",
            }

        cfg = _THRESHOLDS[ratio_name]

        if cfg["direction"] == "lower_is_better":
            # Lower values are safer.
            if value <= cfg["safe_max"]:
                status = "safe"
                threshold = cfg["safe_label"]
            elif value <= cfg["warning_max"]:
                status = "warning"
                threshold = cfg["warning_label"]
            else:
                status = "critical"
                threshold = cfg["critical_label"]
        else:
            # Higher values are safer.
            if value >= cfg["safe_min"]:
                status = "safe"
                threshold = cfg["safe_label"]
            elif value >= cfg["warning_min"]:
                status = "warning"
                threshold = cfg["warning_label"]
            else:
                status = "critical"
                threshold = cfg["critical_label"]

        return {
            "value": value,
            "status": status,
            "threshold": threshold,
        }
