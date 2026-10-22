"""Fundamental ratios from one filing row (``statements_view`` wide + TTM columns) and market data.

All ratios are *yields* or *rates* oriented so that a larger value is better for the investor
(earnings yield instead of P/E, EBITDA yield instead of EV/EBITDA, ...), which keeps the normalisation
sign-free. Growth uses the same-filing comparative column, so it is basis-consistent under TMS-29.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

RATIO_COLUMNS = [
    "earnings_yield",
    "book_to_market",
    "ebitda_yield",
    "fcf_yield",
    "roic",
    "roe",
    "gross_margin",
    "operating_margin",
    "revenue_growth",
    "profit_growth",
    "net_debt_to_ebitda",
    "equity_to_assets",
    "interest_coverage",
    "cost_to_income",
    "nim",
    "loans_to_deposits",
]


def _g(row: Any, key: str) -> float:
    v = row.get(key, np.nan) if hasattr(row, "get") else getattr(row, key, np.nan)
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def _safe_div(a: float, b: float, min_abs_b: float = 1e-9) -> float:
    if np.isnan(a) or np.isnan(b) or abs(b) < min_abs_b:
        return np.nan
    return a / b


def _growth(cur: float, prior: float) -> float:
    """Same-filing growth; undefined when the base is not positive (loss to profit etc.)."""
    if np.isnan(cur) or np.isnan(prior) or prior <= 0:
        return np.nan
    return cur / prior - 1.0


def compute_ratios(row: Any, mcap_tl: float, template: str, tax_rate: float = 0.25) -> dict[str, float]:
    """Ratio dict (``RATIO_COLUMNS``) for one filing row and the market cap on the decision date."""
    out: dict[str, float] = {c: np.nan for c in RATIO_COLUMNS}
    if np.isnan(mcap_tl) or mcap_tl <= 0:
        return out
    net_profit = _g(row, "net_profit_parent_ttm")
    equity = _g(row, "equity_parent_cur")
    assets = _g(row, "assets_cur")
    out["earnings_yield"] = _safe_div(net_profit, mcap_tl)
    out["book_to_market"] = _safe_div(equity, mcap_tl) if not np.isnan(equity) else np.nan
    out["roe"] = _safe_div(net_profit, equity) if equity > 0 else np.nan
    out["equity_to_assets"] = _safe_div(equity, assets) if assets > 0 else np.nan
    out["profit_growth"] = _growth(_g(row, "net_profit_parent_cur"), _g(row, "net_profit_parent_prior"))
    if template == "bank":
        nii, fee, opex = (
            _g(row, "net_interest_income_ttm"),
            _g(row, "net_fee_income_ttm"),
            _g(row, "operating_expenses_ttm"),
        )
        income = (nii if not np.isnan(nii) else 0.0) + (fee if not np.isnan(fee) else 0.0)
        out["cost_to_income"] = _safe_div(abs(opex), income) if not np.isnan(opex) and income > 0 else np.nan
        out["nim"] = _safe_div(nii, assets) if assets > 0 else np.nan
        out["loans_to_deposits"] = _safe_div(_g(row, "loans_cur"), _g(row, "deposits_cur"))
        return out
    if template != "industrial":
        return out
    revenue = _g(row, "revenue_ttm")
    ebit = _g(row, "ebit_ttm")
    da = _g(row, "depreciation_ttm")
    ebitda = ebit + abs(da) if not np.isnan(ebit) and not np.isnan(da) else (ebit if not np.isnan(ebit) else np.nan)
    debt = np.nansum([_g(row, "st_debt_cur"), _g(row, "st_debt_ltportion_cur"), _g(row, "lt_debt_cur")])
    if all(np.isnan(_g(row, k)) for k in ("st_debt_cur", "st_debt_ltportion_cur", "lt_debt_cur")):
        debt = np.nan
    cash = np.nansum([_g(row, "cash_cur"), _g(row, "st_investments_cur")])
    if np.isnan(_g(row, "cash_cur")):
        cash = np.nan
    net_debt = debt - cash if not np.isnan(debt) and not np.isnan(cash) else np.nan
    # claims of minority holders: flows/net debt are 100 % consolidated, so EV and invested capital include NCI
    equity_total = _g(row, "equity_cur")
    nci = max(equity_total - equity, 0.0) if not np.isnan(equity_total) and not np.isnan(equity) else 0.0
    ev = mcap_tl + nci + net_debt if not np.isnan(net_debt) else np.nan
    cfo = _g(row, "cfo_ttm")
    capex = _g(row, "capex_ttm")
    fcf = cfo - abs(capex) if not np.isnan(cfo) and not np.isnan(capex) else np.nan
    out["ebitda_yield"] = _safe_div(ebitda, ev) if not np.isnan(ev) and ev > 0 else np.nan
    # FCF does not depend on net debt: when net debt is unknown, fall back to FCF / (market cap + NCI)
    fcf_den = ev if not np.isnan(ev) else mcap_tl + nci
    out["fcf_yield"] = _safe_div(fcf, fcf_den) if not np.isnan(fcf_den) and fcf_den > 0 else np.nan
    eq_inv = equity_total if not np.isnan(equity_total) else equity
    invested = (eq_inv if not np.isnan(eq_inv) else 0.0) + (net_debt if not np.isnan(net_debt) else 0.0)
    out["roic"] = _safe_div(ebit * (1 - tax_rate), invested) if not np.isnan(ebit) and invested > 0 else np.nan
    out["gross_margin"] = _safe_div(_g(row, "gross_profit_ttm"), revenue) if revenue > 0 else np.nan
    out["operating_margin"] = _safe_div(_g(row, "operating_profit_ttm"), revenue) if revenue > 0 else np.nan
    out["revenue_growth"] = _growth(_g(row, "revenue_cur"), _g(row, "revenue_prior"))
    out["net_debt_to_ebitda"] = _safe_div(net_debt, ebitda) if not np.isnan(ebitda) and ebitda > 0 else np.nan
    interest = abs(_g(row, "interest_expense_ttm"))
    out["interest_coverage"] = _safe_div(ebit, interest) if interest > 0 else np.nan
    return out


def ratios_frame(
    latest: pd.DataFrame, market: pd.DataFrame, tax_rate_by_year: dict[int, float] | None = None
) -> pd.DataFrame:
    """Ratios for every ticker in ``latest`` (one filing per ticker) joined with ``market`` (index ticker, mcap_tl)."""
    records = []
    for row in latest.to_dict("records"):
        t = row["ticker"]
        mcap = float(market.loc[t, "mcap_tl"]) if t in market.index else np.nan
        year = int(row.get("fiscal_year") or 0)
        tax = (tax_rate_by_year or {}).get(year, 0.25)
        rec = {"ticker": t, **compute_ratios(row, mcap, str(row.get("template", "industrial")), tax)}
        rec["statement_period_end"] = row.get("period_end")
        rec["statement_availability_date"] = row.get("availability_date")
        rec["ttm_method"] = row.get("ttm_method", "")
        rec["basis"] = row.get("basis", "")
        records.append(rec)
    return pd.DataFrame(
        records,
        columns=[
            "ticker",
            *RATIO_COLUMNS,
            "statement_period_end",
            "statement_availability_date",
            "ttm_method",
            "basis",
        ],
    )
