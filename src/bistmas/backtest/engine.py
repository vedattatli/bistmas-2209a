"""Walk-forward monthly backtest with next-session VWAP fills.

Timeline for decision date ``t_k`` (last session of the month):
    decide at t_k (information available up to the close) -> fill at VWAP of the next session ``e_k``
    -> hold until ``e_{k+1}`` where the book is rebalanced at that session's VWAP.

Returns are computed on ADJUSTED series (dividends and capital changes are total-return events), costs on
turnover at ``cost_bps_one_way`` per side. A holding whose price series ends before ``e_{k+1}``
(delisting, merger) is liquidated at its last adjusted close times ``(1 - delisting_haircut)`` and the
proceeds stay in cash (0 %) until the next rebalance. The engine is selector-agnostic: it receives a
callable ``selector(decision_date) -> list[str]`` and equal-weights the names it returns.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from bistmas.shared.calendar import TradingCalendar
from bistmas.shared.log import get_logger

log = get_logger("backtest.engine")

Selector = Callable[[pd.Timestamp], list[str]]


@dataclass
class BacktestConfig:
    first_decision: date
    last_decision: date
    cost_bps_one_way: float = 30.0
    delisting_haircut: float = 0.0
    top_n: int | None = None  # None = equal-weight everything the selector returns
    label: str = "strategy"


@dataclass
class BacktestResult:
    label: str
    periods: pd.DataFrame  # one row per holding period
    daily: pd.DataFrame  # date, ret, nav
    holdings: pd.DataFrame  # decision_date, ticker, weight, fill_price, exit_price, period_ret
    config: dict[str, Any] = field(default_factory=dict)


class Backtester:
    def __init__(self, calendar: TradingCalendar, vwap_adj: pd.DataFrame, close_adj: pd.DataFrame) -> None:
        self.calendar = calendar
        # a zero VWAP means no trade took place (halt, suspension): nothing can be filled at that price
        self.vwap = vwap_adj.sort_index().where(vwap_adj.sort_index() > 0)
        self.close = close_adj.sort_index()

    # ------------------------------------------------------------------ schedule
    def decision_dates(self, first: date, last: date) -> list[date]:
        return [d for d in self.calendar.month_ends(first, last) if first <= d <= last]

    def exec_session(self, decision: date) -> date:
        return self.calendar.next_session(decision, strict=True)

    # ------------------------------------------------------------------ core
    def run(self, selector: Selector, cfg: BacktestConfig) -> BacktestResult:
        decisions = self.decision_dates(cfg.first_decision, cfg.last_decision)
        if len(decisions) < 2:
            raise ValueError("need at least two decision dates")
        execs = [self.exec_session(d) for d in decisions]
        # the last holding period ends at the session after the final decision + one month: use the next
        # month-end's execution session if it exists in the price data, else the last available session
        final_exit = self._final_exit(execs[-1])
        period_rows: list[dict[str, Any]] = []
        holding_rows: list[dict[str, Any]] = []
        daily_rets: list[pd.Series] = []
        prev_weights: dict[str, float] = {}
        cost_rate = cfg.cost_bps_one_way / 1e4
        nav = 1.0
        for k, (t, e) in enumerate(zip(decisions, execs, strict=True)):
            e_next = execs[k + 1] if k + 1 < len(execs) else final_exit
            picks = list(selector(pd.Timestamp(t)))
            if cfg.top_n is not None:
                picks = picks[: cfg.top_n]
            # A selected name that never trades in [e, e_next) (halted all month, or no price data at all) keeps
            # its 1/N slice as cash (0 % return); the other weights are never re-normalised, because whether a
            # name trades later in the month is information not available at the decision date.
            idle = [p for p in picks if self._first_fill(p, e, e_next) is None]
            if idle:
                log.warning(
                    "%s %s: %d selected name(s) never trade in the holding period, held as cash: %s",
                    cfg.label,
                    t,
                    len(idle),
                    ",".join(idle[:10]),
                )
            weights = {p: 1.0 / len(picks) for p in picks} if picks else {}
            turnover = 0.5 * sum(
                abs(weights.get(x, 0.0) - prev_weights.get(x, 0.0)) for x in set(weights) | set(prev_weights)
            )
            cost = 2 * turnover * cost_rate  # buys and sells each pay one way
            gross, drifted, rets_daily, per_name = self._period(weights, e, e_next, cfg.delisting_haircut)
            net = (1 + gross) * (1 - cost) - 1
            nav_start = nav
            nav *= 1 + net
            period_rows.append(
                {
                    "decision_date": pd.Timestamp(t),
                    "exec_date": pd.Timestamp(e),
                    "exit_date": pd.Timestamp(e_next),
                    "n_holdings": len(picks),
                    "turnover": turnover,
                    "gross_ret": gross,
                    "cost": cost,
                    "net_ret": net,
                    "nav_start": nav_start,
                    "nav_end": nav,
                    "holdings": ",".join(picks),
                }
            )
            for p, info in per_name.items():
                holding_rows.append({"decision_date": pd.Timestamp(t), "ticker": p, "weight": weights[p], **info})
            # spread the cost over the first day of the period so daily NAV reconciles with period NAV
            if len(rets_daily):
                rets_daily = rets_daily.copy()
                rets_daily.iloc[0] = (1 + rets_daily.iloc[0]) * (1 - cost) - 1
            daily_rets.append(rets_daily)
            prev_weights = drifted
        periods = pd.DataFrame(period_rows)
        if daily_rets:
            stacked = pd.concat(daily_rets)
            stacked.index.name = "date"
            daily = ((1 + stacked).groupby(level=0).prod() - 1).to_frame("ret").sort_index()
        else:
            daily = pd.DataFrame(columns=["ret"])
            daily.index.name = "date"
        daily["nav"] = (1 + daily["ret"]).cumprod()
        daily = daily.reset_index()
        return BacktestResult(cfg.label, periods, daily, pd.DataFrame(holding_rows), cfg.__dict__.copy())

    # ------------------------------------------------------------------ helpers
    def _final_exit(self, last_exec: date) -> date:
        sessions = self.close.index
        after = sessions[sessions > pd.Timestamp(last_exec)]
        if len(after) == 0:
            return last_exec
        # one calendar month after the last execution (or the last available session)
        target = pd.Timestamp(last_exec) + pd.DateOffset(months=1)
        cand = after[after <= target]
        return (cand[-1] if len(cand) else after[0]).date()

    def _first_fill(self, ticker: str, e: date, e_next: date) -> tuple[pd.Timestamp, float] | None:
        """First session in ``[e, e_next)`` with a positive VWAP, or ``None`` when the name never trades.

        A halted or suspended stock (zero VWAP, zero volume) is filled at the first session it trades again
        within the holding period; the slice is held as cash until then.
        """
        if ticker not in self.vwap.columns:
            return None
        e_ts, x_ts = pd.Timestamp(e), pd.Timestamp(e_next)
        window = self.vwap[ticker].loc[(self.vwap.index >= e_ts) & (self.vwap.index < x_ts)].dropna()
        if x_ts == e_ts:  # degenerate final period
            window = self.vwap[ticker].loc[self.vwap.index == e_ts].dropna()
        if len(window) == 0:
            return None
        return window.index[0], float(window.iloc[0])

    def _at(self, frame: pd.DataFrame, ticker: str, d: date) -> float:
        ts = pd.Timestamp(d)
        if ticker not in frame.columns or ts not in frame.index:
            return np.nan
        return float(frame.at[ts, ticker])

    def _period(
        self, weights: dict[str, float], e: date, e_next: date, haircut: float
    ) -> tuple[float, dict[str, float], pd.Series, dict[str, dict[str, Any]]]:
        """Gross period return, drifted end weights, daily return series and per-name fills.

        Value path per name relative to its fill: 1.0 at the fill (VWAP of ``e``), close/fill on every
        session up to the day before the exit, and exit/fill on the exit session (VWAP of ``e_next``).
        The daily series therefore covers ``e .. e_next`` inclusive; the exit session's return is the
        leg close(e_next - 1) -> VWAP(e_next). The caller compounds it with the next period's fill-day
        return that lands on the same date.
        """
        e_ts, x_ts = pd.Timestamp(e), pd.Timestamp(e_next)
        sessions = self.close.index[(self.close.index >= e_ts) & (self.close.index <= x_ts)]
        if not weights or len(sessions) == 0:
            zero = pd.Series(0.0, index=sessions if len(sessions) else pd.DatetimeIndex([e_ts]))
            return 0.0, {}, zero, {}
        per_name: dict[str, dict[str, Any]] = {}
        gross_rets: dict[str, float] = {}
        paths: dict[str, pd.Series] = {}
        for p in weights:
            first = self._first_fill(p, e, e_next)
            if first is None:  # never tradable before the exit session: the slice stays in cash
                gross_rets[p] = 0.0
                paths[p] = pd.Series(1.0, index=sessions)
                per_name[p] = {"fill_price": np.nan, "exit_price": np.nan, "period_ret": 0.0, "liquidated": False}
                continue
            fill_day, fill = first
            closes = self.close.loc[sessions, p].astype(float)
            closes.loc[closes.index < fill_day] = fill  # cash until the (possibly delayed) fill: worth 1.0
            exit_px = self._at(self.vwap, p, e_next)
            liquidated = False
            if np.isnan(exit_px) or x_ts == e_ts:
                last_close = closes.dropna()
                exit_px = float(last_close.iloc[-1]) * (1 - haircut) if len(last_close) else fill
                liquidated = True
            path = closes.copy()
            if liquidated:
                traded = closes.dropna()
                last_idx = traded.index[-1] if len(traded) else sessions[0]
                path.loc[path.index >= last_idx] = exit_px  # cash at the liquidation value from the last trade on
                path = path.ffill().fillna(exit_px)
            else:
                path = path.ffill()
                path.loc[x_ts] = exit_px
            paths[p] = path / fill
            r = exit_px / fill - 1
            gross_rets[p] = r
            per_name[p] = {
                "fill_price": fill,
                "exit_price": exit_px,
                "period_ret": r,
                "liquidated": liquidated,
                "fill_date": fill_day,
            }
        values = pd.DataFrame(paths).ffill().fillna(1.0)
        port = sum(values[p] * w for p, w in weights.items())
        prev = port.shift(1)
        prev.iloc[0] = 1.0  # the book is worth exactly 1.0 at the fill
        daily = port / prev - 1.0
        gross = float(port.iloc[-1]) - 1.0
        end_val = {p: w * (1 + gross_rets[p]) for p, w in weights.items()}
        total = sum(end_val.values())
        # cash slices (names that never traded) are not positions: they stay out of the drifted weights
        drifted = (
            {p: v / total for p, v in end_val.items() if not np.isnan(per_name[p]["fill_price"])} if total > 0 else {}
        )
        return gross, drifted, daily, per_name
