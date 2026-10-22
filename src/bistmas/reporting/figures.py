"""Paper figures from finished runs: NAV curves (TL and USD), calendar-year active returns, jury-score
diagnostics. PNG + SVG under ``data/runs/<run>/figures``; Matplotlib only, no seaborn."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from bistmas.shared.log import get_logger  # noqa: E402

log = get_logger("reporting.figures")

COLOURS = {
    "bench_ew_pit": "#444444",
    "det_top3": "#1f77b4",
    "det_top5": "#6baed6",
    "det_top10": "#c6dbef",
    "hybrid_top3": "#d62728",
    "hybrid_top5": "#fb6a4a",
    "hybrid_top10": "#fcae91",
}


def _save(fig: plt.Figure, out_dir: Path, name: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "svg"):
        p = out_dir / f"{name}.{ext}"
        fig.savefig(p, dpi=160 if ext == "png" else None, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths


def nav_figure(run_dir: Path, usdtry: pd.Series | None = None, log_scale: bool = True) -> list[Path]:
    """Cumulative NAV of every strategy in the run (from the ``*_daily.csv`` files) in TL and, if a USD/TRY
    series is given, in USD."""
    dailies = {
        p.name[: -len("_daily.csv")]: pd.read_csv(p, parse_dates=["date"]) for p in sorted(run_dir.glob("*_daily.csv"))
    }
    dailies = {k: v for k, v in dailies.items() if not k.startswith("random_")}
    if not dailies:
        return []
    panels = [("TL", None)] + ([("USD", usdtry)] if usdtry is not None else [])
    fig, axes = plt.subplots(1, len(panels), figsize=(6.5 * len(panels), 4.2), squeeze=False)
    for ax, (label, conv) in zip(axes[0], panels, strict=True):
        for name, d in dailies.items():
            nav = d.set_index("date")["nav"]
            if conv is not None:
                c = conv.reindex(nav.index).ffill().bfill()
                nav = nav / c * c.iloc[0]
            ax.plot(
                nav.index,
                nav.values,
                label=name,
                color=COLOURS.get(name),
                lw=2 if "top3" in name or name.startswith("bench") else 1.2,
            )
        ax.set_title(f"Cumulative NAV ({label}, net of costs)")
        if log_scale:
            ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    return _save(fig, run_dir / "figures", "nav")


def calendar_year_figure(run_dir: Path) -> list[Path]:
    """Bar chart of calendar-year active returns vs the benchmark from ``report.json['tables']``."""
    rep = run_dir / "report.json"
    if not rep.exists():
        return []
    data: dict[str, Any] = json.loads(rep.read_text(encoding="utf-8"))
    rows = (data.get("tables") or {}).get("calendar_year") or []
    if not rows:
        return []
    t = pd.DataFrame(rows).set_index("year")
    cols = [c for c in t.columns if c.startswith("active_")]
    if not cols:
        return []
    fig, ax = plt.subplots(figsize=(7, 3.8))
    width = 0.8 / len(cols)
    for i, c in enumerate(cols):
        ax.bar(
            t.index + (i - (len(cols) - 1) / 2) * width,
            t[c].values,
            width,
            label=c.replace("active_", ""),
            color=COLOURS.get(c.replace("active_", "")),
        )
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("active return vs PIT equal weight")
    ax.set_title("Calendar-year active returns (net)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    return _save(fig, run_dir / "figures", "calendar_year_active")


def jury_figure(jury_run_dir: Path) -> list[Path]:
    """Histogram of jury scores per agent and the J-vs-composite-rank scatter (echo check)."""
    scores = jury_run_dir / "jury_scores.parquet"
    cons = jury_run_dir / "jury_consensus.parquet"
    if not scores.exists() or not cons.exists():
        return []
    s = pd.read_parquet(scores)
    c = pd.read_parquet(cons)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for agent, g in s.dropna(subset=["score"]).groupby("agent"):
        axes[0].hist(g["score"], bins=20, alpha=0.5, label=f"{agent} (n={len(g)})")
    axes[0].set_title("Jury scores by agent")
    axes[0].legend(fontsize=8)
    cc = c.dropna(subset=["J"]).copy()
    cc["composite_rank"] = pd.to_numeric(cc["composite_rank"], errors="coerce")
    axes[1].scatter(cc["composite_rank"], cc["J"], s=8, alpha=0.4)
    axes[1].set_xlabel("composite rank within the month's screen")
    axes[1].set_ylabel("J (mean jury score)")
    axes[1].set_title("Echo check: J vs composite rank")
    for ax in axes:
        ax.grid(alpha=0.3)
    return _save(fig, jury_run_dir / "figures", "jury_scores")


def write_figures(runs_dir: Path, usdtry: pd.Series | None = None) -> list[Path]:
    out: list[Path] = []
    for run_dir in sorted(Path(runs_dir).iterdir()):
        if not run_dir.is_dir() or run_dir.name in ("jury_cache", "jury_cache_smoke", "results", "audits"):
            continue
        if any(run_dir.glob("*_daily.csv")):
            out += nav_figure(run_dir, usdtry)
            out += calendar_year_figure(run_dir)
        if (run_dir / "jury_consensus.parquet").exists():
            out += jury_figure(run_dir)
    log.info("figures written: %d files", len(out))
    return out
