"""Smoke test: figures are produced from a tiny run directory without a display."""

import json

import numpy as np
import pandas as pd

from bistmas.reporting.figures import write_figures


def test_write_figures_on_tiny_run(tmp_path):
    run = tmp_path / "runs" / "toy"
    run.mkdir(parents=True)
    dates = pd.bdate_range("2018-01-02", periods=40)
    for name, drift in (("det_top3", 0.001), ("bench_ew_pit", 0.0005)):
        ret = np.full(len(dates), drift)
        pd.DataFrame({"date": dates, "ret": ret, "nav": np.cumprod(1 + ret)}).to_csv(
            run / f"{name}_daily.csv", index=False
        )
    (run / "report.json").write_text(
        json.dumps(
            {
                "run": "toy",
                "tables": {
                    "calendar_year": [{"year": 2018, "det_top3": 0.1, "bench_ew_pit": 0.05, "active_det_top3": 0.05}]
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "runs" / "jury_cache").mkdir()
    usdtry = pd.Series(np.linspace(3.8, 4.2, len(dates)), index=dates)
    out = write_figures(tmp_path / "runs", usdtry)
    names = sorted(p.name for p in out)
    assert names == ["calendar_year_active.png", "calendar_year_active.svg", "nav.png", "nav.svg"]
    assert all(p.stat().st_size > 1000 for p in out)
