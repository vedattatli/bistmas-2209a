"""XU100 secondary benchmark: the July-2020 /100 rescale is spliced only when the series really jumps."""

import numpy as np
import pandas as pd

from bistmas.backtest.benchmark import index_daily


def _panel(levels: list[float], start: str = "2020-07-20") -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=len(levels))
    return pd.DataFrame({"date": dates, "ticker": "X", "xu100": levels})


def test_vendor_series_already_in_new_scale_is_left_alone():
    # İş Yatırım's END_DEGER restates the whole history: 1190 -> 1195 across 2020-07-27, no jump
    out = index_daily(_panel([1180.0, 1185.0, 1190.0, 1192.0, 1188.0, 1195.7, 1152.7, 1135.5]))
    assert np.allclose(out["level"].values, [1180.0, 1185.0, 1190.0, 1192.0, 1188.0, 1195.7, 1152.7, 1135.5])
    assert abs(out.loc[out["date"] == pd.Timestamp("2020-07-27"), "ret"].iloc[0]) < 0.05


def test_old_scale_history_is_divided_by_100_at_the_rescale():
    out = index_daily(_panel([118000.0, 118500.0, 119000.0, 119200.0, 118800.0, 1195.7, 1152.7, 1135.5]))
    assert np.allclose(out["level"].values[:5], [1180.0, 1185.0, 1190.0, 1192.0, 1188.0])
    assert abs(out.loc[out["date"] == pd.Timestamp("2020-07-27"), "ret"].iloc[0]) < 0.05
