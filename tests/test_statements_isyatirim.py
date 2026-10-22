import json
from pathlib import Path

import numpy as np
import pandas as pd

from bistmas.data.statements_isyatirim import add_prior_from_previous_year, parse_year

FIX = Path(__file__).parent / "fixtures" / "isyatirim_fin" / "TRKCM_2019.json"


def test_parse_year_maps_codes_and_fallback_availability():
    items = json.loads(FIX.read_text(encoding="utf-8"))["value"]
    w = parse_year(items, "TRKCM", 2019)
    assert list(w["period"]) == [1, 2, 3, 4]
    fy = w[w["period"] == 4].iloc[0]
    assert fy["revenue_cur"] == 6_650_648_000 and fy["assets_cur"] == 14_907_645_000 and fy["cfo_cur"] == 1_140_852_000
    assert fy["period_end"] == pd.Timestamp("2019-12-31") and fy["availability_date"] == pd.Timestamp(
        "2020-04-29"
    )  # +90 +30 COVID
    q1 = w[w["period"] == 1].iloc[0]
    assert q1["availability_date"] == pd.Timestamp("2019-05-15")
    assert w["source"].eq("isyatirim_fallback").all() and (w["disclosure_index"] < 0).all()
    prev = w.copy()
    prev["fiscal_year"] = 2018
    prev["revenue_cur"] = prev["revenue_cur"] * 0.8
    both = add_prior_from_previous_year(pd.concat([prev, w], ignore_index=True))
    cur = both[(both["fiscal_year"] == 2019) & (both["period"] == 4)].iloc[0]
    assert cur["revenue_prior"] == 6_650_648_000 * 0.8 and np.isnan(
        both[both["fiscal_year"] == 2018]["revenue_prior"].iloc[0]
    )
