import pandas as pd

from bistmas.jury.audits.groundedness import scan_rationale
from bistmas.jury.audits.probes import ARMS, date_arm_context


def test_scan_rationale_grounding_and_leaks():
    ok = scan_rationale("roic_pct is high and vol_6m_z is low", {"roic_pct", "vol_6m_z"}, {"THYAO"})
    assert ok["grounded"] and not ok["leak_flag"] and ok["n_metric_tokens"] == 2
    bad = scan_rationale("earnings_yield_pct looks cheap for THYAO in 2021 given the pandemic", {"roic_pct"}, {"THYAO"})
    assert not bad["grounded"] and bad["ungrounded"] == ["earnings_yield_pct"]
    assert bad["leak_flag"] and "2021" in bad["leaks"] and "THYAO" in bad["leaks"]


def test_date_arm_contexts():
    d = pd.Timestamp("2019-06-28")
    assert date_arm_context("masked", d, "THYAO", "Türk Hava Yolları") == ""
    assert "2019-06-28" in date_arm_context("date_only", d, "THYAO", "x")
    assert "THYAO" in date_arm_context("revealed", d, "THYAO", "x")
    assert "2022-06-28" in date_arm_context("transplanted_plus3y", d, "THYAO", "x")
    assert "2012-06-28" in date_arm_context("shifted_minus7y", d, "THYAO", "x")
    assert ARMS[0] == "masked" and len(ARMS) == 5
