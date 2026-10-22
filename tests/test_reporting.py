from datetime import date

import pandas as pd

from bistmas.reporting.certificate import pit_check
from bistmas.reporting.email import send_report
from bistmas.reporting.render import render_decision_markdown


def test_pit_check_flags_rows_after_as_of():
    df = pd.DataFrame({"max_source_available_date": pd.to_datetime(["2019-06-27", "2019-06-28", "2019-07-01"])})
    res = pit_check(df, pd.Timestamp("2019-06-28"))
    assert res["rows"] == 3 and res["rows_after_as_of"] == 1 and not res["ok"]
    assert pit_check(df.iloc[:2], pd.Timestamp("2019-06-28"))["ok"]


def test_render_and_dry_run_email(tmp_path):
    decision = {
        "as_of": "2019-06-28",
        "planned_execution_session": "2019-07-01",
        "run_id": "forward_201906",
        "certificate": {
            "code_commit": "abcdef1234567890",
            "model": {"name": "qwen3.5:9b", "digest_pinned": "6488c96fa5fa"},
            "pit": {"ok": True, "rows": 100},
        },
        "top": [{"rank_J": 1, "ticker": "ABCD", "J": 0.81, "n_votes": 3, "composite_rank": 2, "sector_group": "BANKS"}],
        "screen_top": [
            {
                "rank": 1,
                "ticker": "EFGH",
                "composite": 1.23,
                "S_fundamental": 1.0,
                "S_valuation": None,
                "S_momentum_risk": 0.5,
            }
        ],
        "rationales": [{"ticker": "ABCD", "agent": "value", "score": 0.8, "rationale": "book_to_market_pct is high"}],
        "sha256": "deadbeef",
    }
    md = render_decision_markdown(decision)
    assert "ABCD" in md and "0.810" in md and "not investment advice" in md and "deadbeef" in md
    status = send_report("subject", md, "someone@example.com", "smtp.example.com", 587, dry_run=True)
    assert status.startswith("not sent")
    assert date.fromisoformat(decision["as_of"]) == date(2019, 6, 28)
