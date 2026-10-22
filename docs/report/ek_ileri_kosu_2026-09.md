# bistmas monthly decision — as of 2026-09-30

Planned execution session: **2026-10-21**  
Run: `forward_20260930` · code `c5ff9eb50e` · model `qwen3.5:9b` @ `6488c96fa5fa`  
Point-in-time check: OK (100 feature rows)

## Top picks (hybrid: jury score J)

| rank | ticker | J | votes | composite rank | sector |
|---|---|---|---|---|---|
| 1 | TUPRS | 0.883 | 3 | 3 | MANUFACTURING |
| 2 | KTLEV | 0.867 | 3 | 1 | OTHER_FINANCIAL |
| 3 | THYAO | 0.850 | 3 | 4 | TRANSPORT |

## Deterministic Top-10 (composite)

| rank | ticker | composite | fundamental | valuation | momentum/risk |
|---|---|---|---|---|---|
| 1 | KTLEV | 1.25 | 2.32 | 0.79 | 0.63 |
| 2 | TSKB | 1.10 | 0.64 | 3.00 | -0.33 |
| 3 | TUPRS | 1.08 | 1.06 | 0.94 | 1.25 |
| 4 | THYAO | 0.96 | 0.31 | 1.75 | 0.81 |
| 5 | TRALT | 0.89 | 2.08 | -0.22 | 0.79 |
| 6 | DOHOL | 0.88 | 0.92 | 1.87 | -0.15 |
| 7 | TCELL | 0.87 | 0.17 | 1.92 | 0.51 |
| 8 | TRMET | 0.82 | 1.98 | 0.38 | 0.11 |
| 9 | TTKOM | 0.79 | 0.33 | 1.69 | 0.35 |
| 10 | KRDMD | 0.70 | -0.51 | 1.98 | 0.64 |

## Jury rationales (masked inputs; names re-attached after scoring)

**KTLEV — quality_growth** (score 0.95): This company demonstrates exceptional capital efficiency and profit growth relative to its financials peer group, evidenced by perfect percentile ranks for ROE and profit growth alongside high z-scores. However, the lack of additional dimensions such as margin stability or return consistency prevents a complete assessment of long-term operating durability. Consequently, while the available data points are outstanding, the narrow metric set warrants a slightly tempered confidence in the overall quality verdict.

**TUPRS — quality_growth** (score 0.85): This company demonstrates exceptional sustainable growth driven by superior profit expansion and robust capital returns, evidenced by the high percentiles for profit_growth_pct and roic_pct. However, the broad-based strength is tempered by mediocre gross margin performance, which suggests potential pressure on pricing power or cost control relative to peers. Consequently, while the growth trajectory is outstanding, the overall quality score must be moderated by the lack of excellence in core margin metrics.

**THYAO — quality_growth** (score 0.85): The company demonstrates superior growth consistency with a perfect revenue growth percentile and strong profit expansion, supported by healthy gross and operating margins that place it in the top third of its peer group. However, the ROIC metrics, while positive, lack the exceptional magnitude seen in the growth figures, suggesting that high growth has not yet fully translated into world-class capital efficiency relative to peers. This broad-based strength in margins and growth justifies a high score, though the moderate ROIC prevents it from reaching the highest tier.

**KTLEV — value** (score 0.90): STOCK_E presents a compelling margin of safety primarily driven by its top-tier earnings yield, which suggests the stock is priced well below peer averages. The neutral book-to-market percentile supports this view without indicating severe distress, though missing cash-flow and EV metrics prevent a definitive confirmation of value.

**TUPRS — value** (score 0.85): The company demonstrates a compelling margin of safety driven primarily by its superior free cash flow generation, which ranks in the top tier of peers. While the earnings yield is also attractive, the relatively weak book-to-market metric prevents this from being a universally cheap valuation across all dimensions. Consequently, the investment case relies heavily on the robustness of the cash-flow yield rather than a broad consensus of undervaluation.

**THYAO — value** (score 0.95): The company exhibits a compelling margin of safety, as earnings yield, book-to-market, and EBITDA yield all rank at the 100th percentile, indicating it is significantly cheaper than peers. However, the free cash flow yield lags slightly at the 83rd percentile, suggesting that while asset pricing is attractive, cash generation relative to value may be less pronounced than earnings or book value imply.

**KTLEV — risk_timing** (score 0.75): The stock exhibits exceptional momentum with a z-score of 3.0 and a perfect 100th percentile rank, alongside superior tradability indicated by an advanced 6-month liquidity z-score of 0.99. However, the risk profile is mixed as volatility and maximum drawdown metrics show negative z-scores (25th percentile), suggesting the strong trend is supported by below-average stability rather than robustness. Given the recent statement age, this high-momentum setup with moderate volatility risks warrants a cautious hold stance.

**TUPRS — risk_timing** (score 0.95): The stock exhibits exceptional strength with a momentum_12_1_z of 1.83 and an adv_6m_tl_z of 1.86, indicating robust upward trend and deep liquidity. Risk metrics are also favorable, showing low volatility (vol_6m_pct: 70) and shallow drawdowns (mdd_6m_pct: 83), which supports a confident holding recommendation.

**THYAO — risk_timing** (score 0.75): The stock demonstrates excellent tradability with a perfect percentile rank for advance-decline liquidity (adv_6m_tl_pct) and superior drawdown resilience (mdd_6m_pct). However, the momentum z-score is moderate at 0.46, indicating that while not weak, the upward trend strength is not exceptional compared to peers.


Decision file sha256: `d396e05ae66740dcde5f00a3c08cc7e2a619c9788bac841f02959ff94ee4d9d1`

This is a research prototype (TÜBİTAK 2209-A); it is not investment advice.