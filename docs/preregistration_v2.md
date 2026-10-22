# Pre-registration amendment v2 (frozen and git-tagged as `prereg-v2` before the prereg-v2 jury run)

Status: **FROZEN** at tag `prereg-v2`; the freeze record below is filled at tagging time and never edited
again. Everything in `docs/preregistration.md` (v1) still holds unless this document changes it.

## Why the specification was amended
* On 2026-10-20, while the `prereg-v1` jury was running and **before any 2020–2024 hybrid result was computed
  or any 2020–2024 jury score was inspected**, an independent read-only correctness review of the whole
  pipeline was run (seven areas: point-in-time statements, universe and prices, features and scoring,
  backtest statistics, jury, audits and forward run, end-to-end leakage), with two independent verifiers per
  finding (one reproducing it on the real data, one trying to refute it). It produced 51 distinct findings:
  44 confirmed, 7 plausible (2 further claims were refuted and dropped). Several meant that the code did not
  do what `prereg-v1` promised (flagged ex-date returns were not excluded from price features, small sector
  groups were not pooled into a parent group, the "other financial" template was never applied, the
  pre-registered secondary analyses were missing, TL Sharpe ratios silently used a zero risk-free rate).
* Blinding: `scripts/after_jury.sh` refuses to compute the `prereg-v1` hybrid backtest, audits or reports
  until the tag `prereg-v2` exists. The only jury outputs looked at before this amendment are the 2018-01
  development-slice calls (as in v1). The corrections were decided from the review evidence alone.
* Consequence: **`prereg-v2` is the primary analysis.** `prereg-v1` is reported once, as registered (its
  jury, its frozen store `data/store`, v1 code plus the reporting fixes of commit `3744485`), as a secondary
  analysis next to v2, together with the list of known defects (`docs/known_limitations.md`).

## What does not change
H1 and its statistic (hybrid Top-3 by J rank vs deterministic Top-3 by composite rank; paired Newey-West
t-test with 3 lags on monthly active returns vs the point-in-time equal-weight universe; stationary block
bootstrap 1,000 draws, mean block 4); decision dates 2017-12-29 … 2024-11-29 (84 holding months); next-session
VWAP fills; costs 10 + 20 bps per side; universe = official BIST-100 reconstruction with the same rules;
three equal-weight pillars and the Top-10 screen; jury model `qwen3.5:9b` at the same digest, temperature 0,
seed 20260923, `num_ctx` 8192, `num_predict` 700, thinking off, one candidate per call, three agents with the
same information partition, per-month pseudonyms, the same masking assertions, one validation retry then
abstain; random-jury null with 500 seeds; the H2 operationalisation and the audit list.

## What changes (review finding numbers in brackets; full record in `docs/known_limitations.md`)
**Data layer.**
* Statements: holdings' revenue includes finance-sector revenue so margins stay ≤ 100 % [2]; enterprise value
  = parent market value + book non-controlling interests + net debt, FCF and EBITDA yields over EV, ROIC over
  total equity incl. NCI + net debt, earnings yield and ROE on parent profit and parent equity [5]; a missing
  same-filing comparative is taken from the first-reported same-period filing of the previous year [15];
  leap-day comparatives are matched [16]; participation-bank profit-share income and personnel expenses enter
  the bank ratios [17, 18]; a reported zero parent profit/equity falls back to the next candidate [19];
  interest expense legs use magnitudes [34].
* Universe and prices: delisted or merged members get their KAP sector from `configs/sector_overrides.csv`
  [1]; the LRSHO 2019-09-17 vendor adjustment artefact is corrected through `configs/adjustment_overrides.csv`
  [31]; late vendor share-count updates are aligned [41]; the membership source follows `universe.provider`
  [30]; the alias, override and sector-group files are hashed into the store vintages [49].

**Features and normalisation.**
* The liquidity floor keeps its behaviour (max of 5 mn TL and the 10th percentile of the month's members'
  6-month ADV), now as typed config fields together with the listing-age and statement-age limits [0].
* Flagged corporate-action / adjustment-mismatch sessions are excluded from volatility and drawdown and
  contribute a zero return to 12-1 momentum [4]; stored free-float market value uses an aligned window [32];
  listing age counts from the earlier of the first price and the first official membership start [35].
* Non-bank, non-insurer financials use the other-financial template [3]; a sector group with fewer than 5
  non-null values of a feature in a month is pooled into its parent group (FINANCIALS / NON_FINANCIALS), then
  into all eligible names [20]; growth z-scores are computed within accounting basis in months where nominal
  and TMS-29 statements coexist [21]; eligible names without a composite are labelled `no_composite` [36].

**Jury (prompt version v2).**
* The sector label sent to the model is the pooled normalisation group, and percentiles are ranked within
  that same peer group (as the z-scores) [45, 48]; `prompts/v2` states this; prompt files are checked against
  `prompts/REGISTRY.md` on load [38]; the live model digest is verified in every run [39].
* J is rounded to 6 decimals before ranking and before θ, ties broken by composite rank [9]; J needs at least
  2 of 3 votes (`jury.min_votes = 2`, typed) [37]; every candidate stays in the consensus table (J = NaN when
  votes are missing) so comparators and the random null see the full candidate set [26]; transport failures
  stop the run (resumable) and are never cached or turned into abstentions [10].

**Evaluation.**
* Risk-free rate: TCMB policy rate accrued over the calendar days between sessions and aligned by date;
  Sortino uses the downside deviation over all observations [6, 22, 23].
* Every numeraire (TL, USD, CPI-deflated) carries its own active-return statistics [24]; the CPI index is
  interpolated geometrically between releases [33]; USD/TRY prefers the EVDS series (dated by validity day,
  shifted to the fixing session) and repairs vendor sessions that repeat a neighbouring rate [42, 50].
* Leave-one-year-out groups by holding year [25]; the deflated-Sharpe trial count is taken from the ledger
  without double counting [7]; a selected name that never trades in its holding month stays as cash, weights
  are never re-normalised [47]; an eligible-only equal-weight benchmark is added as a secondary row (the
  headline benchmark stays all point-in-time members) [44]; the results table carries the H1 hybrid-vs-
  deterministic test, PSR and per-numeraire rows [29].
* The pre-registered secondary analyses are implemented in `bistmas.experiments.secondary` [8]:
  Bonferroni-adjusted thresholds next to raw p-values, BUY-only portfolios at θ = 0.75 and θ ± 0.05 (cash when
  no BUY, BUY count reported), Spearman(J, composite) per month, incremental information coefficient of J
  beyond the composite rank (Newey-West on the monthly coefficients), Kish effective votes, pairwise agent
  Spearman, Krippendorff's α (interval), and the cost (0 / 30 / 60 / 100 bps) and 20 % delisting-haircut
  sensitivity reruns for both the deterministic and the hybrid portfolios.

**H2 audits.** The leak scan no longer flags the English modal verb "may" as a month name [14]; audit
samples are stratified by year and composite rank at 10 % of the run's candidates [27]; renamed companies are
probed under the code and title valid at the decision date [28]; the Top-3 robustness test (± 0.25 z input
noise and a paraphrased prompt set `prompts/v2_paraphrase`) runs on a stratified sample of 6 decision months
(reduced from the 10 % share of v1 for machine time); the human blind rubric (30 rationales, three raters,
weighted Cohen's κ) uses `bistmas audit --rubric-export / --rubric-score`.

**Forward run.** Future sessions come from a holiday-aware calendar, an unfinished month is never a month end,
dry runs never persist state that could block the real run, and the scheduled job runs with
`BISTMAS_CONFIG=configs/forward.toml` [11, 12, 13, 40]; the CPI series code and its continuation are config
fields with a staleness warning [43]; the KAP job scripts refuse to write into the frozen store [46].

## Store
The amended specification builds its own store (`BISTMAS_CONFIG=configs/prereg_v2.toml` → `data/store_v2`)
from the same raw layer; `data/store` stays byte-identical for `prereg-v1`.

## Freeze record (filled at tagging time)
| item | value |
|---|---|
| tag / commit | `prereg-v2` @ `d73ab2de3356` (tag applied to the commit that contains this table) |
| features vintage (`features_monthly.parquet` sha256[:16]) | `9db38a7ad8cf533f` |
| prices vintage | `1a6677f46e533d5c` |
| statements vintage | `8fb10fbf0be9a12f` |
| config hash (`bistmas.shared.ledger.config_hash`) | `e0108162ab34` |
| prompt version / registry | `v2` (sha256 rows in `prompts/REGISTRY.md`) |
| deterministic run recorded before the freeze | `det-v2` (ledger line 2026-10-20T12:19:20+00:00, code `d73ab2de3356`) |
