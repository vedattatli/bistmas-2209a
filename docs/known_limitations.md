# Known limitations and the 2026-10-20 correctness review

Before any 2020–2024 hybrid result of the pre-registered run `prereg-v1` was computed, an independent read-only
review of the pipeline (point-in-time statements, universe and prices, features and scoring, backtest statistics,
jury, audits and forward run, end-to-end leakage) was run with two independent verifiers per finding (reproduction
and refutation): 51 distinct findings, 44 confirmed, 7 plausible (2 further claims refuted). They were corrected in
the amended specification `prereg-v2` (`docs/preregistration_v2.md`), which is the primary analysis; `prereg-v1`
is reported as registered, so every item below still applies to the `prereg-v1` numbers.

| # | severity | verdict | affects | finding | prereg-v2 |
|---|---|---|---|---|---|
| 0 | critical | CONFIRMED | features / jury inputs | Undeclared 10th-percentile ADV floor replaces the pre-registered '6-month ADV >= 5 mn TL' liquidity rule and sits outside config_hash (`src/bistmas/scoring/filters.py`) | fixed in prereg-v2 |
| 1 | critical | CONFIRMED | features / jury inputs | Delisted or merged members (SODA, TRKCM, ANACM, ADANA, KIPA) get sector_group OTHER and are z-scored against the wrong peers (survivorship-correlated) (`src/bistmas/backtest/run.py`) | fixed in prereg-v2 |
| 2 | critical | CONFIRMED | features / jury inputs | Holdings with finance-sector operations: revenue excludes finance revenue but gross/operating profit include it (gross margin >100%) (`configs/concepts.toml`) | fixed in prereg-v2 |
| 3 | high | CONFIRMED | features / jury inputs | Pre-registered 'other financial' template never applied: MALİ KURULUŞLAR names (GOZDE, HDFGS, ISMEN, ISFIN, KTLEV) scored with industrial ratios (`configs/sector_groups.csv`) | fixed in prereg-v2 |
| 4 | high | CONFIRMED | features / jury inputs | Pre-registered exclusion of flagged ex-date / corporate-action returns from momentum, vol and drawdown is not implemented (`src/bistmas/scoring/features.py`) | fixed in prereg-v2 |
| 5 | high | CONFIRMED | features / jury inputs | EV, FCF yield and ROIC mix 100 %-consolidated flows and net debt with parent-only market cap and equity (minority interests omitted) (`src/bistmas/scoring/ratios.py`) | fixed in prereg-v2 |
| 6 | high | CONFIRMED | backtest numbers | TL Sharpe/Sortino silently use rf = 0: align_rf returns a RangeIndex series that sharpe()/sortino() reindex by date (`src/bistmas/backtest/tables.py`) | fixed in prereg-v2 (commit 3744485) |
| 7 | high | CONFIRMED | backtest numbers | Deflated-Sharpe trial count double-counts: the current run's strategies are added on top of the ledger even when already there, and det_top{n} is counted again under kind 'hybrid' (`src/bistmas/backtest/run.py`) | fixed in prereg-v2 |
| 8 | high | CONFIRMED | backtest numbers | Pre-registered secondary analyses and controls are not implemented anywhere (Bonferroni, BUY-only θ variant, echo Spearman, incremental IC, Kish votes, agent agreement, hybrid cost/haircut sensitivity, H2 noise/paraphrase robustness) (`src/bistmas/experiments/hybrid.py`) | fixed in prereg-v2 |
| 9 | high | CONFIRMED | backtest numbers | Floating-point noise in J defeats the pre-registered composite-rank tie-break and the θ threshold; one 2018 Top-3 is already wrong, and the random null never ties (`src/bistmas/jury/consensus.py`) | fixed in prereg-v2 |
| 10 | high | CONFIRMED | backtest numbers | Ollama outages and timeouts become permanent cached abstentions and the run still reports 'complete' (`src/bistmas/jury/agents.py`) | fixed in prereg-v2 |
| 11 | high | CONFIRMED | forward run | Forward run can never seal a real-time decision: the execution session is looked up in a calendar built from price data, so it always raises (`src/bistmas/reporting/forward.py`) | fixed in prereg-v2 |
| 12 | high | CONFIRMED | forward run | Forward decision-date selection treats the last price date of an unfinished month as a month end (2026-09-23 now, mid-October after the refresh); 2026-09-30 is never picked (`src/bistmas/reporting/forward.py`) | fixed in prereg-v2 |
| 13 | high | CONFIRMED | forward run | Dry runs persist state that blocks the real run (jury dry-run abstentions count as done; forward --dry-run seals decision.json), and forward_YYYYMM jury state mixes decision dates (`src/bistmas/reporting/forward.py`) | partly fixed in prereg-v2 (see below) |
| 14 | high | CONFIRMED | audits | Leak-flag rate is about 18 % but every flag is a false positive: the month-name regex matches the English modal verb 'may' (`src/bistmas/jury/audits/groundedness.py`) | fixed in prereg-v2 |
| 15 | medium | CONFIRMED | features / jury inputs | 2018 bank interims have no comparative column: banks unranked for ~9 months, or TTM double-counted where the comparative parsed as 0 (`src/bistmas/scoring/statements_view.py`) | fixed in prereg-v2 |
| 16 | medium | CONFIRMED | features / jury inputs | Non-December fiscal years: TTM differenced across the TMS-29 basis change (KAYSE), and a leap-day comparative mismatch (`src/bistmas/scoring/statements_view.py`) | partly fixed in prereg-v2 (see below) |
| 17 | medium | CONFIRMED | features / jury inputs | Participation bank ALBRK: net profit-share income not mapped, so NIM is missing and cost/income is ~5-15x too high every month (`configs/concepts.toml`) | fixed in prereg-v2 |
| 18 | medium | CONFIRMED | features / jury inputs | Bank cost_to_income omits personnel expenses from 2018 (format change splits them out of OtherOperatingExpenses) (`configs/concepts.toml`) | fixed in prereg-v2 |
| 19 | medium | CONFIRMED | features / jury inputs | _pick accepts a reported 0 for ProfitLossAttributableToOwnersOfParent and never falls back to ProfitLoss (`src/bistmas/scoring/statements_view.py`) | fixed in prereg-v2 |
| 20 | medium | CONFIRMED | features / jury inputs | Small groups pooled into one POOLED bucket (not a 'parent group'); group size counted on rows, so many z-scores come from 2–4 values (`src/bistmas/scoring/normalize.py`) | fixed in prereg-v2 |
| 21 | medium | CONFIRMED | features / jury inputs | TMS-29 and nominal statements are mixed in the same month's growth z-scores (Feb–Apr 2024) (`src/bistmas/scoring/normalize.py`) | fixed in prereg-v2 |
| 22 | medium | CONFIRMED | backtest numbers | rf_daily = policy/100/365 is subtracted once per trading session, so the risk-free leg accrues only ~250/365 of the annual rate (`src/bistmas/backtest/tables.py`) | fixed in prereg-v2 (commit 3744485) |
| 23 | medium | CONFIRMED | backtest numbers | Sortino downside deviation averages over negative days only (/N_down instead of /N), understating Sortino by ~1/sqrt(share of down days) (`src/bistmas/backtest/metrics.py`) | fixed in prereg-v2 (commit 3744485) |
| 24 | medium | CONFIRMED | reporting only | Primary statistic computed in TL only: USD and CPI-deflated entries carry TL active NW t/p, bootstrap CI, IR, active mean, hit rate and turnover (`src/bistmas/backtest/run.py`) | fixed in prereg-v2 |
| 25 | medium | CONFIRMED | reporting only | Leave-one-year-out groups by decision-date year: phantom '2017' row (n=83) and every excluded year shifted by one holding month (`src/bistmas/backtest/tables.py`) | fixed in prereg-v2 |
| 26 | medium | CONFIRMED | backtest numbers | consensus() silently drops candidates whose three agents all abstained, so run_hybrid's deterministic comparator and random null depend on jury behaviour (`src/bistmas/jury/consensus.py`) | fixed in prereg-v2 |
| 27 | medium | CONFIRMED | audits | Audit samples are neither stratified nor 10 %: the full run audits only 2017-12…2022-11, and test-retest uses 12 dev-slice payloads pooled by ticker (`src/bistmas/jury/audits/run.py`) | fixed in prereg-v2 |
| 28 | medium | CONFIRMED | audits | Memorisation audits probe renamed companies under their post-2025 codes and titles (15 % of candidates), biasing them toward 'no leakage' (`src/bistmas/jury/audits/run.py`) | fixed in prereg-v2 |
| 29 | medium | CONFIRMED | reporting only | Results table drops the H1 hybrid-vs-deterministic test and the PSR, leaves the hybrid active mean empty, and mislabels the random-null row (`src/bistmas/reporting/results.py`) | fixed in prereg-v2 |
| 30 | medium | CONFIRMED | backtest numbers | universe.provider switch is ignored, and the rule-based replica could only pick from official members (including future ones) (`src/bistmas/backtest/run.py`) | fixed in prereg-v2 |
| 31 | medium | CONFIRMED | features / jury inputs | Uncorrected vendor adjustment artefact: LRSHO 2019-09-17 (+67.6 % adjusted step, flagged class=other) inflates the Sep-2019 benchmark by ~1 pp and LRSHO's vol features (`src/bistmas/data/corporate_actions.py`) | fixed in prereg-v2 |
| 32 | medium | CONFIRMED | data layer (forward period) | Look-ahead in stored ffloat_mcap_tl: positional window from the close_adj index applied to a pivot with fewer rows (`src/bistmas/scoring/features.py`) | fixed in prereg-v2 |
| 33 | medium | CONFIRMED | reporting only | CPI-deflated metrics divide the daily NAV by a step-function point-in-time CPI. This creates 84 artificial one-day real jumps and a one-month misalignment, so every reported real vol, MDD, Sortino and Sharpe is wrong (`src/bistmas/backtest/tables.py`) | fixed in prereg-v2 |
| 34 | low | CONFIRMED | reporting only | TTM legs choose concepts independently: interest_expense mixes the CF adjustment (+) and FinanceCosts (−) in 157 TTMs (`src/bistmas/scoring/statements_view.py`) | fixed in prereg-v2 |
| 35 | low | CONFIRMED | features / jury inputs | Listing-age filter counts from the first vendor price, so long-listed names with a vendor gap are excluded (ERBOS all of 2022; LRSHO momentum NaN for 2019) (`src/bistmas/scoring/features.py`) | fixed in prereg-v2 |
| 36 | low | CONFIRMED | reporting only | Eligible-but-unrankable names keep an empty filter_reason ('no_composite' is never assigned) (`src/bistmas/scoring/screen.py`) | fixed in prereg-v2 |
| 37 | low | CONFIRMED | backtest numbers | min_votes=2 rule is not in the pre-registration ('J = mean of the available scores') (`src/bistmas/jury/consensus.py`) | fixed in prereg-v2 |
| 38 | low | CONFIRMED | backtest numbers | Prompt-registry hash check promised in REGISTRY.md is not enforced (`src/bistmas/jury/agents.py`) | fixed in prereg-v2 |
| 39 | low | CONFIRMED | backtest numbers | With --agent-models, the live digest of the default model is never verified, yet the pinned digest is recorded and used in the cache key (`src/bistmas/jury/run.py`) | fixed in prereg-v2 |
| 40 | low | CONFIRMED | forward run | Scheduled forward job reads the frozen paper store, and the certificate's config hash cannot be matched to the pre-registered hash (`scripts/com.bistmas.forward.plist`) | fixed in prereg-v2 |
| 41 | low | CONFIRMED | data layer (forward period) | align_shares corrects only early SERMAYE updates, not late ones (`src/bistmas/data/prices.py`) | fixed in prereg-v2 |
| 42 | low | CONFIRMED | reporting only | USD numeraire uses the vendor DD_DEGER series unvalidated: 7 sessions carry the previous or next session's rate, including two fill/exit sessions and the 2018-08-10 crash day (-9.1 %) (`src/bistmas/backtest/run.py`) | fixed in prereg-v2 |
| 43 | low | CONFIRMED | data layer (forward period) | cpi_pit silently carries the January-2026 CPI forward for 8 months (TP.FG.J0 has no observations after 2026-01) while macro_meta reports cpi_pit coverage 1.0 (`src/bistmas/data/macro.py`) | fixed in prereg-v2 |
| 44 | medium | PLAUSIBLE | backtest numbers | The 'point-in-time equal-weight universe' benchmark holds every XU100 member, including names the preregistered universe excludes (`src/bistmas/backtest/benchmark.py`) | fixed in prereg-v2 |
| 45 | medium | PLAUSIBLE | features / jury inputs | Payload sends the raw, unpooled sector_group, which pinpoints single-member groups (e.g. 'transport' + 'large' = THYAO) (`src/bistmas/jury/payload.py`) | fixed in prereg-v2 |
| 46 | low | PLAUSIBLE | data layer (forward period) | Frozen store's filing_index.parquet was overwritten by the forward listing after the freeze (`src/bistmas/data/filings.py`) | partly fixed in prereg-v2 (see below) |
| 47 | low | PLAUSIBLE | backtest numbers | Engine skips names that never trade in [e, e_next) and re-weights the rest, using future information; the preregistration says halted names are held as cash (`src/bistmas/backtest/engine.py`) | fixed in prereg-v2 |
| 48 | low | PLAUSIBLE | features / jury inputs | Bank/insurer payloads mix universe-wide percentiles with within-bank z-scores, producing contradictory 'clear weakness' signals (`src/bistmas/jury/payload.py`) | fixed in prereg-v2 |
| 49 | low | PLAUSIBLE | backtest numbers | Canonical membership depends on the live configs/ticker_aliases.csv, which no vintage or config hash records (`src/bistmas/data/universe/official.py`) | fixed in prereg-v2 |
| 50 | low | PLAUSIBLE | data layer (forward period) | macro_daily.usdtry_evds is shifted one session late: EVDS dates TCMB indicative rates by validity day, but fx_to_sessions treats the date as the observation session (`src/bistmas/data/macro.py`) | fixed in prereg-v2 |

## Limitations that remain in prereg-v2
* [16] TTM is not differenced across the TMS-29 basis change only where the statements really are restated; the
  basis label is derived from the period-end date, and for the five non-December filers whose early-2024 interims
  carry that label while still being nominal (KAYSE, FENER, GSRAY, BJKAS, TSPOR) the nominal TTM is kept, because
  annualising would be the worse estimate (verified on the filings).
* [13, 46] The frozen `data/store/filing_index.parquet` of `prereg-v1` was overwritten by the 2025–26 forward
  listing after the v1 freeze; it is a download index (not an input of the v1 features, which are frozen), and the
  job scripts now refuse to write into a frozen store.
* [43] The CPI continuation series (`TP.TUKFIY2025.GENEL`, 2025 = 100) has to be fetched once online before the
  forward run; the 2018–2024 backtest window is unaffected.
* [11, 42, 50] The 2026–2027 public-holiday list of the forward calendar is static; holidays are not used when
  shifting EVDS rates to the fixing session.
* H2: the Top-3 robustness test runs on 6 stratified decision months instead of 10 % of the run; the human blind
  rubric depends on three raters scoring the exported sheet.
* Minor, not affecting selections: a cash slice left by a name that never traded counts as a buy in the next
  rebalance's turnover (slightly higher costs); the BUY-only variants are not added to the ledger trial count;
  `universe.validation_window_end` and the price-rescaling tolerances are not part of the config hash; the
  normalisation group sent to the jury is chosen on the group's row count while z-scores pool on non-null counts.

Full record (claims, evidence, reproduction scripts, verifier reasoning): `data/runs/review_2026-10-20/final.json`.
