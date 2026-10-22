# Data dictionary (skeleton — filled in as tables are created)

## Raw layer (`data/raw/<source>/…`, gzip JSON/CSV, immutable, manifest.jsonl with sha256)

### İş Yatırım `HisseTekil` (one file per ticker-year)
| field | meaning |
|---|---|
| `HGDG_KAPANIS`, `HGDG_AOF`, `HGDG_MIN`, `HGDG_MAX`, `HGDG_HACIM` | adjusted close, VWAP, low, high, volume (adjusted for rights issues, bonus issues and cash dividends; restated retroactively) |
| `HG_KAPANIS`, `HG_AOF`, `HG_MIN`, `HG_MAX`, `HG_HACIM` | raw close, VWAP, low, high, volume |
| `SERMAYE` | paid-in capital (TL, nominal 1 TL per share → share count); updated one day before the ex-date |
| `PD`, `PD_USD`, `HAO_PD`, `HAO_PD_USD` | market cap, free-float market cap (TL, USD) |
| `DD_DEGER` | USD/TRY of the day |
| `END_DEGER` | XU100 index level (÷100 rescale on 2020-07-27) |

### KAP financial report (one file per disclosureIndex)
Header: unit scale (`1.000 TL` / `1.000.000 TL`), consolidated flag, period dates, `publishDate`.
Rows: XBRL concept (e.g. `ifrs-full_Revenue`), Turkish and English labels, current and prior values.

## Feature store (`data/store/`)
| table | key | note |
|---|---|---|
| `security_master` | `mkk_member_oid` | codes with effective dates, sector group, template |
| `universe_membership` | (`mkk_member_oid`, `effective_start`, `effective_end`) | official provider; `membership_known_since` |
| `prices_daily` | (`ticker`, `date`) | raw + adjusted, share count, market cap, flags from the corporate-action audit |
| `statements_long` | (`mkk_member_oid`, `fiscal_year`, `period`, `consolidated`, `concept`) | first-reported current/prior, `publish_ts`, `availability_ts`, `availability_source` (kap / fallback) |
| `features_monthly` | (`as_of`, `ticker`) | ratios, z-scores, composites, `max_source_available_date` |

### KAP disclosure listings (`data/raw/kap/disclosures/<oid>_<from>_<to>.<lang>.json.gz`)
Rows: `publishDate` (dd.MM.yyyy HH:mm:ss), `disclosureClass`, `subject`, `year`, `period`, `ruleType`, `disclosureIndex`, `stockCodes`, `isLate`. Route limits: ≤ 1-year windows, 2000-row cap, per-IP quota (two language buckets).

| table | key | note |
|---|---|---|
| `filing_index` | (`mkk_member_oid`, `disclosure_index`) | periodic financial reports with `publish_ts` |
| `statements_reports` | `disclosure_index` | one row per parsed report: unit multiplier, consolidated flag, `availability_date` |
| `statements_long` | (`disclosure_index`, `statement`, `concept`, `col_index`) | first-reported values in TL; `col_index` 0 = current, 1 = comparative of the same filing |
| `features_monthly` | (`as_of`, `ticker`) | screen output: features, `z_*`, `S_*`, `composite`, `rank`, `eligible`, `filter_reason`, `max_source_available_date` |
| `data/runs/<run>/` | — | backtest artefacts (`*_periods.csv`, `*_daily.csv`, `report.json`), jury (`jury_scores.parquet`, `jury_consensus.parquet`, `jury_summary.json`), audits, forward decisions (`forward/<YYYY-MM>/decision.json`) |

### Macro (`data/store/macro_daily.parquet`)
One row per BIST session: `cpi`, `cpi_available_from`, `cpi_pit` (EVDS, empty without a key), `usdtry_evds`,
`policy_rate` (TCMB one-week repo, %, from the TCMB rate table), `rf_daily` (= policy_rate / 100 / 365; used
as the risk-free leg of the TL Sharpe/Sortino ratios).

### Universe membership (`data/store/universe_membership.parquet`)
`index_code, ticker, effective_start, effective_end (exclusive, NaT = open), membership_known_since, source`.
The file stores the historical codes; `read_membership()` returns the canonical view in which rename
aliases (`configs/ticker_aliases.csv`, kind = rename) are mapped to the successor code and touching
spells are merged, so a renamed issuer is one continuous member under the code its prices and filings
are keyed by.

### Monthly features (`data/store/features_monthly.parquet`)
One row per (`as_of`, member): raw features (`momentum_12_1`, `vol_6m`, `mdd_6m`, `adv_6m_tl`,
`listed_sessions`, `mcap_tl`, ratios per template), statement provenance (`statement_period_end`,
`statement_availability_date`, `statement_age_days`, `ttm_method`, `basis`), `max_source_available_date`
(asserted ≤ `as_of`), `filter_reason` / `eligible`, sector-relative z-scores (`z_*`, sign-adjusted so
higher is better), pillar scores `S_fundamental / S_valuation / S_momentum_risk`, `composite`, and
`rank` (nullable Int64 among eligible names; the Top-10 screen is `rank ≤ 10`).

## Run outputs (`data/runs/`)
* `ledger.jsonl` — append-only experiment ledger: one line per deterministic / hybrid / jury / audit run
  (timestamp, code commit, config hash, prompt version, model digest, store vintages, metric summary).
  Its distinct (kind, strategy, config) count is the trial count of the deflated Sharpe ratio.
* `<run>/report.json` — strategies (TL / USD summaries, Newey-West test and bootstrap CI of the monthly
  active return, information ratio, deflated Sharpe), `tables` (calendar-year returns,
  leave-one-year-out, minimum detectable effect, cost sensitivity), `xu100_index`, `provenance`, `rf`.
* `<run>/<strategy>_periods.csv` / `_daily.csv`, `<run>/screens.parquet`, `<run>/table_*.csv`,
  `<run>/figures/*.png|svg`.
* `<jury run>/jury_scores.parquet` — one row per (month, candidate, agent): score, confidence, abstain
  flag and reason, rationale, cited metrics, cache key, prompt/schema hashes, model and digest, token
  counts, `composite_rank`; `<jury run>/jury_consensus.parquet` — J, votes, `rank_J`, BUY flag;
  `<jury run>/jury_summary.json`.
* `jury_cache/` — content-addressed verdicts keyed by sha256(payload, prompt hash, schema hash, model
  digest, sampling options); a call is never repeated for an identical key.
* `audits/<jury run>.json` — groundedness / leak scan, re-identification, direct elicitation, date arms,
  test-retest, Lookahead-Propensity results.
* `results/results.md|csv` — every run's headline metrics and the secondary tables (`bistmas report`).
