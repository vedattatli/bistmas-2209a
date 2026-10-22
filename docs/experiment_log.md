# Experiment log

Machine-written entries are appended by the pipeline (`data/runs/ledger.jsonl`); this file holds
the human-readable narrative. Every entry that reports a metric must name the data vintage
(raw-manifest hash), the config hash, the prompt version and the model digest.

## 2026-09-23 — M0 scaffold
* Repository scaffold, uv environment (Python 3.12), Ollama installed, `qwen3.5:9b` pulled.
* Decisions and verified data sources recorded in the plan and in `docs/data_governance.md`.
* No metrics yet.
* `bistmas doctor` (2026-09-23 21:20): Python 3.12.14, all packages OK, Ollama server OK, `qwen3.5:9b` digest `6488c96fa5fa…` pinned, JSON smoke call OK (73 prompt / 35 output tokens, ~2 s with `think=false`; thinking mode costs 459 tokens and ~30 s per call, so it is off by default), token log-probabilities available (Lookahead-Propensity audit feasible).

## 2026-09-23 — M1 security master + official universe
* `bistmas fetch universe`: 45 quarterly reviews 2015Q4–2026Q4 parsed (2018Q1–2024Q2: 210 XU100 additions / 210 removals, 2020Q1 = 23/23); 8 official weight files as anchors; KAP 808 codes / 650 sector rows; 207 non-periodic announcement rows (2021+).
* `bistmas build universe`: security master 777 rows (760 current + alias rows). XU100 roll-forward from 2017-07-03: substitute promotion resolves HURGZ (for NTTUR, 2017-10-20) and ITTFH (for KIPA, 2018-09-03) automatically; remaining differences against every 2023–2025 official list are exactly the undated events (ENJSA/SOKM/MPARK fast-track entries, ADANA/CRFSA/DGATE/KLGYO exits, GUSGR→TURSG) → research in progress, then `configs/universe_overrides.csv` is completed.
* Known gap: the English 2020Q1 page lacks the BIST 30/50 tables (recorded as `missing_tables`); XU050/XU030 differences trace back to it and to fast-track displacements.

## 2026-09-24 — M1 closed, M2 price layer
* Universe: every official BIST 100/50/30 list 2023-05-31 … 2025-10-01 is reproduced exactly (XU100 251 tickers over 2017-07-03 … 2026, 101 members at the end of 2024). Mid-period events curated with KAP sources in `configs/universe_overrides.csv`; 2020Q1 BIST 50/30 lists in `configs/index_changes_supplement.csv`. 2026 differences (EFORC→EFOR, watch-list transfers, new listings) are forward-run items.
* Price layer verified on cached raw data: `HGDG_HACIM`/`HG_HACIM` are TL traded value (USD volume field agrees to 1e-13), YKBNK June-2018 rights issue flagged as a capital change with a continuous adjusted series, calendar 752 sessions 2018–2020 from four tickers.
* One-off İş Yatırım pull started 2026-09-24 00:40 (membership tickers, 2017–2026, one-year chunks, ≥ 1.5 s spacing, checkpointed in `data/raw/isyatirim/progress.json`).
* M3 (2026-09-24 01:xx): KAP first-reported statement parser verified on THYAO FY2018 (revenue 62,853 mn TL, net profit 4,045 mn, assets 109,076 mn, CFO 9,613 mn — identical to KAP). Listing route: `/en/api/disclosure/members/byCriteria` per company over the whole range (no subject filter; capped responses split by year). 236 of 251 XU100-history tickers have a KAP oid; 12 delisted names fall back to İş Yatırım (≤ 2022Q3 first-reported) and 3 renames use the successor oid. Listing job started with an auto-resume loop (`data/raw/kap/run_listing.sh`).
* M4 code (2026-09-24): statements view (per-filing wide table, same-filing comparatives, TTM = FY(prev) + YTD − YTD(prior)), ratio set per template, point-in-time feature builder (`max_source_available_date` asserted ≤ as_of), robust sector-relative z-scores with pooled small groups, three equal-weight pillars, eligibility filters, walk-forward engine (t+1 VWAP fills, turnover costs, delisting haircut, daily NAV reconciling with period NAV), metrics in TL/USD, Newey-West, stationary bootstrap, deflated Sharpe; `bistmas build features`, `bistmas screen`, `bistmas backtest --spec deterministic`. Real-data run waits for the price pull and the KAP statement bodies.
* KAP listing route accepts at most one-year windows (longer ranges → "HTTP 400"); the listing job now uses calendar-year windows 2017–2025 (~2,100 POSTs under the per-IP quota, auto-resume loop). Discovery: `GET /tr/api/batch-news/file-by-year/{oid}/{year}` returns a ZIP with a Word-HTML yearbook of the company's general disclosures (timestamped narratives, no financial-report bodies) — a quota-free source for the v2 KAP-text ablation.
* M5–M7 code (2026-09-24): jury layer (per-month pseudonyms, information partition, rationale-first schema, content-addressed cache, retry-then-abstain, consensus, random-jury null), hybrid experiment (H1 paired tests), audits (groundedness/leak scan, re-identification, direct elicitation, date arms, test-retest ICC, LAP), forward run (sealed decision files, certificate, e-mail, launchd). Jury smoke test on the local model: valid JSON, grounded rationales, ~16–23 s per call (238 output tokens).

## 2026-09-24 — real-data shakedown (prices complete, statements partial)
* Price pull finished 03:48: 2,501 ticker-years (244 tickers, 2017-01-02 … 2026-09-23), two transient failures retried; `bistmas build prices` → 479,009 rows, 2,436 sessions, 651 corporate-action events (449 capital changes: 236 bonus, 94 rights/mixed, 100 without price effect, 19 reductions; 70 vendor share-count steps one day early), one unexplained raw jump (TSPOR 2025-06-19, see `docs/data_governance.md`).
* Parser defects found only on the full body set: participation-bank tables (`tbl_par-banks_role_`, ALBRK) were unmapped; every report published before April 2018 (258 in 2017, 67 in early 2018 = all 9M-2017 and FY2017 filings) parsed into zero rows because the value cells carry no column-order class. Both fixed with fixtures (AEFES 9M-2017: revenue 10,188.6 mn TL, net profit 596.5 mn, cash 5,056.6 mn — identical to KAP).
* Engine defect: a suspended session (vendor VWAP 0, e.g. AGHOL 2018-01-02, ASELS 2018-05-23 … 06-06) produced a division by zero in the benchmark fill; zero VWAP is now "no trade" with delayed fills.
* Insurer template verified on cached AKGRT / ANHYT / ANSGR bodies (`tbl_insurance_role_`; net profit, equity, assets present) — no concept-map change needed.
* Shakedown deterministic backtest on partial statements (≈ 40 % of universe rows with fundamentals) runs end to end over 84 months; numbers are not reported because the statement layer is incomplete (KAP listing 219/223 companies, bodies still downloading).
* Renames found on real data (2026-09-24 04:xx): KOZAL/KOZAA/IPEKE trade as TRALT/TRMET/TRENJ since 2025-11-24 and EFORC as EFOR since 2025-11-03; both İş Yatırım and KAP key the whole history under the new codes, so the old codes were benchmark members without any data (with GUSGR→TURSG and YAZIC→AGHOL: up to 6 of 100 names in 2019). Fix: `read_membership()` maps rename aliases to the successor and merges touching spells; TRALT/TRENJ price history pulled (20 ticker-years). The stored membership file keeps historical codes for the official-list validation.
* Jury development-slice checks (2018-01-31, 30 real calls each, prompt v1, `qwen3.5:9b`): first pass abstained on 30 % of calls (verdicts cited the `statement_age` context field, which the validator rejected) and the payload percentiles for "lower is better" metrics (volatility, leverage, cost/income) were ranked on raw values while the prompt says higher is better. Fixes: context fields are citable, percentiles rank the sign-adjusted value, the user template states the orientation explicitly (registry hash refreshed before the freeze). Second pass: 0 % abstentions, 30/30 valid verdicts, 25.7 s per call while a model download ran in parallel, score spread 0.20–1.00, rationales cite only payload fields. A third defect surfaced here: the feature store wrote the screen `rank` as text, so the jury picked ranks 1, 10, 11 … 18 instead of the Top-10 — fixed (nullable integer + numeric coercion + regression test); the deterministic backtest was unaffected.

## 2026-09-24 — final data build, deterministic results, pre-registration freeze, jury launch
* KAP statements complete: 5,483 first-reported reports (243 companies with a KAP id, 1,577 listing POSTs over 8 quota windows, every body cached, 0 unparsed) → 2.75 mn statement rows for 234 tickers, 73 % consolidated; İş Yatırım fallback only for the six delisted names (≈ 0 % of feature rows after 2017). Feature store: 8,400 rows (84 decisions × 100 members), fundamentals present for 97–100 % of rows, 80–88 % eligible per year (liquidity floor 860 rows, < 12 months listed 279, negative equity 118, no statement 21 — DOCO and the first months of the renamed Koza/İpek names), statement age median 91 days (p95 151), TTM from same-filing YTD deltas for 78 % of rows.
* Deterministic backtest `det-v1` (data vintages in `docs/preregistration.md`): equal-weight PIT benchmark TL CAGR 44.3 % (USD 4.8 %, Sharpe 1.51, MDD −36 %); composite Top-3 34.3 % (USD −2.5 %), Top-5 39.8 %, Top-10 41.6 %. Monthly active return vs the benchmark −0.48 % (Top-3, Newey-West t = −0.79), −0.24 % (Top-5), −0.12 % (Top-10); deflated-Sharpe PSR 0.06 / 0.10 / 0.13 with 3 trials. The rule-based composite does not beat the equal-weight universe over 2018–2024; the hybrid question (does the jury add anything on top of the screen) is tested next. XU100 price index (secondary benchmark) 35.7 % TL CAGR after fixing a double rescale in the reporting helper (strategy numbers unaffected).
* Pre-registration frozen: `docs/preregistration.md` freeze record filled, commit c9c6402 tagged `prereg-v1` (features `b169944de7f4…`, prices `b415bf7c06d7…`, statements `8d45b86e3c88…`, config `c91cbe509b8d`).
* Jury run `prereg-v1` launched 06:14 (`scripts/run_jury.sh`, caffeinate + auto-resume): 84 months × 10 candidates × 3 agents = 2,520 calls at ~17–26 s each; v2 models pulled for later ablations (`gemma4:12b-it-qat`, `llama3.1:8b`).
* Post-freeze reporting additions (strategy selections untouched, verified): pre-registered secondary tables — calendar-year net returns, leave-one-year-out Newey-West tests, minimum detectable effect (Top-3: 1.7 % per month ≈ 20 % annualised at 80 % power; Top-10: 1.3 % ≈ 16 %), cost sensitivity 0/30/60/100 bps — and TL Sharpe/Sortino against the TCMB policy rate (`macro_daily.parquet`; CPI columns stay empty until the EVDS key is entered). `cagr()` measured growth from the first NAV point instead of 1.0, which hid the first session's return and the initial transaction cost; fixed and `det-v1` regenerated (TL CAGR Top-3 34.2 %, Top-5 39.8 %, Top-10 41.6 %, benchmark 44.3 %). With a monthly MDE of 1.3–1.7 % the 2018–2024 sample cannot detect active returns below roughly 15 % per year: the H1 test is informative only for large effects, as the pre-registration anticipated.
* Registrations (2026-09-24 07:xx, done with the project lead in the browser): DataStore, MKK API Portal and EVDS accounts; İş Yatırım permission e-mail sent. The 66 official DataStore constituent files gave an independent check of the universe: 135 snapshots (BIST 100/50/30, 2017-07 … 2026-07) compared, two curated events corrected (SOKM fast-track effective 2018-05-18 instead of the KAP notice date 2018-05-24; KONTR → BERA (XU100) / AKSEN (XU050) on 2026-06-18), after which every snapshot is reproduced (the only residual difference is DataStore printing today's code TURSG for the 2020 GUSGR rows, handled by the rename aliases). The SOKM correction moves one fast-track entry by four sessions in 2018; it does not touch the frozen features (universe membership at month-ends is unchanged) — verified by rebuilding the universe and confirming `features_monthly.parquet` would be identical for every decision date.

* CPI-deflated numeraire (EVDS `TP.FG.J0`, point-in-time prints) added next to TL and USD, `det-v1` regenerated: real CAGR 2018–2024 benchmark +6.9 % (max drawdown −47 %), Top-3 −0.6 %, Top-5 +3.5 %, Top-10 +4.9 %. USD and real numbers agree on the ranking: the equal-weight universe beat every composite portfolio.
* Forward-run store: `configs/forward.toml` points `paths.store` to `data/store_forward` (a copy of the frozen store plus the 2025–2026 KAP filings, 6,521 bodies), so the paper's vintages in `data/store` are never rebuilt; the monthly forward decision runs with `BISTMAS_CONFIG=configs/forward.toml`.

## 2026-10-20 — correctness review, pre-registration amendment v2
* The `prereg-v1` jury had stopped on 2026-09-24 after 17 of 84 months (the launching session ended); restarted
  in a detached `screen` session (resumable from the cache).
* Before any 2020–2024 hybrid result or jury score was looked at, a read-only multi-agent correctness review with
  two independent verifiers per finding reported 51 distinct defects (44 confirmed, 7 plausible); the record and
  the per-finding status are in `docs/known_limitations.md`. `scripts/after_jury.sh` keeps the `prereg-v1` hybrid
  results blind until the amendment is tagged.
* The fixes were implemented in seven reviewed work packages, merged, cross-reviewed and rebuilt from the raw
  layer into a separate store (`configs/prereg_v2.toml` → `data/store_v2`; `data/store` of v1 untouched);
  `docs/preregistration_v2.md` lists every change. The jury now uses prompt version v2 (pooled peer group as the
  sector label, percentiles within that group); a dry run over all 84 months built and checked every prompt.
* Deterministic run `det-v2` (TL CAGR, Sharpe over the accrued policy rate): composite Top-3 40.8 % / 0.56,
  Top-5 36.3 % / 0.48, Top-10 43.7 % / 0.67; equal-weight PIT benchmark 44.1 % / 0.73 (eligible-only 42.3 %).
  Monthly active return vs the benchmark −0.05 % (Top-3, NW t −0.07), −0.37 % (Top-5, t −0.60), +0.06 %
  (Top-10, t 0.12): as in v1, the rule-based screen alone does not beat the equal-weight universe. USD CAGR
  Top-3 2.3 % vs benchmark 4.7 %; real (CPI) 4.2 % vs 6.7 %. 57 of 84 monthly Top-3 sets differ from v1.
* Frozen and tagged `prereg-v2`; the v2 jury runs first, then the remaining `prereg-v1` months from a checkout
  of the v1 code (`prereg-v1` stays the as-registered secondary analysis).

## 2026-10-21 — prereg-v2 jury and hybrid results (primary analysis)
* Jury `prereg-v2` (prompt v2, `qwen3.5:9b` @ `6488c96fa5fa`) complete 06:36: 84 months, 2,520 calls, 0 %
  abstentions, ≈ 22 s per call, 596 prompt / 246 output tokens on average.
* `hybrid-v2` (store `data/store_v2`): hybrid Top-3 TL CAGR 35.0 % vs deterministic Top-3 40.8 % (benchmark
  44.1 %); paired hybrid − deterministic −0.43 %/month, NW t −0.56, p 0.58 → **H1 not supported**. Top-5 +0.15 %/month
  (t 0.32); Top-10 identical by construction (10 candidates). Random-jury null (500 seeds): hybrid at the 41st
  percentile. Echo Spearman(J, composite) 0.22; incremental IC of J 0.005 (t 0.10). BUY-only θ = 0.75: 4.3 BUYs per
  month, 2 months in cash, active −0.23 %/month (t −0.30). Agent agreement: mean pairwise Spearman −0.10,
  Krippendorff α −0.19 (information partition). No strategy significant before or after Bonferroni (8 configs).
* Audits `prereg-v2` (84-candidate stratified sample, 15:25): groundedness 99.7 % (2,520 rationales), leak flags
  0 %; re-identification ticker top-1 0 % (chance 1 %), year+quarter 3.6 % (chance 3.4 %); direct elicitation of
  BIST closes MAPE 90 %, 4.8 % within 10 %, next-month direction 45 %; lookahead propensity P(up)+P(down) 0.85
  (confident but wrong); date arms mean Δ < 0.002, mean |Δ| 0.04–0.06, Spearman vs masked 0.89–0.94; test-retest
  ICC(1) at T = 0.7: quality/growth 0.84, risk/timing 0.85, value 0.55.
* Robustness `prereg-v2` (6 stratified months, 360 calls, 17:48): ±0.25 z noise Kendall τ 0.63 (median 0.65), Top-3
  overlap 78 %; paraphrased prompts (`v2_paraphrase`) τ 0.57, Top-3 overlap 78 %.
* prereg-v1 (as originally frozen, secondary analysis) finished 2026-10-22 00:12 (84 months, 2,520 verdicts, 0 % abstentions); `hybrid-v1` produced by the v1 checkout (e991a61). Hybrid Top-3 TL CAGR 31.7 % vs rule-based Top-3 34.2 %; hybrid − rule-based −0.18 %/month (NW t −0.24, p 0.81); hybrid at the 37th percentile of the random-jury null (mean CAGR 34.9 %); turnover 50 % vs 32 %. prereg-v2: −0.43 %/month (t −0.56, p 0.58), random-jury percentile 41. H1 is not supported under either specification; the conclusion does not depend on the v1 → v2 implementation corrections.

