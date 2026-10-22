# bistmas — Hybrid rule + LLM multi-agent stock selection for Borsa İstanbul

> Public snapshot of the development repository, published 2026-10-22 for the TÜBİTAK 2209-A final report.
> The full history, including the pre-registration tags `prereg-v1` (commit c9c6402) and `prereg-v2` (commit
> bda1764) referenced in `docs/`, is kept in the private development repository because it contains personal
> contact details; it is available on request. Raw and derived data (`data/`) are not redistributed.

Research code for the TÜBİTAK 2209-A project *"Kural ve LLM Tabanlı Hibrit Çoklu Ajan Sistemi ile
Hisse Öneri Sistemi"* (Erciyes University, Computer Engineering).

The system screens the BIST-100 universe with three deterministic agents (survivorship-bias-free
universe, point-in-time feature store, composite factor scores) and lets three local LLM "jury"
agents (quality/growth, value, risk/timing) score the Top-10 candidates. A walk-forward backtest
(2018–2024, monthly) compares the hybrid portfolio against deterministic-only and equal-weight
benchmarks, with explicit look-ahead, survivorship and LLM-memorisation controls.

Everything runs on one laptop with free data and a local model (Ollama). No paid APIs.

## Layout

| Package | Work area | Responsibility |
|---|---|---|
| `bistmas.shared` | shared | config, hashing, logging, storage helpers, BIST trading calendar |
| `bistmas.data` | data | source clients, immutable raw layer, security master, universe providers, first-reported statements, point-in-time joins, feature store |
| `bistmas.scoring` | scoring & backtest | ratios per template, robust sector-relative normalisation, composite scores, filters, Top-10 screen |
| `bistmas.backtest` | scoring & backtest | walk-forward engine, execution and costs, benchmarks, metrics, statistics, ablations |
| `bistmas.jury` | jury & reporting | masked payloads, Ollama agents, content-addressed cache, consensus, random-jury null, leakage audits |
| `bistmas.reporting` | jury & reporting | monthly forward run, e-mail report, run certificates |

Import boundaries between the areas are enforced with `import-linter` (see `pyproject.toml`).

## Quick start

```bash
uv sync                      # creates .venv with Python 3.12 and all dependencies
uv run bistmas doctor        # checks Python, packages, Ollama server, model digest, JSON smoke call
uv run pytest                # unit tests + leakage suite
uv run lint-imports          # work-area boundary contracts
```

Pipeline commands (each is idempotent and resumable):

```bash
uv run bistmas fetch universe                 # KAP company/sector pages, index announcements, official weight files
uv run bistmas build universe                 # security master + point-in-time BIST 100/50/30 membership (validated)
uv run bistmas fetch prices [--dry-run]       # one-off İş Yatırım pull (yearly chunks, checkpointed)
uv run bistmas build prices                   # prices_daily, trading calendar, corporate-action audit
uv run bistmas fetch macro                    # TCMB policy rate (+ CPI / USDTRY from EVDS when a key is set)
./scripts/run_listing.sh [END]                # KAP filing listings under the per-IP quota (auto-resume loop)
./scripts/run_bodies.sh                       # KAP report bodies for every listed filing (auto-resume loop)
uv run bistmas build statements               # first-reported statements with availability dates
uv run bistmas build features                 # monthly screens 2017-12 … 2024-11 (features_monthly.parquet)
uv run bistmas screen --as-of 2019-12-31      # one month's deterministic Top-10
uv run bistmas backtest --spec deterministic --run-name det-v1   # Top-3/5/10 vs PIT equal weight, TL/USD, tests, tables
uv run python scripts/freeze_prereg.py det-v1 && git tag prereg-v1   # freeze docs/preregistration.md before the jury
./scripts/run_jury.sh prereg-v1               # local LLM jury over every month (cached, resumable, keeps the Mac awake)
uv run bistmas backtest --spec hybrid --jury-run prereg-v1        # hybrid vs deterministic (same N), random-jury null

# prereg-v2 (primary analysis, docs/preregistration_v2.md): its own store data/store_v2
export BISTMAS_CONFIG=configs/prereg_v2.toml  # every command below then reads/writes data/store_v2
uv run bistmas build universe && uv run bistmas build prices && uv run bistmas build statements && uv run bistmas build features
uv run bistmas backtest --spec deterministic --run-name det-v2
uv run python scripts/freeze_prereg.py det-v2 preregistration_v2.md prereg-v2   # then git tag prereg-v2
./scripts/run_jury.sh prereg-v2 && ./scripts/after_jury.sh prereg-v2   # jury, hybrid-v2, audits, robustness, report
uv run bistmas audit --jury-run prereg-v2     # groundedness, re-identification, date arms, test-retest, LAP
uv run bistmas audit --jury-run prereg-v2 --robustness   # Top-3 under ±0.25 z noise and a paraphrased prompt
uv run bistmas audit --rubric-export prereg-v2           # blind sheet for three human raters; --rubric-score r1.csv r2.csv r3.csv
uv run bistmas jury --run xfam-v2 --agent-models value=gemma4:12b-it-qat,risk_timing=llama3.1:8b   # cross-family ablation
unset BISTMAS_CONFIG
BISTMAS_CONFIG=configs/forward.toml uv run bistmas build statements   # forward store (data/store_forward) with the latest filings
BISTMAS_CONFIG=configs/forward.toml uv run bistmas forward --dry-run   # sealed monthly decision + report on the forward store
uv run bistmas report --figures               # results tables (Markdown/CSV) + NAV / calendar-year / jury figures
```

## Data sources and governance

See `docs/data_governance.md`. Raw pulls live under `data/` and are never committed or
redistributed.

## Status

Code for milestones M0–M9 is in place (≈390 tests, five import contracts). Data pulls are complete (prices for
249 codes, 6,521 first-reported KAP reports incl. 2025–26). The pre-registration was frozen at `prereg-v1`; an
independent correctness review before any hybrid result was computed led to the amendment `prereg-v2`
(`docs/preregistration_v2.md`, review record in `docs/known_limitations.md`), which is the primary analysis.
Both specifications are complete (2026-10-22): deterministic runs `det-v1`/`det-v2`, LLM jury runs `prereg-v1`/
`prereg-v2` (2,520 verdicts each), hybrid backtests `hybrid-v1`/`hybrid-v2`, H2 audits and robustness checks on v2.
Result: H1 is not supported — the hybrid Top-3 trails the rule-based Top-3 by 0.43 %/month in v2 (NW t −0.56) and
0.18 %/month in v1 (t −0.24), and no portfolio beats the point-in-time equal-weight BIST-100 benchmark.
Every run appends a line to `data/runs/ledger.jsonl`. See `docs/experiment_log.md` for the narrative and the
numbers, and `docs/report/` for the TÜBİTAK final report (`sonuc_raporu.pdf`; built with `build_docx.py` and `make_pdf.sh`).

## Acknowledgement

This project was supported by TÜBİTAK (Scientific and Technological Research Council of Türkiye) under the 2209-A
University Students Research Projects Support Programme (application no. 1919B012550729). Bu proje TÜBİTAK 2209-A
Üniversite Öğrencileri Araştırma Projeleri Destek Programı kapsamında desteklenmiştir.
