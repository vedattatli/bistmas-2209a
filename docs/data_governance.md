# Data governance

This document records where every dataset comes from, under which terms it is used, and what is
(not) redistributed. It is part of the TÜBİTAK final report and must be updated whenever a source
changes.

## Principles

1. **Zero-cost, research-only use.** The project budget is 9,000 TL for services; no data or LLM
   service is purchased in v1. All model inference runs locally (Ollama on the developer laptop).
2. **Raw pulls are never redistributed.** `data/` is git-ignored. The public repository contains
   code, configuration, hand-curated tables (aliases, overrides, sector groups), prompts, tests
   and aggregate results — not vendor data.
3. **Polite access.** Every client is rate-limited (≤ 1 request/s İş Yatırım, ≤ 0.5 request/s KAP
   POST routes, ≤ 2 requests/s KAP GET routes), caches every response once and never re-crawls.
   Each pull is a one-off, checkpointed job.
4. **Official routes are requested in parallel** (see `docs/applications/`). When an official
   source becomes available it replaces the unofficial one behind the same client interface.
5. **Disclosure.** The final report and the paper state exactly which sources were used, with
   dates and the terms under which they were accessed.

## Sources

| Dataset | Source | Terms / status | Client |
|---|---|---|---|
| Daily prices, adjusted prices, share count, market cap, free-float cap, USD/TRY, XU100 level | İş Yatırım `HisseTekil` JSON endpoint (unofficial) | Site terms (doc KKİKVKVGP–08082024-2, clause 4) forbid automated access and non-personal use; `robots.txt` disallows `/_layouts/`. **Decision (2026-09-23, with the project lead):** a single, slow, cached research pull is made while a written permission request (`docs/applications/isyatirim_permission_email.md`) and the Borsa İstanbul DataStore academic petition are sent. Data are not redistributed. If permission is refused the source is replaced by DataStore/licensed data and results are re-run. | `data/clients/isyatirim.py` |
| Cross-check prices for live names | Yahoo Finance via `yfinance` | Personal/research use; survivorship-biased (no delisted names); phantom holiday rows dropped | `data/clients/yfinance_xcheck.py` |
| First-reported financial statements with publication timestamps | KAP (kap.org.tr) public JSON routes behind the website (unofficial) | No robots rules or terms page published; MKK states high-volume needs should go through the API service. Access is paced, cached, one-off. The **MKK API Portal** (free, official, aimed at academics and developers) is applied for on day 1 and becomes the primary route once approved. | `data/clients/kap.py` |
| BIST-100/30 historical constituents | Borsa İstanbul DataStore product 3184 (price 0 TL, account required) and the public index-announcements archive | Free official data; DataStore account/academic profile registered by the project lead | `data/clients/datastore_files.py`, `data/clients/bist_announcements.py` |
| Delisted companies, equity code changes, first trading dates | Borsa İstanbul DataStore free files | Free official data | `data/clients/datastore_files.py` |
| Sector classification, company master data | KAP public pages (`/tr/Sektorler`, `/tr/bist-sirketler`) | Public reference data | `data/clients/kap.py` |
| CPI, USD/TRY, policy rate | TCMB EVDS API (free key) and the TCMB 1-week repo table | Free official data | `data/clients/evds.py` |
| KAP disclosure narratives (v2 only) | KAP `attachment-detail` route / MKK API Portal | As above; text is used only as model input, never republished | `data/clients/kap.py` |

## Restatement and point-in-time policy

* Statement values are taken **as first reported** from the KAP filing (current and prior
  columns of the same filing), never from a vendor table that overwrites periods with restated
  comparatives (verified problem with İş Yatırım `MaliTablo` from 2022Q4, TMS-29).
* Every record carries `availability_ts`: the KAP publication timestamp rolled to the next
  trading session (same day if published before 09:55). When the timestamp is unknown a
  conservative lag is used (Q1/Q3 +45 d, +55 d for banks; H1 +55 d; FY +90 d; +30 d for
  2019Q4/2020Q1; FY2023 +10 weeks and 2024Q1/Q2 +6 weeks for the TMS-29 transition). The share
  of fallback rows is reported.
* Adjusted price series are restated by the vendor after every corporate action, so raw prices
  are stored alongside adjustment factors and full histories are re-pulled, never appended.

## Generative-AI disclosure

The code base is developed with an AI coding assistant (Claude) operated by the project lead.
Prompts used by the jury agents are versioned in `prompts/` with hashes. This is disclosed in
the final report in line with TÜBİTAK's guide on the responsible use of generative AI.

## Known coverage gaps (statements)

| Ticker(s) | Membership years | Situation | Treatment |
|---|---|---|---|
| KOZAL → TRALT, KOZAA → TRMET, IPEKE → TRENJ | 2017–2026 | renamed on 2025-11-24 (Koza/İpek group → Türk Altın / TR Anadolu Metal / TR Doğal Enerji); both KAP and İş Yatırım serve the whole history only under the new codes | handled as renames (`configs/ticker_aliases.csv`): the membership table is canonicalised to the successor code, so prices and first-reported statements are continuous across the rename |
| KERVT → BESLR, DGKLB → DGNMO, ITTFH → LRSHO | 2019–2023 / 2018–2020 / 2018–2020 | renamed (Besler Gıda 2025-06-02, Doğanlar Mobilya 2022-01-03, Loras Holding 2023-11-01); history served only under the new codes | handled as renames like the Koza/İpek group |
| ERBOS | 2017–2018, 2020Q1, 2021-07 … 2023-07 | İş Yatırım serves ERBOS only from 2022-01-12 (2017–2021 empty for every request); Yahoo has the earlier history | benchmark and screen omit ERBOS before 2022-01-12 (≈ 18 member-months, ≤ 1 % of the benchmark weight); a Yahoo fill is a documented follow-up |
| LRSHO (ex ITTFH) | 2018-09 … 2019-12 | İş Yatırım answers `Teknik Hata: HisseTekil error` for the 2018 and 2023 windows only (other years served) | 2018-09 … 2018-12 without prices (4 member-months); the checkpointed pull retries the two years on every later run |
| ADANA, ANACM, SODA, TRKCM, KIPA, NTTUR | ≤ 2020 | delisted; İş Yatırım `MaliTablo` (first-reported for ≤ 2022Q3) with the p95-lag availability rule | included through the fallback table |
| DOCO | 2018–2020 | DO & CO (Vienna-listed) files a single free-form block instead of the XBRL-tagged tables; no parseable statement | excluded from the fundamental screen (`filter_reason = no_statement`) |

## Price audit notes

`bistmas build prices` writes `data/store/prices_daily_meta.json` with every flagged day (capital changes,
adjusted/raw mismatches, gaps). Items that stay unexplained after the corporate-action reconciliation:

| Ticker | Date | Observation | Treatment |
|---|---|---|---|
| TSPOR | 2025-06-19 | raw close 0.74 → 1.44 (+94.6 %) while the vendor's paid-in capital field is unchanged; the adjusted series steps −9.6 % (adjustment factor 2.15 → 1.00), i.e. a capital-reduction step the `SERMAYE` field does not show on the ex-date | returns are computed on the adjusted series, so the event is handled; outside the 2018–2024 backtest window; flagged in the forward-run certificate |
| Suspended sessions (26 tickers, 48 rows, e.g. ASELS 2018-05-23 … 06-06, AGHOL 2018-01-02, BAGFS 2021-06-14/15) | — | vendor writes VWAP 0, volume 0 and repeats the last close | a zero VWAP is treated as "no trade": fills are delayed to the first tradable session inside the holding period (cash until then), a name that never trades before the exit is skipped, an exit on a halted session uses the last close |

## Registrations and official access (status 2026-09-24)

| Source | Status | Where it is used |
|---|---|---|
| Borsa İstanbul DataStore (individual profile) | account created; the 66 free "Constituent companies of BIST 30/50/100 since 2000" files (product 3184, one yearly matrix per publication) are in `data/raw/datastore/index_constituents/` (never committed); `bistmas build universe` validates the rolled membership against every official snapshot (135 snapshots 2017-07 … 2026-07, all reproduced) | universe validation; academic petition for daily bulletin data still open |
| MKK API Portal | account + application `bistmas-erciyes-2209a` registered to "KAP Data Dissemination Services" (free plan, 6 calls/min; credentials in `.env`) | official replacement for the KAP web routes in the forward run (throughput too low for the historical pull, which is complete) |
| TCMB EVDS | API key in `.env`; `bistmas fetch macro` now fills CPI (`TP.FG.J0`) and USD/TRY | CPI-deflated metrics, USD cross-check |
| İş Yatırım | permission e-mail sent 2026-09-24 (research use, one-off cached pull, no redistribution); reply pending | disclosure in the report |

