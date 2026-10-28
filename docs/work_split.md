# Work split (three areas)

The repository is organised so that it can be developed and presented as three work areas with
a small shared core. Import boundaries between areas are enforced with `import-linter`
(`uv run lint-imports`).

| Area | Owner | Packages | Deliverables (proposal work packages) |
|---|---|---|---|
| Data layer | Vedat Ali Tatlı | `bistmas.data` (+ `configs/*.csv`, `docs/data_dictionary.md`, `docs/data_governance.md`) | IP1 candidate-pool agent (universe), IP2 explorer agent (ETL, feature store, point-in-time rules) |
| Scoring and backtest | Rabia Yeşil | `bistmas.scoring`, `bistmas.backtest` (+ leakage tests) | IP3 producer-filter agent (composite scores, Top-10), IP5 validation (walk-forward backtest, metrics, statistics) |
| Jury and reporting | Gizem Coşkun | `bistmas.jury` (incl. `audits/`), `bistmas.reporting`, `bistmas.experiments`, `prompts/` (+ e-mail report, paper figures) | IP4 jury agents and consensus, memorisation audits, forward run and reporting |

Shared core (`bistmas.shared`, `bistmas.cli`, `pyproject.toml`, tests infrastructure) is
maintained jointly. This table is updated when ownership changes; the git history records the
actual contributions.

## Module map

| Area | Modules |
|---|---|
| Data layer | `data/clients/{http,kap,isyatirim,bist_announcements,index_anchors,datastore_files,evds,tcmb_rates,yfinance_xcheck}.py`, `data/security_master.py`, `data/universe/{provider,official,rule_based,build}.py`, `data/prices.py`, `data/prices_fetch.py`, `data/calendar_build.py`, `data/corporate_actions.py`, `data/macro.py`, `data/filings.py`, `data/statements_kap.py`, `data/statements_fetch.py`, `data/statements_isyatirim.py`, `data/pit.py`; `configs/{ticker_aliases,universe_overrides,index_changes_supplement,sector_groups,fiscal_calendars}.csv`, `configs/concepts.toml`; `scripts/run_listing.sh`, `scripts/run_bodies.sh` |
| Scoring and backtest | `scoring/{statements_view,ratios,features,normalize,composite,filters,screen}.py`, `backtest/{engine,benchmark,metrics,stats,tables,run}.py`, `experiments/hybrid.py` (H1 experiment), leakage tests |
| Jury and reporting | `jury/{payload,schema,ollama_client,cache,agents,consensus,random_jury,run}.py`, `jury/audits/{groundedness,probes,run}.py`, `prompts/`, `reporting/{forward,certificate,render,email,results,figures}.py`, `scripts/run_jury.sh`, `scripts/freeze_prereg.py`, `scripts/com.bistmas.forward.plist` |
| Shared core | `shared/{config,hashing,log,store,calendar,codes,ledger}.py`, `cli.py`, `configs/base.toml`, `tests/conftest.py`, `pyproject.toml` (import contracts) |

Suggested presentation order for the three-way split: each member owns one row of the first table,
demonstrates their CLI commands end to end (`fetch/build` → `backtest` → `jury/audit/forward`), and the
shared core is presented together.
