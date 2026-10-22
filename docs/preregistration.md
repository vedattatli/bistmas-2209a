# Pre-registration (frozen and git-tagged as `prereg-v1` before the first full jury run)

Status: **FROZEN** (tag `prereg-v1`); the tagged commit, the data vintages and
the prompt hashes are then filled in below and never edited again (later changes go to a new version).

## Primary hypothesis test (H1)
* Headline comparison: **hybrid Top-3 (by jury score J rank) vs deterministic Top-3 (by composite rank)**,
  equal-weight, always invested, monthly rebalance, decisions on the last session of each month from
  2017-12-29 to 2024-11-29 (84 holding months, Jan-2018 … Dec-2024), fills at the next session's VWAP
  (a halted name is filled at its first tradable session, cash until then), costs 10 bps commission + 20 bps
  slippage per side (sensitivity 0 / 30 / 60 / 100 bps), delisting haircut 0 (sensitivity 20 %).
* Primary statistic: paired Newey-West (3 lags) t-test on monthly **active returns vs the point-in-time
  equal-weight universe benchmark** (same dates, same fills, same costs), plus a stationary block-bootstrap
  95 % CI (1,000 draws, mean block 4 months); reported in TL, USD and CPI-deflated terms. The deflated Sharpe
  ratio of the monthly active return uses the trial count of `data/runs/ledger.jsonl`.
* Secondary: the same comparison for Top-5 and Top-10; BUY-only variant with θ = 0.75 (BUY count reported,
  θ ± 0.05 sensitivity); random-jury null (500 seeds through the identical consensus and portfolio code);
  Spearman(J, composite) per month (echo check) and incremental information coefficient of J beyond the
  composite; Kish effective votes and pairwise agent agreement.
* Multiple-comparison control: the number of configurations examined is printed from the experiment
  ledger; Bonferroni-adjusted thresholds are reported next to raw p-values.
* Power: with 84 monthly observations the minimum detectable monthly active return at 80 % power is
  reported next to the estimate; Top-10 is the inference portfolio if Top-3 is under-powered, Top-3 stays the
  illustration the proposal asks for.

## Fixed parameters (not tuned on 2018–2024)
* Universe: official BIST-100 reconstruction (effective-dated quarterly reviews + curated mid-period events),
  one share class per issuer, ≥ 12 months listed, 6-month average daily traded value ≥ 5 mn TL, negative
  book equity excluded. Rule-based universe only as a robustness row.
* Features: ratios per template (industrial / bank / insurer / other financial) from first-reported KAP
  statements (İş Yatırım fallback only for delisted names, ≤ 2022Q3), TTM = FY(prev) + YTD − YTD(prior) from
  the same filing (no differencing across the TMS-29 basis change), price features from adjusted series with
  flagged ex-date returns excluded.
* Normalisation: robust z within sector group and month (median / MAD, clip ± 3, groups < 5 pooled into the
  parent group); three pillars with equal weights (fundamental / valuation / momentum-risk); Top-10 screen by
  the equal-weight composite.
* Jury: model `qwen3.5:9b`, digest `6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7`,
  prompt version v1 (file hashes in `prompts/REGISTRY.md`), temperature 0, seed 20260923, `num_ctx` 8192,
  `num_predict` 700, thinking off, one candidate per call, three agents (quality/growth, value, risk/timing)
  with an information partition (each agent sees only its pillar's z-scores and percentiles), per-month
  pseudonyms, no names / tickers / dates / absolute amounts in any prompt (asserted), one validation retry
  then abstain (abstentions are excluded, never imputed), J = mean of the available scores, Top-3 by J rank.
* Prompt development used only a 2018–2019 development slice (dry runs and the smoke test); nothing in the
  prompts was changed after looking at 2020–2024 outputs.

## H2 (explainability / consistency) operationalisation
* Test-retest ICC at temperature 0.7 on a stratified 10 % sample (5 seeds).
* Robustness of the Top-3 to ± 0.25 z input noise and to a prompt paraphrase (Kendall τ, overlap).
* Rationale groundedness rate (cited fields must exist in the payload) and leak-flag rate (tickers,
  distinctive company-name words, years, month names, event keywords).
* Human blind rubric on 30 rationales (three raters, Cohen's κ).

## Audits committed to
Re-identification probe (which company / which quarter, vs chance), direct-elicitation test of BIST closes,
four-arm date audit (revealed / date-only / masked / transplanted) on a stratified 10 % sample, fake-date
shift test, Lookahead-Propensity probe via token log-probabilities.

## Freeze record (filled at tagging time)
| item | value |
|---|---|
| tag / commit | `prereg-v1` @ `959e8ff47b4d` (tag applied to the commit that contains this table) |
| features vintage (`features_monthly.parquet` sha256[:16]) | `b169944de7f4320a` |
| prices vintage | `b415bf7c06d747b1` |
| statements vintage | `8d45b86e3c88fcb3` |
| config hash (`bistmas.shared.ledger.config_hash`) | `c91cbe509b8d` |
| deterministic run recorded before the freeze | `det-v1` (ledger line 2026-09-24T03:13:09+00:00, code `959e8ff47b4d`) |
