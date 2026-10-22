# v2 ablations (after the pre-registered v1 run)

The v1 run (`prereg-v1`) fixes the design that answers H1/H2. Everything below is exploratory and is
reported as ablation rows, never as the headline.

## (e) Cross-family jury
* Same prompts, same payloads, same cache mechanics; each agent runs on its own model family so that
  the three verdicts are not three draws from one model's prior:
  `bistmas jury --run xfam-v2 --agent-models quality_growth=qwen3.5:9b,value=gemma4:12b-it-qat,risk_timing=llama3.1:8b`
  (models pulled locally; digests recorded per agent in `jury_summary.json` and in every score row).
* Reported next to v1: pairwise Spearman between agents, Krippendorff's alpha, Kish effective votes,
  hybrid Top-3/5/10 vs deterministic with the same tests as v1, random-jury null percentile.
* Thinking mode stays off for all three; `num_ctx`, `num_predict`, temperature and seed as in v1.

## (f) KAP disclosure text
* Source (quota-free): `GET /tr/api/batch-news/file-by-year/{oid}/{year}` returns a ZIP with a Word-HTML
  yearbook of the company's general disclosures (timestamped narratives, no financial-report bodies).
* Point-in-time rule: only disclosures with `publish_ts ≤ decision date 18:00` enter the month's block;
  the block is the last N (= 5) disclosure titles + first sentence, truncated to 600 tokens.
* Masking problem: narratives name the company, its subsidiaries, executives, banks and places, and
  carry dates. Before any text reaches the jury it must pass the same leak assertion as the numeric
  payload. The scrubbing plan: (1) replace every distinctive title word of the issuer and of every
  security-master title with `[ORG]`, (2) replace dates, years and month names with `[DATE]`,
  (3) replace amounts with `[AMOUNT]`, (4) drop sentences that still contain an uppercase token of ≥ 5
  letters not in a Turkish word list; (5) the groundedness audit is extended with an
  "organisation-name recovered?" probe. Only if the residual re-identification rate on a 10 % sample is
  at chance does the block go into an ablation run (`--text-block`).
* Decision: implemented only if the residual leak rate is at chance; otherwise the ablation is reported
  as "not run — masking insufficient" with the measured rate. The proposal's text agent is then
  covered by the forward run, where memorisation is not a concern.

## (g) Rule-based universe
* `configs/base.toml [universe] provider = "rule_based"` rebuilds the universe from the BIST ground
  rules (free-float cap and traded-value rank merge) as a robustness row for the deterministic
  results; it cannot reproduce fast-track, seasoning and substitution rules and is labelled as such.

## (h) Exclude 2024
* Re-run of the H1 tests on decisions ≤ 2023-11-30 (the last year is the most contaminated for any
  LLM trained through 2025), reported as a sensitivity row.
