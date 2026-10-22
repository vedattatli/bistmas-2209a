# Prompt registry

Prompt files are immutable once a version has been used in a run. The runner refuses to load a
version whose file hash differs from the one recorded here. Add a new version directory instead
of editing an existing one.

| version | file | sha256 | first used in run | note |
|---|---|---|---|---|
| v1 | prompts/v1/system_quality_growth.md | 809c9adbd11d4846928a077b86ec910b91f53ec67224217c922089db432d4c02 | (first full run pending) | persona + information partition |
| v1 | prompts/v1/system_value.md | cb535012944ab11e57f66972cda18f398c45d6aedc2de23091a4aff1e7855e8e | (first full run pending) | persona + information partition |
| v1 | prompts/v1/system_risk_timing.md | 255eca23e1767f83a9e50ef5ae319e3223e7b628d1ebae242e80d4e388b7820c | (first full run pending) | persona + information partition |
| v1 | prompts/v1/user_template.md | 40e0af1bb1b207bb6a33e3a88a6a2901fe1cc17672b672f3c87ef610594f1f78 | (first full run pending) | payload wrapper + metric-orientation note (dev slice 2018-01, 2026-09-24) |
| v2 | prompts/v2/system_quality_growth.md | 2b9655c65d4db7204c4539db89882316dbcd86788d0b54efd6feaff1770f6675 | (prereg-v2 run pending) | v1 wording; percentiles are within the peer (normalisation) group |
| v2 | prompts/v2/system_value.md | 0affe2497c76d25c9bae0e1dce5fb95721436e7d1e93f94a04bfcf6eb57eeda7 | (prereg-v2 run pending) | v1 wording; percentiles are within the peer (normalisation) group |
| v2 | prompts/v2/system_risk_timing.md | 34dbb8a12bf506a447b485a27dfbb444c403e2ec9fccd62cf2d5a3acc8856383 | (prereg-v2 run pending) | v1 wording; percentiles are within the peer (normalisation) group |
| v2 | prompts/v2/user_template.md | cb9acacaec0759713c5ba8f2016382fd9c445b9186113d8ad152a0af2c847197 | (prereg-v2 run pending) | peer-group percentiles; sector label is the pooled normalisation group |
| v2_paraphrase | prompts/v2_paraphrase/system_quality_growth.md | d91e7d4d73a26d9e19387ac4b1e53a1a7d3168a8dd5b89cea37e1682d26a60ca | (robustness audit, H2) | reworded v2 for the paraphrase-robustness audit only; same instructions, schema and placeholders |
| v2_paraphrase | prompts/v2_paraphrase/system_value.md | 43e62fbfd91bd4fbda3354d3df3ab720606948bfff94fae9a0d18a36aed2ecc1 | (robustness audit, H2) | reworded v2 for the paraphrase-robustness audit only; same instructions, schema and placeholders |
| v2_paraphrase | prompts/v2_paraphrase/system_risk_timing.md | 107d3cf309144460a098ad13e8019689df5d10cc8e8aa23471591347cd3b4ff2 | (robustness audit, H2) | reworded v2 for the paraphrase-robustness audit only; same instructions, schema and placeholders |
| v2_paraphrase | prompts/v2_paraphrase/user_template.md | 701b728b08493b6612aff57ba1de14e3bfeb46786e5acd1c42e71e8a9007cac0 | (robustness audit, H2) | reworded v2 for the paraphrase-robustness audit only; same instructions, schema and placeholders |
