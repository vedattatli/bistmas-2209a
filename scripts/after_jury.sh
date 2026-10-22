#!/bin/zsh
# Runs once the v1 jury run is complete: hybrid backtest + random-jury null, audits, forward dry run, report.
# Usage: ./scripts/after_jury.sh [jury-run] ; each step logs to data/runs/after_jury.log and continues on failure.
set -u
cd "$(dirname "$0")/.."
run="${1:-prereg-v1}"
log="data/runs/after_jury.log"
step() { echo "$(date '+%F %T') >>> $*" >> "$log"; "$@" >> "$log" 2>&1; echo "$(date '+%F %T') <<< exit $?" >> "$log"; }
# prereg-v1 results stay blind until the corrected specification (prereg-v2) is frozen: see docs/preregistration.md
if [ "$run" = "prereg-v1" ] && ! git rev-parse -q --verify refs/tags/prereg-v2 >/dev/null; then
  echo "$(date '+%F %T') ${run}: jury complete; hybrid/audits deferred until tag prereg-v2 exists (blinding)" >> "$log"; exit 0
fi
if ! grep -q "jury run ${run} complete" "data/runs/${run}/jury.log" 2>/dev/null; then
  echo "$(date '+%F %T') jury run ${run} is not complete; aborting" >> "$log"; exit 1
fi
step uv run bistmas backtest --spec hybrid --jury-run "$run" --run-name "hybrid-${run#prereg-}"
if [ "$run" = "prereg-v1" ]; then
  # prereg-v1 is the secondary (as-registered) analysis: audits and the forward run belong to prereg-v2
  step uv run bistmas report --figures
  echo "$(date '+%F %T') AFTER-JURY CHAIN DONE (${run})" >> "$log"; exit 0
fi
step uv run bistmas audit --jury-run "$run"
step uv run bistmas audit --jury-run "$run" --robustness
step uv run bistmas report --figures
echo "$(date '+%F %T') AFTER-JURY CHAIN DONE" >> "$log"
