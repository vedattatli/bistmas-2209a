#!/bin/zsh
# Resumable local-LLM jury run: keeps the Mac awake, restarts after Ollama hiccups, stops when complete.
# Usage: ./scripts/run_jury.sh <run-name> [max-attempts]
set -u
cd "$(dirname "$0")/.."
run="${1:?run name, e.g. prereg-v1}"
max_attempts="${2:-40}"
log="data/runs/${run}/jury.log"
mkdir -p "data/runs/${run}"
attempt=0
while [ "$attempt" -lt "$max_attempts" ]; do
  attempt=$((attempt + 1))
  echo "$(date '+%F %T') jury attempt ${attempt} (run ${run})" >> "$log"
  caffeinate -i uv run bistmas jury --run "$run" >> "$log" 2>&1
  code=$?
  echo "$(date '+%F %T') attempt ${attempt} exit ${code}" >> "$log"
  if [ "$code" -eq 0 ]; then
    echo "$(date '+%F %T') jury run ${run} complete" >> "$log"
    exit 0
  fi
  sleep 60
done
echo "$(date '+%F %T') gave up after ${max_attempts} attempts" >> "$log"
exit 1
