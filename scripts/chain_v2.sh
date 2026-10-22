#!/bin/zsh
# prereg-v2 run chain, started from a checkout of tag prereg-v2 whose data/ entries link to the main data
# directory: waits until the prereg-v1 jury has finished (one model on the Mac at a time), then runs the
# prereg-v2 jury against the amended store (configs/prereg_v2.toml) and its after-jury chain.
# Usage: screen -dmS bistmas-jury-v2 zsh -c './scripts/chain_v2.sh'
set -u
cd "$(dirname "$0")/.."
export BISTMAS_CONFIG=configs/prereg_v2.toml
until grep -qE "jury run prereg-v1 complete|gave up" data/runs/prereg-v1/jury.log 2>/dev/null; do sleep 120; done
./scripts/run_jury.sh prereg-v2 && ./scripts/after_jury.sh prereg-v2
