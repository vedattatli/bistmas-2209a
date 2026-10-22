#!/bin/zsh
# Resumable KAP filing listing: re-runs after the per-IP quota lock until the listing is complete.
cd "$(dirname "$0")/.."
# Never write into the frozen backtest store by accident: forward runs must set BISTMAS_CONFIG=configs/forward.toml
# (own store data/store_forward). Re-building the frozen store needs BISTMAS_ALLOW_FROZEN=1 on purpose.
if [ -z "$BISTMAS_CONFIG" ] && [ "$BISTMAS_ALLOW_FROZEN" != "1" ]; then
  echo "refusing to run: set BISTMAS_CONFIG=configs/forward.toml (or BISTMAS_ALLOW_FROZEN=1 for the frozen store)" >&2
  exit 2
fi
RAW=$(uv run python -c "from bistmas.shared.config import load_settings; print(load_settings().paths.raw)" 2>/dev/null | tail -1)
for i in {1..120}; do
  uv run python -c "
import json
from bistmas.shared.config import load_settings
from bistmas.data.filings import run_filing_listing
from datetime import date
import sys
end = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else date(2024, 12, 31)
s = run_filing_listing(load_settings(), end=end)
print(json.dumps(s, default=str))
open(load_settings().paths.raw / 'kap' / 'listing_last.json','w').write(json.dumps(s, default=str))
" "${1:-2024-12-31}" 2>&1 | grep -v " INFO " | tail -3
  reason=$(python3 -c "import json;print(json.load(open('${RAW}/kap/listing_last.json'))['stopped_reason'])" 2>/dev/null)
  echo "$(date '+%H:%M:%S') attempt $i -> $reason"
  if [ "$reason" = "complete" ]; then echo "LISTING COMPLETE"; break; fi
  sleep 1200
done
