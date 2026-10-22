#!/bin/zsh
# Fetch financial-report bodies for whatever the filing listing has produced so far; repeat until the
# listing loop has finished and no body is missing.
cd "$(dirname "$0")/.."
# Never write into the frozen backtest store by accident: forward runs must set BISTMAS_CONFIG=configs/forward.toml
# (own store data/store_forward). Re-building the frozen store needs BISTMAS_ALLOW_FROZEN=1 on purpose.
if [ -z "$BISTMAS_CONFIG" ] && [ "$BISTMAS_ALLOW_FROZEN" != "1" ]; then
  echo "refusing to run: set BISTMAS_CONFIG=configs/forward.toml (or BISTMAS_ALLOW_FROZEN=1 for the frozen store)" >&2
  exit 2
fi
RAW=$(uv run python -c "from bistmas.shared.config import load_settings; print(load_settings().paths.raw)" 2>/dev/null | tail -1)
for i in {1..60}; do
  uv run python -c "
import json
from bistmas.shared.config import load_settings
from bistmas.data.filings import build_filing_index
from bistmas.data.statements_fetch import run_statement_fetch
s = load_settings()
idx = build_filing_index(s)
r = run_statement_fetch(s)
r['index_rows'] = int(len(idx))
print(json.dumps(r, default=str))
open(s.paths.raw / 'kap' / 'bodies_last.json','w').write(json.dumps(r, default=str))
" 2>&1 | grep -v " INFO " | tail -2
  listing_running=$(ps aux | grep -c "[r]un_listing.sh")
  reason=$(python3 -c "import json;print(json.load(open('${RAW}/kap/bodies_last.json'))['stopped_reason'])" 2>/dev/null)
  echo "$(date '+%H:%M:%S') bodies attempt $i -> $reason (listing running: $listing_running)"
  if [ "$listing_running" = "0" ] && [ "$reason" = "complete" ]; then
    # the listing may have grown while this attempt ran: rebuild the index and only stop when nothing is missing
    missing=$(uv run python -c "
import glob
from bistmas.shared.config import load_settings
from bistmas.data.filings import build_filing_index
idx = build_filing_index(load_settings())
have = {int(p.split('/')[-1].split('.')[0]) for p in glob.glob(str(load_settings().paths.raw / 'kap' / 'attachment' / '*.json.gz'))}
print(int((~idx.disclosure_index.isin(have)).sum()))
" 2>/dev/null | tail -1)
    if [ "$missing" = "0" ]; then echo "BODIES COMPLETE"; break; fi
    echo "$(date '+%H:%M:%S') index grew: $missing bodies still missing, continuing"
    continue
  fi
  sleep 1200
done
