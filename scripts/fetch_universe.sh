#!/usr/bin/env bash
# Export the 6-symbol universe from Binance.
#
# Run this ON THE VPS, inside your VI-Dashboard checkout:
#
#   ssh vpn-sg
#   cd ~/VI-Dashboard && git fetch origin main && git checkout main && git reset --hard origin/main
#   bash /path/to/fetch_universe.sh          # or paste the loop below
#
# It is a thin loop over backend/tools/export_klines.py, whose defaults
# (--market futures, --intervals 1h,4h,1d, --start 2024-01-01) already match
# the BTCUSDT files in data/.  Nothing here needs to be installed.
#
# This session's egress policy blocks api.binance.com and fapi.binance.com, so
# the fetch cannot run inside Claude Code -- it has to happen on the VPS.

set -uo pipefail
set +H          # history expansion turns "!!" into the previous command when a
                # human pastes this into an interactive shell, which breaks the
                # quoting and strands the terminal at a ">" prompt

EXPORTER="${EXPORTER:-backend/tools/export_klines.py}"
OUT="${OUT:-./exports}"
START="${START:-2024-01-01}"
MARKET="${MARKET:-futures}"        # USDT-M perpetuals, same as the dashboard
SYMBOLS="${SYMBOLS:-BTCUSDT BNBUSDT ETHUSDT HYPEUSDT SOLUSDT TAOUSDT}"

if [[ ! -f "$EXPORTER" ]]; then
  echo "exporter not found at $EXPORTER" >&2
  echo "cd into your VI-Dashboard checkout first, or set EXPORTER=/path/to/export_klines.py" >&2
  exit 1
fi

mkdir -p "$OUT"
ok=(); failed=()
for S in $SYMBOLS; do
  echo "=== $S"
  # A symbol listed after $START just returns less history; ingest reports the
  # real coverage per symbol, so short histories are fine, not errors.
  if python3 "$EXPORTER" --symbol "$S" --market "$MARKET" \
       --intervals 1h,4h,1d --start "$START" --out "$OUT"; then
    ok+=("$S")
  else
    failed+=("$S")
    echo "  FAILED $S — wrong ticker, or not listed on $MARKET. Continuing." >&2
  fi
done

echo
echo "exported OK : ${ok[*]:-none}"
[[ ${#failed[@]} -gt 0 ]] && echo "FAILED      : ${failed[*]}"
[[ ${#ok[@]} -eq 0 ]] && echo "nothing exported — check the exporter path and tickers" >&2
echo "files in $OUT:"
ls -1 "$OUT" | sed 's/^/  /'
echo
echo "next, from your local machine:"
echo "  scp -r vpn-sg:~/VI-Dashboard/exports \"\$env:USERPROFILE\\Desktop\\VI数据\""
echo "  python -m vibt.ingest \"\$env:USERPROFILE\\Desktop\\VI数据\""
