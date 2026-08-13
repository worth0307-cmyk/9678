#!/usr/bin/env bash
# Export the 6-symbol universe from Binance and load it into this repo.
#
# Run this on a machine that can reach Binance (your VPS / the box running
# VI-Dashboard).  This session's egress policy blocks api.binance.com and
# fapi.binance.com, so the fetch cannot happen inside Claude Code.
#
#   VI_DASHBOARD=~/VI-Dashboard ./scripts/fetch_universe.sh
#
# Then copy ./exports here and run:  python -m vibt.ingest ./exports

set -euo pipefail

VI_DASHBOARD="${VI_DASHBOARD:-../VI-Dashboard}"
EXPORTER="$VI_DASHBOARD/backend/tools/export_klines.py"
OUT="${OUT:-./exports}"
START="${START:-2024-01-01}"
MARKET="${MARKET:-futures}"        # matches the dashboard: USDT-M perpetuals
SYMBOLS="${SYMBOLS:-BTCUSDT BNBUSDT ETHUSDT HYPEUSDT SOLUSDT TAOUSDT}"

if [[ ! -f "$EXPORTER" ]]; then
  echo "exporter not found at $EXPORTER" >&2
  echo "set VI_DASHBOARD to your VI-Dashboard checkout, e.g." >&2
  echo "  VI_DASHBOARD=~/VI-Dashboard $0" >&2
  exit 1
fi

mkdir -p "$OUT"
for S in $SYMBOLS; do
  echo "=== $S"
  # Symbols listed after $START simply return less history; that is fine and the
  # ingest step reports the real coverage per symbol.
  python3 "$EXPORTER" --symbol "$S" --market "$MARKET" \
      --intervals 1h,4h,1d --start "$START" --out "$OUT" || {
        echo "  !! $S failed (not listed on $MARKET? wrong ticker?) — continuing" >&2
      }
done

echo
echo "exported to $OUT"
echo "next:  python -m vibt.ingest $OUT"
