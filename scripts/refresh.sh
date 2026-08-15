#!/usr/bin/env bash
# Incremental refresh of the 26-symbol panel.  Run this ON THE VPS.
#
#   ssh vpn-sg
#   cd ~/VI-Dashboard && git fetch origin main && git checkout main && git reset --hard origin/main
#   bash refresh.sh                 # last 10 days, all 26 symbols, all 3 timeframes
#   bash refresh.sh 30              # last 30 days instead
#
# It leaves ONE small zip at ~/vi_refresh_<date>.zip.  Ten days of 26 symbols is
# roughly 200KB against 57MB for the full history, because the repository already
# holds everything up to the last refresh.
#
# Claude Code's container cannot reach api.binance.com or fapi.binance.com (both
# return no route), so the fetch has to happen here.  Nothing else does.

set -uo pipefail
set +H          # history expansion turns "!!" into the previous command when a
                # human pastes this into an interactive shell, which breaks the
                # quoting and strands the terminal at a ">" prompt

DAYS="${1:-10}"
EXPORTER="${EXPORTER:-backend/tools/export_klines.py}"
OUT="${OUT:-./exports_refresh}"
MARKET="${MARKET:-futures}"        # USDT-M perpetuals, same as everything else

# Deliberately overlapping: the previous run's final bar was still forming when
# it was taken, and the exchange occasionally restates settled bars.  The
# overlap is what lets `ingest --merge` notice either one instead of trusting
# the splice blindly.  Ten days is cheap; do not trim this to one.
START="$(date -u -d "${DAYS} days ago" +%Y-%m-%d 2>/dev/null \
       || date -u -v-"${DAYS}"d +%Y-%m-%d)"

SYMBOLS="${SYMBOLS:-BTCUSDT ETHUSDT BNBUSDT SOLUSDT XRPUSDT DOGEUSDT APTUSDT ARBUSDT \
OPUSDT SUIUSDT SEIUSDT TIAUSDT WLDUSDT FETUSDT RENDERUSDT PENDLEUSDT LDOUSDT DYDXUSDT \
ENAUSDT WIFUSDT TAOUSDT HYPEUSDT 1000PEPEUSDT 1000SHIBUSDT 1000BONKUSDT 1000FLOKIUSDT}"

if [[ ! -f "$EXPORTER" ]]; then
  echo "exporter not found at $EXPORTER" >&2
  echo "cd into your VI-Dashboard checkout first (cd ~/VI-Dashboard)," >&2
  echo "or set EXPORTER=/path/to/export_klines.py" >&2
  exit 1
fi

rm -rf "$OUT"; mkdir -p "$OUT"
echo "refreshing from $START (last $DAYS days), $(echo $SYMBOLS | wc -w) symbols"
echo

ok=(); failed=()
for S in $SYMBOLS; do
  printf '  %-16s' "$S"
  if python3 "$EXPORTER" --symbol "$S" --market "$MARKET" \
       --intervals 1h,4h,1d --start "$START" --out "$OUT" >/dev/null 2>&1; then
    ok+=("$S"); echo "ok"
  else
    failed+=("$S"); echo "FAILED"
  fi
done

echo
echo "ok ${#ok[@]}/$(echo $SYMBOLS | wc -w)"
[[ ${#failed[@]} -gt 0 ]] && echo "FAILED: ${failed[*]}"
if [[ ${#ok[@]} -eq 0 ]]; then
  echo "nothing exported -- check the exporter path and tickers" >&2
  exit 1
fi

ZIP=~/vi_refresh_$(date -u +%Y%m%d).zip
rm -f "$ZIP"
if command -v zip >/dev/null 2>&1; then
  (cd "$OUT" && zip -qj "$ZIP" ./*.csv)
else
  # zip is not installed everywhere; tar is.  ingest reads either once unpacked.
  ZIP=~/vi_refresh_$(date -u +%Y%m%d).tar.gz
  tar -czf "$ZIP" -C "$OUT" .
fi

echo
ls -lh "$ZIP"
echo
echo "now, from your local machine (PowerShell):"
echo "  scp vpn-sg:$ZIP \"\$env:USERPROFILE\\Desktop\\\""
echo
echo "then upload it, and it gets spliced onto the stored history with:"
echo "  python -m vibt.ingest <unpacked-dir> --gzip --merge"
