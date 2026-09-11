#!/usr/bin/env bash
# 每天一次：增量抓币安 -> 并入 data/ -> 生成当天的目标持仓
#
#   bash ~/9678/scripts/vps_daily.sh              # 看今天该做什么，不写日志
#   bash ~/9678/scripts/vps_daily.sh --write      # 确认后记进 paper/signals.csv
#   DAYS=30 bash ~/9678/scripts/vps_daily.sh      # 多抓几天（补一段空窗）
#
# 在 00:05 UTC 跑。日线在 00:00 UTC 收盘，而这个仓库测过：执行拖一天，
# Sharpe 掉 0.3~0.6，所以晚跑的代价是真金白银的，不是洁癖。
#
# 抓取窗口默认 10 天而不是 1 天，是故意重叠的：上一次的最后一根 K 线当时
# 还在形成，而交易所偶尔会改写已结算的历史。有重叠，`ingest --merge` 才能
# 分辨「最后一根被补完」和「历史被改写」——后者会让之前所有回测失效。

set -uo pipefail
set +H

HOME_DIR="${HOME_DIR:-$HOME/9678}"
VENV="${VENV:-$HOME_DIR/.venv}"
PY="$VENV/bin/python"
DAYS="${DAYS:-10}"
EQUITY="${EQUITY:-10000}"
INTERVALS="${INTERVALS:-1d}"     # 纸面信号只读日线；要补 4h/1h 就改这里
OUT="${OUT:-$HOME_DIR/.refresh}"
SYMBOLS="${SYMBOLS:-BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,TAOUSDT,HYPEUSDT}"

WRITE=""
[[ "${1:-}" == "--write" ]] && WRITE="--write"

[[ -x "$PY" ]] || { echo "没有 $PY —— 先跑 scripts/vps_setup.sh" >&2; exit 1; }
cd "$HOME_DIR" || exit 1

# 这个脚本里有 rm -rf "$OUT"，而 OUT 可以由环境变量覆盖。无人值守跑的东西
# 不该有一条能被一个手滑的环境变量指到家目录的删除命令。
case "$OUT" in
  "$HOME_DIR"/.*|"$HOME_DIR"/*) : ;;
  *) echo "OUT 必须在 $HOME_DIR 之下，现在是 '$OUT'，拒绝执行" >&2; exit 1 ;;
esac
[[ "$OUT" == "/" || "$OUT" == "$HOME" || "$OUT" == "$HOME_DIR" ]] && {
  echo "OUT 不能是 '$OUT'" >&2; exit 1; }

START="$(date -u -d "${DAYS} days ago" +%Y-%m-%d 2>/dev/null \
       || date -u -v-"${DAYS}"d +%Y-%m-%d)"

echo "=== $(date -u '+%Y-%m-%d %H:%M:%S') UTC   增量抓取 ${START} 起"
rm -rf "$OUT"
"$PY" scripts/fetch_binance.py \
  --symbols "$SYMBOLS" --intervals "$INTERVALS" \
  --start "$START" --out "$OUT" --skip-funding || {
    echo "抓取失败，保留现有数据不动" >&2; exit 1; }

# fetch_binance.py 对单个币的失败是「记进 manifest 然后继续」，整体仍然退出 0。
# 无人值守的时候这会变成：6 个币抓到 4 个，静默并进去，然后信号用一份参差不齐的
# 数据生成。下游其实是安全的（latest_complete_bar 取所有币的最小值，会退回到
# 更早的日期，而 43 号脚本会报数据过期），但这件事必须在这里说出来。
if [[ -f "$OUT/MANIFEST.csv" ]] && grep -q "FAILED" "$OUT/MANIFEST.csv"; then
  echo "⚠️  有币种抓取失败：" >&2
  grep "FAILED" "$OUT/MANIFEST.csv" >&2
  echo "   仍然并入成功的部分；信号会退回到所有币都有数据的最后一天。" >&2
fi

echo
echo "=== 并入 data/（--merge 会报告改写和缺口）"
"$PY" -m vibt.ingest "$OUT" --merge --gzip || {
    echo "ingest 失败，data/ 未改动" >&2; exit 1; }

echo
echo "=== 今天的目标持仓"
"$PY" scripts/43_paper_signals.py --equity "$EQUITY" $WRITE

# 抓下来的原始 CSV 已经并进 data/ 了，留着只会越堆越多
rm -rf "$OUT"

if [[ -n "$WRITE" ]]; then
  echo
  echo "成交后把 fill_price / fill_qty / filled_at / fee_paid 填回 paper/signals.csv，"
  echo "攒够几十笔再跑： $PY scripts/43_paper_signals.py --reconcile"
fi
