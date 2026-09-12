#!/usr/bin/env bash
# 每天一次：增量抓币安 -> 并入 data/ -> 生成当天的目标持仓
#
#   bash ~/9678/scripts/vps_daily.sh              # 看今天该做什么，不写日志
#   bash ~/9678/scripts/vps_daily.sh --write      # 记进 paper/signals.csv（cron 用这个）
#   DAYS=30 bash ~/9678/scripts/vps_daily.sh      # 多抓几天（补一段空窗）
#
# 在 00:05 UTC 跑。日线在 00:00 UTC 收盘，而这个仓库测过：执行拖一天，
# Sharpe 掉 0.3~0.6，所以晚跑的代价是真金白银的，不是洁癖。
#
# 抓取窗口默认 10 天而不是 1 天，是故意重叠的：上一次的最后一根 K 线当时
# 还在形成，而交易所偶尔会改写已结算的历史。有重叠，`ingest --merge` 才能
# 分辨「最后一根被补完」和「历史被改写」——后者会让之前所有回测失效。
#
# 无人值守时最危险的不是报错，是**静默地做错**。所以：任何一步失败都停住而不是
# 带着坏数据往下走，单个币抓失败会显式报出来，成功与失败都会推送通知（配了的话）。

set -uo pipefail
set +H

HOME_DIR="${HOME_DIR:-$HOME/9678}"
VENV="${VENV:-$HOME_DIR/.venv}"
PY="$VENV/bin/python"
DAYS="${DAYS:-10}"
EQUITY="${EQUITY:-10000}"
# 纸面信号只读日线，但三个周期必须一起刷新。只抓 1d 的话，1h/4h 会停在上一次的
# 位置，而 ingest 的交叉校验会拿**完整的日线**去比**半天的 1h 重采样**，于是每天
# 都报一条"1h->1d 不一致"。那不是数据问题，是我自己造出来的假警报——而一个每天
# 都响的告警等于没有告警。10 天 x 6 币的 1h 只有 1440 根，不值得省。
INTERVALS="${INTERVALS:-1h,4h,1d}"
OUT="${OUT:-$HOME_DIR/.refresh}"
SYMBOLS="${SYMBOLS:-BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,TAOUSDT,HYPEUSDT}"

WRITE=""
[[ "${1:-}" == "--write" ]] && WRITE="--write"

[[ -x "$PY" ]] || { echo "没有 $PY —— 先跑 scripts/vps_setup.sh" >&2; exit 1; }
cd "$HOME_DIR" || exit 1

# 通知配置放在 .env（chmod 600，已 gitignore），不进 crontab、不进仓库
# shellcheck disable=SC1091
[[ -f "$HOME_DIR/.env" ]] && . "$HOME_DIR/.env"

# 这个脚本里有 rm -rf "$OUT"，而 OUT 可以由环境变量覆盖。无人值守跑的东西
# 不该有一条能被一个手滑的环境变量指到家目录的删除命令。
case "$OUT" in
  "$HOME_DIR"/.*|"$HOME_DIR"/*) : ;;
  *) echo "OUT 必须在 $HOME_DIR 之下，现在是 '$OUT'，拒绝执行" >&2; exit 1 ;;
esac
[[ "$OUT" == "/" || "$OUT" == "$HOME" || "$OUT" == "$HOME_DIR" ]] && {
  echo "OUT 不能是 '$OUT'" >&2; exit 1; }

LOG="$(mktemp)"
trap 'rm -f "$LOG"' EXIT

# ---------------------------------------------------------------- 单实例
# 两个实例同时跑会互相破坏，而且是**看不出来**的那种：它们并发重写同一批
# data/*.csv.gz，并且都会在还是空的 signals.csv 上判定「今天该再平衡」，
# 于是同一天写进两本账，再平衡的节奏跟着算乱。重复的 crontab 行、或者手动
# 跑撞上 cron，都会造成这个。拿不到锁就安静退出 —— 那一次本来就不该跑。
# 前缀用 "---" 而不是 "==="，才不会被 --verify 当成一次真的运行。
LOCKFILE="$HOME_DIR/.daily.lock"
if command -v flock >/dev/null 2>&1; then
  exec 9>"$LOCKFILE" || { echo "打不开锁文件 $LOCKFILE" >&2; exit 1; }
  if ! flock -n 9; then
    echo "--- $(date -u '+%Y-%m-%d %H:%M:%S') UTC   跳过：另一个实例正在跑（$LOCKFILE）"
    exit 0
  fi
else
  echo "⚠️  这台机器没有 flock，无法保证单实例运行" >&2
fi

# ---------------------------------------------------------------- 通知
# 可选。没配就只写日志——一个没配通知的 cron 任务仍然应该能跑。
notify() {
  local text="$1"
  [[ -n "${TG_TOKEN:-}" && -n "${TG_CHAT:-}" ]] || return 0
  curl -sS --max-time 20 -o /dev/null \
    "https://api.telegram.org/bot${TG_TOKEN}/sendMessage" \
    --data-urlencode "chat_id=${TG_CHAT}" \
    --data-urlencode "text=${text}" \
    --data-urlencode "disable_web_page_preview=true" \
    || echo "（通知发送失败，不影响主流程）" >&2
}

die() {
  local msg="$1"
  echo "$msg" >&2
  # 失败要吵。静默失败的 cron 任务会安静地空跑几个星期，
  # 等你想起来看的时候，那段时间的数据已经永远拿不回来了。
  notify "❌ 纸面信号 $(date -u '+%m-%d %H:%M')UTC 失败
$msg
$(tail -5 "$LOG" 2>/dev/null)"
  exit 1
}

run() { "$@" 2>&1 | tee -a "$LOG"; return "${PIPESTATUS[0]}"; }

START="$(date -u -d "${DAYS} days ago" +%Y-%m-%d 2>/dev/null \
       || date -u -v-"${DAYS}"d +%Y-%m-%d)"

echo "=== $(date -u '+%Y-%m-%d %H:%M:%S') UTC   增量抓取 ${START} 起" | tee -a "$LOG"

# 无人值守（--write）应该在 00:0x UTC 跑。跑在别的时刻，唯一的解释是 cron 按
# 本机时区计时、CRON_TZ 没生效 —— 而那正是最坏的失败方式：它照常出结果、照常
# 推送，只是每天都晚若干小时，而这个仓库测过执行拖一天 Sharpe 掉 0.3~0.6。
if [[ -n "$WRITE" && "$(date -u +%H)" != "00" ]]; then
  {
    echo "⚠️  当前 UTC 时刻是 $(date -u '+%H:%M')，不是 00:0x。"
    echo "    无人值守的任务跑在这个时间，多半是 CRON_TZ 没生效（本机时区 $(date +%Z)）。"
    echo "    确认： crontab -l   修法： timedatectl set-timezone UTC 然后重跑 vps_cron.sh"
  } | tee -a "$LOG" >&2
fi
rm -rf "$OUT"
run "$PY" scripts/fetch_binance.py \
  --symbols "$SYMBOLS" --intervals "$INTERVALS" \
  --start "$START" --out "$OUT" --skip-funding \
  || die "抓取失败，保留现有数据不动"

# fetch_binance.py 对单个币的失败是「记进 manifest 然后继续」，整体仍然退出 0。
# 无人值守的时候这会变成：6 个币抓到 4 个，静默并进去，然后信号用一份参差不齐的
# 数据生成。下游其实是安全的（latest_complete_bar 取所有币的最小值，会退回到
# 更早的日期，而 43 号脚本会报数据过期），但这件事必须在这里说出来。
PARTIAL=""
if [[ -f "$OUT/MANIFEST.csv" ]] && grep -q "FAILED" "$OUT/MANIFEST.csv"; then
  PARTIAL="$(grep FAILED "$OUT/MANIFEST.csv")"
  {
    echo "⚠️  有币种抓取失败："
    echo "$PARTIAL"
    echo "   仍然并入成功的部分；信号会退回到所有币都有数据的最后一天。"
  } | tee -a "$LOG" >&2
fi

echo | tee -a "$LOG"
echo "=== 并入 data/（--merge 会报告改写和缺口）" | tee -a "$LOG"
run "$PY" -m vibt.ingest "$OUT" --merge --gzip || die "ingest 失败，data/ 未改动"

echo | tee -a "$LOG"
echo "=== 今天的目标持仓" | tee -a "$LOG"
# 这一步之前没查退出码。信号生成一崩，$BOOK 里装的就是 traceback，
# 脚本会把它当成持仓表推送出去然后 exit 0 —— 正是这个脚本声称要防的那种静默失败。
if ! BOOK="$("$PY" scripts/43_paper_signals.py --equity "$EQUITY" $WRITE 2>&1)"; then
  echo "$BOOK" | tee -a "$LOG"
  die "信号生成失败"
fi
echo "$BOOK" | tee -a "$LOG"

rm -rf "$OUT"     # 原始 CSV 已经并进 data/ 了，留着只会越堆越多

WARN=""
grep -q "数据过期" <<<"$BOOK" && WARN="${WARN}
⚠️ 数据过期"
[[ -n "$PARTIAL" ]] && WARN="${WARN}
⚠️ 部分币种抓取失败"
# 交易所改写已结算历史会让之前所有回测失效，所以它不该只躺在日志里。
# 匹配 ingest 的 "  RESTATED <symbol> <tf>:" —— 只在真的发生时打印，
# 而表头里的 restated 那一列是每次都有的，不能拿它当判据。
if grep -q "^  RESTATED " "$LOG"; then
  WARN="${WARN}
🔴 交易所改写了已结算的历史：
$(grep '^  RESTATED ' "$LOG")"
fi

notify "📋 纸面信号 $(date -u '+%m-%d')UTC${WARN}

$(sed -n '/目标持仓/,/本次成交名义/p' <<<"$BOOK")"

if [[ -n "$WRITE" ]]; then
  echo
  echo "成交后把 fill_price / fill_qty / filled_at / fee_paid 填回 paper/signals.csv，"
  echo "攒够几十笔再跑： $PY scripts/43_paper_signals.py --reconcile"
fi
