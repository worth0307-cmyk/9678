#!/usr/bin/env bash
# 把每日运行挂进 cron，顺带处理时区、日志轮转、通知配置
#
#   bash ~/9678/scripts/vps_cron.sh            # 安装
#   bash ~/9678/scripts/vps_cron.sh --status   # 看现在装了什么
#   bash ~/9678/scripts/vps_cron.sh --remove   # 卸载
#   bash ~/9678/scripts/vps_cron.sh --test     # 立刻跑一次（不写日志），验证 cron 环境
#   bash ~/9678/scripts/vps_cron.sh --probe    # 两分钟测定它会不会在 UTC 时刻触发
#
# 四件容易踩的事，这个脚本都处理了：
#
#   时区   `5 0 * * *` 是**本机时区**的 00:05，不是 UTC。VPS 默认多半是 UTC，
#          但不能假设。不是 UTC 时会写 CRON_TZ=UTC。
#   注释   crontab 的环境行不认行尾注释 —— "CRON_TZ=UTC # mark" 会把时区设成
#          字面量 "UTC # mark"。所以标记用独立的 BEGIN/END 行，不贴在行尾。
#   PATH   cron 的 PATH 极简，通常只有 /usr/bin:/bin。脚本里用到 curl/date/grep，
#          所以显式设一个。
#   日志   一条 cron 任务写一个文件，一年之后那个文件会很大。装个 logrotate 配置。

set -uo pipefail
set +H

HOME_DIR="${HOME_DIR:-$HOME/9678}"
SCRIPT="$HOME_DIR/scripts/vps_daily.sh"
LOGFILE="${LOGFILE:-$HOME/paper.log}"
# 用 BEGIN/END 成对标记划出自己的区块，而不是在每行尾部加注释。
# 原因是 crontab 的环境行（CRON_TZ=...）不认行尾注释：Vixie cron 把 "=" 之后
# 一直到行尾都当成值，所以 "CRON_TZ=UTC # mark" 会把时区设成字面量
# "UTC # mark" —— 要么报错，要么**静默退回本机时区**，任务就在错误的时间跑了。
BEGIN_MARK="# vibt-paper-daily BEGIN"
END_MARK="# vibt-paper-daily END"
MARK="vibt-paper-daily"
EQUITY="${EQUITY:-10000}"

[[ -f "$SCRIPT" ]] || { echo "找不到 $SCRIPT —— 先跑 scripts/vps_setup.sh" >&2; exit 1; }

current() { crontab -l 2>/dev/null || true; }

# 删掉 BEGIN..END 之间（含两端）的所有行，其余原样保留
without_ours() {
  current | awk -v b="$BEGIN_MARK" -v e="$END_MARK" '
    $0 == b { skip = 1; next }
    $0 == e { skip = 0; next }
    !skip   { print }
  '
}

case "${1:-}" in
  --status)
    echo "=== 当前 crontab 中属于本项目的区块"
    current | awk -v b="$BEGIN_MARK" -v e="$END_MARK" '
      $0 == b { inb = 1 } inb { print } $0 == e { inb = 0 }' \
      | grep . || echo "  （没有）"
    echo
    echo "=== 本机时区: $(date +%Z) ($(date))"
    echo "=== UTC 现在: $(date -u)"
    echo
    if [[ -f "$LOGFILE" ]]; then
      echo "=== 日志 $LOGFILE ($(du -h "$LOGFILE" | cut -f1))，最后 15 行："
      tail -15 "$LOGFILE"
    else
      echo "=== 还没有日志 $LOGFILE（说明 cron 还没跑过，或者跑之前就失败了）"
    fi
    exit 0 ;;
  --remove)
    without_ours | crontab -
    echo "已移除。现在的 crontab："
    current || echo "  （空）"
    exit 0 ;;
  --probe)
    # 探针只回答一个问题，但那是唯一重要的那个：**装好的这条任务，会不会在我
    # 以为的那个绝对时刻触发？** 有两种失败会让它跑在别的时刻，而且都是静默的：
    #
    #   本机不是 UTC     任务靠 CRON_TZ 计时，而 CRON_TZ 是 cron 的扩展语法，
    #                    不认的版本会**默默**退回本机时区
    #   本机刚改成 UTC   守护进程可能还缓存着旧时区，不重启就还按旧的算
    #
    # 两种都表现为「照常出结果、照常推 Telegram，只是每天都晚若干小时」，而这个
    # 仓库测过执行拖一天 Sharpe 掉 0.3~0.6。等明天看日志也能发现，但要浪费一天。
    #
    # 做法：用**和正式任务完全相同的时区写法**，把一个只 touch 文件的任务排在
    # 「UTC 当前时刻 + 2 分钟」。触发了，就说明这套写法在这台机器上确实成立。
    PROBE_FILE="$HOME_DIR/.cron_probe"
    PB="# vibt-cron-probe BEGIN"
    PE="# vibt-cron-probe END"
    TZNAME="$(date +%Z)"

    # 先把当前 crontab 存成文件，收尾时原样写回 —— 而不是收尾时再从 crontab -l
    # 重新推导一遍。因为那条命令万一在收尾的时刻失败（管道左边空了也照样成立），
    # 推导出来的就是个空 crontab，正式任务会跟着一起没掉。
    # 顺手滤掉上一次异常退出可能留下的探针区块。
    BACKUP="$(mktemp)"
    crontab -l 2>/dev/null | awk -v b="$PB" -v e="$PE" '
      $0 == b { skip = 1; next } $0 == e { skip = 0; next } !skip { print }' >"$BACKUP"
    cleanup_probe() {
      crontab "$BACKUP" 2>/dev/null || crontab -r 2>/dev/null
      rm -f "$BACKUP" "$PROBE_FILE"
    }
    trap cleanup_probe EXIT INT TERM

    rm -f "$PROBE_FILE"
    HH="$(date -u -d '+2 minutes' +%H 2>/dev/null || date -u -v+2M +%H)"
    MM="$(date -u -d '+2 minutes' +%M 2>/dev/null || date -u -v+2M +%M)"
    { cat "$BACKUP"
      echo "$PB"
      [[ "$TZNAME" != "UTC" ]] && echo "CRON_TZ=UTC"
      echo "${MM#0} ${HH#0} * * * touch $PROBE_FILE"
      echo "$PE"
    } | crontab -

    echo "=== cron 触发时刻探针"
    echo "  本机时区 $TZNAME，UTC 现在 $(date -u '+%H:%M:%S')"
    if [[ "$TZNAME" != "UTC" ]]; then
      echo "  探针带 CRON_TZ=UTC（和正式任务一样），排在 UTC ${HH}:${MM}"
    else
      echo "  探针排在 UTC ${HH}:${MM}（本机就是 UTC，正式任务不需要 CRON_TZ）"
    fi
    echo "  等待最多 3 分钟 ..."
    for _ in $(seq 1 36); do
      [[ -f "$PROBE_FILE" ]] && break
      sleep 5
    done
    echo
    if [[ -f "$PROBE_FILE" ]]; then
      echo "✅ 触发了 —— 这台机器上 00:05 UTC 就是 00:05 UTC，不用再改。"
      rc=0
    elif [[ "$TZNAME" != "UTC" ]]; then
      echo "❌ 没触发 —— 这版 cron **不认 CRON_TZ**，任务会按本机时区（$TZNAME）的 00:05 跑。"
      echo "   修法（把整机设成 UTC，最省事也最不容易再出错）："
      echo "     timedatectl set-timezone UTC && systemctl restart cron"
      echo "     bash $HOME_DIR/scripts/vps_cron.sh && bash $HOME_DIR/scripts/vps_cron.sh --probe"
      rc=1
    else
      echo "❌ 没触发 —— 本机是 UTC，但 cron 在这个时刻没跑任务。"
      echo "   最常见的原因是**刚改过时区，守护进程还缓存着旧的**："
      echo "     systemctl restart cron   # 或 crond / cronie，看发行版"
      echo "     bash $HOME_DIR/scripts/vps_cron.sh --probe   # 再验一次"
      echo "   其次是 cron 根本没在跑： systemctl status cron"
      rc=1
    fi
    exit "$rc" ;;
  --test)
    echo "=== 用 cron 那套环境跑一次（--write 不加，不会写日志）"
    # env -i 模拟 cron 的空环境，这是「手动跑得通、cron 跑不通」的常见原因
    env -i HOME="$HOME" PATH=/usr/local/bin:/usr/bin:/bin SHELL=/bin/sh \
      /bin/sh -c "bash $SCRIPT" 2>&1 | tail -25
    # 必须立刻取。PIPESTATUS 只保留**最近一条**管线的状态，中间插一个 echo
    # 就会被它自己的 0 覆盖掉 —— 第一版就是这么把一次失败报成了"退出码 0"。
    rc="${PIPESTATUS[0]}"
    echo
    if [[ "$rc" == "0" ]]; then
      echo "=== 退出码 0 —— 这条 cron 能跑"
    else
      echo "=== 退出码 $rc —— **cron 跑不通**，先修好再等明天" >&2
    fi
    exit "$rc" ;;
esac

# ---------------------------------------------------------------- 时区
TZNAME="$(date +%Z)"
TZLINE=""
if [[ "$TZNAME" != "UTC" ]]; then
  TZLINE="CRON_TZ=UTC"
  echo "⚠️  本机时区是 $TZNAME，不是 UTC。"
  echo "    已在 crontab 里写 CRON_TZ=UTC，让这条任务按 UTC 计时。"
  echo "    （想让整台机器用 UTC： timedatectl set-timezone UTC）"
fi

# ---------------------------------------------------------------- 安装
# --write：无人值守就是要记录。那个「先看再确认」的两步流程是给人用的，
# 纸面阶段没有真金白银，记下意图才是整件事的目的。
JOB="5 0 * * * PATH=/usr/local/bin:/usr/bin:/bin EQUITY=$EQUITY bash $SCRIPT --write >> $LOGFILE 2>&1"
{
  without_ours
  echo "$BEGIN_MARK"
  [[ -n "$TZLINE" ]] && echo "$TZLINE"
  echo "$JOB"
  echo "$END_MARK"
} | crontab -

echo
echo "=== 已安装"
current | awk -v b="$BEGIN_MARK" -v e="$END_MARK" '
  $0 == b { inb = 1 } inb { print } $0 == e { inb = 0 }'

if [[ -n "$TZLINE" ]]; then
  echo
  echo "⚠️  CRON_TZ 是 cron 的扩展语法，不是所有版本都支持，而不支持的表现是"
  echo "    **静默**退回本机时区 —— 任务照常出结果、照常推送，只是每天晚若干小时。"
  echo "    别等明天看日志，两分钟就能测定： bash \$0 --probe"
else
  echo
  echo "本机时区已经是 UTC，不需要 CRON_TZ —— 这是最稳的形态。"
  echo "如果时区是**刚刚**改成 UTC 的，cron 守护进程可能还缓存着旧的，先重启再验："
  echo "  systemctl restart cron && bash \$0 --probe"
fi

# ---------------------------------------------------------------- 日志轮转
if [[ -d /etc/logrotate.d && -w /etc/logrotate.d ]]; then
  cat > /etc/logrotate.d/vibt-paper <<EOF
$LOGFILE {
    weekly
    rotate 8
    compress
    missingok
    notifempty
    copytruncate
}
EOF
  echo "=== 已装日志轮转 /etc/logrotate.d/vibt-paper（周轮转，留 8 份）"
else
  echo "=== 没有 /etc/logrotate.d 的写权限，跳过日志轮转"
  echo "    $LOGFILE 会一直长；偶尔手动清一下，或者用 root 重跑这一步。"
fi

# ---------------------------------------------------------------- 通知
if [[ -f "$HOME_DIR/.env" ]] && grep -q TG_TOKEN "$HOME_DIR/.env" 2>/dev/null; then
  echo "=== 通知已配置（$HOME_DIR/.env）"
else
  cat <<EOF

=== 通知没配（可选，但强烈建议）

  没有通知的话，你只能靠 SSH 上来看日志才知道今天该做什么，
  而且**任务静默失败几个星期都不会有人发现** —— 那段时间的数据永远拿不回来了。

  Telegram 配法：
    1. 和 @BotFather 对话，/newbot，拿到 token
    2. 给你的新 bot 发一条消息，然后打开
       https://api.telegram.org/bot<你的token>/getUpdates 找到 chat id
    3. 写进配置（600 权限，已 gitignore）：

       cat > $HOME_DIR/.env <<'ENVEOF'
       TG_TOKEN=123456:AAxxxxxxxxxxxxxxxxxxxxxxxxx
       TG_CHAT=123456789
       ENVEOF
       chmod 600 $HOME_DIR/.env

    4. 验证： bash $SCRIPT
EOF
fi

cat <<EOF

=== 下一步

  立刻验一次 cron 环境跑不跑得通（这和手动跑不是一回事）：
    bash $HOME_DIR/scripts/vps_cron.sh --test

  再验一次它会不会在**对的时刻**触发（两分钟，不用等明天）：
    bash $HOME_DIR/scripts/vps_cron.sh --probe

  看状态 / 日志：
    bash $HOME_DIR/scripts/vps_cron.sh --status

  明天 00:05 UTC 之后确认它真的跑了：
    tail -30 $LOGFILE
EOF
