#!/usr/bin/env bash
# 把每日运行挂进 cron，顺带处理时区、日志轮转、通知配置
#
#   bash ~/9678/scripts/vps_cron.sh            # 安装
#   bash ~/9678/scripts/vps_cron.sh --status   # 看现在装了什么
#   bash ~/9678/scripts/vps_cron.sh --remove   # 卸载
#   bash ~/9678/scripts/vps_cron.sh --test     # 立刻跑一次（不写日志），验证 cron 环境
#
# 三件容易踩的事，这个脚本都处理了：
#
#   时区   `5 0 * * *` 是**本机时区**的 00:05，不是 UTC。VPS 默认多半是 UTC，
#          但不能假设。不是 UTC 时会写 CRON_TZ=UTC（Debian/Ubuntu 的 cron 支持）。
#   PATH   cron 的 PATH 极简，通常只有 /usr/bin:/bin。脚本里用到 curl/date/grep，
#          所以显式设一个。
#   日志   一条 cron 任务写一个文件，一年之后那个文件会很大。装个 logrotate 配置。

set -uo pipefail
set +H

HOME_DIR="${HOME_DIR:-$HOME/9678}"
SCRIPT="$HOME_DIR/scripts/vps_daily.sh"
LOGFILE="${LOGFILE:-$HOME/paper.log}"
MARK="# vibt-paper-daily"          # 用它找到自己的那一行，不碰别人的
EQUITY="${EQUITY:-10000}"

[[ -f "$SCRIPT" ]] || { echo "找不到 $SCRIPT —— 先跑 scripts/vps_setup.sh" >&2; exit 1; }

current() { crontab -l 2>/dev/null || true; }
without_ours() { current | grep -v "$MARK" | grep -v "^CRON_TZ=UTC $MARK" || true; }

case "${1:-}" in
  --status)
    echo "=== 当前 crontab 中属于本项目的行"
    current | grep "$MARK" || echo "  （没有）"
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
  TZLINE="CRON_TZ=UTC $MARK"
  echo "⚠️  本机时区是 $TZNAME，不是 UTC。"
  echo "    已在 crontab 里写 CRON_TZ=UTC，让这条任务按 UTC 计时。"
  echo "    （想让整台机器用 UTC： timedatectl set-timezone UTC）"
fi

# ---------------------------------------------------------------- 安装
# --write：无人值守就是要记录。那个「先看再确认」的两步流程是给人用的，
# 纸面阶段没有真金白银，记下意图才是整件事的目的。
JOB="5 0 * * * PATH=/usr/local/bin:/usr/bin:/bin EQUITY=$EQUITY bash $SCRIPT --write >> $LOGFILE 2>&1 $MARK"
{ without_ours; [[ -n "$TZLINE" ]] && echo "$TZLINE"; echo "$JOB"; } | crontab -

echo
echo "=== 已安装"
current | grep "$MARK"

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

  看状态 / 日志：
    bash $HOME_DIR/scripts/vps_cron.sh --status

  明天 00:05 UTC 之后确认它真的跑了：
    tail -30 $LOGFILE
EOF
