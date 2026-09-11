#!/usr/bin/env bash
# 在 VPS 上装一次，之后每天只跑 vps_daily.sh
#
#   curl -sSL https://raw.githubusercontent.com/worth0307-cmyk/9678/claude/btcusdt-trading-system-hl6119/scripts/vps_setup.sh | bash
#
# 为什么需要这个：这个项目跨两台机器，而两台各缺一半。
#   VPS      能连币安，但没有仓库、没有 pandas，python 也没有别名
#   Claude   有全部代码和数据，但出口策略把币安封了
# 之前是靠手动传 zip 打通的。这个脚本把 VPS 补成能独立跑完整条链路。
#
# 它是幂等的：重复跑只会更新代码、补齐缺的依赖，不会重建已有的东西。

set -euo pipefail
set +H          # 交互式 shell 里 "!!" 会被历史展开吃掉，粘贴多行命令时会断在 ">" 提示符

REPO="${REPO:-https://github.com/worth0307-cmyk/9678}"
BRANCH="${BRANCH:-claude/btcusdt-trading-system-hl6119}"
HOME_DIR="${HOME_DIR:-$HOME/9678}"
VENV="${VENV:-$HOME_DIR/.venv}"

say() { printf '\n=== %s\n' "$*"; }

# ---------------------------------------------------------------- python
say "检查 python"
PY=""
for c in python3 python; do
  if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
done
if [[ -z "$PY" ]]; then
  echo "没有找到 python3。先装：" >&2
  echo "  apt-get update && apt-get install -y python3 python3-venv python3-pip git" >&2
  exit 1
fi
echo "  $PY -> $($PY -VV | head -1)"

# ---------------------------------------------------------------- repo
say "取仓库到 $HOME_DIR"
if [[ -d "$HOME_DIR/.git" ]]; then
  git -C "$HOME_DIR" fetch origin "$BRANCH" --depth 1
  git -C "$HOME_DIR" checkout -B "$BRANCH" "origin/$BRANCH"
  echo "  已更新到 $(git -C "$HOME_DIR" rev-parse --short HEAD)"
else
  command -v git >/dev/null 2>&1 || { echo "没有 git：apt-get install -y git" >&2; exit 1; }
  git clone --depth 1 --branch "$BRANCH" "$REPO" "$HOME_DIR"
  echo "  已 clone 到 $(git -C "$HOME_DIR" rev-parse --short HEAD)"
fi

# ---------------------------------------------------------------- venv
# 用 venv 而不是直接 pip install：新版 Ubuntu/Debian 带 PEP 668，
# 对系统 python 直接 pip 会被拒（externally-managed-environment），
# 而 --break-system-packages 顾名思义不是个好主意。
say "准备虚拟环境 $VENV"
if [[ ! -x "$VENV/bin/python" ]]; then
  "$PY" -m venv "$VENV" 2>/dev/null || {
    echo "创建 venv 失败，多半缺 python3-venv：" >&2
    echo "  apt-get install -y python3-venv" >&2
    exit 1
  }
fi
"$VENV/bin/python" -m pip install --quiet --upgrade pip
# 纸面信号只要这两个；scipy 只有研究脚本用得到，日常链路不装
"$VENV/bin/python" -m pip install --quiet pandas numpy
echo "  $("$VENV/bin/python" -c 'import pandas,numpy;print("pandas",pandas.__version__,"numpy",numpy.__version__)')"

# ---------------------------------------------------------------- 自检
say "自检"
cd "$HOME_DIR"
"$VENV/bin/python" - <<'PY'
from vibt import data as D
syms = D.available_symbols(require=("1d",))
print(f"  可加载币种: {len(syms)}  {' '.join(syms)}")
d = D.load("1d", symbol="BTCUSDT")
print(f"  BTCUSDT 日线 {len(d)} 根，{d.index[0].date()} -> {d.index[-1].date()}")
PY

# 顺手确认这台机器确实能连币安 —— 这是它存在的全部理由
code="$(curl -sS -o /dev/null -w '%{http_code}' --max-time 15 \
        https://fapi.binance.com/fapi/v1/time || echo 000)"
if [[ "$code" == "200" ]]; then
  echo "  币安可达 (HTTP $code)"
else
  echo "  ⚠️  币安不可达 (HTTP $code) —— 这台机器跑不了抓取，换一台能连的" >&2
fi

cat <<EOF

=== 装好了

每天跑：
  bash $HOME_DIR/scripts/vps_daily.sh

加到 cron（00:05 UTC，日线刚收盘）：
  (crontab -l 2>/dev/null; echo "5 0 * * * bash $HOME_DIR/scripts/vps_daily.sh >> \$HOME/paper.log 2>&1") | crontab -

EOF
