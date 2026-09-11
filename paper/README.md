# 纸面交易日志

这个目录存 `scripts/43_paper_signals.py` 产出的信号日志。**它的目的是测量，不是赚钱。**

## 为什么要做

样本外检验里，这套策略的打平成本是 **6.38bp/边**，而回测假设的成本是 **6.5bp/边**。
两个数贴在一起——**它成不成立，完全取决于你实际拿到什么成交价**，
而这件事用任何 OHLC 回测都答不出来。所以要做的是：
把打算做什么、按什么价格算的，写下来；等成交了，把真实价格填回去；攒够笔数再比。

跨币种的干净样本已经用完了（181 个从未参与开发的币在 `REPORT_RESMOM.md` 里用掉了）。
**剩下唯一没被污染的资源是未来的时间。**

## 两台机器

这条链路横跨两台机器，而两台各缺一半：

| | 能连币安 | 有仓库 + pandas |
|---|---|---|
| **VPS** | ✅ | 装之前没有 |
| **Claude 的容器** | ❌（出口策略封了） | ✅ |

所以日常运行放在 VPS 上。**一次性安装：**

```bash
curl -sSL https://raw.githubusercontent.com/worth0307-cmyk/9678/claude/btcusdt-trading-system-hl6119/scripts/vps_setup.sh | bash
```

它会 clone 仓库、建 venv（新版 Debian/Ubuntu 有 PEP 668，直接 pip 会被拒）、
装 pandas+numpy、自检，并确认这台机器确实连得上币安。幂等，可以重复跑。

### 在 VPS 上 `git pull` 失败时

`data/*.csv.gz` 这 18 个文件是**被 git 跟踪的**，而 VPS 每天都会重写它们。
于是 VPS 的工作区从第一次跑完就一直是「脏」的，而且和仓库里那份快照
**每天多差一天**。平时无所谓——只要上游没动 `data/`，`git pull` 照常快进。

但只要仓库那边提交过 `data/`，下一次 `git pull` 就会硬失败：

```
error: Your local changes to the following files would be overwritten by merge:
        data/BTCUSDT_1d.csv.gz
```

这是**好事**：它宁可停下也不覆盖你的新数据。丢掉本地改动再拉就行——

```bash
cd ~/9678 && git checkout -- data/ && git pull
```

之所以安全，是因为下一次 `vps_daily.sh` 默认回抓 10 天并 `--merge` 合并。
**前提是仓库那份快照落后不超过 10 天**；落后更多就会留一个洞（`ingest` 会报缺口），
那时候用 `DAYS=60 bash scripts/vps_daily.sh` 把窗口开大一次补回来。

## 每天怎么跑

信号在日线收盘时形成，计划在**同一时刻**（00:00 UTC）成交。
这个仓库测过：执行拖一天，Sharpe 掉 0.3~0.6。所以 00:00 UTC 之后尽快跑。

```bash
bash ~/9678/scripts/vps_daily.sh            # 抓增量 -> 并库 -> 打印目标持仓
bash ~/9678/scripts/vps_daily.sh --write    # 确认后记进日志
```

## 挂成全自动

```bash
bash ~/9678/scripts/vps_cron.sh            # 安装
bash ~/9678/scripts/vps_cron.sh --test     # 用 cron 的空环境验一次
bash ~/9678/scripts/vps_cron.sh --probe    # 验它会不会在对的时刻触发（两分钟）
bash ~/9678/scripts/vps_cron.sh --status   # 看装了什么、日志到哪了
bash ~/9678/scripts/vps_cron.sh --remove   # 卸载
```

不要手写那行 crontab，有四个坑：

| 坑 | 后果 | 脚本怎么处理 |
|---|---|---|
| **时区** | `5 0 * * *` 是**本机时区**的 00:05，不是 UTC | 本机不是 UTC 时自动写 `CRON_TZ=UTC`；`--probe` 当场验它生没生效 |
| **行尾注释** | crontab 的环境行不认，`CRON_TZ=UTC # mark` 会把时区设成字面量 `UTC # mark` | 标记用独立的 `BEGIN`/`END` 行，不贴行尾 |
| **PATH** | cron 的 PATH 极简，`curl` 之类可能找不到 | 在任务行里显式设 PATH |
| **日志** | 一年后那个文件会很大 | 装 logrotate 配置（周轮转，留 8 份） |

装完有两件事值得当场验，它们查的不是同一个问题：

`--test` 用 `env -i` 模拟 cron 的空环境跑一次——**「手动跑得通、cron 跑不通」几乎
总是环境差异**。它查的是「跑起来会不会崩」。

`--probe` 把一个只 `touch` 文件的任务排在「UTC 当前时刻 + 2 分钟」，用**和正式任务
完全相同的时区写法**，然后等它触发。它查的是「会不会在**我以为的那个时刻**跑」。
这件事有两种静默的失败法：本机不是 UTC 时 `CRON_TZ` 可能压根不被这版 cron 认；
本机**刚**被改成 UTC 时守护进程可能还缓存着旧时区（`systemctl restart cron`）。
两种都表现为**照常出结果、照常推 Telegram，只是每天晚若干小时**——而执行拖一天
Sharpe 掉 0.3~0.6。等明天看日志也能发现，但那要浪费一天。

> 最稳的形态是整机就用 UTC（`timedatectl set-timezone UTC`），
> 那样 crontab 里连 `CRON_TZ` 都不需要，少一个能静默失效的东西。

cron 里用的是 `--write`：无人值守就是要记录。那个「先看再确认」的两步是给人用的，
纸面阶段没有真金白银，**记下意图本身才是目的**。

## 通知（可选，但强烈建议）

没有通知的话，你只能 SSH 上去看日志才知道今天该做什么，而且**任务静默失败几个星期
都不会有人发现**——那段时间的数据永远拿不回来了。

Telegram：和 `@BotFather` 建个 bot 拿 token，给它发条消息后从
`https://api.telegram.org/bot<token>/getUpdates` 找到 chat id，然后：

```bash
cat > ~/9678/.env <<'EOF'
TG_TOKEN=123456:AAxxxxxxxxxxxxxxxxxxxxx
TG_CHAT=123456789
EOF
chmod 600 ~/9678/.env
```

`.env` 已经在 `.gitignore` 里，不会进仓库。配好之后每天会推送目标持仓，
并在这三种情况下告警：抓取失败、部分币种缺失、**交易所改写了已结算的历史**
（最后这条会让之前所有回测失效，不该只躺在日志里）。

## 关于 `signals.csv` 为什么进 git

它**故意没有**被 gitignore。git 历史本身就是这份日志的防篡改时间证明——
能证明每条信号是在结果揭晓**之前**写下的。对一个「事前记录、事后对账」的实验来说，
这个性质比省掉合并冲突重要得多。VPS 上定期 `git add paper/signals.csv && git commit` 即可。

`vps_daily.sh` 做的三件事，每一件失败都会停住而不是带着坏数据往下走：

1. **增量抓取**，窗口默认 10 天而不是 1 天——**故意重叠**。上一次的最后一根 K 线
   当时还在形成，而交易所偶尔改写已结算的历史；有重叠，`ingest --merge` 才能
   区分「最后一根被补完」和「历史被改写」，后者会让之前所有回测失效。
2. **并入 data/**，逐字段比对重叠区间并报告改写、缺口、异常 OHLC。
3. **生成目标持仓**。

数据过期超过 6 小时时会警告——**过期数据会产出一张格式完全正常、但属于过去某一天的
持仓表**，下游没有任何东西能分辨，所以只能在生成处提示。

单个币抓失败时脚本会显式报出来。下游其实是安全的（`latest_complete_bar` 取所有币
的最小值，会退回到所有币都有数据的最后一天），但不说出来就成了静默降级。

## 成交之后

打开 `signals.csv`，给已成交的行填这四列：

| 列 | 填什么 |
|---|---|
| `fill_price` | 实际成交均价 |
| `fill_qty` | 实际成交数量 |
| `filled_at` | 实际成交时间（UTC） |
| `fee_paid` | 实付手续费（计价货币） |

然后：

```bash
python scripts/43_paper_signals.py --reconcile
```

输出每笔的**有符号滑点**（买贵了和卖便宜了都记为正成本，所以可以直接和 6.5bp 比）
以及执行延迟。

## 攒多少笔才有意义

每 3 天调仓一次、6 个币，一次大约 4~6 笔有效成交，所以一个月约 40~60 笔。
按 `scripts/41_edge_arithmetic.py` 的算法，要把滑点中位数估到 ±1bp 以内，
几十笔就够了——**这比确认策略本身有没有边际（需要上万笔）容易得多**，
这也正是先做这件事的理由。

## 要记住的事

这套策略在**这 6 个币上是样本内的**（BTC/ETH/SOL/BNB 是整个项目的原始开发集）。
6 币面板上 Sharpe +1.05，和当年 26 币样本内的 +1.13 是同一类数字，
而它在 181 个新币上是 **−0.63**。

**所以不要把纸面阶段的盈亏当成验证。** 这个阶段要回答的只有一个问题：
**真实成交价和参考价差多少。**
