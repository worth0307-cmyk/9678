# 数据导出 Runbook（6 币种）

沿用你现在的流程，只是把单币扩成 6 个币。

> Claude Code 这个会话的出口策略封了 `api.binance.com` / `fapi.binance.com`
> （还有 coingecko / kraken / coinbase / bybit / okx，全部 403），所以导出必须在 VPS 上跑。

---

## 1. VPS 上导出

```bash
ssh vpn-sg
cd ~/VI-Dashboard && git fetch origin main && git checkout main && git reset --hard origin/main
```

然后直接贴这一段（不依赖本仓库的任何文件）：

```bash
set +H
cd ~/VI-Dashboard || { echo "找不到 ~/VI-Dashboard"; return 2>/dev/null || exit 1; }
[ -f backend/tools/export_klines.py ] || {
  echo "当前在 $PWD，这里没有 backend/tools/export_klines.py"
  return 2>/dev/null || exit 1
}

for S in BTCUSDT BNBUSDT ETHUSDT HYPEUSDT SOLUSDT TAOUSDT; do
  echo "=== $S"
  python3 backend/tools/export_klines.py --symbol "$S" \
    --market futures --intervals 1h,4h,1d --start 2024-01-01 --out ./exports \
    || echo "  FAILED $S，继续下一个"
done
ls -1 ./exports
```

> 开头那句 `set +H` 是必需的：交互式 bash 默认开启历史展开，
> **双引号里出现 `!!` 会被替换成上一条命令**，引号随即不配对，终端就卡在 `>` 提示符上。
> 真卡住了按 `Ctrl+C` 退出，没有副作用。
> 前两行的 `cd` 和文件检查也不能省：exporter 是相对路径，
> 在 `/root` 下直接跑会每个币都报 "No such file"。

跑完应该是 **18 个文件**（6 币 × 3 周期），命名形如 `ETHUSDT_4h_2024-01-01_to_now.csv`。

`--market futures`、`--intervals 1h,4h,1d`、`--start 2024-01-01` 就是 `export_klines.py` 的默认值，
也就是你之前裸跑 `python3 backend/tools/export_klines.py` 得到 BTCUSDT 那批数据的口径（**U 本位合约**）。
这里写全只是为了明确，不是改了什么。

> 如果本仓库已经在 VPS 上，也可以直接 `bash /path/to/9678/scripts/fetch_universe.sh`，效果一样。

---

## 2. 拉回本地

```powershell
scp -r vpn-sg:~/VI-Dashboard/exports "$env:USERPROFILE\Desktop\VI数据"
```

（原来那条是 `...\Desktop\BTCUSDT数据`；现在里面是 6 个币了，换个名字免得误会。）

---

## 3. 导入本仓库

```bash
python -m vibt.ingest "$env:USERPROFILE\Desktop\VI数据"
```

`ingest` 会做四件事：

1. 归一化命名 → `data/{SYMBOL}_{1h,4h,1d}.csv`
2. 逐币逐周期报告：K 线根数、真实起止时间、**缺口数、不可能的 OHLC、非正价格**
3. 用 1h 重采样交叉校验 4h / 1d 是否自洽
4. 打印最终可加载的币种列表

先看一遍报告再往下跑。想先看不写盘就加 `--dry-run`。

---

## 3b. 4h / 1h 数据（已入库）

**26 个币 × 1h/4h/1d 全部在 `data/` 里了**，共 78 个文件 13MB。
1h/4h 以 `.csv.gz` 存储，日线保持明文（小，且 git 里可读 diff）。
`data.py` 会**透明读取** `.csv.gz`，分析代码一行都不用改，两种格式可以混存。

```bash
python -m vibt.ingest "<导出目录>" --gzip     # 以后补数据时这样跑
```

压缩比约 **3.7x**（1h 单个币从 1.8MB 降到 0.49MB）。
重新 ingest 时会自动删掉另一种格式，避免同一个币同时存在过期的 `.csv` 和新的 `.csv.gz`
——那种情况下加载器会静默读到旧数据。有测试守着这条。

**主策略仍然只用日线**（`scripts/20_frequency.py`：26 币零成本 Sharpe 1.30 / 1.00 / 0.74，
日线在每一档成本下都最好）。存着是为了以后测别的想法时不用再抓一次。

### 已知的两处数据瑕疵

入库时做了三向交叉校验（1h→4h、1h→1d、4h→1d）。29,847 个币-日全部精确重建，
7,900 根 4h 里只有 2 根对不上，而且两处都是**交易所侧**的，不是导出错误：

| 时间 (UTC) | 影响 | 说明 |
|---|---|---|
| 2024-10-28 20:00–23:00 | **全部 26 个币** | 交易所整段数据空洞，1h 和 4h 都受影响，1h 抓到的成交更完整 |
| 2023-11-10 12:00 | 19 个币 | 4h 的**收盘价**错了，被它自己下一根 4h 的开盘价打脸；**信 1h** |

需要严格自洽的多周期数据时，用 1h 重采样出 4h，不要直接用 4h 文件。
差异只有 2/7900 根，绝大多数用途可以忽略。

> `cross_check` 报告的是**有多少根对不上**，不只是最大相对偏差——
> 一根错误的收盘价会让"最大偏差"和整份数据损坏长得一模一样。

## 3c. 以后取新数据：增量刷新（`scripts/refresh.sh`）

历史已经在库里了，所以以后**只需要取增量**，不用再全量导一次。

**在 VPS 上：**

```bash
ssh vpn-sg
cd ~/VI-Dashboard && git fetch origin main && git checkout main && git reset --hard origin/main
bash refresh.sh            # 默认最近 10 天，26 个币，三个周期
bash refresh.sh 30         # 想多取一点就传天数
```

（`refresh.sh` 在本仓库 `scripts/` 下，拷到 VPS 上或直接 `bash /path/to/9678/scripts/refresh.sh`。）

它会留下一个 `~/vi_refresh_<日期>.zip`：

| 取多久 | 大小 |
|---|---|
| 全量历史 | ~57MB |
| **最近 10 天** | **~200KB** |
| 最近 30 天 | ~600KB |

**拉回本地：**

```powershell
scp vpn-sg:~/vi_refresh_20260815.zip "$env:USERPROFILE\Desktop\"
```

**然后上传，我这边接上去：**

```bash
python -m vibt.ingest <解压目录> --gzip --merge
```

### `--merge` 为什么是必需的

**不加 `--merge`，那份 10 天的文件会把整段历史覆盖掉**——而且所有完整性检查都会通过，
因为一份只有 10 天的文件本身完全合法。有测试专门守着这条
（`test_merge_without_it_would_have_truncated`）。

`--merge` 做三件事：

1. **拼接**：新旧按时间并集，重叠部分以新数据为准。
2. **区分两种重叠不一致**：
   - 上次导出时**最后一根 K 线还没走完**（币安会返回进行中的那根）——正常，静默用新的覆盖。
   - **已经收盘的历史 K 线被改了**——交易所回填/订正，这会让之前所有回测失效，
     所以单独计数并打印 `RESTATED` 警告。
3. **报告**：每个币每个周期新增了多少根、重叠多少根、有几根被订正。

刻意留 10 天重叠就是为了让第 2 条有东西可查。**别把它压到 1 天。**

---

## 3d. 拉样本外 + 成交量 + 资金费率（`scripts/fetch_binance.py`）

这一步和上面的 `refresh.sh` 不同：**它不依赖你的 exporter**，直接打币安公开 API，
只用 Python 标准库（VPS 上不用装任何东西）。

```bash
scp scripts/fetch_binance.py vpn-sg:~/          # 从本地传上去
ssh vpn-sg
python3 ~/fetch_binance.py --dry-run            # 先看它打算拉什么
python3 ~/fetch_binance.py                      # 真跑，约 20 分钟
```

一次拿三样：

| | 内容 | 为什么 |
|---|---|---|
| 1 | **40 个从没用过的币**，1d+4h，含成交量 | 干净的样本外横截面 |
| 2 | 现有 26 个币的**成交量** | 补上缺口——6.5bp 成本假设从没验证过 |
| 3 | 全部 66 个币的**资金费率**历史 | 和价格动量机制不同的 carry 信号 |

### 样本外名单是怎么选的（这条很重要）

选择规则**写死在脚本里，且不使用任何来自价格的信息**：

> 所有 TRADING 状态的 USDT 本位永续，上市日期 ≤ 2024-01-01，
> 排除已有的 26 个，**按上市日期排序取最早的 40 个**。

按成交量、波动率或历史表现排序都会把这次检验要测量的选择效应重新引进来。
脚本还会在 MANIFEST 里标注每个币是否在 UNIVERSE.md 里被我事先点过名，
这样结果可以按「我提前挑过 vs 完全没提过」拆开看。

### 已验证的部分

抓取脚本在本仓库里用合成数据测过（连不上币安也能测）：

- **时间戳约定**：写出来的北京时间和 `data/` 里现有文件逐字符一致
- **分页**：3800 根 K 线跨 3 页、2600 条资金费率跨 3 页，无缺口、无重复、有序
- **未收盘的最后一根会被丢掉**——否则写进磁盘的是半根 K 线，下次刷新时会被误报成"交易所订正"
- **输出能被 `ingest` 直接吃掉**，成交量字段一路保留到 `D.load()`（有测试守着）

### 大小与打包

约 66 个币 × 2 个周期 + 资金费率，原始约 80MB，压缩后约 25MB。
如果一次传不动就分两个包：

```bash
cd ~/binance_pull
zip -qr ~/pull_klines.zip  ./*_1d_*.csv ./*_4h_*.csv MANIFEST.csv
zip -qr ~/pull_funding.zip ./*_funding.csv
ls -lh ~/pull_*.zip
```

脚本**支持断点续传**：已经写好的文件会跳过，中途断了直接重跑同一条命令。

---

## 4. 跑池化研究

```bash
python -m pytest tests/ -q          # 16 个测试，含无前视检验
python scripts/12_multi_symbol.py   # 六币种池化的 VI+ 双突破事件研究
```

这一步才是加币种的意义：现在单 BTC 只有 61 个事件，什么都证明不了；6 个币有 300+ 个。
脚本会在三种通道口径（固定 / 滚动分位 / 滚动区间）下**池化统计**，并给出**逐币一致性检查**。

**怎么读结果：** 如果"上破后继续上涨"在 6 个币上都成立，它就不是 BTC 的偶然；
如果只在 BTC 上成立、其他 5 个币各说各话，那它本来就是噪声。

---

## 预期会遇到的两件事

**HYPE / TAO 的历史短于 2024-01-01。** HYPE 是 2024 年 11 月底才上线的。
`ingest` 会如实报告每个币的真实起始时间，池化统计自动只用各自有数据的区间。
但要注意：**HYPE 的样本基本只覆盖了 2025-2026 的下跌，不覆盖 2024 那波牛市**，
而第 5 节已经证明这个信号的盈亏符号跟着大趋势走——所以别把 HYPE 的结果和 BTC 的直接平均。

**Ticker 可能对不上。** HYPEUSDT / TAOUSDT 在币安 U 本位合约上的确切交易对名称，
我在这个会话里连不上币安没法验证。哪个失败了就看 Dashboard 里 `symbols` 配置用的字符串，
或者直接改上面循环里的名字。
