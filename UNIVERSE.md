# 候选币池 — 用于横截面策略扩展

选币原则只有一条：**和现有池子的相关性低**。不是市值大，不是"我看好它"。

理由见 [REPORT_XSEC.md](REPORT_XSEC.md) 第 3 节 —— BTC/ETH/SOL/BNB 之间相关性 0.815，
横截面策略在它们身上 Sharpe 是 **−0.07**；加上相关性 0.48~0.70 的 TAO/HYPE 之后才有东西可赚。
**相关性 > 0.8 的币加进来是纯粹的稀释。**

工作流是「**广撒网导出 → 用相关性定量筛选**」，不是先挑好再导。
先把下面全导出来（多导几个不花钱），然后 `python scripts/16_screen_universe.py` 按数据说话。

---

## 币安 U 本位合约代码的坑（先看这个）

**导出失败最常见的原因是代码写错，不是币不存在：**

| 你以为的代码 | 币安 U 本位合约实际代码 | 原因 |
|---|---|---|
| PEPEUSDT | **1000PEPEUSDT** | 单价太小，合约按 1000 枚计价 |
| SHIBUSDT | **1000SHIBUSDT** | 同上 |
| BONKUSDT | **1000BONKUSDT** | 同上 |
| FLOKIUSDT | **1000FLOKIUSDT** | 同上 |
| RNDRUSDT | **RENDERUSDT** | 2024 年代币更名 |
| MATICUSDT | **POLUSDT** | 2024 年迁移到 POL |

这几个前缀/更名规则会随时间变，**以 Dashboard 里 ccxt 拉到的交易对列表为准**。
`fetch_universe.sh` 单个币失败会跳过继续，所以名单里混几个错的不影响其他币。

---

## 推荐执行名单（从 34 个候选里挑）

> **重要限制**：我无法在这个会话里验证币安的实际上市状态（连不上币安，知识截止 2026-05）。
> 下面是先验判断，**导出本身就是验证**——代码错了或未上市的会失败跳过，不影响其他币。

### 结构性矛盾：历史长度 vs 低相关性

**历史最长的币恰恰和 BTC 相关性最高，相关性最低的恰恰是新上市的。**
这不是巧合——新币的独立走势正来自它的上市动态。所以"2023-01-01 起 + 低相关"有内在冲突。

框架对参差不齐的历史是**原生支持**的（`xsec.cross_sectional_weights` 只对当天有有效数据的币排序，
HYPE/TAO 在上市前根本不进排序），所以**短历史不是排除理由**，只是权重上要心里有数。

### 实际导出结果（2026-08-14，20/20 全部成功）

下面是**实测的**日线根数和起始日期，用来修正上面那些"预期上市"的猜测：

| 币种 | 日线根数 | 实际起始 | 与我的预期 |
|---|---|---|---|
| DOGEUSDT | 1321 | 2023-01-01 | ✓ 全历史 |
| 1000SHIBUSDT | 1321 | 2023-01-01 | ✓ 全历史 |
| XRPUSDT | 1321 | 2023-01-01 | ✓ 全历史 |
| DYDXUSDT | 1321 | 2023-01-01 | ✓ 全历史 |
| OPUSDT | 1321 | 2023-01-01 | ✓ 全历史 |
| APTUSDT | 1321 | 2023-01-01 | ✓ 全历史 |
| LDOUSDT | 1321 | 2023-01-01 | ✓ 全历史 |
| FETUSDT | 1305 | 2023-01-17 | ✓ |
| ARBUSDT | 1240 | 2023-03-23 | ✓ |
| SUIUSDT | 1199 | 2023-05-04 | ✓ |
| 1000PEPEUSDT | 1197 | 2023-05-06 | ✓ |
| 1000FLOKIUSDT | 1196 | 2023-05-07 | ✓ |
| WLDUSDT | 1117 | 2023-07-24 | ✓ |
| PENDLEUSDT | 1113 | 2023-07-28 | ✓ |
| SEIUSDT | 1093 | 2023-08-17 | ✓ |
| TIAUSDT | 1018 | 2023-11-01 | ✓ |
| 1000BONKUSDT | 996 | 2023-11-22 | ✓ |
| WIFUSDT | 939 | 2024-01-18 | ✓ |
| ENAUSDT | 864 | 2024-04-02 | ✓ |
| **RENDERUSDT** | **749** | **2024-07-26** | **✗ 我预期 2022** |

**RENDER 是唯一猜错的**：RNDR → RENDER 更名时币安开了**新的合约代码**，
2024-07 之前的历史留在旧的 `RNDRUSDT` 下面。想补齐得单独导 `RNDRUSDT`
（如果还没下架），然后在 ingest 前把两段拼起来——但两个合约的价格序列未必连续，
拼接要小心。**更省事的做法是接受 RENDER 只有 2 年历史。**

> 这条经验适用于所有更名过的币：**更名 = 新合约 = 历史断点**。
> 遇到日线根数明显短于预期的币，先怀疑是不是改过名。

### 第一优先（12 个）——建议一定要跑

| 币种 | 组 | 预期上市 | 选它的理由 |
|---|---|---|---|
| `DOGEUSDT` | A | 2020 | 全历史 meme，流动性最好的 meme，独立资金流 |
| `1000PEPEUSDT` | A | 2023-05 | 特质波动极高，2023 起就有 |
| `1000SHIBUSDT` | A | 2021 | 全历史 meme，流动性好 |
| `XRPUSDT` | E | 2020 | **E 组里唯一我看好的**：监管事件驱动，和 BTC 脱钩明显 |
| `FETUSDT` | B | 2021 | AI 赛道，全历史，TAO 的同类但历史长得多 |
| `RENDERUSDT` | B | 2022 | AI/DePIN，注意是 RENDER 不是 RNDR |
| `WLDUSDT` | B | 2023-07 | 特质性极强（自己的解锁/叙事周期） |
| `SUIUSDT` | C | 2023-05 | 新公链里流动性最好的之一 |
| `SEIUSDT` | C | 2023-08 | 独立叙事 |
| `TIAUSDT` | C | 2023-10 | 独立叙事，波动大 |
| `DYDXUSDT` | D | 2021 | 全历史，衍生品赛道 |
| `PENDLEUSDT` | D | 2023 | 收益率赛道，和大盘关联弱 |

这 12 个的组合逻辑：**4 个全历史低相关（DOGE/SHIB/XRP/FET/DYDX）打底，
8 个 2023 年内上市的高特质波动提供离散度**。

### 第二优先（8 个）——一起跑，边际成本很低

```
WIFUSDT  1000BONKUSDT  1000FLOKIUSDT     # meme，2023-2024 上市，波动最大
ARBUSDT  OPUSDT  APTUSDT                  # L2/L1，历史较长
ENAUSDT  LDOUSDT                          # DeFi
```

### 建议跳过（14 个）——理由是"我预期它们会被相关性筛掉"

```
ADAUSDT  LTCUSDT  DOTUSDT  ATOMUSDT  LINKUSDT  AVAXUSDT  AAVEUSDT  NEARUSDT
```
这 8 个是典型的"高相关大市值"，和 BTC 大概 0.75~0.85。
**它们进来只会把有效独立标的数往下拉**——正是 REPORT_XSEC 第 3 节证明没用的那一类。

```
GRTUSDT  GMXUSDT  ARKMUSDT  AKTUSDT  JUPUSDT  STRKUSDT
```
这 6 个不是不好，是**流动性存疑**。横截面策略在 30bp 成本下就转负，
点差大的小币会直接吃掉 alpha。**宁可要 20 个流动性好的，不要 34 个含一半垃圾的。**

> 如果你想要更彻底：把跳过的这 14 个也导出来（只导 1d，很快），
> 让 `scripts/16_screen_universe.py` 用数据否掉它们，比信我的先验更可靠。
> 我上面的分组只是省你时间，不是定论。

---

## 原始候选名单（约 34 个，按预期与 BTC 的相关性从低到高）

### A 组 · Meme（特质波动最高，和 BTC 相关性通常最低）

```
DOGEUSDT  1000PEPEUSDT  WIFUSDT  1000BONKUSDT  1000SHIBUSDT  1000FLOKIUSDT
```

这组是**横截面策略最需要的燃料**：它们由自己的资金流驱动，和大盘的关联最弱，
离散度贡献最大。DOGE 相对成熟、流动性好；PEPE/WIF/BONK 波动极大。

### B 组 · AI / DePIN（TAO 的同类，你已经验证过这类有效）

```
RENDERUSDT  FETUSDT  ARKMUSDT  WLDUSDT  AKTUSDT  GRTUSDT
```

TAO 是现有池子里贡献最大的一个（等风险下权重 0.097 贡献 +37.1%）。
同赛道的币值得优先测——但注意它们**彼此之间**可能高度相关，筛选时要看两两矩阵不是只看对 BTC。

### C 组 · 新公链 / L2（2023 年后上市，独立叙事）

```
SUIUSDT  SEIUSDT  TIAUSDT  APTUSDT  ARBUSDT  OPUSDT  STRKUSDT  JUPUSDT
```

### D 组 · DeFi / 衍生品（HYPE 的同类）

```
DYDXUSDT  GMXUSDT  LDOUSDT  ENAUSDT  PENDLEUSDT  AAVEUSDT
```

### E 组 · 老牌大市值（相关性可能偏高，但 XRP 有独立的监管驱动）

```
XRPUSDT  ADAUSDT  LTCUSDT  LINKUSDT  AVAXUSDT  DOTUSDT  ATOMUSDT  NEARUSDT
```

这组我预期大部分会被相关性筛掉（和 BTC 大概 0.75~0.85），
但**必须导出来才知道**——XRP 尤其可能因为监管事件而有独立走势。

---

## 导出命令（20 个币 × 1h/4h/1d，2023-01-01 起）

> **粘贴进终端前先看这条**：交互式 bash 默认开启历史展开，
> **双引号里的 `!!` 会被替换成上一条命令**，导致引号不配对、终端卡在 `>` 提示符。
> 所以下面的版本 (a) 开头 `set +H` 关掉历史展开，(b) 符号表写成一行避免
> 反斜杠续行后面跟空格的问题，(c) 提示语里不含 `!!`。
> 卡住了按 `Ctrl+C` 退出即可，不会有副作用。
>
> **另外必须先 `cd` 到 VI-Dashboard 目录**——`backend/tools/export_klines.py` 是相对路径，
> 在 `/root` 下跑会 20 个币全部报 "No such file"。下面第一行已经带上了，
> 并且加了前置检查：路径不对时**一次就退出**，不会刷 20 条一样的报错。

在 VPS 上（`~/VI-Dashboard` 目录里）：

```bash
set +H
cd ~/VI-Dashboard || { echo "找不到 ~/VI-Dashboard"; return 2>/dev/null || exit 1; }
[ -f backend/tools/export_klines.py ] || {
  echo "当前在 $PWD，这里没有 backend/tools/export_klines.py"
  return 2>/dev/null || exit 1
}

SYMS="DOGEUSDT 1000PEPEUSDT 1000SHIBUSDT XRPUSDT FETUSDT RENDERUSDT WLDUSDT SUIUSDT SEIUSDT TIAUSDT DYDXUSDT PENDLEUSDT WIFUSDT 1000BONKUSDT 1000FLOKIUSDT ARBUSDT OPUSDT APTUSDT ENAUSDT LDOUSDT"

OK=""; FAIL=""
for S in $SYMS; do
  echo "=== $S"
  if python3 backend/tools/export_klines.py --symbol "$S" \
       --market futures --intervals 1h,4h,1d --start 2023-01-01 --out ./exports; then
    OK="$OK $S"
  else
    FAIL="$FAIL $S"
  fi
done
echo
echo "成功:$OK"
echo "失败:$FAIL"
ls -1 ./exports | wc -l    # 期望 60 个文件（20 币 × 3 周期）
```

或直接用仓库里的脚本（已经处理好上面这些坑）：

```bash
SYMBOLS="DOGEUSDT 1000PEPEUSDT 1000SHIBUSDT XRPUSDT FETUSDT RENDERUSDT WLDUSDT SUIUSDT SEIUSDT TIAUSDT DYDXUSDT PENDLEUSDT WIFUSDT 1000BONKUSDT 1000FLOKIUSDT ARBUSDT OPUSDT APTUSDT ENAUSDT LDOUSDT" \
  bash scripts/fetch_universe.sh
```

**数据量预估**：3.6 年的 1h 数据每个币约 1.8MB，20 个币三周期合计约 **45MB**
（现有 6 币是 11MB）。git 扛得住，但会明显变大。

> **一个诚实的提醒**：横截面分析实际只用 **1d**。1h 占了 80% 的体积，
> 而且日线换手在 6.5bp 下就已经吃掉三分之一的 Sharpe（REPORT_XSEC 第 4 节），
> 更高频的横截面大概率是负期望。你要求三个周期我照做了，
> 但如果想省事，**先只导 1d 跑筛选**，等确认哪些币留下来再补 1h/4h 也完全可以。

拉回本地后：

```bash
python -m vibt.ingest "<你的导出目录>"
python scripts/16_screen_universe.py       # 相关性筛选 + 有效独立标的数
python scripts/14_cross_section.py         # 横截面重测
python scripts/15_xsec_robustness.py       # 稳健性
```

---

## 筛选标准（脚本会自动算，这里说清楚判据）

1. **与现有 6 币池子的平均相关性 < 0.75** —— 这是硬门槛，高于此的加进来是稀释
2. **历史长度 ≥ 400 个日线** —— 太短的币在早期不参与排序，且容易被单段行情主导
3. **流动性** —— 这条脚本算不出来，需要你看 Dashboard 里的成交量。
   横截面策略在 30bp 成本下就转负（REPORT_XSEC 第 4 节），
   **点差大的小币会直接吃掉 alpha**。宁可要 15 个流动性好的，不要 30 个含一半垃圾的。
4. **加入后组合的"有效独立标的数"要上升** —— 这是最终判据，脚本会直接报这个数。
   现在是 1.37；如果扩到 20 个币后这个数没超过 3，说明加的都是同一个东西。

---

## 幸存者偏差：这个名单最大的问题

**上面每一个币都是我知道它活到了 2026 年才写进来的。**
2023-2024 期间上市后归零、被下架、或者跌到没有流动性的币，一个都不在名单里。

对横截面策略来说这不是小问题：**幸存下来的币，其"相对强弱"天然偏正**，
做多强者的策略会被系统性地高估。

**怎么缓解（按可行性排序）：**

1. **用币安的历史合约列表，而不是当前列表。** Dashboard 用的是 ccxt，
   `exchange.fetch_markets()` 只给当前在交易的。要拿历史列表得查
   币安的公告 API 或者 `/fapi/v1/exchangeInfo` 的历史快照。这一步最有价值也最麻烦。
2. **至少把已下架的加回来测一遍**。如果你记得 2023-2024 有哪些币安合约后来下架了
   （比如一些被摘牌的小币），把它们的历史也导出来 —— 有数据的那段照样能进横截面。
3. **退而求其次：接受偏差，但在结论里明确标注**，并且**只看多空组合的净值**
   （多头和空头都受同样的幸存者偏差影响，部分相互抵消），
   不要单看多头腿的表现。
4. **看 2026 年的表现**。2026 是熊市，幸存者偏差在下跌年份里的影响方向相反，
   如果策略在 2026 依然为正（现有池子是 +2.42 Sharpe），说明它不完全是偏差驱动的。

我建议先做第 3 条（标注 + 只看多空净值）把流程跑通，第 1 条作为后续改进。
**别因为偏差消不掉就不做——要做的是知道它有多大、往哪个方向。**
