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

## 候选名单（约 34 个，按预期与 BTC 的相关性从低到高）

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

## 导出命令

在 VPS 上（`~/VI-Dashboard` 目录里）：

```bash
SYMS="DOGEUSDT 1000PEPEUSDT WIFUSDT 1000BONKUSDT 1000SHIBUSDT 1000FLOKIUSDT \
RENDERUSDT FETUSDT ARKMUSDT WLDUSDT AKTUSDT GRTUSDT \
SUIUSDT SEIUSDT TIAUSDT APTUSDT ARBUSDT OPUSDT STRKUSDT JUPUSDT \
DYDXUSDT GMXUSDT LDOUSDT ENAUSDT PENDLEUSDT AAVEUSDT \
XRPUSDT ADAUSDT LTCUSDT LINKUSDT AVAXUSDT DOTUSDT ATOMUSDT NEARUSDT"

for S in $SYMS; do
  echo "=== $S"
  python3 backend/tools/export_klines.py --symbol "$S" \
    --market futures --intervals 1d --start 2023-01-01 --out ./exports \
    || echo "  !! $S 失败（代码不对？未上合约？），继续"
done
ls -1 ./exports | wc -l
```

**注意这里只导 `--intervals 1d`。** 筛选阶段只需要日线算相关性，34 个币 × 3 周期
会是几百 MB，没必要。**等筛完选出 15~20 个之后，再单独把它们的 1h/4h 补齐。**

拉回本地后：

```bash
python -m vibt.ingest "<你的导出目录>"
python scripts/16_screen_universe.py
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
