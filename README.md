# BTCUSDT 区间型择时系统

基于 1H / 4H / 1D OHLC 数据的 BTCUSDT 自动交易研究框架，围绕 **Vortex Indicator (VI)** 展开，
最终落到一套**以 VI + DMI 投票 + 波动率标定**为核心的日线仓位分配系统。

完整研究过程和全部证伪测试见 **[REPORT.md](REPORT.md)**。

---

## 三句话结论

1. **1 小时级别做不了。** 测过的每一个 1H 变体全部亏损（最差 −95.6%）——不是指标问题，是 13bp 往返成本占了 1H 中位振幅的 11%。
2. **单参数回测全是幻觉。** 日线 VI 仅多组合样本内平均 Sharpe +1.09，样本外 −0.74；而 VI(14) 那个漂亮的 1.06 是一根孤立尖刺，n=16 就掉到 0.25。
3. **真正稳的是"降暴露"。** 样本外那波 −50% 熊市里，所有家族、所有参数、100% 的格子都跑赢了拿现货。这不是 alpha，但它可复制。

## 系统表现（2024-01 ~ 2026-08，含 6.5bp/边成本）

| | 系统 | 拿现货 |
|---|---|---|
| 总收益 | +33.9% | +44.5% |
| 最大回撤 | **−22.1%** | −53.0% |
| Sharpe | 0.59 | 0.53 |
| Calmar | **0.53** | 0.29 |
| 2026 熊市 | **−6.2%** | −27.0% |

对现货 beta 0.29，年化 alpha +6.6%。
**但：t = 0.96，bootstrap Sharpe 90% 区间 [−0.59, +1.69]，deflated Sharpe 1.2%。**
超额收益在统计上未被证明——2.6 年数据证明 Sharpe 0.59 需要约 11.4 年。详见 REPORT.md 第 8 节。

![净值](reports/fig1_equity.png)

## 快速开始

```bash
pip install pandas numpy scipy matplotlib pytest
python -m pytest tests/ -q     # 11 个无前视偏差测试
python scripts/08_final.py     # 最终系统 + 全套验证
```

实盘信号（每天 UTC 00:00 / 北京 08:00 后跑）：

```bash
python -m vibt.fetch                        # 增量拉币安 K 线，公开接口无需 key
python -m vibt.live --equity 10000          # 打印目标仓位和下单计划
```

`vibt/live.py` 只输出下单计划，不替你发单。先纸上跑几个月再说。

## 目录结构

```
vibt/
  data.py         数据加载、多周期对齐（严格无前视）
  indicators.py   Vortex / DMI / ATR / Parkinson 波动率等，全部因果
  backtest.py     回测引擎：信号收盘产生、下一根开盘成交、成本按换手计
  metrics.py      绩效统计 + block bootstrap + deflated Sharpe
  strategies.py   信号库（VI 各变体、趋势、均值回归）
  system.py       最终系统
  live.py         实盘日线运行器
  fetch.py        币安 K 线增量拉取
scripts/          01~09，研究过程按顺序可复现
tests/            无前视偏差与引擎正确性测试
reports/          图表与全部扫描结果 CSV
```

## 核心设计

```
目标仓位 = clip(看多票比 × 40% / 已实现波动率, 0, 1) − 确认空头 × 25% × 同一标定
```

- **12 票**：Vortex 和 DMI 各取 6 个 lookback（10/14/20/28/36/48）
- **只用高低点**：VI 和 DMI 由区间关系构成，日线上稳定优于收盘价均线（家族中位 Sharpe 0.49/0.64 vs EMA 0.20）
- **Parkinson 波动率**做仓位标定，比收盘价估计效率高约 5 倍
- **空头是保险不是 alpha**：整体拉低 Sharpe，但把熊市亏损从 −15.4% 收窄到 −11.5%
- **仓位按 20% 网格取整**，避免噪声换手。年换手 24.7 倍，年化成本拖累 1.60%

## 免责声明

研究代码，非投资建议。回测不包含永续资金费率、交易所故障、极端滑点。
按报告中的置信区间，未来 12 个月跑输拿现货是完全可能的结果之一。
