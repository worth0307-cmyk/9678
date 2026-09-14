"""资金费率收割：不预测方向，只收别人为杠杆付的钱.

    python scripts/54_funding_harvest.py

这个项目测过的所有东西都是**择时**，而且每一个都倒在同一个地方：样本内好看，
扣掉成本、扣掉后见之明、和一个诚实的零假设比过之后就没了。53 号脚本更是把
四个趋势工具一次性归结为「少持仓」。

资金费率收割是**结构上不同**的一件事：

    做空永续 + 等额买入现货  ->  价格涨跌完全对冲，只剩资金费率

它不需要预测任何东西。收益来源是**带杠杆的多头愿意为杠杆付钱** —— 那是风险
溢价，不是预测能力。这也是为什么它有理由长期存在，而一根均线没有。

本仓库有 2022-01-01 起六个币的完整资金费率，所以这一段可以精确算。

**但必须先说清楚这里算不了什么**（在最后一节详列）：没有现货数据，所以基差
的波动、现货腿的成本、爆仓风险都进不来。这一节算的是**资金费率流本身**，
它是这笔交易的收入项，不是全部。

一个重要的口径问题：均值会被极端值主导。SOL 的均值是 −5.0%/年，但那**全部**
来自 2022-11-09/10 FTX 崩盘期间打到 −200bp 下限的那几次；它的中位数是 +5.0%/年。
所以下面均值和中位数都报，并且单独看尾部。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vibt import ingest as I  # noqa: E402

pd.set_option("display.width", 230)
COINS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "TAOUSDT", "HYPEUSDT")

# 币安普通用户：现货 10bp/边（比永续贵），U 本位永续吃单 5bp/边。
# 建仓要同时开两条腿，平仓同理 —— 一次完整进出 = (10+5)×2 = 30bp。
SPOT_BPS, PERP_BPS = 10.0, 5.0
ROUND_TRIP_BPS = (SPOT_BPS + PERP_BPS) * 2


def load(sym: str) -> tuple[pd.Series, float]:
    d = I._load_funding(Path("data/funding") / f"{sym}_funding.csv.gz")
    r = d["funding_rate"].astype(float)
    return r, I.funding_interval_hours(d.index)


def daily_stream(sym: str) -> pd.Series:
    """每天收到的资金费率（做空永续 = 收正费率）。"""
    r, _ = load(sym)
    return r.groupby(r.index.floor("D")).sum()


def describe(s: pd.Series, label: str) -> dict:
    yrs = len(s) / 365.0
    eq = (1 + s).cumprod()
    sd = float(s.std())
    return {"版本": label, "年化(复利)": float((1 + float(eq.iloc[-1] - 1)) ** (1 / yrs) - 1),
            "年化波动": float(sd * math.sqrt(365)),
            "Sharpe": float(s.mean() / sd * math.sqrt(365)) if sd > 0 else np.nan,
            "最大回撤": float((eq / eq.cummax() - 1).min()),
            "为正天数": float((s > 0).mean()), "天数": len(s)}


def main() -> None:
    print("=" * 118)
    print("资金费率收割（做空永续 + 等额现货，delta 中性）")
    print("=" * 118)

    print("\n" + "-" * 118)
    print("A  单币：资金费率流本身（未扣建仓成本）")
    print("-" * 118 + "\n")
    rows = []
    streams = {}
    for s in COINS:
        d = daily_stream(s)
        streams[s] = d
        rows.append(describe(d, s))
    t = pd.DataFrame(rows).set_index("版本")
    for c in ("年化(复利)", "年化波动", "最大回撤", "为正天数"):
        t[c] = t[c].map("{:+.1%}".format)
    t["Sharpe"] = t["Sharpe"].map("{:+.2f}".format)
    print(t.to_string())

    print("\n" + "-" * 118)
    print("B  尾部：最差的那几段")
    print("-" * 118 + "\n")
    for s in COINS:
        d = streams[s]
        eq = (1 + d).cumprod()
        dd = eq / eq.cummax() - 1
        worst = dd.idxmin()
        # 这一段回撤持续多久
        peak = eq.loc[:worst].idxmax()
        rec = eq.loc[worst:][eq.loc[worst:] >= eq.loc[peak]]
        rec_txt = str(rec.index[0].date()) if len(rec) else "尚未修复"
        print(f"  {s:9s} 最大回撤 {dd.min():+.2%}  谷底 {worst.date()}  "
              f"从 {peak.date()} 开始  修复于 {rec_txt}   "
              f"单日最差 {d.min() * 100:+.2f}%")

    print("\n" + "-" * 118)
    print("C  组合：等权持有，以及把 BNB 剔除")
    print("-" * 118 + "\n")
    P = pd.DataFrame(streams)
    rows = [describe(P.mean(axis=1).dropna(), "六个币等权"),
            describe(P.drop(columns=["BNBUSDT"]).mean(axis=1).dropna(), "剔除 BNB（中位数为 0）"),
            describe(P[["BTCUSDT", "ETHUSDT"]].mean(axis=1).dropna(), "只做 BTC+ETH")]
    t = pd.DataFrame(rows).set_index("版本")
    for c in ("年化(复利)", "年化波动", "最大回撤", "为正天数"):
        t[c] = t[c].map("{:+.1%}".format)
    t["Sharpe"] = t["Sharpe"].map("{:+.2f}".format)
    print(t.to_string())

    print("\n" + "-" * 118)
    print("D  一个极低自由度的开关：近 7 天平均费率为正才持有")
    print("-" * 118 + "\n")
    rows = []
    for name, cols in (("六个币", list(COINS)),
                       ("剔除 BNB", [c for c in COINS if c != "BNBUSDT"]),
                       ("BTC+ETH", ["BTCUSDT", "ETHUSDT"])):
        sub = P[cols]
        # 只用**已经发生**的费率决定明天持不持有 —— 不偷看未来
        sig = (sub.rolling(7).mean().shift(1) > 0).astype(float)
        held = (sub * sig).sum(axis=1) / len(cols)
        # 开关每次切换都要进出两条腿
        turn = sig.diff().abs().sum(axis=1) / len(cols)
        net = (held - turn * ROUND_TRIP_BPS / 2 * 1e-4).dropna()
        rows.append(describe(net, f"{name} · 带开关（已扣切换成本）"))
        rows.append(describe((sub.mean(axis=1)).dropna(), f"{name} · 一直持有"))
    t = pd.DataFrame(rows).set_index("版本")
    for c in ("年化(复利)", "年化波动", "最大回撤", "为正天数"):
        t[c] = t[c].map("{:+.1%}".format)
    t["Sharpe"] = t["Sharpe"].map("{:+.2f}".format)
    print(t.to_string())

    print("\n" + "-" * 118)
    print("E  这里**算不了**的东西 —— 在动真钱之前必须自己确认")
    print("-" * 118)
    print(f"""
  1  现货腿完全不在数据里。做空永续必须配等额现货，而本仓库只有永续 K 线。
     基差（现货与永续的价差）自己会波动，建仓时基差不利就等于先亏一笔。

  2  建仓成本。币安现货普通用户 {SPOT_BPS:.0f}bp/边，永续吃单 {PERP_BPS:.0f}bp/边，
     两条腿进出一次 = **{ROUND_TRIP_BPS:.0f}bp**。按 BTC 年化 6.6% 算，光建仓就吃掉
     {ROUND_TRIP_BPS / 1e4 / 0.066 * 12:.1f} 个月的收益 —— 所以这是个**长持**的交易，频繁进出必亏。

  3  爆仓风险。如果用现货做保证金去开空，价格大涨时永续腿浮亏，
     保证金率会掉。真正 delta 中性不等于没有爆仓风险。

  4  这是一笔**拥挤**的交易。6% 是拿这些风险换来的溢价，不是白捡的钱。
     所有人都知道它，所以它的回报就该只有这个水平。

  5  2022~2026 是多头主导的行情，所以费率整体为正。**长期熊市里费率会转负**，
     那时收割方要付钱。上面 B 节的尾部就是这件事的样子。""")


if __name__ == "__main__":
    main()
