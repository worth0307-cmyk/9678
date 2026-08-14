"""Entry and exit, spelled out.

A cross-sectional book has no entry signal and no exit signal.  It has a target
portfolio, recomputed on a schedule; "opening" and "closing" are just the
difference between two consecutive targets.  That difference is what gets sent
to the exchange, and it is the only thing that costs money.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vibt import backtest as B, data as D, metrics as M, xsec as X  # noqa: E402

pd.set_option("display.width", 200)

COSTS = B.Costs(fee_bps=4.5, slip_bps=2.0)
ANN = 365.0
LOOKBACK, N_SIDE, REBAL = 14, 5, 3
UNIT = 0.5 / N_SIDE          # 10% of equity per name


def classify(prev: pd.Series, cur: pd.Series) -> dict[str, list[str]]:
    out = {"新开多": [], "新开空": [], "平多": [], "平空": [],
           "多翻空": [], "空翻多": [], "保持": []}
    for s in prev.index:
        a, b = prev.get(s, 0.0), cur.get(s, 0.0)
        if a == 0 and b > 0:
            out["新开多"].append(s)
        elif a == 0 and b < 0:
            out["新开空"].append(s)
        elif a > 0 and b == 0:
            out["平多"].append(s)
        elif a < 0 and b == 0:
            out["平空"].append(s)
        elif a > 0 > b:
            out["多翻空"].append(s)
        elif a < 0 < b:
            out["空翻多"].append(s)
        elif a != 0 and np.sign(a) == np.sign(b):
            out["保持"].append(s)
    return out


def main() -> None:
    syms = D.available_symbols(require=("1d",))
    U = D.load_universe(syms, tfs=("1d",))
    px = X.price_panel(U, "1d", "open")
    feat = X.feature_panel(U, lambda d: d["close"] / d["close"].shift(LOOKBACK) - 1)
    w = X.cross_sectional_weights(feat, n_side=N_SIDE, mode="long_short")
    res = X.run(px, w, COSTS, ANN, "", REBAL)
    held = res.weights

    print("=" * 120)
    print("1. 没有入场信号，也没有出场信号")
    print("=" * 120)
    print(f"""
  这套策略里不存在"某个条件触发就开仓"。每 {REBAL} 天做一次同样的事：

    1) 用截至昨天收盘的数据，算每个币的 {LOOKBACK} 日收益率
    2) 对当天有价格、有信号的币排序
    3) 目标组合 = 前 {N_SIDE} 名各 +{UNIT:.0%}，后 {N_SIDE} 名各 -{UNIT:.0%}，其余 0
    4) 拿目标组合减去手上的组合，差额就是要发的单

  所以"平仓"只有一个原因：**这个币掉出了前 {N_SIDE} / 后 {N_SIDE} 名**。
  没有止损，没有止盈，没有时间止损。持仓期内价格怎么走都不动手。
""")

    print("=" * 120)
    print("2. 每次调仓只有 6 种动作")
    print("=" * 120)
    rebal_days = held.index[::REBAL]
    counts = {k: 0 for k in ("新开多", "新开空", "平多", "平空", "多翻空", "空翻多", "保持")}
    n_rb = 0
    for i in range(1, len(rebal_days)):
        prev, cur = held.loc[rebal_days[i - 1]], held.loc[rebal_days[i]]
        if (cur != 0).sum() == 0:
            continue
        n_rb += 1
        for k, v in classify(prev, cur).items():
            counts[k] += len(v)
    print(f"\n  统计 {n_rb} 次调仓，每次平均：")
    for k, v in counts.items():
        print(f"    {k:<8} {v/max(n_rb,1):5.2f} 个/次   （总计 {v}）")
    turnover = sum(counts[k] for k in ("新开多", "新开空", "平多", "平空")) \
        + 2 * sum(counts[k] for k in ("多翻空", "空翻多"))
    print(f"\n  平均每次调仓要动 {turnover/max(n_rb,1):.1f} 条腿，"
          f"其中约 {counts['保持']/max(n_rb,1):.1f} 个仓位原样留着不动")

    print("\n" + "=" * 120)
    print("3. 一次真实调仓的完整动作")
    print("=" * 120)
    valid = [d for d in rebal_days if (held.loc[d] != 0).sum() == 2 * N_SIDE]
    day = valid[len(valid) // 2]
    prev_day = rebal_days[list(rebal_days).index(day) - 1]
    prev, cur = held.loc[prev_day], held.loc[day]
    print(f"\n  上次调仓 {prev_day.date()}  ->  本次调仓 {day.date()}\n")
    acts = classify(prev, cur)
    sig = feat.loc[day] if day in feat.index else pd.Series(dtype=float)
    for k in ("保持", "平多", "平空", "新开多", "新开空", "多翻空", "空翻多"):
        if not acts[k]:
            continue
        for s in acts[k]:
            rank = int(sig.rank(ascending=False).get(s, -1)) if s in sig.index else -1
            n_valid = int(sig.notna().sum())
            print(f"    {k:<6} {s:<15} 信号 {sig.get(s, np.nan):+7.1%}  "
                  f"排名 {rank}/{n_valid}   仓位 {prev.get(s,0):+.0%} -> {cur.get(s,0):+.0%}")
    traded = float((cur - prev).abs().sum())
    print(f"\n    本次成交名义额 = {traded:.0%} 的净值   成本 = {traded*COSTS.per_side:.3%}")

    print("\n" + "=" * 120)
    print("4. 一个仓位能活多久")
    print("=" * 120)
    lives = []
    for s in held.columns:
        v = held[s].to_numpy()
        i = 0
        while i < len(v):
            if v[i] == 0:
                i += 1
                continue
            j = i
            while j + 1 < len(v) and np.sign(v[j + 1]) == np.sign(v[i]) and v[j + 1] != 0:
                j += 1
            lives.append(j - i + 1)
            i = j + 1
    lives = np.array(lives)
    print(f"\n  共 {len(lives)} 段持仓，持仓天数分布：")
    print(f"    中位 {np.median(lives):.0f} 天   平均 {lives.mean():.1f} 天   "
          f"p75 {np.percentile(lives,75):.0f} 天   p90 {np.percentile(lives,90):.0f} 天   "
          f"最长 {lives.max()} 天")
    print(f"    只活一个调仓周期（<= {REBAL} 天）的占 {(lives<=REBAL).mean():.0%}")
    print(f"    活过 2 周的占 {(lives>14).mean():.0%}")

    print("\n" + "=" * 120)
    print("5. 没有止损意味着什么")
    print("=" * 120)
    bar = (px.shift(-1) / px - 1.0)
    per_leg = (held * bar).stack()
    worst_leg = per_leg.nsmallest(5)
    print("\n  单腿单日最差的 5 次（占总净值的百分比）：")
    for (d, s), v in worst_leg.items():
        print(f"    {d.date()}  {s:<15} {v:+.2%}   (仓位 {held.loc[d, s]:+.0%}，"
              f"当日价格 {bar.loc[d, s]:+.1%})")
    print(f"\n  组合单日最差：{res.rets.min():+.2%} on {res.rets.idxmin().date()}")
    print(f"  组合最大回撤：{res.stats.max_dd:.1%}")
    print(f"\n  单个仓位最多占 {UNIT:.0%} 净值，所以就算某个币一天腰斩，"
          f"组合当天最多亏 {UNIT*0.5:.0%}。")
    print("  这就是不设止损的底气：**用仓位上限代替止损**。")

    print("\n" + "=" * 120)
    print("6. 一个必须说清楚的建模细节")
    print("=" * 120)
    print(f"""
  引擎在两次调仓之间把权重当成恒定的 {UNIT:.0%}。这隐含了"每天把仓位调回 {UNIT:.0%}"，
  但**没有为这个日内调整收费**。真实操作有两种选择：

    A. 让它漂：3 天里不管，涨的仓位变大、跌的变小。省手续费，但敞口会偏离。
    B. 每天调回 {UNIT:.0%}：和回测一致，但要多付手续费。

  下面量化 B 比 A 多花多少钱：""")
    drift_turn = 0.0
    for i in range(len(held) - 1):
        h = held.iloc[i]
        if (h != 0).sum() == 0:
            continue
        r = bar.iloc[i].reindex(h.index).fillna(0.0)
        drifted = h * (1 + r)
        nxt = held.iloc[i + 1]
        if (nxt - h).abs().sum() > 1e-9:      # a rebalance day, already charged
            continue
        drift_turn += float((h - drifted).abs().sum())
    years = len(res.rets) / ANN
    extra = drift_turn * COSTS.per_side
    print(f"    日内漂移需要的额外换手 = {drift_turn:.1f} 倍净值 / {years:.1f} 年")
    print(f"    对应额外成本 = {extra:.1%} 总计，即 {extra/years:.2%} / 年")
    print(f"    当前记录的年化成本 {res.costs.sum()/years:.2%}，"
          f"加上后是 {(res.costs.sum()+extra)/years:.2%}")
    print(f"    对 CAGR 的影响约 -{extra/years:.2%}，即从 {res.stats.cagr:.1%} 降到 "
          f"约 {res.stats.cagr - extra/years:.1%}")
    print("\n  换句话说：选 A（让它漂）就不用付这笔钱，但敞口会偏离目标；")
    print("  选 B（每天调回）要付，回测数字应该按上面调低。这笔账不大，但不该藏着。")


if __name__ == "__main__":
    main()
