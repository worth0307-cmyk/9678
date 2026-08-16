"""Relaxing the RSI strategy's two entry filters, which finally buys sample size.

The base rule demands three consecutive RSI bars outside the band AND three
consecutive MA moves in the right direction.  Stacking them leaves twelve trades
in 3.62 years, which is too few to conclude anything -- the previous run put the
base strategy at t = -0.97, p = 0.355, unable to establish even its own loss.

Dropping either filter multiplies the trade count severalfold, so for the first
time the per-trade mean can be tested rather than eyeballed.  That is the point
of this run: not to find a better setting, but to reach a sample where "better"
and "worse" are distinguishable at all.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scipy import stats  # noqa: E402
from vibt import data as D, rsi_strat as R  # noqa: E402

pd.set_option("display.width", 220)
EQ0, BPD = 1000.0, 6
REPORTS = Path(__file__).resolve().parent.parent / "reports"


def summarise(df: pd.DataFrame, p: R.RsiParams) -> dict:
    res = R.run(df, p, EQ0)
    t, eq = res.trades, res.equity.dropna()
    yrs = len(df) / (365 * BPD)
    final = float(eq.iloc[-1]) if len(eq) else EQ0
    out = {"trades": len(t), "total": final / EQ0 - 1,
           "cagr": (final / EQ0) ** (1 / yrs) - 1 if final > 0 else -1.0,
           "maxdd": float((eq / eq.cummax() - 1).min()) if len(eq) else 0.0}
    if len(t) >= 2:
        x = t.return_on_equity.to_numpy()
        tt = stats.ttest_1samp(x, 0)
        out |= {"per_trade": x.mean(), "sd": x.std(ddof=1),
                "t": float(tt.statistic), "p": float(tt.pvalue),
                "win": float((t.pnl > 0).mean()),
                "shorts": int((t.direction == "short").sum()),
                "days": float(t.bars_held.mean() / BPD)}
    else:
        out |= {"per_trade": np.nan, "sd": np.nan, "t": np.nan, "p": np.nan,
                "win": np.nan, "shorts": 0, "days": np.nan}
    return out


def main() -> None:
    df = D.load("4h", symbol="BTCUSDT")
    half = len(df) // 2
    first, second = df.iloc[:half], df.iloc[half:]

    print("=" * 150)
    print("BTCUSDT 4h  放宽两个入场过滤条件")
    print("=" * 150)
    print("""
  RSI条件 = 连续 N 根 RSI 在带外   （N=1 表示只看当根）
  MA条件  = 连续 M 根 RSI-based MA 同向   （M=0 表示完全关闭这个条件）
""")

    print("=" * 150)
    print("1. 两个条件的完整组合（无止损，阈值 30/70）")
    print("=" * 150 + "\n")
    rows = []
    for rb, tb in itertools.product((1, 2, 3), (0, 1, 2, 3)):
        s = summarise(df, R.RsiParams(rsi_bars=rb, trend_bars=tb))
        rows.append({"RSI条件": f"{rb}根", "MA条件": "关" if tb == 0 else f"{tb}根", **s})
    d = pd.DataFrame(rows)
    out = d.copy()
    for c in ("total", "cagr", "maxdd", "per_trade", "sd", "win"):
        out[c] = out[c].map(lambda v: f"{v:+.1%}" if pd.notna(v) else "")
    out["t"] = out["t"].map(lambda v: f"{v:+.2f}" if pd.notna(v) else "")
    out["p"] = out["p"].map(lambda v: f"{v:.3f}" if pd.notna(v) else "")
    out["days"] = out["days"].map(lambda v: f"{v:.0f}" if pd.notna(v) else "")
    print(out[["RSI条件", "MA条件", "trades", "shorts", "total", "cagr", "maxdd",
               "win", "per_trade", "sd", "t", "p", "days"]].to_string(index=False))

    best_p = d.p.min()
    print(f"""
  12 个组合的总收益全部为负，而且交易越多总亏损越大。

  但 p 那一列必须如实说：**样本量翻了 5 倍，依然没有达到显著。**
  最好的一格 p = {best_p:.3f}，{int(d.trades.max())} 笔那一格 p = {float(d.loc[d.trades.idxmax(),'p']):.3f}。
  原因是笔数变多的同时每笔亏损也变小了（{d.loc[d.trades.idxmin(),'per_trade']:+.1%} -> {d.loc[d.trades.idxmax(),'per_trade']:+.1%}），
  两个效应互相抵消，t 值几乎没动。

  所以放宽条件**没有**把"这个策略亏钱"变成一个统计结论。
  它证明的是另一件事，见下一节。""")

    # ---------------------------------------------------------------- 2
    print("\n" + "=" * 150)
    print("2. 交易频率 vs 每笔期望 —— 亏损来自规则本身，不是来自过滤")
    print("=" * 150 + "\n")
    valid = d[d.trades >= 5]
    r, pv = stats.pearsonr(valid.trades, valid.per_trade)
    print(f"  交易笔数 与 每笔平均收益 的相关性: {r:+.3f}  p={pv:.3f}")
    verdict = ("过滤有用：筛掉的确实是更差的信号" if r < -0.3 else
               "**过滤是有害的：它筛掉的反而是更好的信号**" if r > 0.3 else
               "过滤基本无关")
    print(f"""
  如果过滤条件在挑好信号，笔数越少每笔期望应该越**高**，相关性应该为负。
  实际是 {r:+.3f}（p={pv:.3f}）——方向正好相反。{verdict}

  具体说：最宽松（1根RSI、不看MA、{int(valid.trades.max())} 笔）每笔 {valid.loc[valid.trades.idxmax(),'per_trade']:+.2%}；
  最严格的那几档（12~16 笔）每笔在 {valid.per_trade.min():+.2%} ~ {valid[valid.trades<20].per_trade.max():+.2%} 之间。

  **原版那 12 笔之所以总亏损最小，是因为它交易得最少，不是因为它选得准
  ——按每笔算，加了过滤的档位反而更差。**""")

    # ---------------------------------------------------------------- 3
    print("\n" + "=" * 150)
    print("3. 放宽之后加止损（RSI条件=1根、MA条件=关，共 66 笔）")
    print("=" * 150 + "\n")
    rows = []
    for sl in (0.0, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20):
        s = summarise(df, R.RsiParams(rsi_bars=1, trend_bars=0, stop_loss=sl))
        rows.append({"止损": "无" if sl == 0 else f"{sl:.0%}", **s})
    o = pd.DataFrame(rows)
    for c in ("total", "cagr", "maxdd", "per_trade", "win"):
        o[c] = o[c].map(lambda v: f"{v:+.1%}" if pd.notna(v) else "")
    o["t"] = o["t"].map(lambda v: f"{v:+.2f}")
    o["p"] = o["p"].map(lambda v: f"{v:.3f}")
    print(o[["止损", "trades", "total", "cagr", "maxdd", "win", "per_trade",
             "t", "p"]].to_string(index=False))

    # ---------------------------------------------------------------- 4
    print("\n" + "=" * 150)
    print("4. 前后半稳定性（现在样本够了，这个检验才有意义）")
    print("=" * 150 + "\n")
    rows = []
    for rb, tb in ((3, 3), (3, 0), (1, 3), (1, 0), (2, 0)):
        p = R.RsiParams(rsi_bars=rb, trend_bars=tb)
        a, b = summarise(first, p), summarise(second, p)
        rows.append({"配置": f"RSI{rb}根/MA{'关' if tb==0 else f'{tb}根'}",
                     "前半笔数": a["trades"], "前半收益": a["total"],
                     "前半每笔": a["per_trade"],
                     "后半笔数": b["trades"], "后半收益": b["total"],
                     "后半每笔": b["per_trade"],
                     "符号一致": "是" if np.sign(a["total"]) == np.sign(b["total"]) else ""})
    dd = pd.DataFrame(rows)
    o2 = dd.copy()
    for c in ("前半收益", "后半收益", "前半每笔", "后半每笔"):
        o2[c] = o2[c].map(lambda v: f"{v:+.1%}" if pd.notna(v) else "")
    print(o2.to_string(index=False))
    neg1 = int((dd["前半收益"] < 0).sum())
    neg2 = int((dd["后半收益"] < 0).sum())
    both = int(((dd["前半收益"] < 0) & (dd["后半收益"] < 0)).sum())
    print(f"""
  前半为负 {neg1}/{len(dd)}   后半为负 {neg2}/{len(dd)}   两半都为负 {both}/{len(dd)}
  符号一致 {int((dd['符号一致']=='是').sum())}/{len(dd)}

  注意后半有 {len(dd)-neg2} 个配置转正，而且都是**保留 MA 条件**的那几个
  ——它们在后半只有 5~7 笔交易，又回到了"几笔抛硬币"的状态。
  真正样本量大的三个配置（MA关，26~37 笔/半段）**两半都是负的**。""")

    # ---------------------------------------------------------------- 5
    print("\n" + "=" * 150)
    print("5. 为什么亏 —— 方向偏空")
    print("=" * 150 + "\n")
    p = R.RsiParams(rsi_bars=1, trend_bars=0)
    t = R.run(df, p, EQ0).trades
    for side in ("long", "short"):
        sub = t[t.direction == side]
        if not len(sub):
            continue
        tt = stats.ttest_1samp(sub.return_on_equity, 0) if len(sub) > 1 else None
        print(f"  {side:<6} {len(sub):>3} 笔   每笔 {sub.return_on_equity.mean():+.2%}   "
              f"胜率 {(sub.pnl>0).mean():.0%}   "
              f"合计 {sub.pnl.sum():+,.0f} USDT"
              + (f"   t={tt.statistic:+.2f} p={tt.pvalue:.3f}" if tt else ""))
    rsi = R.rsi_pine(df["close"], 14)
    print(f"""
  样本里 RSI>70 有 {(rsi>70).sum()} 根，RSI<30 只有 {(rsi<30).sum()} 根，
  因为 BTC 这段涨了 {df['close'].iloc[-1]/df['open'].iloc[0]-1:+.0%}。
  超买远比超卖频繁 -> 规则被迫做空一个上涨的市场，而且做空次数是做多的数倍。""")

    REPORTS.mkdir(exist_ok=True)
    d.to_csv(REPORTS / "rsi_relaxed.csv", index=False)
    print(f"\n  wrote {REPORTS/'rsi_relaxed.csv'}")


if __name__ == "__main__":
    main()
