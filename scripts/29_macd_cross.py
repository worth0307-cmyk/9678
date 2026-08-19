"""MACD golden/death cross, long-flat, across timeframes and the whole universe.

The strategy as supplied: MACD(12,26,9) golden cross buys the full allocation,
death cross sells all of it, no shorting, limit order at the bar's close.

Two things about the source worth carrying into the test.  The signal runs on
10-minute bars -- the H1 in global_variables is overwritten before it is used --
and the code is written for equities (RTH sessions, lot_size, DAY orders), which
have no meaning on a 24/7 market.

There is no 10-minute data here, so this runs 1d, 4h and 1h.  That is the useful
shape anyway: if performance degrades monotonically as the bar shrinks, 10-minute
sits past the end of that trend rather than somewhere new, and the reason is
mechanical -- MACD(12,26,9) on 10-minute bars is a 2-hour against a 4.3-hour
average, so it crosses constantly and pays a spread every time.

Every timeframe is run across all 26 symbols rather than BTC alone, because a
single asset's cross count is too small to separate a rule from its sample.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vibt import backtest as B, data as D, metrics as M  # noqa: E402

pd.set_option("display.width", 200)
REPORTS = Path(__file__).resolve().parent.parent / "reports"
FAST, SLOW, SIG = 12, 26, 9


def macd_position(close: pd.Series) -> pd.Series:
    """1 while MACD is above its signal line, 0 below -- long or flat, never short."""
    macd = (close.ewm(span=FAST, adjust=False).mean()
            - close.ewm(span=SLOW, adjust=False).mean())
    signal = macd.ewm(span=SIG, adjust=False).mean()
    return (macd > signal).astype(float)


def run_one(df: pd.DataFrame, tf: str, costs: B.Costs) -> tuple[B.Result, int]:
    pos = macd_position(df["close"])
    res = B.run(df, pos, costs, D.bars_per_year(tf), "")
    flips = int((pos.diff().abs() > 0).sum())
    return res, flips


def main() -> None:
    syms = D.available_symbols(require=("1h", "4h", "1d"))
    print("=" * 140)
    print(f"MACD({FAST},{SLOW},{SIG}) 金叉做多 / 死叉空仓   {len(syms)} 个币")
    print("=" * 140)
    print("""
  规则完全按你给的逻辑：金叉满仓买入，死叉全部卖出，不做空。
  信号在 K 线收盘确认，下一根开盘成交（原策略是收盘价限价单，见文末讨论）。
""")

    for costs, label in ((B.Costs(0, 0), "零成本"), (B.Costs(4.5, 2.0), "6.5bp/边")):
        print("=" * 140)
        print(f"{label}")
        print("=" * 140 + "\n")
        print(f"  {'周期':<6}{'币数':>5}{'平均换手次数':>14}{'中位年化':>11}{'平均年化':>11}"
              f"{'为正的币':>10}{'中位Sharpe':>12}{'中位回撤':>10}")
        for tf in ("1d", "4h", "1h"):
            rows = []
            for s in syms:
                df = D.load(tf, symbol=s)
                res, flips = run_one(df, tf, costs)
                yrs = len(df) / D.bars_per_year(tf)
                tot = res.stats.total_return
                rows.append({"cagr": (1 + tot) ** (1 / yrs) - 1 if tot > -1 else -1.0,
                             "sharpe": res.stats.sharpe, "dd": res.stats.max_dd,
                             "flips": flips})
            d = pd.DataFrame(rows)
            print(f"  {tf:<6}{len(d):>5}{d.flips.mean():>14.0f}{d.cagr.median():>+11.1%}"
                  f"{d.cagr.mean():>+11.1%}{(d.cagr > 0).mean():>10.0%}"
                  f"{d.sharpe.median():>+12.2f}{d.dd.median():>10.1%}")

    # ------------------------------------------------------------------ vs hold
    print("\n" + "=" * 140)
    print("对照：同期买入持有")
    print("=" * 140 + "\n")
    rows = []
    for s in syms:
        df = D.load("1d", symbol=s)
        yrs = len(df) / 365.0
        bh = float(df["close"].iloc[-1] / df["open"].iloc[0] - 1)
        res, _ = run_one(df, "1d", B.Costs(4.5, 2.0))
        rows.append({"symbol": s, "MACD_1d": res.stats.total_return, "买入持有": bh,
                     "差": res.stats.total_return - bh})
    d = pd.DataFrame(rows).sort_values("差", ascending=False)
    o = d.copy()
    for c in ("MACD_1d", "买入持有", "差"):
        o[c] = o[c].map(lambda v: f"{v:+.1%}")
    print(o.to_string(index=False))
    print(f"""
  MACD 日线跑赢买入持有的: {int((d['差'] > 0).sum())}/{len(d)} 个币
  中位差 {d['差'].median():+.1%}""")

    # ------------------------------------------------------------------ decay
    print("\n" + "=" * 140)
    print("周期越快，成本吃掉的越多")
    print("=" * 140 + "\n")
    print(f"  {'周期':<6}{'平均换手':>10}{'零成本中位年化':>16}{'6.5bp中位年化':>16}{'成本吃掉':>12}")
    for tf in ("1d", "4h", "1h"):
        a, b, fl = [], [], []
        for s in syms:
            df = D.load(tf, symbol=s)
            yrs = len(df) / D.bars_per_year(tf)
            r0, f0 = run_one(df, tf, B.Costs(0, 0))
            r1, _ = run_one(df, tf, B.Costs(4.5, 2.0))
            for lst, r in ((a, r0), (b, r1)):
                t = r.stats.total_return
                lst.append((1 + t) ** (1 / yrs) - 1 if t > -1 else -1.0)
            fl.append(f0)
        m0, m1 = np.median(a), np.median(b)
        print(f"  {tf:<6}{np.mean(fl):>10.0f}{m0:>+16.1%}{m1:>+16.1%}{m1-m0:>+12.1%}")
    print("""
  换手次数随周期缩短大致翻倍，成本损耗跟着翻倍。
  MACD(12,26,9) 在 10 分钟线上是「2 小时均线 vs 4.3 小时均线」——
  按这个趋势外推，10 分钟的换手会是 1h 的数倍，成本损耗也是。""")

    REPORTS.mkdir(exist_ok=True)
    d.to_csv(REPORTS / "macd_cross.csv", index=False)
    print(f"\n  wrote {REPORTS/'macd_cross.csv'}")


if __name__ == "__main__":
    main()
