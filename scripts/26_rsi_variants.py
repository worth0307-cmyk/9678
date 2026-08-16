"""Stop losses and RSI thresholds for the 4h strategy -- the whole surface.

Two requested changes, run as a grid rather than a search.  The distinction
matters here more than anywhere else in this repository: the base strategy makes
twelve trades in 3.62 years, so a first-half/second-half split leaves roughly six
trades a side, and six trades cannot separate skill from luck at any threshold
worth using.  Every number below is reported for the shape of the surface, not
so the best cell can be adopted.

The honest test at this sample size is not "which cell wins" but "does the whole
surface sit above zero".  A single cell standing out among dozens is what a null
surface looks like.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vibt import data as D, rsi_strat as R  # noqa: E402

pd.set_option("display.width", 220)
EQ0 = 1000.0
BPD = 6
REPORTS = Path(__file__).resolve().parent.parent / "reports"


def stats_of(df: pd.DataFrame, p: R.RsiParams) -> dict:
    res = R.run(df, p, EQ0)
    eq = res.equity.dropna()
    t = res.trades
    yrs = len(df) / (365 * BPD)
    final = float(eq.iloc[-1]) if len(eq) else EQ0
    if len(t) == 0:
        return {"trades": 0, "total": 0.0, "cagr": 0.0, "maxdd": 0.0,
                "win": np.nan, "worst_exc": np.nan, "stopped": 0, "avg_days": np.nan}
    return {
        "trades": len(t),
        "total": final / EQ0 - 1,
        "cagr": (final / EQ0) ** (1 / yrs) - 1,
        "maxdd": float((eq / eq.cummax() - 1).min()),
        "win": float((t.pnl > 0).mean()),
        "worst_exc": float(t.worst_excursion.min()),
        "stopped": int(t["stopped"].sum()) if "stopped" in t else 0,
        "avg_days": float(t.bars_held.mean() / BPD),
        "shorts": int((t.direction == "short").sum()),
    }


def show(rows: list[dict], keys: list[str]) -> None:
    d = pd.DataFrame(rows)
    for c in ("total", "cagr", "maxdd", "win", "worst_exc"):
        if c in d:
            d[c] = d[c].map(lambda v: f"{v:+.1%}" if pd.notna(v) else "")
    if "avg_days" in d:
        d["avg_days"] = d["avg_days"].map(lambda v: f"{v:.0f}" if pd.notna(v) else "")
    print(d[keys].to_string(index=False))


def main() -> None:
    df = D.load("4h", symbol="BTCUSDT")
    n = len(df)
    half = n // 2
    first, second = df.iloc[:half], df.iloc[half:]

    print("=" * 150)
    print("BTCUSDT 4h RSI 策略 —— 止损与阈值的完整曲面")
    print("=" * 150)
    print(f"""
  样本 {df.index[0].date()} -> {df.index[-1].date()}，{n} 根 4h
  前半 {first.index[0].date()} -> {first.index[-1].date()}
  后半 {second.index[0].date()} -> {second.index[-1].date()}

  基准（原规则、无止损、30/70）：期末 714.80，即 -28.5%
""")

    # ---------------------------------------------------------------- 1 stops
    print("=" * 150)
    print("1. 只加止损（阈值保持 30/70）")
    print("=" * 150 + "\n")
    rows = []
    for sl in (0.0, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30):
        st = stats_of(df, R.RsiParams(stop_loss=sl))
        rows.append({"止损": "无" if sl == 0 else f"{sl:.0%}", **st})
    show(rows, ["止损", "trades", "stopped", "total", "cagr", "maxdd",
                "win", "worst_exc", "avg_days"])
    print("""
  worst_exc 被止损精确压住，说明机制生效。
  但注意 total 那一列：止损没有把策略从亏损变成盈利，只是改变了亏损的形状。""")

    # ---------------------------------------------------------------- 2 bands
    print("\n" + "=" * 150)
    print("2. 只改阈值（无止损）")
    print("=" * 150 + "\n")
    rows = []
    for lo, hi in ((30, 70), (25, 75), (20, 80), (35, 65), (25, 70), (30, 75), (20, 75)):
        st = stats_of(df, R.RsiParams(lower=lo, upper=hi))
        rows.append({"下轨/上轨": f"{lo}/{hi}", **st})
    for w, ql, qh in ((500, 0.10, 0.90), (500, 0.05, 0.95), (250, 0.10, 0.90),
                      (1000, 0.10, 0.90)):
        st = stats_of(df, R.RsiParams(band_mode="quantile", band_window=w,
                                      q_lo=ql, q_hi=qh))
        rows.append({"下轨/上轨": f"分位 q{ql:.2f}/q{qh:.2f} w{w}", **st})
    show(rows, ["下轨/上轨", "trades", "shorts", "total", "cagr", "maxdd",
                "win", "worst_exc", "avg_days"])

    # ---------------------------------------------------------------- 3 grid
    print("\n" + "=" * 150)
    print("3. 两者一起：完整网格")
    print("=" * 150 + "\n")
    bands = [("30/70", dict(lower=30, upper=70)),
             ("25/75", dict(lower=25, upper=75)),
             ("20/80", dict(lower=20, upper=80)),
             ("分位10/90", dict(band_mode="quantile", q_lo=0.10, q_hi=0.90)),
             ("分位05/95", dict(band_mode="quantile", q_lo=0.05, q_hi=0.95))]
    stops = (0.0, 0.05, 0.10, 0.15, 0.20)
    grid = []
    print(f"  {'':<12}" + "".join(f"{('无' if s==0 else f'{s:.0%}'):>12}" for s in stops))
    for bname, bkw in bands:
        cells = []
        for sl in stops:
            st = stats_of(df, R.RsiParams(stop_loss=sl, **bkw))
            grid.append({"band": bname, "stop": sl, **st})
            cells.append(f"{st['total']:>+11.1%} ")
        print(f"  {bname:<12}" + "".join(cells))
    g = pd.DataFrame(grid)
    pos = (g.total > 0).mean()
    print(f"""
  共 {len(g)} 格。为正的 {int((g.total>0).sum())}/{len(g)} = {pos:.0%}
  中位 {g.total.median():+.1%}   平均 {g.total.mean():+.1%}
  最好 {g.total.max():+.1%} ({g.loc[g.total.idxmax(),'band']} / 止损 {g.loc[g.total.idxmax(),'stop']:.0%})
  最差 {g.total.min():+.1%}""")

    # ---------------------------------------------------------------- 4 split
    print("\n" + "=" * 150)
    print("4. 前后半稳定性 —— 这一节才是判定依据")
    print("=" * 150 + "\n")
    rows = []
    for bname, bkw in bands:
        for sl in (0.0, 0.10, 0.20):
            p = R.RsiParams(stop_loss=sl, **bkw)
            a, b = stats_of(first, p), stats_of(second, p)
            rows.append({"配置": f"{bname} / 止损{'无' if sl==0 else f'{sl:.0%}'}",
                         "前半笔数": a["trades"], "前半收益": a["total"],
                         "后半笔数": b["trades"], "后半收益": b["total"],
                         "符号一致": "是" if np.sign(a["total"]) == np.sign(b["total"]) else ""})
    d = pd.DataFrame(rows)
    out = d.copy()
    for c in ("前半收益", "后半收益"):
        out[c] = out[c].map(lambda v: f"{v:+.1%}")
    print(out.to_string(index=False))
    same = (d["符号一致"] == "是").mean()
    print(f"""
  前后半符号一致的配置: {int((d['符号一致']=='是').sum())}/{len(d)} = {same:.0%}
  前半平均笔数 {d['前半笔数'].mean():.1f}   后半平均笔数 {d['后半笔数'].mean():.1f}

  **这就是问题所在。** 每半段平均只有 {d['前半笔数'].mean():.0f} 笔交易。
  6 笔交易的胜负几乎全部由运气决定——把一枚硬币抛 6 次，
  出现 5 正 1 反的概率是 9.4%，而这已经足够让一个配置在表里"看起来很好"。""")

    corr = np.corrcoef(d["前半收益"], d["后半收益"])[0, 1]
    print(f"  前半收益 vs 后半收益 的相关性: {corr:+.3f}"
          f"   （{'前半完全预测不了后半' if abs(corr) < 0.3 else '存在一定关联'}）")

    REPORTS.mkdir(exist_ok=True)
    g.to_csv(REPORTS / "rsi_grid.csv", index=False)
    print(f"\n  wrote {REPORTS/'rsi_grid.csv'}")


if __name__ == "__main__":
    main()
