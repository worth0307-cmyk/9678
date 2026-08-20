"""The fixed-anchor equal-size grid, started at every possible date.

This is the classic stock grid, not a martingale: five equal layers, no doubling,
no stop.  A single backtest of it is close to meaningless because everything
depends on where the anchor happens to sit relative to what price does next, so
this launches the grid on every date in the sample and reports the distribution.

The supplied parameters are 235 base, 1.00 buy step, 1.00 sell step, 100 shares
a layer, 500 max -- five layers spanning 235 down to 230, which is 2.13% of
coverage.  That number is the whole analysis.  Above the band the grid holds
nothing and earns nothing; below it the grid is fully loaded, has no orders left,
and is simply a long position with no stop.  Coverage is tested from 2% to 40%
so the trade-off between grid income and the size of the hole underneath is
visible rather than assumed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vibt import data as D, grid as G  # noqa: E402

pd.set_option("display.width", 210)
REPORTS = Path(__file__).resolve().parent.parent / "reports"
HOLD_DAYS = 180


def launch_everywhere(df: pd.DataFrame, step: float, layers: int,
                      hold: int = HOLD_DAYS, stride: int = 5) -> pd.DataFrame:
    """Start the grid on every stride-th bar and hold it for `hold` bars."""
    rows = []
    per_layer = 1.0 / layers          # fully invested when all layers fill
    for i in range(0, len(df) - hold, stride):
        sub = df.iloc[i:i + hold]
        anchor = float(sub["open"].iloc[0])
        p = G.AnchoredParams(anchor=anchor, step=step, up=step, layers=layers,
                             per_layer=per_layer, fee=0.00065, ratio_mode=True)
        r = G.run_anchored(sub, p)
        px_end = float(sub["close"].iloc[-1])
        rows.append({
            "start": sub.index[0], "trips": r.round_trips,
            "realised": r.realised.iloc[-1] - 1.0,
            "total": r.mark.iloc[-1] - 1.0,
            "full_frac": r.bars_full / len(sub),
            "empty_frac": r.bars_empty / len(sub),
            "px_move": px_end / anchor - 1.0,
        })
    return pd.DataFrame(rows)


def main() -> None:
    syms = D.available_symbols(require=("1d",))
    syms = [s for s in syms if s not in ("USDCUSDT", "BTCDOMUSDT")][:40]

    print("=" * 150)
    print("固定锚点等量网格 —— 在每个可能的启动日各跑一次")
    print("=" * 150)
    print(f"""
  你给的参数：基准价 235，每跌 1 买，每涨 1 卖，每笔 100 股，最大 500 股
    -> 5 层，覆盖 235~230，宽度 **2.13%**，每格利润 0.43%

  下面用比例模式复现同样的形状，并把覆盖宽度从 2% 拉到 40%，
  每个币在每 5 天启动一次网格、持有 {HOLD_DAYS} 天，汇总所有启动点。
""")

    grid_specs = [(0.0043, 5, "2.1%  (你的参数)"),
                  (0.01, 5, "4.9%"),
                  (0.02, 5, "9.6%"),
                  (0.02, 10, "18.3%"),
                  (0.03, 10, "26.3%"),
                  (0.05, 10, "40.1%")]

    out = []
    for step, layers, label in grid_specs:
        allr = []
        for s in syms:
            df = D.load("1d", symbol=s)
            if len(df) < HOLD_DAYS + 50:
                continue
            allr.append(launch_everywhere(df, step, layers))
        d = pd.concat(allr, ignore_index=True)
        cover = 1 - (1 - step) ** layers
        out.append({
            "覆盖": label, "启动次数": len(d),
            "中位完成格数": d.trips.median(),
            "满仓时间占比": d.full_frac.mean(),
            "空仓时间占比": d.empty_frac.mean(),
            "中位已实现": d.realised.median(),
            "中位实际": d.total.median(),
            "实际为正占比": (d.total > 0).mean(),
            "5%分位": d.total.quantile(0.05),
            "95%分位": d.total.quantile(0.95),
        })

    t = pd.DataFrame(out)
    o = t.copy()
    for c in ("满仓时间占比", "空仓时间占比", "中位已实现", "中位实际",
              "实际为正占比", "5%分位", "95%分位"):
        o[c] = o[c].map(lambda v: f"{v:+.1%}")
    o["中位完成格数"] = o["中位完成格数"].map(lambda v: f"{v:.0f}")
    print(o.to_string(index=False))

    print(f"""
  「满仓时间占比」= 五层全部买满、没有任何买单在挂、网格收入为零的时间。
  「空仓时间占比」= 价格在最高一层之上、什么都没持有的时间。
  两者相加就是**网格完全不工作的时间**。

  「中位已实现」和「中位实际」的差 = 藏在未平仓头寸里的浮亏。
""")

    # ------------------------------------------------------------------ 2
    print("=" * 150)
    print("按你的参数（2.1% 覆盖）细看")
    print("=" * 150 + "\n")
    allr = []
    for s in syms:
        df = D.load("1d", symbol=s)
        if len(df) < HOLD_DAYS + 50:
            continue
        r = launch_everywhere(df, 0.0043, 5)
        r["symbol"] = s
        allr.append(r)
    d = pd.concat(allr, ignore_index=True)
    dead = d.full_frac > 0.9
    print(f"  {len(d)} 次启动，持有 {HOLD_DAYS} 天：")
    print(f"    立刻买满并一直满仓（>90% 时间）的比例: {dead.mean():.1%}")
    print(f"    完成 0 个格子的比例:                  {(d.trips == 0).mean():.1%}")
    print(f"    中位完成格数:                          {d.trips.median():.0f}")
    print(f"\n  收益分布（实际，含浮亏）:")
    for q in (5, 25, 50, 75, 95):
        print(f"    {q:>2}% 分位  {d.total.quantile(q/100):+7.1%}")
    print(f"\n  实际收益 与 期间价格涨跌 的相关性: {d.total.corr(d.px_move):+.3f}")
    print(f"""
  最后这个相关性是关键：{d.total.corr(d.px_move):+.2f} 说明这套网格的盈亏
  {'几乎完全由标的涨跌决定' if d.total.corr(d.px_move) > 0.8 else '和标的涨跌高度相关'}——
  它不是一个"震荡获利"的策略，它是一个**被网格包装过的多头持仓**。""")

    REPORTS.mkdir(exist_ok=True)
    t.to_csv(REPORTS / "anchored_grid.csv", index=False)
    print(f"\n  wrote {REPORTS/'anchored_grid.csv'}")


if __name__ == "__main__":
    main()
