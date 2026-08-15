"""Does the cross-sectional edge survive at 4h and 1h?

Controlled comparison: same universe, same rule, same calendar-time lookback and
holding period, only the bar size changes.  14 daily bars = 84 4h bars = 336 1h
bars; rebalancing every 3 days = every 18 4h bars = every 72 1h bars.

All 26 symbols now carry all three timeframes, so this runs on the same universe
and the same top/bottom-5 construction as the headline daily strategy.  Earlier
versions were stuck with the six symbols that happened to have intraday data,
which was too small for the cross-section to work at all -- so the numbers here
supersede that run rather than extending it.
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
N_SIDE = 5   # matches the headline daily strategy
REPORTS = Path(__file__).resolve().parent.parent / "reports"

# (timeframe, bars per day, lookback bars = 14 days, rebalance bars = 3 days)
GRID = [("1d", 1, 14, 3), ("4h", 6, 84, 18), ("1h", 24, 336, 72)]


def main() -> None:
    syms = D.available_symbols(require=("1h", "4h", "1d"))
    print("=" * 140)
    print(f"FREQUENCY COMPARISON   universe = {syms}")
    print("=" * 140)
    print("""
  同一个币池、同一条规则、同样的日历回看期（14 天）和持仓期（3 天），
  只改变 K 线粒度。频率越高，同样的日历周期需要越多根 K 线，
  但**再平衡次数不变**——所以成本差异只来自信号本身的抖动。
""")
    rows = []
    print(f"  {'周期':<6}{'K线数':>8}{'回看':>7}{'再平衡':>8}{'Sharpe':>9}{'收益':>10}"
          f"{'波动':>8}{'回撤':>9}{'年化成本':>10}{'年换手':>9}")
    for tf, bpd, lb, rb in GRID:
        U = D.load_universe(syms, tfs=(tf,))
        ann = D.bars_per_year(tf)
        px = X.price_panel(U, tf, "open")
        feat = X.feature_panel(U, lambda d, k=lb: d["close"] / d["close"].shift(k) - 1, tf=tf)
        w = X.cross_sectional_weights(feat, n_side=N_SIDE, mode="long_short")
        res = X.run(px, w, COSTS, ann, "", rb)
        yrs = len(res.rets) / ann
        rows.append({"tf": tf, "sharpe": res.stats.sharpe, "ret": res.stats.total_return,
                     "cost_yr": res.costs.sum() / yrs})
        print(f"  {tf:<6}{len(px):>8}{lb:>7}{rb:>8}{res.stats.sharpe:>+9.2f}"
              f"{res.stats.total_return:>+10.1%}{res.stats.ann_vol:>8.0%}"
              f"{res.stats.max_dd:>9.1%}{res.costs.sum()/yrs:>10.2%}"
              f"{res.stats.turnover_ann:>9.1f}")

    print("""
  如果三行差不多，说明日线粒度没丢信息，用日线就行；
  如果高频明显更好，说明日线在丢东西，值得把 4h/1h 全导过来。""")

    print("\n" + "=" * 140)
    print("拆开看：是信号变好了，还是只是交易更频繁了？")
    print("=" * 140)
    print("""
  上面每个频率的再平衡间隔都是 3 天。现在固定用【日线信号】，
  只改变再平衡频率，看纯粹的交易频率效应：
""")
    U1 = D.load_universe(syms, tfs=("1d",))
    px1 = X.price_panel(U1, "1d", "open")
    f1 = X.feature_panel(U1, lambda d: d["close"] / d["close"].shift(14) - 1)
    w1 = X.cross_sectional_weights(f1, n_side=N_SIDE, mode="long_short")
    print(f"  {'再平衡间隔':<12}{'Sharpe':>9}{'收益':>10}{'年化成本':>10}")
    for rb in (1, 2, 3, 5, 7, 14):
        res = X.run(px1, w1, COSTS, 365.0, "", rb)
        yrs = len(res.rets) / 365.0
        print(f"  每 {rb:>2} 天{'':<6}{res.stats.sharpe:>+9.2f}{res.stats.total_return:>+10.1%}"
              f"{res.costs.sum()/yrs:>10.2%}")

    print("\n" + "=" * 140)
    print("成本压力：高频对成本的敏感度高得多")
    print("=" * 140)
    print(f"\n  {'周期':<6}" + "".join(f"{c:>12}" for c in
          ("0bp", "3bp", "6.5bp", "12.5bp", "30bp")))
    for tf, bpd, lb, rb in GRID:
        U = D.load_universe(syms, tfs=(tf,))
        ann = D.bars_per_year(tf)
        px = X.price_panel(U, tf, "open")
        feat = X.feature_panel(U, lambda d, k=lb: d["close"] / d["close"].shift(k) - 1, tf=tf)
        w = X.cross_sectional_weights(feat, n_side=N_SIDE, mode="long_short")
        cells = []
        for fee, slip in ((0, 0), (2, 1), (4.5, 2), (4.5, 8), (15, 15)):
            r = X.run(px, w, B.Costs(fee, slip), ann, "", rb)
            cells.append(f"{r.stats.sharpe:>+12.2f}")
        print(f"  {tf:<6}" + "".join(cells))

    print("\n" + "=" * 140)
    print("结论")
    print("=" * 140)
    best = max(rows, key=lambda x: x["sharpe"])
    d1 = next(x for x in rows if x["tf"] == "1d")
    print(f"""
  最好的粒度是 {best['tf']}（Sharpe {best['sharpe']:+.2f}），日线是 {d1['sharpe']:+.2f}。
  差距 {best['sharpe']-d1['sharpe']:+.2f}。

  这是 26 个币、top/bottom-5 的结果，和主策略同一个口径，
  所以绝对水平和 REPORT_XSEC 的主结果可比，不只是频率之间的相对差。
""")


if __name__ == "__main__":
    main()
