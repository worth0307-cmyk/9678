"""BTCUSDT 4h RSI mean-reversion with pyramiding -- the strategy as specified.

Rules, sizing and the absence of a stop are all as instructed; see vibt/rsi_strat
for the implementation.  Nothing here is fitted: one parameter set, one run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vibt import data as D, rsi_strat as R  # noqa: E402

pd.set_option("display.width", 200)
REPORTS = Path(__file__).resolve().parent.parent / "reports"
EQ0 = 1000.0
BARS_PER_DAY = 6


def main() -> None:
    df = D.load("4h", symbol="BTCUSDT")
    p = R.RsiParams()
    res = R.run(df, p, EQ0)
    t = res.trades
    eq = res.equity.dropna()
    yrs = len(df) / (365 * BARS_PER_DAY)

    print("=" * 118)
    print("BTCUSDT 4h  RSI 均值回归 + 金字塔加仓")
    print("=" * 118)
    print(f"""
  数据    {df.index[0]} -> {df.index[-1]}   {len(df)} 根 4h K线   {yrs:.2f} 年
  本金    {EQ0:,.0f} USDT   无杠杆（满仓 = 50% + 5x10% = 100% 权益）
  成本    {p.fee*1e4:.1f}bp/边（4.5bp 手续费 + 2bp 滑点）
  信号    收盘确认，下一根开盘成交；加仓同理
  止损    无（按你的指定：一直持有到触及平仓线）
""")

    print("=" * 118)
    print("1. 总体")
    print("=" * 118)
    final = float(eq.iloc[-1])
    total = final / EQ0 - 1
    cagr = (final / EQ0) ** (1 / yrs) - 1
    dd = float((eq / eq.cummax() - 1).min())
    wins = t[t.pnl > 0]
    losses = t[t.pnl <= 0]
    print(f"""
  期末权益          {final:,.2f} USDT      （本金 {EQ0:,.0f}）
  总盈亏            {final - EQ0:+,.2f} USDT   = {total:+.2%}
  年化              {cagr:+.2%}
  最大回撤（盯市）  {dd:+.2%}

  开仓次数          {len(t)}   （多 {int((t.direction=='long').sum())} / 空 {int((t.direction=='short').sum())}）
  盈利 / 亏损       {len(wins)} / {len(losses)}     胜率 {len(wins)/len(t):.1%}
  平均持仓          {t.bars_held.mean():.0f} 根 4h = {t.bars_held.mean()/BARS_PER_DAY:.1f} 天
  持仓中位          {t.bars_held.median():.0f} 根 = {t.bars_held.median()/BARS_PER_DAY:.1f} 天
  最长 / 最短持仓   {t.bars_held.max()/BARS_PER_DAY:.1f} 天 / {t.bars_held.min()/BARS_PER_DAY:.1f} 天

  单笔最大盈利      {t.pnl.max():+,.2f} USDT  ({t.return_on_equity.max():+.2%} of equity)
  单笔最大亏损      {t.pnl.min():+,.2f} USDT  ({t.return_on_equity.min():+.2%} of equity)
  平均每笔          {t.pnl.mean():+,.2f} USDT
  盈亏比            {abs(wins.pnl.mean()/losses.pnl.mean()) if len(losses) else float('nan'):.2f}

  持仓期间最差浮亏  {t.worst_excursion.min():+.2%} of equity
    （无止损，所以这个数字才是真实承受的风险，不是上面的单笔最大亏损）
""")

    print("=" * 118)
    print("2. 逐年")
    print("=" * 118 + "\n")
    yr = eq.groupby(eq.index.year).agg(["first", "last"])
    yr["盈亏%"] = yr["last"] / yr["first"] - 1
    yr["盈亏USDT"] = yr["last"] - yr["first"]
    tr_yr = t.groupby(t.exit_time.dt.year).agg(平仓笔数=("pnl", "size"), 该年已实现=("pnl", "sum"))
    tab = yr[["first", "last", "盈亏USDT", "盈亏%"]].join(tr_yr).fillna(0)
    tab.columns = ["年初权益", "年末权益", "盈亏USDT", "盈亏%", "平仓笔数", "已实现盈亏"]
    print(tab.to_string(float_format=lambda v: f"{v:,.2f}"))

    print("\n" + "=" * 118)
    print("3. 逐笔明细")
    print("=" * 118 + "\n")
    show = t.copy()
    show["entry"] = show.entry_time.dt.strftime("%Y-%m-%d %H:%M")
    show["exit"] = show.exit_time.dt.strftime("%Y-%m-%d %H:%M")
    show["天数"] = (show.bars_held / BARS_PER_DAY).round(1)
    show["加仓层"] = show.legs - 1
    cols = ["direction", "entry", "exit", "天数", "entry_rsi", "exit_rsi",
            "加仓层", "notional", "pnl", "return_on_equity", "worst_excursion"]
    print(show[cols].to_string(index=False, float_format=lambda v: f"{v:,.2f}"))

    print("\n" + "=" * 118)
    print("4. 对照")
    print("=" * 118)
    bh = float(df["close"].iloc[-1] / df["open"].iloc[0] - 1)
    print(f"""
  同期 BTC 买入持有        {bh:+.1%}   （年化 {(1+bh)**(1/yrs)-1:+.1%}）
  本策略                   {total:+.1%}   （年化 {cagr:+.1%}）

  策略在场时间             {(t.bars_held.sum()/len(df)):.1%} 的 K 线
""")

    zero = R.run(df, R.RsiParams(fee=0.0), EQ0)
    print(f"  零成本下的期末权益       {zero.equity.dropna().iloc[-1]:,.2f} "
          f"({zero.equity.dropna().iloc[-1]/EQ0-1:+.2%})")
    print(f"  成本吃掉                 {zero.equity.dropna().iloc[-1] - final:,.2f} USDT")

    REPORTS.mkdir(exist_ok=True)
    t.to_csv(REPORTS / "rsi_trades.csv", index=False)
    res.equity.to_csv(REPORTS / "rsi_equity.csv")
    print(f"\n  wrote {REPORTS/'rsi_trades.csv'}")


if __name__ == "__main__":
    main()
