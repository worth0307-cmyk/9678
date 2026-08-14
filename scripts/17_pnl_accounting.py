"""Exactly what the cross-sectional backtest does, and exactly what it earned.

Written to be auditable rather than impressive: every number below can be traced
to a line of vibt/xsec.py, and the worked example prints the actual positions on
a real date so the arithmetic can be checked by hand.
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


def main() -> None:
    syms = D.available_symbols(require=("1d",))
    U = D.load_universe(syms, tfs=("1d",))
    px = X.price_panel(U, "1d", "open")
    feat = X.feature_panel(U, lambda d: d["close"] / d["close"].shift(LOOKBACK) - 1)
    w = X.cross_sectional_weights(feat, n_side=N_SIDE, mode="long_short")
    res = X.run(px, w, COSTS, ANN, "", REBAL)

    print("=" * 130)
    print("THE RULE, IN FULL")
    print("=" * 130)
    print(f"""
  universe   : {len(syms)} Binance USDT-M perpetuals, daily bars from 2023-01-01
  signal     : {LOOKBACK}-day return, close[t] / close[t-{LOOKBACK}] - 1, per symbol
  ranking    : every {REBAL}rd day, rank all symbols that have a price AND a valid
               signal that day.  Symbols not yet listed simply are not ranked.
  positions  : LONG  the top {N_SIDE}  at +{0.5/N_SIDE:.0%} of equity each
               SHORT the bottom {N_SIDE} at -{0.5/N_SIDE:.0%} of equity each
               -> 50% long + 50% short = 100% gross, 0% net.  No leverage beyond 1x gross.
  execution  : the signal uses data through the CLOSE of day t;
               the position is entered at the OPEN of day t+1;
               the return earned is open-to-open.  The engine applies that shift
               itself (xsec.run: held = w.shift(1)), so no signal can peek.
  costs      : {COSTS.fee_bps}bp fee + {COSTS.slip_bps}bp slippage = {COSTS.per_side*1e4:.1f}bp per side,
               charged on |change in weight| every day a weight changes.
  holding    : weights are held for {REBAL} days, then re-ranked.
""")

    print("=" * 130)
    print("WHAT IT EARNED")
    print("=" * 130)
    r = res.rets
    years = len(r) / ANN
    print(f"  period            : {r.index[0].date()} to {r.index[-1].date()}  ({years:.2f} years)")
    print(f"  total return      : {res.stats.total_return:+.1%}")
    print(f"  CAGR              : {res.stats.cagr:+.1%}   <- the 'annual profit' number")
    print(f"  annualised vol    : {res.stats.ann_vol:.1%}")
    print(f"  max drawdown      : {res.stats.max_dd:.1%}")
    print(f"  Sharpe            : {res.stats.sharpe:+.2f}")
    print(f"  Calmar            : {res.stats.calmar:+.2f}")

    print("\n  calendar-year returns (what you would actually have booked):")
    for y in (2023, 2024, 2025, 2026):
        seg = r[r.index.year == y]
        if len(seg) < 30:
            continue
        eq = float(np.prod(1 + seg) - 1)
        dd = M.compute(seg, ANN).max_dd
        tag = "  (partial year)" if y == 2026 else ""
        print(f"    {y}: {eq:+7.1%}   最大回撤 {dd:6.1%}{tag}")

    print("\n  on a 100,000 USDT account, compounding:")
    eq = 100_000 * float(np.prod(1 + r))
    print(f"    ending equity {eq:,.0f} USDT   profit {eq-100_000:+,.0f} USDT over {years:.1f} years")
    for y in (2023, 2024, 2025, 2026):
        seg = r[r.index.year == y]
        if len(seg) < 30:
            continue
        print(f"    {y} P&L on a constant 100k base: {100_000*float(np.prod(1+seg)-1):+,.0f} USDT")

    print("\n" + "=" * 130)
    print("WHERE THE COSTS GO")
    print("=" * 130)
    turn = (res.position.diff().abs()).sum()
    print(f"  total cost paid          : {res.costs.sum():.1%} of equity over {years:.1f} years")
    print(f"  annual cost drag         : {res.costs.sum()/years:.2%} per year")
    print(f"  gross return before cost : {float(np.prod(1+res.gross)-1):+.1%}")
    print(f"  net return after cost    : {res.stats.total_return:+.1%}")
    print(f"  rebalances               : {int(len(r)/REBAL)} over the period "
          f"(~{ANN/REBAL:.0f} per year)")

    print("\n" + "=" * 130)
    print("A WORKED EXAMPLE -- check the arithmetic by hand")
    print("=" * 130)
    # Use the weights the ENGINE held, not a hand-rebuilt copy: with rebalance>1
    # the engine resamples the weight panel before shifting it, so w.shift(1)
    # is a different portfolio and the arithmetic will not reconcile.
    held = res.weights
    active = held[(held != 0).sum(axis=1) == 2 * N_SIDE]
    day = active.index[len(active) // 2]
    pos = held.loc[day]
    nxt = px.index[px.index.get_loc(day) + 1]
    ret_day = (px.loc[nxt] / px.loc[day] - 1)
    print(f"\n  date {day.date()}  (positions entered at this day's open, "
          f"held to {nxt.date()}'s open)\n")
    print(f"    {'symbol':<15}{'weight':>9}{'signal 14d':>13}{'open today':>14}"
          f"{'open next':>14}{'move':>9}{'P&L':>10}")
    sig_day = feat.loc[day] if day in feat.index else pd.Series(dtype=float)
    for s in pos[pos != 0].sort_values(ascending=False).index:
        print(f"    {s:<15}{pos[s]:>+9.2%}{sig_day.get(s, np.nan):>12.1%}"
              f"{px.loc[day, s]:>14.4f}{px.loc[nxt, s]:>14.4f}"
              f"{ret_day[s]:>+9.2%}{pos[s]*ret_day[s]:>+10.3%}")
    gross_day = float((pos * ret_day.fillna(0)).sum())
    cost_day = float(res.costs.get(day, 0.0))
    net_day = float(res.rets.get(day, np.nan))
    print(f"\n    hand-computed gross = {gross_day:+.3%}")
    print(f"    engine gross        = {float(res.gross.get(day, np.nan)):+.3%}")
    print(f"    cost that day       = {cost_day:.3%}")
    print(f"    engine net          = {net_day:+.3%}")
    ok = abs(gross_day - cost_day - net_day) < 1e-9
    print(f"    reconciles: {ok}   (gross - cost - net = "
          f"{gross_day - cost_day - net_day:+.2e})")
    assert ok, "worked example does not reconcile with the equity curve"

    print("\n" + "=" * 130)
    print("WHAT THE BACKTEST DOES *NOT* CHARGE YOU FOR")
    print("=" * 130)
    print("""  1. Perpetual funding.  Set to zero.  A dollar-neutral book pays funding on the
     longs and receives it on the shorts, so it largely cancels -- but not exactly,
     and in a strong bull market the crowded longs pay more.  This is the single
     largest un-modelled cost.
  2. Market impact beyond a flat 2bp.  Fine for BTC, optimistic for a 19% position
     in BONK or PENDLE at size.
  3. Liquidation / margin calls.  The book is 1x gross, so this is mostly moot,
     but exchange margin rules on 26 simultaneous positions are not modelled.
  4. Survivorship.  Every symbol here is one that still traded in 2026.  Coins that
     delisted or died in 2023-2024 are absent, which flatters a long-the-winners
     strategy.  See UNIVERSE.md.
  5. Borrow availability, API downtime, partial fills, exchange outages.""")

    print("\n" + "=" * 130)
    print("HOW MUCH OF THIS SHOULD YOU BELIEVE")
    print("=" * 130)
    p5, p50, p95 = M.block_bootstrap_sharpe(r, ANN, block=20, n_boot=5000)
    print(f"  bootstrap Sharpe 90% CI  : [{p5:+.2f}, {p95:+.2f}]   (median {p50:+.2f})")
    print(f"  t-statistic              : {res.stats.t_stat:+.2f}")
    print(f"  deflated Sharpe (~120 cfg): "
          f"{M.deflated_sharpe(res.stats.sharpe, 120, len(px), ANN):.1%}")
    lo = (1 + p5 * res.stats.ann_vol) ** 1 - 1
    hi = (1 + p95 * res.stats.ann_vol) ** 1 - 1
    print(f"\n  Translating the Sharpe CI into an annual-return range at {res.stats.ann_vol:.0%} vol:")
    print(f"    pessimistic {p5 * res.stats.ann_vol:+.1%}/yr   ...   "
          f"optimistic {p95 * res.stats.ann_vol:+.1%}/yr")
    print(f"    point estimate {res.stats.cagr:+.1%}/yr")
    print("\n  That range is the honest answer to 'what is the annual profit'.")


if __name__ == "__main__":
    main()
