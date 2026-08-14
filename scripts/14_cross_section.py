"""Cross-sectional research: the one direction this dataset unlocks and I never tested.

Time-series signals on a single asset spend their life fighting crypto beta.  A
dollar-neutral cross-sectional book never takes that bet at all.  With six names
the cross-section is small, so treat everything here as a first pass -- but the
question "do relative moves persist or revert?" is answerable and orthogonal to
everything in reports 1 and 2.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vibt import backtest as B, data as D, indicators as I, metrics as M, xsec as X  # noqa: E402

pd.set_option("display.width", 220)

COSTS = B.Costs(fee_bps=4.5, slip_bps=2.0)
ANN = 365.0
REPORTS = Path(__file__).resolve().parent.parent / "reports"


def show(name: str, res: B.Result, bench: pd.Series, rows: list) -> None:
    beta, alpha = X.beta_to(res.rets, bench)
    p5, _, p95 = M.block_bootstrap_sharpe(res.rets, ANN, block=20, n_boot=3000)
    rows.append({"name": name, "sharpe": res.stats.sharpe, "ret": res.stats.total_return,
                 "cagr": res.stats.cagr, "vol": res.stats.ann_vol, "maxdd": res.stats.max_dd,
                 "beta": beta, "alpha": alpha, "t": res.stats.t_stat,
                 "boot_p5": p5, "boot_p95": p95})
    print(f"  {name:<40} Sh {res.stats.sharpe:+5.2f} [{p5:+.2f},{p95:+.2f}] | "
          f"ret {res.stats.total_return:+8.1%} | vol {res.stats.ann_vol:5.1%} | "
          f"DD {res.stats.max_dd:6.1%} | beta {beta:+5.2f} | alpha {alpha:+6.1%} | "
          f"t {res.stats.t_stat:+5.2f}")


def main() -> None:
    symbols = D.available_symbols(require=("1d",))
    universe = D.load_universe(symbols, tfs=("1d",))
    px = X.price_panel(universe, "1d", "open")
    print("=" * 190)
    print(f"CROSS-SECTIONAL STUDY   {symbols}   {px.index[0].date()} .. {px.index[-1].date()}")
    print("=" * 190)
    avail = px.notna().sum()
    print("  bars available per symbol: " + "  ".join(f"{s}={int(avail[s])}" for s in symbols))

    eq_bench = px.pct_change().shift(-1).mean(axis=1).iloc[:-1]     # equal-weight crypto
    bstat = M.compute(eq_bench.dropna(), ANN)
    print(f"\n  benchmark: equal-weight long-only basket   Sharpe {bstat.sharpe:+.2f}  "
          f"ret {bstat.total_return:+.1%}  vol {bstat.ann_vol:.1%}  maxDD {bstat.max_dd:.1%}")
    print("  every strategy below is measured for beta against THAT basket.\n")

    rows: list[dict] = []

    print("=" * 190)
    print("1. CROSS-SECTIONAL MOMENTUM vs REVERSAL -- do relative moves persist or revert?")
    print("=" * 190)
    print("  long the strongest 2, short the weakest 2, dollar-neutral, daily rebalance\n")
    for lb in (1, 3, 7, 14, 30, 60, 90):
        feat = X.feature_panel(universe, lambda d, n=lb: d["close"] / d["close"].shift(n) - 1)
        w = X.cross_sectional_weights(feat, n_side=2, mode="long_short")
        show(f"momentum {lb}d (long winners)", X.run(px, w, COSTS, ANN, "", 1), eq_bench, rows)
    print()
    for lb in (1, 3, 7, 14):
        feat = X.feature_panel(universe, lambda d, n=lb: d["close"] / d["close"].shift(n) - 1)
        w = X.cross_sectional_weights(-feat, n_side=2, mode="long_short")
        show(f"reversal {lb}d (long losers)", X.run(px, w, COSTS, ANN, "", 1), eq_bench, rows)

    print("\n" + "=" * 190)
    print("2. THE SAME TEST USING YOUR INDICATOR -- rank by VI spread instead of by return")
    print("=" * 190)
    for n in (7, 14, 30):
        feat = X.feature_panel(universe, lambda d, k=n: I.vortex(d, k)["vi_spread"])
        for mode, tag in (("long_short", "long high VI"), ("long_short_rev", "long low VI")):
            f = feat if mode == "long_short" else -feat
            w = X.cross_sectional_weights(f, n_side=2, mode="long_short")
            show(f"VI({n}) spread, {tag}", X.run(px, w, COSTS, ANN, "", 1), eq_bench, rows)

    print("\n" + "=" * 190)
    print("3. LOW-VOLATILITY / DISPERSION")
    print("=" * 190)
    for n in (14, 30, 60):
        feat = X.feature_panel(universe, lambda d, k=n: I.parkinson_vol(d, k, ANN))
        w = X.cross_sectional_weights(-feat, n_side=2, mode="long_short")
        show(f"long low-vol / short high-vol ({n}d)", X.run(px, w, COSTS, ANN, "", 1), eq_bench, rows)

    print("\n" + "=" * 190)
    print("4. REBALANCE FREQUENCY -- costs vs signal decay, on the best momentum lookback")
    print("=" * 190)
    feat = X.feature_panel(universe, lambda d: d["close"] / d["close"].shift(30) - 1)
    for rb in (1, 3, 7, 14):
        w = X.cross_sectional_weights(feat, n_side=2, mode="long_short")
        show(f"momentum 30d, rebalance every {rb}d",
             X.run(px, w, COSTS, ANN, "", rb), eq_bench, rows)

    print("\n" + "=" * 190)
    print("5. LONG-ONLY ROTATION -- keep the beta, just pick better names")
    print("=" * 190)
    for lb in (14, 30, 60, 90):
        feat = X.feature_panel(universe, lambda d, n=lb: d["close"] / d["close"].shift(n) - 1)
        w = X.cross_sectional_weights(feat, n_side=2, mode="long_only")
        show(f"long-only top-2 by {lb}d momentum",
             X.run(px, w, COSTS, ANN, "", 7), eq_bench, rows)

    print("\n" + "=" * 190)
    print("6. CONTINUOUS RANK WEIGHTS -- use all six names instead of a top/bottom cut")
    print("=" * 190)
    for lb in (14, 30, 60):
        feat = X.feature_panel(universe, lambda d, n=lb: d["close"] / d["close"].shift(n) - 1)
        w = X.cross_sectional_weights(feat, mode="rank")
        show(f"rank-weighted momentum {lb}d", X.run(px, w, COSTS, ANN, "", 3), eq_bench, rows)

    out = pd.DataFrame(rows)
    out.to_csv(REPORTS / "cross_section.csv", index=False)

    print("\n" + "=" * 190)
    print("SUMMARY -- sorted by Sharpe, with the honesty columns attached")
    print("=" * 190)
    top = out.sort_values("sharpe", ascending=False).head(12)
    print(top[["name", "sharpe", "ret", "vol", "maxdd", "beta", "alpha", "t",
               "boot_p5", "boot_p95"]].round(3).to_string(index=False))
    n_trials = len(out)
    best = out.iloc[int(out["sharpe"].idxmax())]
    dsr = M.deflated_sharpe(best["sharpe"], n_trials, len(px), ANN)
    print(f"\n  configurations tested here: {n_trials}   best: {best['name']} "
          f"(Sharpe {best['sharpe']:+.2f})")
    print(f"  deflated Sharpe probability: {dsr:.1%}")
    print(f"  how many have a bootstrap CI entirely above zero: "
          f"{int((out['boot_p5'] > 0).sum())} / {n_trials}")


if __name__ == "__main__":
    main()
