"""Does cross-sectional momentum survive the tests that killed everything else?

The pattern that kills a finding in this repo is always the same: the winning
parameter turns out to be an isolated spike, or the result rests on one symbol,
or it lives in one sub-period.  Same gauntlet here.
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
YEARS = [("2023", "2023-01-01", "2024-01-01"), ("2024", "2024-01-01", "2025-01-01"),
         ("2025", "2025-01-01", "2026-01-01"), ("2026", "2026-01-01", "2027-01-01")]


def mom_feature(universe, n):
    return X.feature_panel(universe, lambda d: d["close"] / d["close"].shift(n) - 1)


def vi_feature(universe, n):
    return X.feature_panel(universe, lambda d: I.vortex(d, n)["vi_spread"])


def sharpe_of(universe, px, feat, n_side=2, rebalance=1, costs=COSTS):
    w = X.cross_sectional_weights(feat, n_side=n_side, mode="long_short")
    return X.run(px, w, costs, ANN, "", rebalance)


def main() -> None:
    symbols = D.available_symbols(require=("1d",))
    universe = D.load_universe(symbols, tfs=("1d",))
    px = X.price_panel(universe, "1d", "open")

    print("=" * 180)
    print("TEST 1 -- LOOKBACK SURFACE.  A spike is curve-fitting; a plateau is a finding.")
    print("=" * 180)
    grid = [2, 3, 5, 7, 10, 12, 14, 16, 18, 21, 25, 30, 40, 50, 60, 90]
    rows = []
    print(f"  {'lookback':<10}{'momentum Sharpe':>18}{'VI-spread Sharpe':>20}")
    for n in grid:
        m = sharpe_of(universe, px, mom_feature(universe, n))
        v = sharpe_of(universe, px, vi_feature(universe, n))
        rows.append({"lookback": n, "mom": m.stats.sharpe, "vi": v.stats.sharpe})
        print(f"  {n:<10}{m.stats.sharpe:>+18.2f}{v.stats.sharpe:>+20.2f}")
    g = pd.DataFrame(rows)
    for col, nm in (("mom", "momentum"), ("vi", "VI spread")):
        pos = (g[col] > 0).mean()
        print(f"\n  {nm}: median Sharpe {g[col].median():+.2f}   "
              f"share of lookbacks positive {pos:.0%}   "
              f"best {g[col].max():+.2f} at n={int(g.loc[g[col].idxmax(),'lookback'])}")

    print("\n" + "=" * 180)
    print("TEST 2 -- ARE THE TWO SIGNALS THE SAME THING?")
    print("=" * 180)
    for n in (7, 14, 30):
        a = mom_feature(universe, n).rank(axis=1)
        b = vi_feature(universe, n).rank(axis=1)
        common = a.index.intersection(b.index)
        cors = [a.loc[t].corr(b.loc[t]) for t in common
                if a.loc[t].notna().sum() >= 3 and b.loc[t].notna().sum() >= 3]
        print(f"  n={n:<3} mean cross-sectional rank correlation between "
              f"{n}d momentum and VI({n}) spread: {np.nanmean(cors):+.3f}")
    print("  (high correlation = one finding, not two)")

    print("\n" + "=" * 180)
    print("TEST 3 -- DOES IT DEPEND ON THE LATE-LISTING NAMES?")
    print("=" * 180)
    subsets = {
        "all 6": symbols,
        "drop HYPE": [s for s in symbols if s != "HYPEUSDT"],
        "drop HYPE+TAO": [s for s in symbols if s not in ("HYPEUSDT", "TAOUSDT")],
        "BTC/ETH/SOL/BNB only": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"],
    }
    for label, subs in subsets.items():
        u = {s: universe[s] for s in subs if s in universe}
        p = X.price_panel(u, "1d", "open")
        m14 = sharpe_of(u, p, mom_feature(u, 14), n_side=min(2, len(subs) // 2))
        v14 = sharpe_of(u, p, vi_feature(u, 14), n_side=min(2, len(subs) // 2))
        print(f"  {label:<24} n={len(subs)}   momentum14 Sharpe {m14.stats.sharpe:+5.2f}   "
              f"VI(14) Sharpe {v14.stats.sharpe:+5.2f}   "
              f"ret {m14.stats.total_return:+7.1%} / {v14.stats.total_return:+7.1%}")

    print("\n" + "=" * 180)
    print("TEST 4 -- SUB-PERIODS.  This is where most of this repo's findings died.")
    print("=" * 180)
    for label, feat_fn in (("momentum 14d", lambda: mom_feature(universe, 14)),
                           ("VI(14) spread", lambda: vi_feature(universe, 14))):
        res = sharpe_of(universe, px, feat_fn())
        r = res.rets
        cells = []
        for yr, lo, hi in YEARS:
            seg = r[(r.index >= lo) & (r.index < hi)]
            if len(seg) > 60:
                st = M.compute(seg, ANN)
                cells.append(f"{yr} Sh {st.sharpe:+5.2f} ret {st.total_return:+7.1%}")
        print(f"  {label:<16} " + " | ".join(cells))

    print("\n" + "=" * 180)
    print("TEST 5 -- PORTFOLIO CONSTRUCTION AND COSTS")
    print("=" * 180)
    for n_side in (1, 2, 3):
        r = sharpe_of(universe, px, mom_feature(universe, 14), n_side=n_side)
        print(f"  long/short top-{n_side} each side   Sharpe {r.stats.sharpe:+5.2f}   "
              f"ret {r.stats.total_return:+7.1%}   turnover {r.stats.turnover_ann:.0f}x/yr")
    print()
    for rb in (1, 2, 3, 5, 10):
        r = sharpe_of(universe, px, mom_feature(universe, 14), rebalance=rb)
        print(f"  rebalance every {rb:>2}d            Sharpe {r.stats.sharpe:+5.2f}   "
              f"ret {r.stats.total_return:+7.1%}")
    print()
    for fee, slip, tag in ((0, 0, "frictionless"), (2, 1, "maker/VIP"),
                           (4.5, 2, "retail taker"), (4.5, 8, "bad fills"), (15, 15, "disaster")):
        r = sharpe_of(universe, px, mom_feature(universe, 14), costs=B.Costs(fee, slip))
        print(f"  {tag:<14} ({fee}+{slip}bp)   Sharpe {r.stats.sharpe:+5.2f}   "
              f"ret {r.stats.total_return:+7.1%}")

    print("\n" + "=" * 180)
    print("TEST 6 -- ROLLING 180-DAY SHARPE.  Is it always working, or was it one stretch?")
    print("=" * 180)
    for label, feat in (("momentum 14d", mom_feature(universe, 14)),
                        ("VI(14) spread", vi_feature(universe, 14))):
        r = sharpe_of(universe, px, feat).rets
        roll = r.rolling(180).mean() / r.rolling(180).std() * np.sqrt(ANN)
        roll = roll.dropna()
        print(f"  {label:<16} share of rolling windows positive {(roll > 0).mean():.0%}   "
              f"min {roll.min():+.2f}   median {roll.median():+.2f}   max {roll.max():+.2f}")

    print("\n" + "=" * 180)
    print("TEST 7 -- EQUAL-WEIGHT COMBINATION of the two ranking signals")
    print("=" * 180)
    combo = (mom_feature(universe, 14).rank(axis=1) + vi_feature(universe, 14).rank(axis=1)) / 2
    for n_side in (1, 2):
        r = sharpe_of(universe, px, combo, n_side=n_side)
        p5, _, p95 = M.block_bootstrap_sharpe(r.rets, ANN, block=20, n_boot=4000)
        beta, alpha = X.beta_to(r.rets, px.pct_change().shift(-1).mean(axis=1).iloc[:-1])
        print(f"  combined rank, top-{n_side}   Sharpe {r.stats.sharpe:+5.2f} "
              f"[90% CI {p5:+.2f},{p95:+.2f}]   ret {r.stats.total_return:+7.1%}   "
              f"vol {r.stats.ann_vol:.1%}   DD {r.stats.max_dd:6.1%}   "
              f"beta {beta:+.2f}   alpha {alpha:+.1%}")


if __name__ == "__main__":
    main()
