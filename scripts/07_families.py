"""Fair fight: sweep EVERY filter family over its own grid and compare distributions.

Comparing a hand-picked VI(14) against one arbitrary EMA(50,200) is how people
fool themselves.  What matters is whether the VI family as a whole beats the
moving-average family as a whole -- the median cell, not the best cell.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vibt import backtest as B, data as D, indicators as I, metrics as M, strategies as S  # noqa: E402

pd.set_option("display.width", 220)

COSTS = B.Costs(fee_bps=4.5, slip_bps=2.0)
SPLIT = pd.Timestamp("2025-07-01")
REPORTS = Path(__file__).resolve().parent.parent / "reports"


def family_signals(df: pd.DataFrame) -> dict[str, list[tuple[str, pd.Series]]]:
    c = df["close"]
    fam: dict[str, list[tuple[str, pd.Series]]] = {}

    fam["VI"] = [(f"VI({n})", (I.vortex(df, n)["vi_spread"] > 0).astype(float))
                 for n in range(8, 61, 2)]

    fam["SMA"] = [(f"C>SMA({n})", (c > I.sma(c, n)).astype(float))
                  for n in range(20, 221, 8)]

    fam["EMAx"] = [(f"EMA({f},{s})", (I.ema(c, f) > I.ema(c, s)).astype(float))
                   for f, s in itertools.product((10, 20, 30, 50), (60, 100, 150, 200))
                   if f < s]

    fam["Donchian"] = [(f"DC({n})mid", (c > I.donchian(df, n)["dc_mid"]).astype(float))
                       for n in range(20, 121, 5)]

    fam["ROC"] = [(f"ROC({n})>0", (I.roc(c, n) > 0).astype(float))
                  for n in range(10, 131, 5)]

    fam["ADX+DI"] = []
    for n in range(10, 41, 2):
        up = I.high_low_di(df, n) if hasattr(I, "high_low_di") else None
        if up is None:
            pdm = df["high"].diff().clip(lower=0)
            mdm = (-df["low"].diff()).clip(lower=0)
            f = (pdm.rolling(n).sum() > mdm.rolling(n).sum()).astype(float)
            fam["ADX+DI"].append((f"DI({n})", f))
    return fam


def main() -> None:
    frames = D.load_all()
    df = frames["1d"]
    ann = D.bars_per_year("1d")
    bench = B.buy_and_hold(df, ann, COSTS)
    pv = I.parkinson_vol(df, 30, ann)
    bench_oos = M.compute(bench.rets[bench.rets.index >= SPLIT], ann)

    print("=" * 190)
    print("FAMILY COMPARISON -- daily, long-only, costs on.  Benchmark buy & hold: "
          f"Sharpe {bench.stats.sharpe:+.2f}, ret {bench.stats.total_return:+.1%}, "
          f"maxDD {bench.stats.max_dd:.1%}")
    print("=" * 190)

    fam = family_signals(df)
    all_rows = []
    for name, members in fam.items():
        rows = []
        for label, sig in members:
            res = B.run(df, sig, COSTS, ann, label, 1.0)
            r = res.rets
            rows.append({
                "family": name, "label": label, "sharpe": res.stats.sharpe,
                "ret": res.stats.total_return, "maxdd": res.stats.max_dd,
                "calmar": res.stats.calmar, "expo": res.stats.exposure,
                "trades": res.stats.trades,
                "sharpe_is": M.compute(r[r.index < SPLIT], ann).sharpe,
                "sharpe_oos": M.compute(r[r.index >= SPLIT], ann).sharpe,
                "ret_oos": M.compute(r[r.index >= SPLIT], ann).total_return,
            })
        g = pd.DataFrame(rows)
        all_rows.append(g)
        print(f"\n  {name:<10} cells={len(g):3d}  "
              f"Sharpe  min {g['sharpe'].min():+.2f} / p25 {g['sharpe'].quantile(.25):+.2f} / "
              f"MED {g['sharpe'].median():+.2f} / p75 {g['sharpe'].quantile(.75):+.2f} / "
              f"max {g['sharpe'].max():+.2f}")
        print(f"  {'':<10} maxDD median {g['maxdd'].median():+.1%}   "
              f"Calmar median {g['calmar'].median():+.2f}   "
              f"exposure median {g['expo'].median():.0%}   "
              f"%cells beating B&H Sharpe: {(g['sharpe'] > bench.stats.sharpe).mean():.0%}   "
              f"%cells beating B&H in the OOS bear: "
              f"{(g['ret_oos'] > bench_oos.total_return).mean():.0%}")
        best = g.loc[g["sharpe"].idxmax()]
        print(f"  {'':<10} best cell {best['label']} Sharpe {best['sharpe']:+.2f} "
              f"(IS {best['sharpe_is']:+.2f} / OOS {best['sharpe_oos']:+.2f})"
              f"   <- the number a curve-fitter would quote")

    full = pd.concat(all_rows)
    full.to_csv(REPORTS / "families.csv", index=False)

    n_trials = len(full)
    full = full.reset_index(drop=True)
    best_overall = full.iloc[int(full["sharpe"].idxmax())]
    dsr = M.deflated_sharpe(best_overall["sharpe"], n_trials, len(df), ann)
    print("\n" + "=" * 190)
    print("MULTIPLE-TESTING CORRECTION")
    print("=" * 190)
    print(f"  configurations tested in this script alone: {n_trials}")
    print(f"  best Sharpe found: {best_overall['sharpe']:+.2f} ({best_overall['label']})")
    print(f"  deflated Sharpe probability (prob. it beats the best of {n_trials} random "
          f"strategies): {dsr:.1%}")
    print("  reference: you want this above ~95%.  Below ~50% the 'winner' is indistinguishable "
          "from the luckiest coin in the bag.")

    print("\n" + "=" * 190)
    print("SAME TEST, VI FAMILY, WITH THE VOL-TARGET OVERLAY ADDED")
    print("=" * 190)
    rows = []
    for n in range(8, 61, 2):
        base = (I.vortex(df, n)["vi_spread"] > 0).astype(float)
        sig = S.discretise(base * (0.40 / pv).clip(0, 1.0), 0.2)
        res = B.run(df, sig, COSTS, ann, f"VI({n})+vt", 1.0)
        r = res.rets
        rows.append({"n": n, "sharpe": res.stats.sharpe, "ret": res.stats.total_return,
                     "maxdd": res.stats.max_dd, "calmar": res.stats.calmar,
                     "sharpe_oos": M.compute(r[r.index >= SPLIT], ann).sharpe,
                     "ret_oos": M.compute(r[r.index >= SPLIT], ann).total_return})
    g = pd.DataFrame(rows)
    print(f"  Sharpe   MED {g['sharpe'].median():+.2f}  (raw VI family median was "
          f"{all_rows[0]['sharpe'].median():+.2f})")
    print(f"  maxDD    MED {g['maxdd'].median():+.1%}  (raw VI family median "
          f"{all_rows[0]['maxdd'].median():+.1%})")
    print(f"  Calmar   MED {g['calmar'].median():+.2f}  (raw VI family median "
          f"{all_rows[0]['calmar'].median():+.2f})")
    print(f"  OOS ret  MED {g['ret_oos'].median():+.1%}  vs buy & hold OOS "
          f"{M.compute(bench.rets[bench.rets.index >= SPLIT], ann).total_return:+.1%}")
    print(f"  %cells beating B&H Sharpe: {(g['sharpe'] > bench.stats.sharpe).mean():.0%}")
    print(f"  %cells beating B&H in the OOS bear: "
          f"{(g['ret_oos'] > bench_oos.total_return).mean():.0%}")
    print("\n  per-lookback detail:")
    print("   " + g.round(3).to_string(index=False).replace("\n", "\n   "))


if __name__ == "__main__":
    main()
