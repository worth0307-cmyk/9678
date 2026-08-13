"""Validation of the final system, and the variants that lost to it.

Includes the tests a backtest should have to pass before anyone risks money:
walk-forward, cost stress, parameter perturbation, randomised-signal control,
and a bootstrap confidence interval on the Sharpe.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vibt import backtest as B, data as D, indicators as I, metrics as M, system as SYS  # noqa: E402

pd.set_option("display.width", 220)

COSTS = B.Costs(fee_bps=4.5, slip_bps=2.0)
SPLIT = pd.Timestamp("2025-07-01")
REPORTS = Path(__file__).resolve().parent.parent / "reports"
ANN = 365.0


def show(name: str, df: pd.DataFrame, sig: pd.Series, bench: B.Result) -> B.Result:
    res = B.run(df, sig, COSTS, ANN, name, 1.0)
    r = res.rets
    is_s = M.compute(r[r.index < SPLIT], ANN)
    oos_s = M.compute(r[r.index >= SPLIT], ANN)
    print(f"  {name:<42} Sh {res.stats.sharpe:+5.2f} | ret {res.stats.total_return:+7.1%} | "
          f"CAGR {res.stats.cagr:+6.1%} | vol {res.stats.ann_vol:5.1%} | DD {res.stats.max_dd:6.1%} | "
          f"Cal {res.stats.calmar:+5.2f} | IS {is_s.sharpe:+5.2f} | OOS {oos_s.sharpe:+5.2f} "
          f"({oos_s.total_return:+6.1%}) | trd {res.stats.trades:3d}")
    return res


def main() -> None:
    frames = D.load_all()
    df = frames["1d"]
    bench = B.buy_and_hold(df, ANN, COSTS)
    p = SYS.Params()

    print("=" * 200)
    print("FINAL SYSTEM vs EVERYTHING IT BEAT")
    print("=" * 200)
    print(f"  {'buy & hold (benchmark)':<42} Sh {bench.stats.sharpe:+5.2f} | "
          f"ret {bench.stats.total_return:+7.1%} | CAGR {bench.stats.cagr:+6.1%} | "
          f"vol {bench.stats.ann_vol:5.1%} | DD {bench.stats.max_dd:6.1%} | "
          f"Cal {bench.stats.calmar:+5.2f} | "
          f"IS {M.compute(bench.rets[bench.rets.index < SPLIT], ANN).sharpe:+5.2f} | "
          f"OOS {M.compute(bench.rets[bench.rets.index >= SPLIT], ANN).sharpe:+5.2f} "
          f"({M.compute(bench.rets[bench.rets.index >= SPLIT], ANN).total_return:+6.1%})")
    print("-" * 200)

    final = show("FINAL: range-vote + vol target + short", df, SYS.target_exposure(df, p), bench)

    print("\n  --- ablation: remove one ingredient at a time ---")
    show("  no short book", df, SYS.target_exposure(df, SYS.Params(short_size=0.0)), bench)
    show("  no vol target (binary vote)", df,
         SYS.target_exposure(df, SYS.Params(target_vol=99.0, short_size=0.0)), bench)
    show("  vortex votes only", df, SYS.target_exposure(df, SYS.Params(use_dmi=False)), bench)
    show("  DMI votes only", df, SYS.target_exposure(df, SYS.Params(use_vortex=False)), bench)
    show("  single lookback (14) only", df, SYS.target_exposure(df, SYS.Params(lookbacks=(14,))), bench)
    show("  no exposure grid (step=0)", df, SYS.target_exposure(df, SYS.Params(exposure_step=0.0)), bench)

    print("\n" + "=" * 200)
    print("TEST 1 -- PARAMETER PERTURBATION.  A real system should degrade gently.")
    print("=" * 200)
    rows = []
    for tv in (0.30, 0.35, 0.40, 0.45, 0.50):
        for ss in (0.0, 0.25, 0.5):
            for vl in (20, 30, 45):
                pp = SYS.Params(target_vol=tv, short_size=ss, vol_lookback=vl)
                res = B.run(df, SYS.target_exposure(df, pp), COSTS, ANN, "", 1.0)
                r = res.rets
                rows.append({"target_vol": tv, "short": ss, "vol_lb": vl,
                             "sharpe": res.stats.sharpe, "ret": res.stats.total_return,
                             "maxdd": res.stats.max_dd, "calmar": res.stats.calmar,
                             "oos_ret": M.compute(r[r.index >= SPLIT], ANN).total_return})
    g = pd.DataFrame(rows)
    print(f"  {len(g)} perturbations of the three risk parameters:")
    print(f"    Sharpe   min {g['sharpe'].min():+.2f}  p25 {g['sharpe'].quantile(.25):+.2f}  "
          f"MED {g['sharpe'].median():+.2f}  p75 {g['sharpe'].quantile(.75):+.2f}  "
          f"max {g['sharpe'].max():+.2f}")
    print(f"    maxDD    min {g['maxdd'].min():+.1%}  MED {g['maxdd'].median():+.1%}  "
          f"max {g['maxdd'].max():+.1%}")
    print(f"    %beating B&H Sharpe ({bench.stats.sharpe:+.2f}): "
          f"{(g['sharpe'] > bench.stats.sharpe).mean():.0%}")
    print(f"    %beating B&H maxDD  ({bench.stats.max_dd:+.1%}): "
          f"{(g['maxdd'] > bench.stats.max_dd).mean():.0%}")
    g.to_csv(REPORTS / "final_perturbation.csv", index=False)

    print("\n" + "=" * 200)
    print("TEST 2 -- LOOKBACK-SET PERTURBATION.  Drop lookbacks from the vote at random.")
    print("=" * 200)
    rng = np.random.default_rng(11)
    pool = list(range(8, 61, 2))
    srs = []
    for _ in range(60):
        k = rng.integers(3, 9)
        lbs = tuple(sorted(rng.choice(pool, size=int(k), replace=False).tolist()))
        res = B.run(df, SYS.target_exposure(df, SYS.Params(lookbacks=lbs)), COSTS, ANN, "", 1.0)
        srs.append(res.stats.sharpe)
    srs = np.array(srs)
    print(f"  60 random lookback sets: Sharpe min {srs.min():+.2f}  p10 {np.percentile(srs, 10):+.2f}  "
          f"MED {np.median(srs):+.2f}  p90 {np.percentile(srs, 90):+.2f}  max {srs.max():+.2f}")
    print(f"  %beating B&H: {(srs > bench.stats.sharpe).mean():.0%}   "
          f"%positive: {(srs > 0).mean():.0%}")

    print("\n" + "=" * 200)
    print("TEST 3 -- RANDOM-SIGNAL CONTROL.  Shuffle the signal, keep the exposure profile.")
    print("=" * 200)
    sig = SYS.target_exposure(df, p)
    rng = np.random.default_rng(3)
    ctrl = []
    for _ in range(500):
        shift = int(rng.integers(20, len(sig) - 20))
        rolled = pd.Series(np.roll(sig.to_numpy(), shift), index=sig.index)
        ctrl.append(B.run(df, rolled, COSTS, ANN, "", 1.0).stats.sharpe)
    ctrl = np.array(ctrl)
    pct = (ctrl < final.stats.sharpe).mean()
    print(f"  500 circular-shifted copies of the same signal (same turnover, same exposure):")
    print(f"    control Sharpe  MED {np.median(ctrl):+.2f}   p95 {np.percentile(ctrl, 95):+.2f}")
    print(f"    real system     {final.stats.sharpe:+.2f}  ->  percentile {pct:.1%}")
    print("    (this asks: is the TIMING doing the work, or just the average exposure?)")

    print("\n" + "=" * 200)
    print("TEST 4 -- WALK-FORWARD.  No parameter is chosen with future data.")
    print("=" * 200)
    print("  The system has no fitted parameter to walk forward -- the vote spans the whole")
    print("  lookback grid by construction.  What we CAN walk forward is the risk target.")
    wf = pd.Series(0.0, index=df.index)
    train, test = 300, 90
    grid = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60]
    cache = {tv: B.run(df, SYS.target_exposure(df, SYS.Params(target_vol=tv)),
                       COSTS, ANN, "", 1.0).rets for tv in grid}
    sigs = {tv: SYS.target_exposure(df, SYS.Params(target_vol=tv)) for tv in grid}
    picks = []
    start = train
    while start < len(df) - 1:
        end = min(start + test, len(df) - 1)
        tr = df.index[start - train:start]
        best, best_sr = grid[0], -np.inf
        for tv in grid:
            seg = cache[tv].reindex(tr).dropna()
            if len(seg) < 30 or seg.std() == 0:
                continue
            sr = seg.mean() / seg.std() * np.sqrt(ANN)
            if sr > best_sr:
                best_sr, best = sr, tv
        idx = df.index[start:end]
        wf.loc[idx] = sigs[best].reindex(idx).fillna(0.0)
        picks.append((df.index[start].date(), best))
        start = end
    print("  picks: " + ", ".join(f"{d}:{tv:.0%}" for d, tv in picks))
    show("walk-forward risk target", df, wf, bench)

    print("\n" + "=" * 200)
    print("TEST 5 -- COST STRESS")
    print("=" * 200)
    for fee, slip, tag in ((0, 0, "frictionless (fantasy)"),
                           (2.0, 1.0, "VIP / limit orders"),
                           (4.5, 2.0, "retail taker (baseline)"),
                           (4.5, 8.0, "bad fills"),
                           (15.0, 15.0, "disaster")):
        res = B.run(df, sig, B.Costs(fee_bps=fee, slip_bps=slip), ANN, "", 1.0)
        print(f"  {tag:<26} ({fee:.1f}+{slip:.1f}bp)  Sharpe {res.stats.sharpe:+5.2f}  "
              f"ret {res.stats.total_return:+7.1%}")

    print("\n" + "=" * 200)
    print("TEST 6 -- HOW SURE ARE WE, REALLY?")
    print("=" * 200)
    p5, p50, p95 = M.block_bootstrap_sharpe(final.rets, ANN, block=20, n_boot=5000)
    print(f"  block-bootstrap Sharpe:  90% CI [{p5:+.2f}, {p95:+.2f}]   median {p50:+.2f}")
    print(f"  t-statistic of mean daily return: {final.stats.t_stat:+.2f}  "
          f"(need ~2.0 for 95% confidence)")
    n_trials_total = 224 * 3 + 131 + len(g) + 60
    dsr = M.deflated_sharpe(final.stats.sharpe, n_trials_total, len(final.rets), ANN)
    print(f"  configurations examined across this whole study: ~{n_trials_total}")
    print(f"  deflated Sharpe probability: {dsr:.1%}")
    print(f"  years of data: {len(df) / 365:.1f}.  Years needed to prove Sharpe "
          f"{final.stats.sharpe:.2f} at 95% confidence: ~{(2.0 / max(final.stats.sharpe, .01))**2:.1f}")

    print("\n" + "=" * 200)
    print("YEAR-BY-YEAR AND CURRENT STATE")
    print("=" * 200)
    eq = final.equity
    beq = bench.equity.reindex(eq.index).ffill()
    for year, grp in final.rets.groupby(final.rets.index.year):
        bg = bench.rets.reindex(grp.index)
        s = M.compute(grp, ANN)
        bs = M.compute(bg.dropna(), ANN)
        print(f"  {year}:  system {s.total_return:+7.1%} (DD {s.max_dd:6.1%})   "
              f"B&H {bs.total_return:+7.1%} (DD {bs.max_dd:6.1%})   "
              f"avg exposure {final.position.reindex(grp.index).mean():+.0%}")

    print("\n  monthly returns (%):")
    mt = M.monthly_table(final.rets) * 100
    print("   " + mt.round(1).to_string().replace("\n", "\n   "))

    print("\n  current system state (last 10 daily bars):")
    print("   " + SYS.explain(df, p, 10).to_string().replace("\n", "\n   "))

    out = pd.DataFrame({
        "equity_system": final.equity, "equity_bh": beq,
        "position": final.position, "close": df["close"].reindex(final.equity.index),
    })
    out.to_csv(REPORTS / "final_equity.csv")
    print(f"\n  wrote {REPORTS / 'final_equity.csv'}")


if __name__ == "__main__":
    main()
