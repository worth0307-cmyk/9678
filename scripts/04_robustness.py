"""The test that matters: does the parameter surface hold up out of sample?

A strategy whose in-sample Sharpe tells you nothing about its out-of-sample
Sharpe is a curve-fit, no matter how good the best cell looks.  We measure the
rank correlation between the two across the whole parameter grid.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vibt import backtest as B, data as D, metrics as M, strategies as S  # noqa: E402

pd.set_option("display.width", 220)

COSTS = B.Costs(fee_bps=4.5, slip_bps=2.0)
SPLIT = pd.Timestamp("2025-07-01")          # OOS holds the Oct-25 top and the 2026 bear
REPORTS = Path(__file__).resolve().parent.parent / "reports"


def segment_stats(res: B.Result, ann: float) -> dict:
    r = res.rets
    is_r, oos_r = r[r.index < SPLIT], r[r.index >= SPLIT]
    out = {"sharpe_full": res.stats.sharpe, "ret_full": res.stats.total_return,
           "dd_full": res.stats.max_dd, "trades": res.stats.trades,
           "expo": res.stats.exposure}
    for tag, seg in (("is", is_r), ("oos", oos_r)):
        if len(seg) > 30:
            st = M.compute(seg, ann)
            out[f"sharpe_{tag}"] = st.sharpe
            out[f"ret_{tag}"] = st.total_return
            out[f"dd_{tag}"] = st.max_dd
        else:
            out[f"sharpe_{tag}"] = np.nan
            out[f"ret_{tag}"] = np.nan
            out[f"dd_{tag}"] = np.nan
    return out


def sweep_vi(frames: dict, tf: str) -> pd.DataFrame:
    df = frames[tf]
    ann = D.bars_per_year(tf)
    bpd = ann / 365.0
    rows = []
    for n in range(6, 61, 2):
        for mode in ("ls", "long_only"):
            for band in (0.0, 0.02, 0.04, 0.06):
                sig = (S.vi_cross(df, n, mode) if band == 0
                       else S.vi_cross_band(df, n, band, mode))
                res = B.run(df, sig, COSTS, ann, f"VI{n}_{mode}_b{band}", bpd)
                rows.append({"tf": tf, "n": n, "mode": mode, "band": band,
                             **segment_stats(res, ann)})
    return pd.DataFrame(rows)


def stability_report(g: pd.DataFrame, label: str) -> None:
    m = g.dropna(subset=["sharpe_is", "sharpe_oos"])
    if len(m) < 5:
        return
    rho, p = sps.spearmanr(m["sharpe_is"], m["sharpe_oos"])
    best_is = m.loc[m["sharpe_is"].idxmax()]
    print(f"\n  {label}: cells={len(m)}  "
          f"IS->OOS Sharpe rank corr = {rho:+.3f} (p={p:.3f})")
    print(f"    mean Sharpe   IS {m['sharpe_is'].mean():+.2f}   OOS {m['sharpe_oos'].mean():+.2f}")
    print(f"    %cells OOS>0  {(m['sharpe_oos'] > 0).mean():.0%}")
    print(f"    best-IS cell: n={int(best_is['n'])} band={best_is['band']} -> "
          f"IS Sharpe {best_is['sharpe_is']:+.2f}, OOS Sharpe {best_is['sharpe_oos']:+.2f}"
          f"  <- what you would actually have traded")


def main() -> None:
    frames = D.load_all()
    print("=" * 190)
    print(f"IN-SAMPLE  2024-01-01 .. {SPLIT.date()}      OUT-OF-SAMPLE  {SPLIT.date()} .. 2026-08-12")
    print("=" * 190)

    all_rows = []
    for tf in ("1d", "4h", "1h"):
        g = sweep_vi(frames, tf)
        g.to_csv(REPORTS / f"sweep_vi_{tf}.csv", index=False)
        all_rows.append(g)
        print(f"\n### {tf} -- Vortex parameter sweep ({len(g)} configurations)")
        for mode in ("ls", "long_only"):
            stability_report(g[g["mode"] == mode], f"{tf} {mode}")

        top = g.sort_values("sharpe_full", ascending=False).head(6)
        print(f"\n  top 6 by FULL-sample Sharpe ({tf}) -- note how they do in each half:")
        print("    " + top[["n", "mode", "band", "sharpe_is", "sharpe_oos",
                            "sharpe_full", "ret_full", "dd_full", "trades"]]
              .to_string(index=False).replace("\n", "\n    "))

    print("\n" + "=" * 190)
    print("PARAMETER-NEIGHBOURHOOD TEST (daily VI): a real edge should not vanish "
          "when you nudge the lookback")
    print("=" * 190)
    d = all_rows[0]
    piv = d[d["band"] == 0].pivot_table(index="n", columns="mode",
                                        values=["sharpe_is", "sharpe_oos", "sharpe_full"])
    print(piv.round(2).to_string())

    print("\n" + "=" * 190)
    print("ENSEMBLE vs SINGLE PARAMETER -- averaging the signal across lookbacks")
    print("=" * 190)
    for tf in ("1d", "4h"):
        df = frames[tf]
        ann = D.bars_per_year(tf)
        bpd = ann / 365.0
        lens = list(range(8, 41, 4))
        for mode in ("ls", "long_only"):
            sigs = pd.concat([S.vi_cross(df, n, mode) for n in lens], axis=1)
            avg = sigs.mean(axis=1)
            for step, tag in ((1.0, "vote>0 (binary)"), (0.25, "graded 1/4"), (0.0, "raw average")):
                sig = np.sign(avg) if step == 1.0 else (
                    S.discretise(avg, step) if step > 0 else avg)
                sig = pd.Series(sig, index=df.index).fillna(0.0)
                if mode == "long_only":
                    sig = sig.clip(lower=0)
                res = B.run(df, sig, COSTS, ann, f"{tf} VI ens {mode} {tag}", bpd)
                seg = segment_stats(res, ann)
                print(f"  {tf} {mode:<10} {tag:<16} Sharpe full {seg['sharpe_full']:+.2f}  "
                      f"IS {seg['sharpe_is']:+.2f}  OOS {seg['sharpe_oos']:+.2f}  "
                      f"ret {seg['ret_full']:+7.1%}  DD {seg['dd_full']:6.1%}  "
                      f"trades {seg['trades']:4d}")
        print()


if __name__ == "__main__":
    main()
