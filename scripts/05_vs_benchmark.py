"""Everything relative to buy & hold, plus a walk-forward that re-picks parameters.

Judging a long-only crypto strategy against zero is unfair in a bear market and
flattering in a bull one.  The benchmark is holding the coin.  And the only
selection test that means anything is one where the parameter is chosen using
data available at the time.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vibt import backtest as B, data as D, indicators as I, metrics as M, strategies as S  # noqa: E402

pd.set_option("display.width", 200)

COSTS = B.Costs(fee_bps=4.5, slip_bps=2.0)
SPLIT = pd.Timestamp("2025-07-01")
REPORTS = Path(__file__).resolve().parent.parent / "reports"


def capture(strat: pd.Series, bench: pd.Series) -> tuple[float, float]:
    """Up-capture / down-capture vs the benchmark, on benchmark-up and -down bars."""
    m = strat.index.intersection(bench.index)
    s, b = strat.loc[m], bench.loc[m]
    up, dn = b > 0, b < 0
    uc = s[up].sum() / b[up].sum() if b[up].sum() != 0 else np.nan
    dc = s[dn].sum() / b[dn].sum() if b[dn].sum() != 0 else np.nan
    return float(uc), float(dc)


def seg_line(tag: str, r: pd.Series, bench: pd.Series, ann: float) -> str:
    st = M.compute(r, ann)
    bst = M.compute(bench.reindex(r.index).dropna(), ann)
    uc, dc = capture(r, bench)
    return (f"    {tag:<22} ret {st.total_return:+8.1%} (B&H {bst.total_return:+8.1%})  "
            f"Sharpe {st.sharpe:+5.2f} (B&H {bst.sharpe:+5.2f})  "
            f"maxDD {st.max_dd:7.1%} (B&H {bst.max_dd:7.1%})  "
            f"up-cap {uc:+5.1%}  down-cap {dc:+6.1%}")


def evaluate(name: str, df: pd.DataFrame, sig: pd.Series, ann: float, bench: B.Result) -> dict:
    res = B.run(df, sig, COSTS, ann, name, ann / 365.0)
    r, br = res.rets, bench.rets
    print(f"\n  {name}")
    for tag, lo, hi in (("full sample", r.index[0], r.index[-1]),
                        ("in-sample", r.index[0], SPLIT),
                        ("out-of-sample", SPLIT, r.index[-1])):
        seg = r[(r.index >= lo) & (r.index < hi)] if tag != "full sample" else r
        if len(seg) > 30:
            print(seg_line(tag, seg, br, ann))
    p5, p50, p95 = M.block_bootstrap_sharpe(r, ann, block=int(max(5, ann / 52)))
    print(f"    bootstrap Sharpe 90% CI [{p5:+.2f}, {p95:+.2f}]  median {p50:+.2f}   "
          f"trades {res.stats.trades}  exposure {res.stats.exposure:.0%}")
    return {"name": name, **res.stats.as_row(), "boot_p5": p5, "boot_p95": p95}


def walk_forward(df: pd.DataFrame, tf: str, ann: float, mode: str,
                 train_bars: int, test_bars: int, grid: list[int]) -> pd.Series:
    """Re-pick the VI lookback every `test_bars` using only the prior `train_bars`."""
    sigs = {n: S.vi_cross(df, n, mode) for n in grid}
    cache = {n: B.run(df, sigs[n], COSTS, ann, "", ann / 365.0).rets for n in grid}
    out = pd.Series(0.0, index=df.index)
    picks = []
    start = train_bars
    while start < len(df) - 1:
        end = min(start + test_bars, len(df) - 1)
        tr = df.index[start - train_bars: start]
        best_n, best_sr = grid[0], -np.inf
        for n in grid:
            seg = cache[n].reindex(tr).dropna()
            if len(seg) < 30 or seg.std() == 0:
                continue
            sr = seg.mean() / seg.std() * np.sqrt(ann)
            if sr > best_sr:
                best_sr, best_n = sr, n
        idx = df.index[start:end]
        out.loc[idx] = sigs[best_n].reindex(idx).fillna(0.0)
        picks.append((df.index[start], best_n, best_sr))
        start = end
    print(f"    walk-forward picks ({tf} {mode}): " +
          ", ".join(f"{t.date()}:n={n}" for t, n, _ in picks[:14]) +
          (" ..." if len(picks) > 14 else ""))
    return out


def main() -> None:
    frames = D.load_all()

    for tf, wf_train, wf_test in (("1d", 250, 90), ("4h", 1500, 540)):
        df = frames[tf]
        ann = D.bars_per_year(tf)
        bench = B.buy_and_hold(df, ann, COSTS)
        print("=" * 200)
        print(f"TIMEFRAME {tf}")
        print("=" * 200)
        print(f"  benchmark buy & hold: {bench.stats}")
        br = bench.rets
        for tag, lo, hi in (("in-sample", br.index[0], SPLIT), ("out-of-sample", SPLIT, br.index[-1])):
            seg = br[(br.index >= lo) & (br.index < hi)]
            st = M.compute(seg, ann)
            print(f"    B&H {tag:<16} ret {st.total_return:+8.1%}  Sharpe {st.sharpe:+5.2f}  "
                  f"maxDD {st.max_dd:7.1%}")

        rows = []
        rows.append(evaluate("VI(14) long-only", df, S.vi_cross(df, 14, "long_only"), ann, bench))
        rows.append(evaluate("VI(14) long/short", df, S.vi_cross(df, 14, "ls"), ann, bench))
        rows.append(evaluate("EMA50/200 long-only", df, S.ema_cross(df, 50, 200, "long_only"), ann, bench))

        vol_sig = S.apply_vol_target(S.vi_cross(df, 14, "long_only"), df, ann, 30, 0.40, 1.0)
        rows.append(evaluate("VI(14) long-only + vol target 40%",
                             df, S.discretise(vol_sig, 0.2), ann, bench))

        grid = list(range(8, 49, 2))
        for mode in ("long_only", "ls"):
            wf = walk_forward(df, tf, ann, mode, wf_train, wf_test, grid)
            rows.append(evaluate(f"VI walk-forward ({mode})", df, wf, ann, bench))

        pd.DataFrame(rows).to_csv(REPORTS / f"vs_benchmark_{tf}.csv", index=False)
        print()


if __name__ == "__main__":
    main()
