"""Data sanity + regime survey.  Run before trusting any backtest number."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vibt import data as D  # noqa: E402

pd.set_option("display.width", 140)


def main() -> None:
    frames = D.load_all()

    print("=" * 78)
    print("1. COVERAGE / INTEGRITY")
    print("=" * 78)
    for tf, df in frames.items():
        gaps = D.gap_report(df, tf)
        print(
            f"{tf:>3}  bars={len(df):6d}  {df.index[0]} -> {df.index[-1]}  "
            f"gaps={len(gaps):3d}  missing_bars={int(gaps['bars_missing'].sum()) if len(gaps) else 0}"
        )
        if len(gaps):
            print(gaps.head(10).to_string())

    print("\ncross-check: 1h resampled to 4h/1d vs the supplied files")
    for tf, rule in (("4h", "4h"), ("1d", "1D")):
        built = D.resample_from(frames["1h"], rule)
        given = frames[tf][["open", "high", "low", "close"]]
        common = built.index.intersection(given.index)
        diff = (built.loc[common] - given.loc[common]).abs()
        rel = (diff / given.loc[common]).max()
        print(f"  {tf}: {len(common)} shared bars, max rel diff per column:")
        print("   ", {c: f"{rel[c]:.2e}" for c in rel.index})

    d = frames["1d"]
    print("\n" + "=" * 78)
    print("2. PRICE REGIMES (daily)")
    print("=" * 78)
    print(f"first close {d['close'].iloc[0]:,.0f}   last close {d['close'].iloc[-1]:,.0f}")
    print(f"buy & hold total return: {d['close'].iloc[-1] / d['close'].iloc[0] - 1:+.1%}")
    print(f"all-time high in sample: {d['high'].max():,.0f} on {d['high'].idxmax().date()}")
    print(f"all-time low  in sample: {d['low'].min():,.0f} on {d['low'].idxmin().date()}")

    runmax = d["close"].cummax()
    dd = d["close"] / runmax - 1
    print(f"max drawdown of buy & hold: {dd.min():.1%} on {dd.idxmin().date()}")

    print("\nyear-by-year (close-to-close):")
    yr = d["close"].resample("YE")
    for period, grp in d.groupby(d.index.year):
        r = grp["close"].iloc[-1] / grp["open"].iloc[0] - 1
        vol = grp["close"].pct_change().std() * np.sqrt(365)
        mdd = (grp["close"] / grp["close"].cummax() - 1).min()
        print(f"  {period}: return {r:+7.1%}   ann.vol {vol:6.1%}   maxDD {mdd:7.1%}   bars {len(grp)}")

    print("\nquarter-by-quarter close-to-close return:")
    q = d["close"].resample("QE").last()
    qr = q.pct_change().dropna()
    for ts, v in qr.items():
        print(f"  {ts.year}Q{ts.quarter}: {v:+7.1%}")

    print("\n" + "=" * 78)
    print("3. RETURN STATISTICS (what any strategy has to beat)")
    print("=" * 78)
    for tf, df in frames.items():
        r = np.log(df["close"]).diff().dropna()
        ann = D.bars_per_year(tf)
        print(
            f"{tf:>3}  n={len(r):6d}  ann.vol={r.std() * np.sqrt(ann):6.1%}  "
            f"skew={r.skew():+6.2f}  kurt={r.kurtosis():7.2f}  "
            f"ann.drift={r.mean() * ann:+7.1%}"
        )

    print("\nautocorrelation of log returns (trend vs mean-reversion tell):")
    for tf, df in frames.items():
        r = np.log(df["close"]).diff().dropna()
        acs = [r.autocorr(lag) for lag in (1, 2, 3, 5, 10, 20)]
        print(f"  {tf:>3}  " + "  ".join(f"lag{l}={a:+.3f}" for l, a in zip((1, 2, 3, 5, 10, 20), acs)))

    print("\nvariance ratio (>1 trending, <1 mean-reverting), daily log returns:")
    r = np.log(d["close"]).diff().dropna().to_numpy()
    for k in (2, 3, 5, 10, 20):
        agg = np.add.reduceat(r[: len(r) // k * k], np.arange(0, len(r) // k * k, k))
        vr = agg.var(ddof=1) / (k * r.var(ddof=1))
        print(f"  k={k:2d}: VR={vr:.3f}")

    print("\n" + "=" * 78)
    print("4. COST CONTEXT")
    print("=" * 78)
    for tf, df in frames.items():
        rng = (df["high"] - df["low"]) / df["close"]
        print(
            f"{tf:>3}  median bar range = {rng.median():.3%} of price   "
            f"p25={rng.quantile(.25):.3%}  p75={rng.quantile(.75):.3%}"
        )
    print("  a 6bp round-trip (taker+slippage) eats this share of a median bar range:")
    for tf, df in frames.items():
        rng = ((df["high"] - df["low"]) / df["close"]).median()
        print(f"   {tf:>3}: {0.0006 / rng:.1%}")


if __name__ == "__main__":
    main()
