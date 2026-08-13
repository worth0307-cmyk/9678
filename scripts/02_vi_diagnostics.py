"""Does the Vortex Indicator carry any information about future BTC returns?

No strategy, no costs, no optimisation -- just conditional forward returns.
If there is no edge here, no amount of stop-loss engineering will create one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vibt import data as D, indicators as I  # noqa: E402

pd.set_option("display.width", 160)


def fwd_returns(close: pd.Series, horizons: list[int]) -> pd.DataFrame:
    return pd.DataFrame({f"h{h}": close.shift(-h) / close - 1.0 for h in horizons})


def ic(sig: pd.Series, fwd: pd.Series) -> tuple[float, float]:
    """Spearman IC and its t-stat."""
    m = sig.notna() & fwd.notna()
    if m.sum() < 50:
        return np.nan, np.nan
    r = sig[m].rank()
    f = fwd[m].rank()
    c = np.corrcoef(r, f)[0, 1]
    n = m.sum()
    t = c * np.sqrt((n - 2) / max(1e-12, 1 - c**2))
    return c, t


def state_table(state: pd.Series, fwd: pd.DataFrame, label: str) -> None:
    print(f"\n  {label}")
    print(f"    {'state':<14}{'n':>7}  " + "  ".join(f"{c:>18}" for c in fwd.columns))
    for s in sorted(state.dropna().unique()):
        m = state == s
        cells = []
        for c in fwd.columns:
            v = fwd.loc[m, c].dropna()
            if len(v) < 30:
                cells.append(f"{'--':>18}")
                continue
            mu = v.mean()
            t = mu / (v.std(ddof=1) / np.sqrt(len(v)))
            cells.append(f"{mu:+8.3%} (t{t:+5.1f})")
        print(f"    {str(s):<14}{int(m.sum()):>7}  " + "  ".join(cells))


def main() -> None:
    frames = D.load_all()
    horizons = {"1h": [1, 4, 12, 24, 72], "4h": [1, 3, 6, 12, 30], "1d": [1, 2, 3, 5, 10]}
    lengths = [7, 14, 21, 34, 55]

    for tf in ("1d", "4h", "1h"):
        df = frames[tf]
        fwd = fwd_returns(df["close"], horizons[tf])
        print("=" * 150)
        print(f"TIMEFRAME {tf}   (n={len(df)})   forward horizons in bars: {horizons[tf]}")
        print("=" * 150)

        base = fwd.mean()
        print("  unconditional mean forward return: " +
              "  ".join(f"{c}={base[c]:+.3%}" for c in fwd.columns))

        print("\n  -- Spearman IC of VI spread (VI+ minus VI-) vs forward return --")
        print(f"    {'n_vi':<6}" + "  ".join(f"{c:>16}" for c in fwd.columns))
        for n in lengths:
            v = I.vortex(df, n)
            cells = []
            for c in fwd.columns:
                c_ic, t = ic(v["vi_spread"], fwd[c])
                cells.append(f"{c_ic:+.4f}(t{t:+5.1f})")
            print(f"    {n:<6}" + "  ".join(f"{x:>16}" for x in cells))

        print("\n  -- forward return by VI regime (n_vi=14) --")
        v14 = I.vortex(df, 14)
        state_table(np.sign(v14["vi_spread"]).map({1.0: "VI+ > VI-", -1.0: "VI+ < VI-"}),
                    fwd, "sign of VI spread")

        q = pd.qcut(v14["vi_spread"], 5, labels=["Q1 low", "Q2", "Q3", "Q4", "Q5 high"])
        state_table(q.astype(object), fwd, "VI spread quintile")

        print("\n  -- VI sum (VI+ + VI-) as a regime gauge, n=14 --")
        qs = pd.qcut(v14["vi_sum"], 4, labels=["S1 coiled", "S2", "S3", "S4 expanding"])
        state_table(qs.astype(object), fwd, "VI sum quartile: does it predict direction?")
        vs = pd.DataFrame({"vi_sum": v14["vi_sum"], "absfwd": fwd.iloc[:, -1].abs()})
        c_ic, t = ic(vs["vi_sum"], vs["absfwd"])
        print(f"    IC(VI sum, |forward return {fwd.columns[-1]}|) = {c_ic:+.4f} (t{t:+.1f})"
              "   <- positive means it forecasts MOVEMENT, not direction")

        print("\n  -- crossover event study (n_vi=14), mean forward return after the cross --")
        x = I.crossover(v14["vi_plus"], v14["vi_minus"])
        state_table(x.replace({0: np.nan}).map({1.0: "bull cross", -1.0: "bear cross"}),
                    fwd, "crossover bar")

        print("\n  -- benchmark: same IC test for plain momentum and RSI --")
        cands = {
            "ROC(14)": I.roc(df["close"], 14),
            "close/EMA50-1": df["close"] / I.ema(df["close"], 50) - 1,
            "EMA20-EMA50": I.ema(df["close"], 20) - I.ema(df["close"], 50),
            "RSI(14)-50": I.rsi(df["close"], 14) - 50,
            "EffRatio(20)*sign": I.efficiency_ratio(df["close"], 20) * np.sign(I.roc(df["close"], 20)),
        }
        print(f"    {'signal':<20}" + "  ".join(f"{c:>16}" for c in fwd.columns))
        for nm, s in cands.items():
            cells = []
            for c in fwd.columns:
                c_ic, t = ic(s, fwd[c])
                cells.append(f"{c_ic:+.4f}(t{t:+5.1f})")
            print(f"    {nm:<20}" + "  ".join(f"{x:>16}" for x in cells))
        print()


if __name__ == "__main__":
    main()
