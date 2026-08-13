"""Is the VI+ double breakout a VOLATILITY event rather than a direction event?

Direction failed the drift-controlled test in script 12.  VI+ is built from bar
ranges, so the natural next hypothesis is that it forecasts how far price
travels, not which way.

Two traps this script is built to avoid:

1. Drift/composition -- same fix as script 12: the null is random dates drawn
   with identical per-symbol event counts.
2. Volatility clustering -- "vol is high after the event" is nearly worthless if
   vol was already high before it.  So the statistic of interest is the RATIO of
   forward vol to trailing vol, compared against that same ratio on random
   dates.  Anything left is genuine forecasting power.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vibt import backtest as B, data as D, indicators as I, metrics as M, system as SYS, vi_band as VB  # noqa: E402

pd.set_option("display.width", 220)

REPORTS = Path(__file__).resolve().parent.parent / "reports"
COSTS = B.Costs(fee_bps=4.5, slip_bps=2.0)
ANN4 = D.bars_per_year("4h")
TRAIL = 30          # 4H bars of trailing window (5 days)
HORIZONS = [6, 12, 30]


def bar_stats(frames: dict) -> dict[str, np.ndarray]:
    f4 = frames["4h"]
    o, h, l, c = (f4[x].to_numpy() for x in ("open", "high", "low", "close"))
    tr = I.true_range(f4).to_numpy()
    return {"o": o, "h": h, "l": l, "c": c, "tr": tr, "rel_tr": tr / c}


def window_metrics(bs: dict, i: int, k: int) -> dict | None:
    """Volatility metrics over the k bars starting at the bar after the event."""
    s, e = i + 1, i + 1 + k
    if e >= len(bs["o"]) or s - TRAIL < 0:
        return None
    entry = bs["o"][s]
    fwd_tr = bs["rel_tr"][s:e]
    trail_tr = bs["rel_tr"][s - TRAIL:s]
    if not np.isfinite(fwd_tr).all() or not np.isfinite(trail_tr).all() or trail_tr.mean() <= 0:
        return None
    hi, lo = bs["h"][s:e].max(), bs["l"][s:e].min()
    return {
        "fwd_atr": fwd_tr.mean(),
        "trail_atr": trail_tr.mean(),
        "ratio": fwd_tr.mean() / trail_tr.mean(),
        "span": (hi - lo) / entry,
        "mfe_long": hi / entry - 1,
        "mae_long": lo / entry - 1,
        "absret": abs(bs["o"][e] / entry - 1),
    }


def collect(symbols: list[str], p: VB.BandParams, k: int):
    """Event metrics and a matched random-date null, per band mode."""
    ev_rows, null_rows = [], []
    rng = np.random.default_rng(23)
    for sym in symbols:
        frames = D.load_all(symbol=sym)
        bs = bar_stats(frames)
        df = VB.build_frame(frames, p)
        d = df["double"]
        new = (d != d.shift(1)) & (d != 0)
        idx = [df.index.get_loc(t) for t in df.index[new]]
        n_by_side = {1: 0, -1: 0}
        for i in idx:
            m = window_metrics(bs, i, k)
            if m is None:
                continue
            side = int(d.iloc[i])
            n_by_side[side] += 1
            ev_rows.append({"symbol": sym, "side": side, **m})
        # matched null: same count, random bars from the same symbol
        total = sum(n_by_side.values())
        if total:
            pool = np.arange(TRAIL, len(bs["o"]) - k - 2)
            if len(pool):
                for j in rng.choice(pool, size=total * 30, replace=True):
                    m = window_metrics(bs, int(j), k)
                    if m is not None:
                        null_rows.append({"symbol": sym, **m})
    return pd.DataFrame(ev_rows), pd.DataFrame(null_rows)


def compare(ev: pd.DataFrame, null: pd.DataFrame, field: str, label: str) -> None:
    if ev.empty or null.empty:
        return
    e, n = ev[field].dropna().to_numpy(), null[field].dropna().to_numpy()
    if len(e) < 20:
        return
    # p-value from the sampling distribution of the null mean at this sample size
    rng = np.random.default_rng(5)
    draws = np.array([rng.choice(n, len(e), replace=True).mean() for _ in range(4000)])
    pct = float((draws < e.mean()).mean())
    p = 2 * min(pct, 1 - pct)
    print(f"    {label:<26} event {e.mean():8.3f}   random {n.mean():8.3f}   "
          f"ratio {e.mean()/n.mean():5.2f}x   p={p:.3f}{'  *' if p < 0.05 else ''}")


def main() -> None:
    symbols = D.available_symbols()
    print("=" * 170)
    print(f"VI+ DOUBLE BREAKOUT AS A VOLATILITY EVENT   symbols: {symbols}")
    print("=" * 170)

    modes = [("fixed", VB.BandParams()),
             ("rolling q86/q14", VB.BandParams(band_mode="quantile"))]

    for mode_name, p in modes:
        print(f"\n{'=' * 170}\nBAND MODE: {mode_name}\n{'=' * 170}")
        for k in HORIZONS:
            ev, null = collect(symbols, p, k)
            if ev.empty:
                continue
            print(f"\n  horizon {k*4}h ({k} bars)   events {len(ev)}   null draws {len(null)}")
            compare(ev, null, "fwd_atr", "mean ATR/price ahead")
            compare(ev, null, "ratio", "ATR ahead / ATR before")
            compare(ev, null, "span", "high-low span of window")
            compare(ev, null, "absret", "|return| over window")
            for side, nm in ((1, "up-break only"), (-1, "down-break only")):
                sub = ev[ev["side"] == side]
                if len(sub) >= 20:
                    print(f"      -- {nm} (n={len(sub)})")
                    compare(sub, null, "ratio", "  ATR ahead / ATR before")
                    compare(sub, null, "span", "  high-low span")

    print("\n" + "=" * 170)
    print("MFE / MAE SHAPE  (pooled, rolling q86/q14, 48h window)")
    print("=" * 170)
    ev, null = collect(symbols, VB.BandParams(band_mode="quantile"), 12)
    for side, nm in ((1, "up-break"), (-1, "down-break")):
        sub = ev[ev["side"] == side]
        if len(sub) < 20:
            continue
        print(f"\n  {nm}  n={len(sub)}")
        for lab, col, sign in (("if you go LONG ", "mfe_long", 1), ("if you go SHORT", "mae_long", -1)):
            mfe = sub["mfe_long"] if sign > 0 else -sub["mae_long"]
            mae = sub["mae_long"] if sign > 0 else -sub["mfe_long"]
            print(f"    {lab}  MFE median {mfe.median():+6.2%}  p75 {mfe.quantile(.75):+6.2%}  "
                  f"p90 {mfe.quantile(.90):+6.2%}   |   "
                  f"MAE median {mae.median():+6.2%}  p25 {mae.quantile(.25):+6.2%}  "
                  f"p10 {mae.quantile(.10):+6.2%}")
    print(f"\n  random-date baseline (n={len(null)}):")
    print(f"    long-side MFE median {null['mfe_long'].median():+6.2%}  "
          f"MAE median {null['mae_long'].median():+6.2%}  "
          f"span median {null['span'].median():6.2%}")

    print("\n" + "=" * 170)
    print("SO WHAT?  Use the band state to SIZE the daily system, not to pick a side.")
    print("=" * 170)
    print("  If VI+ extremes forecast bigger ranges, the payoff is smaller positions")
    print("  into them -- a risk overlay, which needs no directional edge at all.\n")

    ann = 365.0
    for sym in symbols:
        frames = D.load_all(symbol=sym)
        daily = frames["1d"]
        base_sig = SYS.target_exposure(daily)
        base = B.run(daily, base_sig, COSTS, ann, "", 1.0)

        df = VB.build_frame(frames, VB.BandParams(band_mode="quantile"))
        extreme_4h = (df["double"] != 0).astype(float)
        dd = daily.copy()
        # carry the 4H band state onto the daily grid by close time (no lookahead)
        state = extreme_4h.to_frame("x")
        state["close_time"] = df.index + pd.Timedelta(hours=4)
        aligned = D.align_higher_tf(daily.index, state, "4h", ["x"])["x_4h"].fillna(0.0)

        for cut in (0.5, 0.0):
            gated = base_sig * np.where(aligned > 0, cut, 1.0)
            res = B.run(daily, pd.Series(gated, index=daily.index), COSTS, ann, "", 1.0)
            tag = f"size x{cut:g} while VI+ extreme"
            print(f"  {sym:<10} {tag:<30} Sharpe {res.stats.sharpe:+5.2f} "
                  f"(base {base.stats.sharpe:+5.2f})  ret {res.stats.total_return:+8.1%} "
                  f"(base {base.stats.total_return:+8.1%})  DD {res.stats.max_dd:6.1%} "
                  f"(base {base.stats.max_dd:6.1%})")


def independence_check(symbols: list[str]) -> None:
    """Six correlated crypto assets are not six independent experiments."""
    print("\n" + "=" * 170)
    print("HOW MUCH DID SIX SYMBOLS ACTUALLY BUY US?")
    print("=" * 170)
    R = pd.DataFrame({s: np.log(D.load("1d", symbol=s)["close"]).diff()
                      for s in symbols}).dropna()
    C = R.corr()
    off = C.values[np.triu_indices(len(symbols), 1)]
    rho, n = off.mean(), len(symbols)
    neff = n / (1 + (n - 1) * rho)
    Rz = (R - R.mean()) / R.std()
    ev = np.linalg.eigvalsh(np.cov(Rz.T.values))[::-1]
    print(f"  mean pairwise correlation of daily returns: {rho:.3f}  "
          f"(range {off.min():.2f} to {off.max():.2f})")
    print(f"  first principal component explains {ev[0]/ev.sum():.0%} of variance -- crypto beta")
    print(f"  effective independent bets = n/(1+(n-1)rho) = {neff:.2f}, not {n}")
    print("  So 61 -> 359 events is real, but the INDEPENDENT information grew ~1.4x, not 6x.")
    print("  That is why the confidence intervals barely moved.")


def cluster_robust(symbols: list[str]) -> None:
    """Events fire on the same days across symbols; treat weeks as the unit."""
    print("\n" + "=" * 170)
    print("CLUSTER-ROBUST TEST -- events on the same day are not independent draws")
    print("=" * 170)
    for mode in ("fixed", "quantile"):
        rows = []
        for sym in symbols:
            frames = D.load_all(symbol=sym)
            p = VB.BandParams(band_mode=mode)
            df = VB.build_frame(frames, p)
            o = df["open"].to_numpy()
            n = len(df)
            d = df["double"]
            for ts in df.index[(d != d.shift(1)) & (d != 0)]:
                i = df.index.get_loc(ts)
                if i + 13 < n:
                    rows.append({"symbol": sym, "time": ts, "side": int(d.loc[ts]),
                                 "h12": o[i + 13] / o[i + 1] - 1})
        P = pd.DataFrame(rows)
        up = P[P["side"] == 1].copy()
        up["week"] = up["time"].dt.to_period("W")
        short = -up["h12"].to_numpy()
        rng = np.random.default_rng(0)
        naive = np.array([rng.choice(short, len(short), replace=True).mean()
                          for _ in range(20_000)])
        groups = [g["h12"].to_numpy() for _, g in up.groupby("week")]
        k = len(groups)
        clus = np.array([-np.concatenate([groups[j] for j in rng.integers(0, k, k)]).mean()
                         for _ in range(20_000)])
        print(f"\n  {mode}: {len(up)} up-breaks over {up['week'].nunique()} distinct weeks "
              f"({len(up)/up['time'].dt.floor('D').nunique():.2f} symbols firing per active day)")
        print(f"    shorting the up-break, mean 48h P&L {short.mean():+.2%}")
        print(f"      naive bootstrap (events independent): P(mean>0) = {np.mean(naive>0):.1%}")
        print(f"      week-clustered bootstrap            : P(mean>0) = {np.mean(clus>0):.1%}"
              f"   <- the honest one")


if __name__ == "__main__":
    main()
    syms = D.available_symbols()
    if len(syms) > 1:
        independence_check(syms)
        cluster_robust(syms)
