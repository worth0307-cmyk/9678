"""Your actual strategy, tested: what happens AFTER a VI+ double breakout?

The dashboard's rule (backend/data_service.py):
    vip > upper -> "above";  vip < lower -> "below"
    double breakout = 4H and 1D are in the SAME state
Your trade: double-above -> SHORT, double-below -> LONG.  I.e. fade the extreme.

This script does not trade.  It measures the forward return after every event,
so we find out whether fading is the right sign before building anything.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vibt import data as D, indicators as I  # noqa: E402

pd.set_option("display.width", 200)

BANDS = {"4h": (1.19, 0.728), "1d": (1.155, 0.60)}
VI_LEN = 14
HORIZONS = [1, 3, 6, 12, 30, 60, 90]        # in 4H bars: 4h .. 15 days


def state(vip: pd.Series, upper: float, lower: float) -> pd.Series:
    return pd.Series(np.where(vip > upper, "above",
                     np.where(vip < lower, "below", "inside")), index=vip.index)


def build(frames: dict) -> pd.DataFrame:
    f4, f1 = frames["4h"], frames["1d"]
    vi4 = I.vortex(f4, VI_LEN)["vi_plus"]
    vi1 = I.vortex(f1, VI_LEN)["vi_plus"]

    # daily VI aligned onto the 4H grid using CLOSE times -> no lookahead
    d = f1.copy()
    d["vip"] = vi1
    a = D.align_higher_tf(f4.index, d, "1d", ["vip"])["vip_1d"]

    df = pd.DataFrame({"open": f4["open"], "high": f4["high"], "low": f4["low"],
                       "close": f4["close"], "vi4": vi4, "vi1d": a})
    df["s4"] = state(df["vi4"], *BANDS["4h"])
    df["s1d"] = state(df["vi1d"], *BANDS["1d"])
    df["double"] = np.where((df["s4"] == df["s1d"]) & (df["s4"] != "inside"), df["s4"], "none")
    return df


def episodes(df: pd.DataFrame) -> pd.DataFrame:
    """First bar of each new double-breakout episode."""
    d = df["double"]
    new = (d != d.shift(1)) & (d != "none")
    ev = df.loc[new, ["double", "close", "vi4", "vi1d"]].copy()
    ev["idx"] = [df.index.get_loc(t) for t in ev.index]
    return ev


def main() -> None:
    frames = D.load_all()
    df = build(frames)
    o = df["open"].to_numpy()
    n = len(df)

    print("=" * 170)
    print(f"VI+ DOUBLE BREAKOUT STUDY   bands 4H {BANDS['4h']}  1D {BANDS['1d']}  VI length {VI_LEN}")
    print("=" * 170)

    occ = df["double"].value_counts()
    print(f"\n  bar-level occupancy over {n} 4H bars ({n/6:.0f} days):")
    for k in ("above", "below", "none"):
        c = int(occ.get(k, 0))
        print(f"    {k:<6} {c:6d} bars  {c/n:6.1%}")
    print(f"\n  single-timeframe occupancy:")
    for col, tf in (("s4", "4H"), ("s1d", "1D")):
        vc = df[col].value_counts()
        print(f"    {tf}: " + "  ".join(f"{k}={int(vc.get(k,0)):5d} ({vc.get(k,0)/n:5.1%})"
                                        for k in ("above", "inside", "below")))

    ev = episodes(df)
    print(f"\n  distinct episodes: {len(ev)}  "
          f"(above {int((ev['double']=='above').sum())}, below {int((ev['double']=='below').sum())})")
    if len(ev) == 0:
        print("\n  NO EVENTS -- the bands never fire on this sample.  Nothing to test.")
        return

    print("\n" + "=" * 170)
    print("FORWARD RETURN AFTER EACH EPISODE  (entry at the NEXT 4H open, horizons in 4H bars)")
    print("  Your trade: 'above' -> SHORT (so you want NEGATIVE numbers)")
    print("              'below' -> LONG  (so you want POSITIVE numbers)")
    print("=" * 170)
    hdr = "  ".join(f"{h*4:>4}h" .rjust(17) for h in HORIZONS)
    print(f"    {'event':<8}{'n':>5}   " + "  ".join(f"{f'+{h*4}h':>16}" for h in HORIZONS))
    for kind in ("above", "below"):
        sub = ev[ev["double"] == kind]
        cells = []
        for h in HORIZONS:
            rs = []
            for i in sub["idx"]:
                if i + 1 + h < n:
                    rs.append(o[i + 1 + h] / o[i + 1] - 1.0)
            if len(rs) < 5:
                cells.append(f"{'--':>16}")
                continue
            rs = np.array(rs)
            t = rs.mean() / (rs.std(ddof=1) / np.sqrt(len(rs))) if rs.std(ddof=1) > 0 else 0
            cells.append(f"{rs.mean():+7.2%} (t{t:+4.1f})")
        print(f"    {kind:<8}{len(sub):>5}   " + "  ".join(cells))

    print("\n  same thing as HIT RATES (share of episodes where your trade was in profit):")
    print(f"    {'event':<8}{'side':<7}   " + "  ".join(f"{f'+{h*4}h':>10}" for h in HORIZONS))
    for kind, side in (("above", -1), ("below", +1)):
        sub = ev[ev["double"] == kind]
        cells = []
        for h in HORIZONS:
            rs = [side * (o[i + 1 + h] / o[i + 1] - 1.0) for i in sub["idx"] if i + 1 + h < n]
            cells.append(f"{np.mean(np.array(rs) > 0):>9.0%}" if len(rs) >= 5 else f"{'--':>10}")
        print(f"    {kind:<8}{'short' if side < 0 else 'long':<7}   " + "  ".join(cells))

    print("\n" + "=" * 170)
    print("MFE / MAE  -- how far did price run FOR you and AGAINST you, in your direction?")
    print("  (this is the 'use high/low as a profit/loss percentage' idea)")
    print("=" * 170)
    hi, lo = df["high"].to_numpy(), df["low"].to_numpy()
    for kind, side in (("above", -1), ("below", +1)):
        sub = ev[ev["double"] == kind]
        print(f"\n  {kind} -> {'SHORT' if side < 0 else 'LONG'}  (n={len(sub)})")
        for h in (6, 12, 30, 60):
            mfe, mae = [], []
            for i in sub["idx"]:
                if i + 1 + h >= n:
                    continue
                entry = o[i + 1]
                w_hi, w_lo = hi[i + 1: i + 1 + h].max(), lo[i + 1: i + 1 + h].min()
                if side > 0:
                    mfe.append(w_hi / entry - 1); mae.append(w_lo / entry - 1)
                else:
                    mfe.append(entry / w_lo - 1); mae.append(entry / w_hi - 1)
            if len(mfe) < 5:
                continue
            print(f"    within {h*4:>3}h:  median MFE {np.median(mfe):+6.2%}  "
                  f"median MAE {np.median(mae):+6.2%}   "
                  f"MFE p75 {np.percentile(mfe, 75):+6.2%}  MAE p25 {np.percentile(mae, 25):+6.2%}")

    print("\n" + "=" * 170)
    print("CONTROL: is the SIGN right?  Same events, both directions, at +12 bars (2 days)")
    print("=" * 170)
    for kind in ("above", "below"):
        sub = ev[ev["double"] == kind]
        rs = np.array([o[i + 13] / o[i + 1] - 1 for i in sub["idx"] if i + 13 < n])
        if len(rs) < 5:
            continue
        print(f"    after '{kind}':  raw price move {rs.mean():+.2%}  median {np.median(rs):+.2%}  "
              f"  fade P&L {-np.sign(1 if kind=='above' else -1)*0:.0%}"
              f"  -> SHORT earns {-rs.mean():+.2%}, LONG earns {rs.mean():+.2%}")

    print("\n" + "=" * 170)
    print("BAND SENSITIVITY -- were 1.19/0.728 and 1.155/0.60 lucky picks?")
    print("=" * 170)
    print(f"    {'4H up':>7}{'1D up':>8}{'n':>5}  {'+2d move':>12}{'t':>7}   "
          f"{'4H dn':>7}{'1D dn':>8}{'n':>5}  {'+2d move':>12}{'t':>7}")
    for scale in (0.94, 0.97, 1.00, 1.03, 1.06):
        u4, u1 = 1.19 * scale, 1.155 * scale
        l4, l1 = 0.728 / scale, 0.60 / scale
        d2 = df.copy()
        d2["s4"] = state(d2["vi4"], u4, l4)
        d2["s1d"] = state(d2["vi1d"], u1, l1)
        d2["double"] = np.where((d2["s4"] == d2["s1d"]) & (d2["s4"] != "inside"), d2["s4"], "none")
        e2 = episodes(d2)
        row = []
        for kind in ("above", "below"):
            sub = e2[e2["double"] == kind]
            rs = np.array([o[i + 13] / o[i + 1] - 1 for i in sub["idx"] if i + 13 < n])
            if len(rs) >= 5:
                t = rs.mean() / (rs.std(ddof=1) / np.sqrt(len(rs)))
                row.append((len(rs), rs.mean(), t))
            else:
                row.append((len(rs), np.nan, np.nan))
        print(f"    {u4:7.3f}{u1:8.3f}{row[0][0]:5d}  {row[0][1]:+11.2%}{row[0][2]:+7.1f}   "
              f"{l4:7.3f}{l1:8.3f}{row[1][0]:5d}  {row[1][1]:+11.2%}{row[1][2]:+7.1f}")


if __name__ == "__main__":
    main()
