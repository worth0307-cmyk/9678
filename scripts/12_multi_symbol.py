"""Pooled multi-symbol study of the VI+ double-breakout signal.

The whole reason for adding symbols: 61 BTC events cannot settle anything, but
six symbols over the same window can.  Every symbol is treated as an independent
draw of the same rule, then the events are POOLED and the statistics computed on
the pool.

Runs on whatever symbols are present, so it works today with BTCUSDT alone and
gets sharper with each file you add.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vibt import backtest as B, data as D, metrics as M, vi_band as VB  # noqa: E402

pd.set_option("display.width", 220)

COSTS = B.Costs(fee_bps=4.5, slip_bps=2.0)
ANN = D.bars_per_year("4h")
REPORTS = Path(__file__).resolve().parent.parent / "reports"
HORIZONS = [3, 6, 12, 30]           # 4H bars -> 12h, 24h, 48h, 120h


def events(frames: dict, p: VB.BandParams) -> pd.DataFrame:
    df = VB.build_frame(frames, p)
    o = df["open"].to_numpy()
    n = len(df)
    d = df["double"]
    new = (d != d.shift(1)) & (d != 0)
    rows = []
    for ts in df.index[new]:
        i = df.index.get_loc(ts)
        if i + 1 >= n:
            continue
        row = {"time": ts, "side": int(d.loc[ts]), "entry": o[i + 1]}
        for h in HORIZONS:
            row[f"h{h}"] = o[i + 1 + h] / o[i + 1] - 1 if i + 1 + h < n else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def summarise(pool: pd.DataFrame, label: str) -> None:
    print(f"\n  {label}")
    print(f"    {'event':<24}{'n':>5}   " + "  ".join(f"{f'+{h*4}h':>17}" for h in HORIZONS))
    for side, name in ((1, "up-break (you SHORT)"), (-1, "down-break (you LONG)")):
        sub = pool[pool["side"] == side]
        cells = []
        for h in HORIZONS:
            v = sub[f"h{h}"].dropna().to_numpy()
            if len(v) < 8:
                cells.append(f"{'--':>17}")
                continue
            t = v.mean() / (v.std(ddof=1) / np.sqrt(len(v)))
            cells.append(f"{v.mean():+8.2%} (t{t:+4.1f})")
        print(f"    {name:<24}{len(sub):>5}   " + "  ".join(cells))

    for side, trade in ((1, -1), (-1, 1)):
        sub = pool[pool["side"] == side]
        v = sub["h12"].dropna().to_numpy() * trade
        if len(v) < 8:
            continue
        rng = np.random.default_rng(0)
        boot = np.array([rng.choice(v, len(v), replace=True).mean() for _ in range(10_000)])
        名 = "SHORT the up-break" if side == 1 else "LONG the down-break"
        print(f"      your trade '{名}' over 48h: mean {v.mean():+.2%}  "
              f"win {np.mean(v>0):.0%}  P(edge>0) = {np.mean(boot>0):.1%}")


def main() -> None:
    symbols = D.available_symbols()
    print("=" * 170)
    print(f"MULTI-SYMBOL VI+ DOUBLE-BREAKOUT STUDY   symbols found: {symbols}")
    print("=" * 170)
    if len(symbols) < 2:
        print("\n  Only one symbol present.  Export the others with the dashboard tool:")
        print("    for S in BNBUSDT ETHUSDT HYPEUSDT SOLUSDT TAOUSDT; do \\")
        print("      python3 backend/tools/export_klines.py --symbol $S --market futures \\")
        print("        --intervals 1h,4h,1d --start 2024-01-01 --out ./exports; done")
        print("    python -m vibt.ingest ./exports")
        print("\n  Running on what is available so the numbers below are still reproducible.\n")

    modes = [("fixed", VB.BandParams()),
             ("rolling q86/q14", VB.BandParams(band_mode="quantile")),
             ("rolling range 80/20", VB.BandParams(band_mode="range"))]

    all_rows = []
    for mode_name, p in modes:
        print("\n" + "=" * 170)
        print(f"BAND MODE: {mode_name}")
        print("=" * 170)
        pool = []
        for sym in symbols:
            frames = D.load_all(symbol=sym)
            ev = events(frames, p)
            ev["symbol"] = sym
            pool.append(ev)
            up = int((ev["side"] == 1).sum())
            dn = int((ev["side"] == -1).sum())
            span = f"{frames['4h'].index[0].date()} .. {frames['4h'].index[-1].date()}"
            print(f"  {sym:<10} {len(frames['4h']):5d} 4H bars  {span}   "
                  f"up-breaks {up:3d}  down-breaks {dn:3d}")
        pool = pd.concat(pool, ignore_index=True) if pool else pd.DataFrame()
        if pool.empty:
            continue
        pool["mode"] = mode_name
        all_rows.append(pool)
        summarise(pool, f"POOLED across {len(symbols)} symbol(s)")

        if len(symbols) > 1:
            print("\n    per-symbol +48h mean, up-breaks (consistency check):")
            for sym in symbols:
                s = pool[(pool["symbol"] == sym) & (pool["side"] == 1)]["h12"].dropna()
                if len(s) >= 5:
                    print(f"      {sym:<10} n={len(s):3d}  mean {s.mean():+.2%}  "
                          f"share up {np.mean(s > 0):.0%}")

    if all_rows:
        out = pd.concat(all_rows, ignore_index=True)
        out.to_csv(REPORTS / "multi_symbol_events.csv", index=False)
        print(f"\n  wrote {REPORTS/'multi_symbol_events.csv'}  ({len(out)} event rows)")

    print("\n" + "=" * 170)
    print("PER-SYMBOL BACKTEST of the as-specified strategy (fade, pyramid, exit on re-entry)")
    print("=" * 170)
    rows = []
    for sym in symbols:
        frames = D.load_all(symbol=sym)
        for mode_name, p in modes:
            df = VB.build_frame(frames, p)
            res = B.run(frames["4h"], VB.target_position(df, p), COSTS, ANN, "", 6.0)
            rows.append({"symbol": sym, "band": mode_name, "sharpe": res.stats.sharpe,
                         "ret": res.stats.total_return, "maxdd": res.stats.max_dd,
                         "trades": res.stats.trades})
            print(f"  {sym:<10} {mode_name:<20} Sharpe {res.stats.sharpe:+5.2f}  "
                  f"ret {res.stats.total_return:+8.1%}  DD {res.stats.max_dd:6.1%}  "
                  f"trades {res.stats.trades:3d}")
    if rows:
        pd.DataFrame(rows).to_csv(REPORTS / "multi_symbol_backtest.csv", index=False)


if __name__ == "__main__":
    main()
