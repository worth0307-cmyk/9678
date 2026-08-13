"""Screen candidate symbols by correlation, not by narrative.

REPORT_XSEC section 3: BTC/ETH/SOL/BNB correlate at 0.815 and the cross-sectional
strategy earns nothing on them.  What the strategy needs is dispersion, so the
selection criterion is "low correlation to what we already hold" -- market cap and
conviction are irrelevant.

Run after ingesting the candidate exports:
    python -m vibt.ingest <export dir>
    python scripts/16_screen_universe.py
    python scripts/16_screen_universe.py --core BTCUSDT,ETHUSDT,SOLUSDT --max-corr 0.7
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vibt import data as D  # noqa: E402

pd.set_option("display.width", 200)

DEFAULT_CORE = ["BTCUSDT", "BNBUSDT", "ETHUSDT", "HYPEUSDT", "SOLUSDT", "TAOUSDT"]
REPORTS = Path(__file__).resolve().parent.parent / "reports"


def effective_bets(corr: pd.DataFrame) -> float:
    """n / (1 + (n-1) * mean pairwise correlation)."""
    n = len(corr)
    if n < 2:
        return float(n)
    off = corr.values[np.triu_indices(n, 1)]
    rho = np.nanmean(off)
    return float(n / (1 + (n - 1) * rho))


def pc1_share(rets: pd.DataFrame) -> float:
    r = rets.dropna()
    if r.shape[1] < 2 or len(r) < 30:
        return np.nan
    z = (r - r.mean()) / r.std()
    ev = np.linalg.eigvalsh(np.cov(z.T.values))[::-1]
    return float(ev[0] / ev.sum())


def main() -> None:
    ap = argparse.ArgumentParser(description="rank candidate symbols by diversification value")
    ap.add_argument("--core", default=",".join(DEFAULT_CORE),
                    help="comma-separated symbols already in the book")
    ap.add_argument("--max-corr", type=float, default=0.75,
                    help="reject candidates above this mean correlation to the core")
    ap.add_argument("--min-bars", type=int, default=400, help="minimum daily bars")
    ap.add_argument("--target", type=int, default=20, help="target universe size")
    args = ap.parse_args()

    available = D.available_symbols()
    core = [s for s in args.core.split(",") if s in available]
    cands = [s for s in available if s not in core]

    print("=" * 150)
    print(f"UNIVERSE SCREEN   {len(available)} symbols with complete data")
    print(f"  core       : {core}")
    print(f"  candidates : {len(cands)}")
    print("=" * 150)
    if not cands:
        print("\n  No candidates beyond the core yet.  Export more symbols first -- see UNIVERSE.md:")
        print("    for S in DOGEUSDT 1000PEPEUSDT WIFUSDT RENDERUSDT SUIUSDT ... ; do")
        print("      python3 backend/tools/export_klines.py --symbol $S --market futures \\")
        print("        --intervals 1d --start 2023-01-01 --out ./exports; done")
        print("\n  Note the ticker traps: PEPE/SHIB/BONK/FLOKI are 1000-prefixed on")
        print("  USDT-M futures, RNDR is now RENDER, MATIC is now POL.")
        return

    rets = pd.DataFrame({s: np.log(D.load("1d", symbol=s)["close"]).diff()
                         for s in available})

    core_r = rets[core]
    rows = []
    for s in cands:
        pair = rets[[s] + core].dropna()
        if len(pair) < 60:
            rows.append({"symbol": s, "bars": int(rets[s].notna().sum()),
                         "corr_core": np.nan, "corr_max": np.nan, "verdict": "too short"})
            continue
        c = pair.corr()[s].drop(s)
        bars = int(rets[s].notna().sum())
        vol = float(rets[s].std() * np.sqrt(365))
        idio = vol * float(np.sqrt(max(0.0, 1 - c.mean() ** 2)))
        ok = (c.mean() <= args.max_corr) and (bars >= args.min_bars)
        rows.append({
            "symbol": s, "bars": bars, "ann_vol": vol,
            "corr_core": float(c.mean()), "corr_max": float(c.max()),
            "closest": c.idxmax(), "idio_vol": idio,
            "verdict": "KEEP" if ok else ("too correlated" if c.mean() > args.max_corr
                                          else "too short"),
        })

    g = pd.DataFrame(rows).sort_values("corr_core")
    print("\n  ranked by mean correlation to the core (lower is better):\n")
    print("   " + g.round(3).to_string(index=False).replace("\n", "\n   "))

    keep = g[g["verdict"] == "KEEP"]["symbol"].tolist()
    print(f"\n  passing both filters (corr <= {args.max_corr}, bars >= {args.min_bars}): "
          f"{len(keep)}")
    print(f"    {keep}")

    print("\n" + "=" * 150)
    print("DIVERSIFICATION GAIN -- the number that actually decides this")
    print("=" * 150)
    base_corr = rets[core].dropna().corr()
    print(f"  core only ({len(core)} names):  mean pairwise corr "
          f"{base_corr.values[np.triu_indices(len(core),1)].mean():.3f}   "
          f"effective bets {effective_bets(base_corr):.2f}   "
          f"PC1 {pc1_share(rets[core]):.0%}")

    steps = sorted({min(k, len(keep)) for k in (5, 10, 15, args.target)} - {0})
    for k in steps:
        sel = core + keep[:k]
        if len(sel) <= len(core):
            continue
        sub = rets[sel]
        c = sub.corr()
        print(f"  core + best {min(k, len(keep)):2d} ({len(sel):2d} names): "
              f"mean pairwise corr {c.values[np.triu_indices(len(sel),1)].mean():.3f}   "
              f"effective bets {effective_bets(c):.2f}   PC1 {pc1_share(sub):.0%}")

    print("\n  Target: effective bets above ~3.  Currently the six-name book is 1.37,")
    print("  which is why its confidence intervals stay wide no matter how many events")
    print("  the event studies count.")

    print("\n" + "=" * 150)
    print("CROSS-SECTIONAL DISPERSION -- the fuel the strategy actually runs on")
    print("=" * 150)
    for label, sel in (("core only", core), ("core + all keepers", core + keep)):
        sub = rets[sel].dropna(how="all")
        disp = sub.std(axis=1).dropna()
        if len(disp):
            print(f"  {label:<22} daily cross-sectional stdev: median {disp.median():.3%}   "
                  f"p25 {disp.quantile(.25):.3%}   p75 {disp.quantile(.75):.3%}")

    g.to_csv(REPORTS / "universe_screen.csv", index=False)
    print(f"\n  wrote {REPORTS/'universe_screen.csv'}")
    print("\n  Next: export 1h and 4h for the keepers only, then rerun")
    print("  scripts/14_cross_section.py and scripts/15_xsec_robustness.py.")
    print("  The prediction under test: more dispersion should STRENGTHEN the effect.")
    print("  If it weakens instead, the dispersion mechanism is wrong and this path ends.")


if __name__ == "__main__":
    main()
