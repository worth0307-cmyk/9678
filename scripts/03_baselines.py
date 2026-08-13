"""Head-to-head of the obvious candidates, with real costs, on the full sample.

This is the "is anything here even worth optimising?" pass.  Nothing is tuned;
every parameter is a textbook default.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vibt import backtest as B, data as D, indicators as I, strategies as S  # noqa: E402

pd.set_option("display.width", 200)

COSTS = B.Costs(fee_bps=4.5, slip_bps=2.0)


def build(tf: str, frames: dict) -> dict[str, pd.Series]:
    df = frames[tf]
    out: dict[str, pd.Series] = {}

    for n in (14, 21, 34):
        out[f"VI({n}) long/short"] = S.vi_cross(df, n, "ls")
        out[f"VI({n}) long only"] = S.vi_cross(df, n, "long_only")
    out["VI(14) band .04 L/S"] = S.vi_cross_band(df, 14, 0.04, "ls")
    out["VI(14) proportional"] = S.vi_strength(df, 14, 0.15, "ls")
    out["VI(14) + ER>0.3"] = S.vi_filtered(df, 14, 20, 0.30, "ls")
    out["VI(14) contrarian"] = S.vi_contrarian(df, 14, 0.8, 500)

    if tf != "1d":
        htf = "1d" if tf == "4h" else "4h"
        out[f"VI mtf ({htf} regime) L/S"] = S.vi_mtf(df, frames[htf], htf, 14, 14, "ls")
        out[f"VI mtf ({htf} regime) L"] = S.vi_mtf(df, frames[htf], htf, 14, 14, "long_only")

    out["EMA 20/50 L/S"] = S.ema_cross(df, 20, 50, "ls")
    out["EMA 20/50 long only"] = S.ema_cross(df, 20, 50, "long_only")
    out["EMA 50/200 long only"] = S.ema_cross(df, 50, 200, "long_only")
    out["Donchian 20/10 L/S"] = S.donchian_break(df, 20, 10, "ls")
    out["Donchian 55/20 L/S"] = S.donchian_break(df, 55, 20, "ls")
    out["TSMOM(30) L/S"] = S.tsmom(df, 30, "ls")
    out["price > SMA100 (long)"] = S.price_vs_ma(df, 100, "long_only")
    out["z-revert(24, 2.0)"] = S.zscore_revert(df, 24, 2.0, 0.5, "ls")
    out["RSI revert(14)"] = S.rsi_revert(df, 14, 30, 70, "ls")
    return out


def main() -> None:
    frames = D.load_all()
    for tf in ("1d", "4h", "1h"):
        df = frames[tf]
        ann = D.bars_per_year(tf)
        bpd = ann / 365.0
        print("=" * 168)
        print(f"TIMEFRAME {tf}   costs = {COSTS.fee_bps}bp fee + {COSTS.slip_bps}bp slip per side "
              f"({(COSTS.fee_bps + COSTS.slip_bps) * 2:.0f}bp round trip)")
        print("=" * 168)

        bh = B.buy_and_hold(df, ann, COSTS)
        print(f"{'buy & hold':<34} {bh.stats}")
        print("-" * 168)

        rows = []
        for name, sig in build(tf, frames).items():
            r = B.run(df, sig, COSTS, ann, name, bpd)
            print(r)
            rows.append({"tf": tf, "name": name, **r.stats.as_row()})
        print()

        pd.DataFrame(rows).to_csv(
            Path(__file__).resolve().parent.parent / "reports" / f"baselines_{tf}.csv",
            index=False,
        )


if __name__ == "__main__":
    main()
