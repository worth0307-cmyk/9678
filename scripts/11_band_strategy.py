"""Backtest of the VI-Dashboard band strategy: as specified, and repaired.

Order of business:
  A. your rules exactly as described, with real costs
  B. the same rules with the sign flipped
  C. the same rules with a daily trend filter (the regime allocator from §7)
  D. bands recalibrated by percentile, which revives the dead long side
  E. per-regime attribution, so you can see WHERE it makes and loses money
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
ANN = D.bars_per_year("4h")          # 2190 bars/yr
BPD = 6.0
REPORTS = Path(__file__).resolve().parent.parent / "reports"
REGIMES = [("2024 bull", "2024-01-01", "2025-01-01"),
           ("2025 chop", "2025-01-01", "2026-01-01"),
           ("2026 bear", "2026-01-01", "2027-01-01")]


def run(frames, p: VB.BandParams, name: str, rows: list) -> B.Result:
    df = VB.build_frame(frames, p)
    sig = VB.target_position(df, p)
    res = B.run(frames["4h"], sig, COSTS, ANN, name, BPD)
    r = res.rets
    seg = {}
    for lab, lo, hi in REGIMES:
        s = r[(r.index >= lo) & (r.index < hi)]
        seg[lab] = M.compute(s, ANN).total_return if len(s) > 30 else np.nan
    rows.append({"name": name, "sharpe": res.stats.sharpe, "ret": res.stats.total_return,
                 "maxdd": res.stats.max_dd, "trades": res.stats.trades,
                 "expo": res.stats.exposure, **seg})
    print(f"  {name:<46} Sh {res.stats.sharpe:+5.2f} | ret {res.stats.total_return:+8.1%} | "
          f"DD {res.stats.max_dd:6.1%} | trd {res.stats.trades:3d} | expo {res.stats.exposure:4.0%} | "
          + " | ".join(f"{lab.split()[0]} {seg[lab]:+6.1%}" for lab, _, _ in REGIMES))
    return res


def main() -> None:
    frames = D.load_all()
    bench = B.buy_and_hold(frames["4h"], ANN, COSTS)
    rows: list[dict] = []

    print("=" * 210)
    print("A. YOUR RULES AS DESCRIBED   (double-above -> SHORT, double-below -> LONG, "
          "pyramid, exit when VI+ falls back)")
    print("=" * 210)
    print(f"  {'buy & hold':<46} Sh {bench.stats.sharpe:+5.2f} | ret {bench.stats.total_return:+8.1%} | "
          f"DD {bench.stats.max_dd:6.1%}")
    print("-" * 210)

    base = VB.BandParams()
    run(frames, base, "as-specified (fade, pyramid, exit on re-entry)", rows)
    run(frames, VB.BandParams(pyramid_steps=0), "  without pyramiding", rows)
    run(frames, VB.BandParams(exit_mode="retrace", exit_retrace=0.5),
        "  exit on 50% VI+ retrace instead", rows)
    run(frames, VB.BandParams(stop_atr=2.0), "  + 2xATR stop", rows)
    run(frames, VB.BandParams(max_hold_bars=30), "  + 5-day time stop", rows)

    print("\n" + "=" * 210)
    print("B. SAME RULES, SIGN FLIPPED  (double-above -> LONG)")
    print("=" * 210)
    run(frames, VB.BandParams(direction="follow"), "follow instead of fade", rows)
    run(frames, VB.BandParams(direction="follow", stop_atr=2.0), "  + 2xATR stop", rows)
    run(frames, VB.BandParams(direction="follow", exit_mode="retrace", exit_retrace=0.5),
        "  exit on 50% VI+ retrace", rows)

    print("\n" + "=" * 210)
    print("C. WITH A DAILY TREND FILTER  (take the trade only when the daily regime agrees)")
    print("=" * 210)
    run(frames, VB.BandParams(trend_filter=True), "fade + trend filter", rows)
    run(frames, VB.BandParams(direction="follow", trend_filter=True), "follow + trend filter", rows)
    run(frames, VB.BandParams(direction="follow", trend_filter=True, stop_atr=2.0),
        "follow + trend filter + 2xATR stop", rows)

    print("\n" + "=" * 210)
    print("D. PERCENTILE-CALIBRATED BANDS  (your fixed lower bands almost never fire)")
    print("=" * 210)
    for up_q, dn_q in ((0.85, 0.15), (0.80, 0.20), (0.90, 0.10)):
        pb = VB.percentile_bands(frames, 14, up_q, dn_q)
        print(f"\n  q{up_q:.0%}/q{dn_q:.0%}  ->  4H bands ({pb['up_4h']:.3f}, {pb['dn_4h']:.3f})   "
              f"1D bands ({pb['up_1d']:.3f}, {pb['dn_1d']:.3f})")
        for tag, extra in (("fade", {}), ("follow", {"direction": "follow"}),
                           ("follow+trend", {"direction": "follow", "trend_filter": True})):
            run(frames, VB.BandParams(**pb, **extra), f"  q{up_q:.0%}/q{dn_q:.0%} {tag}", rows)

    print("\n" + "=" * 210)
    print("E. LONG-SIDE vs SHORT-SIDE ATTRIBUTION of the as-specified strategy")
    print("=" * 210)
    df = VB.build_frame(frames, base)
    sig = VB.target_position(df, base)
    res = B.run(frames["4h"], sig, COSTS, ANN, "", BPD)
    pos = res.position
    for lab, mask in (("short book", pos < 0), ("long book", pos > 0)):
        rr = res.rets.where(mask.reindex(res.rets.index).fillna(False), 0.0)
        bars = int(mask.sum())
        print(f"  {lab:<12} bars held {bars:5d} ({bars/len(pos):5.1%})   "
              f"cumulative P&L {np.prod(1+rr)-1:+7.2%}")

    print("\n  trade log summary (as-specified):")
    tl = res.trades
    if len(tl):
        for side in ("short", "long"):
            s = tl[tl["side"] == side]
            if not len(s):
                print(f"    {side}: no trades")
                continue
            print(f"    {side}: n={len(s):3d}  win rate {(s['pnl']>0).mean():5.0%}  "
                  f"mean {s['pnl'].mean():+.2%}  median {s['pnl'].median():+.2%}  "
                  f"best {s['pnl'].max():+.2%}  worst {s['pnl'].min():+.2%}  "
                  f"mean hold {s['bars'].mean()*4:.0f}h")
        tl.to_csv(REPORTS / "band_trades_asspecified.csv", index=False)

    print("\n" + "=" * 210)
    print("D2. ROLLING ADAPTIVE BANDS -- what you actually do: re-read the VI+ high/low over")
    print("    the last ~6 months and put the rails at a relatively high / low value.")
    print("    Recomputed every bar from PRIOR bars only, so unlike section D there is no lookahead.")
    print("=" * 210)
    for mode, tag in (("quantile", "q86/q14"), ("range", "range 80/20")):
        for direction in ("fade", "follow"):
            run(frames, VB.BandParams(band_mode=mode, direction=direction),
                f"rolling {tag}, {direction}", rows)
        run(frames, VB.BandParams(band_mode=mode, pyramid_steps=0),
            f"rolling {tag}, fade, no pyramid", rows)
        run(frames, VB.BandParams(band_mode=mode, stop_atr=2.0),
            f"rolling {tag}, fade, 2xATR stop", rows)

    print("\n  event counts and forward returns by band definition:")
    for mode, tag in (("fixed", "fixed (your screenshot)"), ("quantile", "rolling q86/q14"),
                      ("range", "rolling range 80/20")):
        d = VB.build_frame(frames, VB.BandParams(band_mode=mode))
        o = d["open"].to_numpy()
        dd = d["double"]
        print(f"\n    {tag}")
        for kind, val in (("up-break (you SHORT)", 1), ("down-break (you LONG)", -1)):
            idx = [d.index.get_loc(t) for t in d.index[(dd != dd.shift(1)) & (dd == val)]]
            cells = []
            for h in (3, 6, 12, 30):
                rs = np.array([o[i + 1 + h] / o[i + 1] - 1 for i in idx if i + 1 + h < len(d)])
                if len(rs) < 5:
                    cells.append(f"{'--':>16}")
                    continue
                t = rs.mean() / (rs.std(ddof=1) / np.sqrt(len(rs)))
                cells.append(f"{rs.mean():+7.2%} (t{t:+4.1f})")
            print(f"      {kind:<24}n={len(idx):3d}   " + "  ".join(cells))

    print("\n" + "=" * 210)
    print("F. REGIME-DIRECTED: the band event is only a TRIGGER, the daily trend picks the side")
    print("=" * 210)
    run(frames, VB.BandParams(direction="regime"), "regime-directed", rows)
    run(frames, VB.BandParams(direction="regime", pyramid_steps=0), "  no pyramid", rows)
    run(frames, VB.BandParams(direction="regime", stop_atr=2.0), "  + 2xATR stop", rows)
    for q in (0.85, 0.80):
        pb = VB.percentile_bands(frames, 14, q, 1 - q)
        run(frames, VB.BandParams(**pb, direction="regime"),
            f"  q{q:.0%}/q{1-q:.0%} bands, regime-directed", rows)

    print("\n" + "=" * 210)
    print("G. STATISTICAL REALITY CHECK -- is the as-specified edge distinguishable from noise?")
    print("=" * 210)
    sh = tl[tl["side"] == "short"]["pnl"].to_numpy()
    t = sh.mean() / (sh.std(ddof=1) / np.sqrt(len(sh)))
    rng = np.random.default_rng(0)
    boot = np.array([rng.choice(sh, len(sh), replace=True).mean() for _ in range(20_000)])
    print(f"  short trades: n={len(sh)}  mean {sh.mean():+.2%}  t={t:+.2f}")
    print(f"  bootstrap 90% CI of the mean trade: "
          f"[{np.percentile(boot, 5):+.2%}, {np.percentile(boot, 95):+.2%}]")
    print(f"  P(true mean > 0) = {np.mean(boot > 0):.1%}   "
          "<- the ONE thing in this study that is statistically solid, and it is bad news")
    print(f"  win rate {np.mean(sh > 0):.0%}   median {np.median(sh):+.2%}   "
          f"mean win {sh[sh>0].mean():+.2%}   mean loss {sh[sh<0].mean():+.2%}")
    print(f"  worst three trades: " + ", ".join(f"{x:+.2%}" for x in np.sort(sh)[:3]))
    print(f"  drop the single worst trade -> mean {np.sort(sh)[1:].mean():+.2%} (still negative)")

    pd.DataFrame(rows).to_csv(REPORTS / "band_variants.csv", index=False)
    print(f"\n  wrote {REPORTS/'band_variants.csv'}")


if __name__ == "__main__":
    main()
