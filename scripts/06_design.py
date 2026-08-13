"""Design study: what actually contributes, measured by ablation.

Question 1: is VI adding anything a moving average does not, at matched exposure?
Question 2: does an ensemble of filters beat any single filter?
Question 3: what does the vol-target overlay contribute on its own?
Question 4: is the short book worth carrying?
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vibt import backtest as B, data as D, indicators as I, metrics as M, strategies as S  # noqa: E402

pd.set_option("display.width", 220)

COSTS = B.Costs(fee_bps=4.5, slip_bps=2.0)
SPLIT = pd.Timestamp("2025-07-01")
REPORTS = Path(__file__).resolve().parent.parent / "reports"


def report(name: str, df: pd.DataFrame, sig: pd.Series, ann: float,
           bench: B.Result, rows: list) -> B.Result:
    res = B.run(df, sig, COSTS, ann, name, ann / 365.0)
    r = res.rets
    is_r = r[r.index < SPLIT]
    oos_r = r[r.index >= SPLIT]
    st_is, st_oos = M.compute(is_r, ann), M.compute(oos_r, ann)
    b = bench.rets
    up = b > 0
    dn = b < 0
    uc = r[up].sum() / b[up].sum()
    dc = r[dn].sum() / b[dn].sum()
    rows.append({
        "name": name, "sharpe": res.stats.sharpe, "ret": res.stats.total_return,
        "cagr": res.stats.cagr, "vol": res.stats.ann_vol, "maxdd": res.stats.max_dd,
        "calmar": res.stats.calmar, "expo": res.stats.exposure, "trades": res.stats.trades,
        "sharpe_is": st_is.sharpe, "ret_is": st_is.total_return,
        "sharpe_oos": st_oos.sharpe, "ret_oos": st_oos.total_return, "dd_oos": st_oos.max_dd,
        "up_cap": uc, "down_cap": dc,
    })
    print(f"  {name:<44} Sh {res.stats.sharpe:+5.2f} | ret {res.stats.total_return:+7.1%} | "
          f"DD {res.stats.max_dd:6.1%} | Cal {res.stats.calmar:+5.2f} | "
          f"IS {st_is.sharpe:+5.2f}/{st_is.total_return:+7.1%} | "
          f"OOS {st_oos.sharpe:+5.2f}/{st_oos.total_return:+7.1%} | "
          f"up {uc:+5.1%} dn {dc:+6.1%} | expo {res.stats.exposure:4.0%} | trd {res.stats.trades:3d}")
    return res


def filters(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Binary long-permission filters, all roughly 50% duty cycle."""
    c = df["close"]
    return {
        "VI(14)": (I.vortex(df, 14)["vi_spread"] > 0).astype(float),
        "VI(30)": (I.vortex(df, 30)["vi_spread"] > 0).astype(float),
        "price>SMA100": (c > I.sma(c, 100)).astype(float),
        "EMA50>EMA200": (I.ema(c, 50) > I.ema(c, 200)).astype(float),
        "Donchian50 mid": (c > I.donchian(df, 50)["dc_mid"]).astype(float),
        "ROC(60)>0": (I.roc(c, 60) > 0).astype(float),
    }


def main() -> None:
    frames = D.load_all()
    df = frames["1d"]
    ann = D.bars_per_year("1d")
    bench = B.buy_and_hold(df, ann, COSTS)
    rows: list[dict] = []

    print("=" * 210)
    print("Q1. SINGLE LONG-ONLY FILTERS AT MATCHED (~50%) EXPOSURE -- is VI special?")
    print("=" * 210)
    print(f"  {'buy & hold':<44} Sh {bench.stats.sharpe:+5.2f} | ret {bench.stats.total_return:+7.1%} | "
          f"DD {bench.stats.max_dd:6.1%} | Cal {bench.stats.calmar:+5.2f}")
    F = filters(df)
    for nm, f in F.items():
        report(f"long-only: {nm}", df, f, ann, bench, rows)

    print("\n" + "=" * 210)
    print("Q2. ENSEMBLE -- exposure = share of filters that agree")
    print("=" * 210)
    vote = pd.concat(F.values(), axis=1).mean(axis=1)
    report("ensemble raw (0..1)", df, vote, ann, bench, rows)
    report("ensemble step 1/3", df, S.discretise(vote, 1 / 3), ann, bench, rows)
    report("ensemble >=1/2 binary", df, (vote >= 0.5).astype(float), ann, bench, rows)
    report("ensemble >=2/3 binary", df, (vote >= 2 / 3).astype(float), ann, bench, rows)

    print("\n" + "=" * 210)
    print("Q3. VOL-TARGET OVERLAY -- what does risk sizing alone buy you?")
    print("=" * 210)
    pv = I.parkinson_vol(df, 30, ann)
    print(f"  realised 30d Parkinson vol: p10={pv.quantile(.1):.0%} median={pv.median():.0%} "
          f"p90={pv.quantile(.9):.0%}")
    for tgt in (0.30, 0.40, 0.50):
        lev = (tgt / pv).clip(0, 1.0)
        report(f"B&H + vol target {tgt:.0%} (cap 1x)", df, S.discretise(lev, 0.2), ann, bench, rows)
    for tgt in (0.30, 0.40, 0.50):
        sig = S.discretise(vote * (tgt / pv).clip(0, 1.0), 0.2)
        report(f"ensemble + vol target {tgt:.0%}", df, sig, ann, bench, rows)

    print("\n" + "=" * 210)
    print("Q4. IS THE SHORT BOOK WORTH CARRYING?")
    print("=" * 210)
    bear = (vote <= 0.0).astype(float)          # every filter bearish
    bear_confirmed = ((vote <= 0.0) & (df["close"] < I.sma(df["close"], 200))).astype(float)
    lev = (0.40 / pv).clip(0, 1.0)
    base = S.discretise(vote * lev, 0.2)
    for size in (0.0, 0.25, 0.5, 1.0):
        sig = base - S.discretise(bear_confirmed * lev * size, 0.2)
        report(f"ensemble + vol tgt, short size {size:.2f}", df, sig, ann, bench, rows)
    for size in (0.5, 1.0):
        sig = base - S.discretise(bear * lev * size, 0.2)
        report(f"  (unconfirmed bear) short size {size:.2f}", df, sig, ann, bench, rows)

    print("\n" + "=" * 210)
    print("Q5. COST SENSITIVITY of the leading candidate (ensemble + 40% vol target)")
    print("=" * 210)
    cand = S.discretise(vote * (0.40 / pv).clip(0, 1.0), 0.2)
    for fee, slip in ((0.0, 0.0), (2.0, 1.0), (4.5, 2.0), (4.5, 5.0), (10.0, 10.0)):
        c = B.Costs(fee_bps=fee, slip_bps=slip)
        res = B.run(df, cand, c, ann, "", 1.0)
        print(f"  {fee:>4.1f}bp fee + {slip:>4.1f}bp slip  -> Sharpe {res.stats.sharpe:+5.2f}  "
              f"ret {res.stats.total_return:+7.1%}  (annual cost drag "
              f"{res.costs.sum() / (len(res.rets) / ann):.2%})")

    print("\n" + "=" * 210)
    print("Q6. EXECUTION-LAG SENSITIVITY -- how fragile is the signal to being slow?")
    print("=" * 210)
    for lag in (0, 1, 2, 3):
        res = B.run(df, cand.shift(lag).fillna(0.0), COSTS, ann, "", 1.0)
        print(f"  extra lag {lag} day(s) -> Sharpe {res.stats.sharpe:+5.2f}  "
              f"ret {res.stats.total_return:+7.1%}  DD {res.stats.max_dd:6.1%}")

    pd.DataFrame(rows).to_csv(REPORTS / "design_ablation.csv", index=False)


if __name__ == "__main__":
    main()
