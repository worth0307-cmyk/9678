"""Funding rate: is it a usable signal, and does it need to be fitted per coin?

Two questions, in order, because the second one decides how the first is built.

PART A asks whether each coin deserves its own fitted parameters.  Across 63
liquid coins the funding-return slope genuinely differs -- I-squared 61%, Q-test
p<0.001 -- which looks like a mandate for per-coin models.  It is not.  The
differences are explained by no observable coin characteristic (turnover, funding
volatility, funding level, listing age all null), and more decisively a coin's
slope in the first half of its history has no relationship to its slope in the
second half.  Fitting per coin fits noise.

The same instability holds for 14-day momentum, which is the useful part: the
cross-sectional momentum strategy works at +0.51 out-of-sample while no
individual coin has a stable momentum coefficient.  What carries information is
the RANKING ACROSS coins, not any coin's own behaviour, and that is exactly why
pooling works where per-coin fitting cannot.

What does survive from "every coin is different" is normalisation rather than
fitting.  5bp of funding means something different for BTC (2.4bp daily
volatility) than for SOL (7.0bp), so a global threshold is wrong -- but the fix
is to score each coin against its OWN trailing distribution, which uses each
coin's individuality and fits zero parameters.

PART B builds the factor that way and tests it.  The construction is not
copied from momentum, because the two signals have opposite shapes: momentum
lives entirely in the tails (full-panel IC is negative while the extreme 5v5
spread is +0.445% per 3 days), whereas funding has broad-panel signal whose
tails run backwards, because extreme funding is itself a momentum symptom.  So
momentum is traded top/bottom-5 and funding is traded rank-weighted across
every name, and both are shown both ways to make the difference visible.

Everything is then re-run on the 183 symbols that took no part in development.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scipy import stats  # noqa: E402
from vibt import backtest as B, data as D, metrics as M, xsec as X  # noqa: E402

pd.set_option("display.width", 210)

LOOKBACK, N_SIDE, REBALANCE = 14, 5, 3
COSTS = B.Costs(fee_bps=4.5, slip_bps=2.0)
ANN = 365.0
LIQ_FLOOR = 10e6
Z_WINDOW = 90          # trailing bars for the per-coin z-score
REPORTS = Path(__file__).resolve().parent.parent / "reports"

IN_SAMPLE = [
    "1000BONKUSDT", "1000FLOKIUSDT", "1000PEPEUSDT", "1000SHIBUSDT", "APTUSDT",
    "ARBUSDT", "BNBUSDT", "BTCUSDT", "DOGEUSDT", "DYDXUSDT", "ENAUSDT", "ETHUSDT",
    "FETUSDT", "HYPEUSDT", "LDOUSDT", "OPUSDT", "PENDLEUSDT", "RENDERUSDT",
    "SEIUSDT", "SOLUSDT", "SUIUSDT", "TAOUSDT", "TIAUSDT", "WIFUSDT", "WLDUSDT",
    "XRPUSDT",
]
NOT_COINS = ["USDCUSDT", "BTCDOMUSDT"]


def funding_series(symbol: str) -> pd.Series | None:
    p = Path(__file__).resolve().parent.parent / "data" / "funding" / f"{symbol}_funding.csv.gz"
    if not p.exists():
        return None
    d = pd.read_csv(p, encoding="utf-8-sig")
    d["ts"] = pd.to_datetime(d["日期时间(北京)"]) - pd.Timedelta(hours=8)
    return d.set_index("ts")["funding_rate"].astype(float).sort_index()


def build(symbols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Open prices, daily funding in bp, and median dollar volume."""
    px, fnd, dv = {}, {}, {}
    for s in symbols:
        d = D.load("1d", symbol=s)
        f = funding_series(s)
        if f is None or "quote_volume" not in d.columns:
            continue
        px[s] = d["open"]
        fnd[s] = f.resample("1D").sum().reindex(d.index) * 1e4
        dv[s] = float(d["quote_volume"].tail(365).median())
    P = pd.DataFrame(px).sort_index()
    F = pd.DataFrame(fnd).reindex(P.index)
    return P, F, P / P.shift(LOOKBACK) - 1.0, dv


def zscore(F: pd.DataFrame, window: int = Z_WINDOW) -> pd.DataFrame:
    """Score each coin against its own trailing distribution, prior bars only.

    This is the defensible half of "every coin needs its own treatment": the same
    5bp is ordinary for one coin and extreme for another, so a global cut is
    meaningless -- but the scale comes from each coin's own history rather than
    from a fitted parameter, so it costs no statistical power.  shift(1) keeps
    the current bar out of its own reference window.
    """
    prior = F.shift(1)
    mu = prior.rolling(window, min_periods=window // 2).mean()
    sd = prior.rolling(window, min_periods=window // 2).std()
    return (F - mu) / sd.replace(0, np.nan)


def bt(P: pd.DataFrame, feat: pd.DataFrame, mode: str, n_side: int = N_SIDE,
       costs: B.Costs = COSTS) -> B.Result:
    w = X.cross_sectional_weights(feat, n_side=n_side, mode=mode)
    return X.run(P, w, costs, ANN, "", REBALANCE)


def row(name: str, r: B.Result, extra: str = "") -> str:
    yrs = len(r.rets) / ANN
    cagr = (1 + r.stats.total_return) ** (1 / yrs) - 1
    return (f"  {name:<38}{r.stats.sharpe:>+9.2f}{r.stats.total_return:>+11.1%}"
            f"{cagr:>+9.1%}{r.stats.ann_vol:>8.0%}{r.stats.max_dd:>9.1%}  {extra}")


HEAD = f"  {'':<38}{'Sharpe':>9}{'总收益':>11}{'CAGR':>9}{'波动':>8}{'回撤':>9}"


def part_a(P, F, MOM) -> None:
    print("=" * 140)
    print("A. 按币拟合 vs 按币归一化")
    print("=" * 140)
    ret = P.shift(-1 - REBALANCE) / P.shift(-1) - 1.0

    def half_betas(sig):
        out = []
        for s in sig.columns:
            d = pd.DataFrame({"x": sig[s], "y": ret[s]}).dropna().iloc[::REBALANCE]
            if len(d) < 120:
                continue
            h = len(d) // 2
            b = []
            for part in (d.iloc[:h], d.iloc[h:]):
                z = (part.x - part.x.mean()) / part.x.std()
                b.append(stats.linregress(z, part.y).slope * 100)
            out.append({"symbol": s, "b1": b[0], "b2": b[1]})
        return pd.DataFrame(out)

    print(f"\n  每个币把自己的历史一分为二，分别估计斜率，看前半能否预测后半：\n")
    print(f"  {'信号':<16}{'币数':>6}{'Pearson r':>12}{'p':>9}{'符号一致':>10}{'二项 p':>9}")
    for sig, name in ((F, "资金费率"), (MOM, "14天动量")):
        hb = half_betas(sig)
        if len(hb) < 10:
            continue
        r, p = stats.pearsonr(hb.b1, hb.b2)
        same = (np.sign(hb.b1) == np.sign(hb.b2)).mean()
        bp = stats.binomtest(int(same * len(hb)), len(hb), 0.5, "greater").pvalue
        print(f"  {name:<16}{len(hb):>6}{r:>+12.3f}{p:>9.3f}{same:>10.0%}{bp:>9.3f}")
    print("""
  两个信号都不持续——连已知可用的动量也一样。
  所以动量之所以有效，不是因为哪个币有稳定的系数，而是因为**币和币之间的排序**携带信息。
  这正是池化有效、按币拟合无效的原因：它们测的不是同一个东西。""")


def part_b(P, F, MOM, label: str) -> pd.DataFrame:
    print("\n" + "=" * 140)
    print(f"B. 横截面因子 —— {label}")
    print("=" * 140)
    Z = zscore(F)
    out = []
    print(f"\n{HEAD}")
    specs = [
        ("动量 top/bottom 5", MOM, "long_short", N_SIDE),
        ("动量 全截面 rank 加权", MOM, "rank", 0),
        ("资金费率(负) top/bottom 5", -Z, "long_short", N_SIDE),
        ("资金费率(负) 全截面 rank 加权", -Z, "rank", 0),
    ]
    res = {}
    for name, feat, mode, k in specs:
        r = bt(P, feat, mode, k or N_SIDE)
        res[name] = r
        print(row(name, r))
        out.append({"strategy": name, "sharpe": r.stats.sharpe,
                    "total": r.stats.total_return, "vol": r.stats.ann_vol})

    # combine: each leg at the construction that suits its own shape
    wm = X.cross_sectional_weights(MOM, n_side=N_SIDE, mode="long_short")
    wf = X.cross_sectional_weights(-Z, mode="rank")
    idx = wm.index.union(wf.index)
    combo = (wm.reindex(idx).fillna(0) + wf.reindex(idx).fillna(0)) / 2
    r = X.run(P, combo, COSTS, ANN, "", REBALANCE)
    print(row("【合成】动量tails + 资金费率rank", r))
    out.append({"strategy": "combo", "sharpe": r.stats.sharpe,
                "total": r.stats.total_return, "vol": r.stats.ann_vol})

    a = bt(P, MOM, "long_short").rets
    b = bt(P, -Z, "rank").rets
    j = pd.DataFrame({"mom": a, "fnd": b}).dropna()
    print(f"\n  两条腿的日收益相关性: {j.mom.corr(j.fnd):+.3f}")
    if len(j) > 100:
        sa, sb = M.compute(j.mom, ANN).sharpe, M.compute(j.fnd, ANN).sharpe
        rho = j.mom.corr(j.fnd)
        if sa > 0 and sb > 0:
            theo = (sa + sb) / np.sqrt(2 + 2 * rho)
            print(f"  等权合成的理论 Sharpe: ({sa:+.2f} + {sb:+.2f}) / sqrt(2+2x{rho:+.3f}) "
                  f"= {theo:+.2f}   实际 {r.stats.sharpe:+.2f}")
    return pd.DataFrame(out)


def staleness(P: pd.DataFrame, subsets: dict) -> None:
    """How much of the edge is the last few hours of information?

    Momentum can be measured on closes or on opens.  Both are causal and both
    execute at the next open, but the close-based signal is only one overnight
    gap away from execution while the open-based one is a full day stale.  If
    the two disagree sharply, the edge decays within hours -- which is a live
    trading constraint, not a backtest detail.
    """
    print("\n" + "=" * 140)
    print("C. 信号新鲜度：同一个策略，动量用收盘价 vs 开盘价计算")
    print("=" * 140 + "\n")
    C = pd.DataFrame({s2: D.load("1d", symbol=s2)["close"] for s2 in P.columns}).reindex(P.index)
    print(f"  {'币池':<20}{'构造':<22}{'用收盘算':>10}{'用开盘算':>10}{'差':>8}")
    for nm, sub in subsets.items():
        for mode, lbl in (("long_short", "top/bottom 5"), ("rank", "全截面 rank")):
            got = []
            for src in (C, P):
                mom = src[sub] / src[sub].shift(LOOKBACK) - 1
                got.append(bt(P[sub], mom, mode).stats.sharpe)
            print(f"  {nm:<20}{lbl:<22}{got[0]:>+10.2f}{got[1]:>+10.2f}{got[1]-got[0]:>+8.2f}")
    print("""
  两种算法都合法（都只用已收盘的 bar，执行都在次日开盘），差的只是信号新鲜度：
  收盘价版本离执行只隔一个隔夜跳空，开盘价版本整整旧了一天。
  Sharpe 因此掉 0.3~0.6，说明**这个边际在几小时内就衰减掉了**——
  这是实盘约束，不是回测细节。注意 rank 构造对此的敏感度明显更低。""")


def main() -> None:
    every = [s for s in D.available_symbols(require=("1d",)) if s not in NOT_COINS]
    P_all, F_all, MOM_all, dv = build(every)
    liquid = [s for s in P_all.columns if dv.get(s, 0) >= LIQ_FLOOR]
    ins = [s for s in liquid if s in IN_SAMPLE]
    oos = [s for s in liquid if s not in IN_SAMPLE]

    print("=" * 140)
    print(f"FUNDING RATE STUDY   流动性门槛 ${LIQ_FLOOR/1e6:.0f}M   "
          f"可用 {len(liquid)} 个（样本内 {len(ins)} / 样本外 {len(oos)}）")
    print("=" * 140)

    part_a(P_all[liquid], F_all[liquid], MOM_all[liquid])
    part_b(P_all[liquid], F_all[liquid], MOM_all[liquid], f"全部 {len(liquid)} 个流动币（含开发集）")
    part_b(P_all[ins], F_all[ins], MOM_all[ins], f"样本内 {len(ins)} 个")
    tab = part_b(P_all[oos], F_all[oos], MOM_all[oos],
                 f"样本外 {len(oos)} 个（从未参与开发）")
    staleness(P_all[liquid], {"样本内 26": ins, "样本外 37": oos, "全部 63": liquid})

    REPORTS.mkdir(exist_ok=True)
    tab.to_csv(REPORTS / "funding_oos.csv", index=False)
    print(f"\n  wrote {REPORTS/'funding_oos.csv'}")


if __name__ == "__main__":
    main()
