"""Daily MACD cross, long-only, as an equal-weight portfolio -- with an OOS test.

The single-symbol version was buried by costs at fast timeframes and, on daily
bars, beat buy-and-hold mainly on coins that collapsed.  A portfolio changes two
things that matter: it averages 26 independent signals instead of trusting one,
and it lets the rule sit in cash on some names while holding others.

Sizing is one instance of the rule per symbol, each on 1/N of capital, so
exposure is simply the fraction of symbols currently bullish.  That is what
running the supplied strategy 26 times side by side would actually do.  A
concentrated variant that splits capital only among the bullish names is shown
too, since it is the other reasonable reading.

The out-of-sample section is the part that decides anything.  183 symbols in this
repository took no part in building anything, so the rule -- which has no fitted
parameters at all, just MACD(12,26,9) -- can be dropped on them unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vibt import backtest as B, data as D, metrics as M, xsec as X  # noqa: E402

pd.set_option("display.width", 200)
REPORTS = Path(__file__).resolve().parent.parent / "reports"
FAST, SLOW, SIG = 12, 26, 9
ANN = 365.0

IN_SAMPLE = [
    "1000BONKUSDT", "1000FLOKIUSDT", "1000PEPEUSDT", "1000SHIBUSDT", "APTUSDT",
    "ARBUSDT", "BNBUSDT", "BTCUSDT", "DOGEUSDT", "DYDXUSDT", "ENAUSDT", "ETHUSDT",
    "FETUSDT", "HYPEUSDT", "LDOUSDT", "OPUSDT", "PENDLEUSDT", "RENDERUSDT",
    "SEIUSDT", "SOLUSDT", "SUIUSDT", "TAOUSDT", "TIAUSDT", "WIFUSDT", "WLDUSDT",
    "XRPUSDT",
]
NOT_COINS = ["USDCUSDT", "BTCDOMUSDT"]


def panels(symbols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Open-price panel and a long/flat signal panel, both on the shared index."""
    op, sig = {}, {}
    for s in symbols:
        d = D.load("1d", symbol=s)
        c = d["close"]
        macd = (c.ewm(span=FAST, adjust=False).mean()
                - c.ewm(span=SLOW, adjust=False).mean())
        line = macd.ewm(span=SIG, adjust=False).mean()
        op[s] = d["open"]
        sig[s] = (macd > line).astype(float).where(c.notna())
    P = pd.DataFrame(op).sort_index()
    S = pd.DataFrame(sig).reindex(P.index)
    return P, S


def portfolio(P: pd.DataFrame, S: pd.DataFrame, costs: B.Costs,
              concentrated: bool = False) -> B.Result:
    live = P.notna() & S.notna()
    if concentrated:
        # capital split among the bullish names only; fully invested when any fire
        n_on = S.where(live).sum(axis=1).replace(0, np.nan)
        w = S.where(live).div(n_on, axis=0).fillna(0.0)
    else:
        # one instance per symbol on 1/N of capital; exposure = share bullish
        n = live.sum(axis=1).replace(0, np.nan)
        w = S.where(live).div(n, axis=0).fillna(0.0)
    return X.run(P, w, costs, ANN, "", 1)


def hold(P: pd.DataFrame, scale: float = 1.0) -> pd.Series:
    live = P.notna()
    n = live.sum(axis=1).replace(0, np.nan)
    w = live.astype(float).div(n, axis=0).fillna(0.0) * scale
    return X.run(P, w, B.Costs(0, 0), ANN, "", 1).rets


def describe(name: str, rets: pd.Series, expo: float | None = None) -> dict:
    st = M.compute(rets, ANN)
    yrs = len(rets) / ANN
    return {"策略": name, "Sharpe": st.sharpe, "总收益": st.total_return,
            "年化": (1 + st.total_return) ** (1 / yrs) - 1 if st.total_return > -1 else -1.0,
            "波动": st.ann_vol, "回撤": st.max_dd,
            "平均仓位": expo if expo is not None else np.nan}


def show(rows: list[dict]) -> None:
    d = pd.DataFrame(rows)
    for c in ("总收益", "年化", "波动", "回撤", "平均仓位"):
        d[c] = d[c].map(lambda v: f"{v:+.1%}" if pd.notna(v) else "")
    d["Sharpe"] = d["Sharpe"].map(lambda v: f"{v:+.2f}")
    print(d.to_string(index=False))


def block(symbols: list[str], label: str) -> pd.Series:
    P, S = panels(symbols)
    costs = B.Costs(4.5, 2.0)
    print("\n" + "=" * 130)
    print(f"{label}   {len(symbols)} 个币   {P.index[0].date()} -> {P.index[-1].date()}")
    print("=" * 130 + "\n")

    r_eq = portfolio(P, S, costs)
    r_cn = portfolio(P, S, costs, concentrated=True)
    # the engine's own held weights, not the signal panel: they already carry the
    # execution lag, so the exposure quoted here is the one actually financed
    expo = float(r_eq.weights.abs().sum(axis=1).mean())
    bh = hold(P)

    rows = [describe("MACD 日线 等权(每币1/N)", r_eq.rets, expo),
            describe("MACD 日线 集中(只分给做多的)", r_cn.rets, 1.0),
            describe("等权买入持有", bh, 1.0),
            describe(f"等权持有 x{expo:.2f}（等仓位对照）", bh * expo, expo)]
    show(rows)

    beta, alpha = X.beta_to(r_eq.rets, bh)
    p5, _, p95 = M.block_bootstrap_sharpe(r_eq.rets, ANN, block=20)
    print(f"\n  等权版对大盘: beta {beta:+.3f}   年化 alpha {alpha:+.2%}")
    print(f"  Sharpe 的 block bootstrap 95% 区间: [{p5:+.2f}, {p95:+.2f}]")
    print(f"  年换手 {r_eq.stats.turnover_ann:.1f}   "
          f"年化成本 {r_eq.costs.sum()/(len(r_eq.rets)/ANN):.2%}")
    return r_eq.rets


def main() -> None:
    every = [s for s in D.available_symbols(require=("1d",)) if s not in NOT_COINS]
    ins = [s for s in every if s in IN_SAMPLE]
    oos = [s for s in every if s not in IN_SAMPLE]

    print("=" * 130)
    print(f"MACD({FAST},{SLOW},{SIG}) 日线 金叉做多 / 死叉空仓   等权组合")
    print("=" * 130)
    print("""
  规则零拟合参数：只有 MACD 的 12/26/9，全部是原策略的默认值。
  信号在日线收盘确认，次日开盘成交。成本 6.5bp/边。
""")

    r_ins = block(ins, "① 样本内（开发用过的 26 个）")
    r_oos = block(oos, "② 样本外（183 个，从未参与开发）")
    block(every, "③ 全部合并")

    # ------------------------------------------------------------------ annual
    print("\n" + "=" * 130)
    print("逐年（样本外 183 币，等权版）")
    print("=" * 130 + "\n")
    P, S = panels(oos)
    bh = hold(P)
    tab = pd.DataFrame({"MACD策略": r_oos, "等权买入持有": bh}).dropna()
    yr = tab.groupby(tab.index.year).apply(
        lambda g: pd.Series({c: (1 + g[c]).prod() - 1 for c in g.columns}),
        include_groups=False)
    yr["超额"] = yr["MACD策略"] - yr["等权买入持有"]
    print(yr.to_string(float_format=lambda v: f"{v:+.1%}"))

    # ------------------------------------------------------------------ costs
    print("\n" + "=" * 130)
    print("成本敏感度（样本外 183 币，等权版）")
    print("=" * 130 + "\n")
    print(f"  {'成本/边':<12}{'Sharpe':>9}{'总收益':>11}{'年化':>9}")
    for fee, slip in ((0, 0), (2, 1), (4.5, 2), (4.5, 8), (15, 15)):
        c = B.Costs(fee, slip)
        r = portfolio(P, S, c)
        yrs = len(r.rets) / ANN
        t = r.stats.total_return
        print(f"  {c.per_side*1e4:>6.1f}bp{'':<4}{r.stats.sharpe:>+9.2f}{t:>+11.1%}"
              f"{(1+t)**(1/yrs)-1:>+9.1%}")

    REPORTS.mkdir(exist_ok=True)
    pd.DataFrame({"oos": r_oos, "ins": r_ins}).to_csv(REPORTS / "macd_portfolio.csv")
    print(f"\n  wrote {REPORTS/'macd_portfolio.csv'}")


if __name__ == "__main__":
    main()
