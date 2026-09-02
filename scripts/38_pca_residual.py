"""主成分对冲、交易残差 -- 在这个仓库的 207 个币上验一遍.

The framework in the video is the standard institutional decomposition: data,
alpha model, risk model, cost model, execution model.  The one piece of it that
this repository has measured but never acted on is the risk model, and the video
names the exact tool -- hedge the principal components, trade what is left.

For crypto that is not a refinement, it is the whole problem.  An earlier run
here found roughly 1.6 effective independent bets across 26 coins, which is
another way of saying almost every position is the same position.  If that is
true then a cross-sectional signal is mostly a market-timing signal wearing a
disguise, and the only way to find out is to remove the common factors and look
at the remainder.

So this file does three things in order:

  ① how much of crypto is one factor, and what that factor is
  ② what is left after hedging it out, and whether the remainder mean-reverts
  ③ whether trading that remainder survives costs, an out-of-sample split, and
     a control -- the same bar every other idea in this repository had to clear

The PCA is fitted on a ROLLING PAST WINDOW, never on the full sample.  Fitting
components on all the data and then "trading the residual" is one of the most
common ways to produce a beautiful backtest of nothing: the loadings already
know which coins were about to diverge.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vibt import backtest as B, data as D, metrics as M, xsec as X  # noqa: E402

pd.set_option("display.width", 240)
REPORTS = Path(__file__).resolve().parent.parent / "reports"
ANN = 365.0
NOT_COINS = ("USDCUSDT", "BTCDOMUSDT")
IN_SAMPLE = [
    "1000BONKUSDT", "1000FLOKIUSDT", "1000PEPEUSDT", "1000SHIBUSDT", "APTUSDT",
    "ARBUSDT", "BNBUSDT", "BTCUSDT", "DOGEUSDT", "DYDXUSDT", "ENAUSDT", "ETHUSDT",
    "FETUSDT", "HYPEUSDT", "LDOUSDT", "OPUSDT", "PENDLEUSDT", "RENDERUSDT",
    "SEIUSDT", "SOLUSDT", "SUIUSDT", "TAOUSDT", "TIAUSDT", "WIFUSDT", "WLDUSDT",
    "XRPUSDT",
]
LOOKBACK = 120      # bars of history the PCA is fitted on
MIN_NAMES = 30      # a cross-section smaller than this is not a cross-section


def panels() -> tuple[pd.DataFrame, pd.DataFrame]:
    syms = [s for s in D.available_symbols(require=("1d",)) if s not in NOT_COINS]
    op, cl, qv = {}, {}, {}
    for s in syms:
        d = D.load("1d", symbol=s)
        op[s], cl[s] = d["open"], d["close"]
        if "quote_volume" in d:
            qv[s] = d["quote_volume"]
    P = pd.DataFrame(op).sort_index()
    C = pd.DataFrame(cl).reindex(P.index)
    V = pd.DataFrame(qv).reindex(P.index)
    return P, C, V


def effective_bets(corr: np.ndarray) -> float:
    """n / (1 + (n-1)*rho_bar) -- how many independent positions you really hold.

    Once the average correlation is at or below zero the formula's denominator
    goes non-positive and the quantity stops meaning anything; the honest answer
    there is "all n of them", so it is capped rather than returned as nan.
    """
    n = corr.shape[0]
    iu = np.triu_indices(n, 1)
    rho = np.nanmean(corr[iu])
    denom = 1 + (n - 1) * rho
    return min(float(n), n / denom) if denom > 0 else float(n)


def main() -> None:
    P, C, V = panels()
    R = np.log(C).diff()                      # log returns, for the factor algebra
    R = R.replace([np.inf, -np.inf], np.nan)

    print("=" * 150)
    print("① 币圈里「一个因子」占多少")
    print("=" * 150)

    # a window where enough coins are live at once
    live = R.notna().sum(axis=1)
    idx = R.index[live >= MIN_NAMES]
    sub = R.loc[idx]
    rows = []
    for label, seg in (("全样本", sub),
                       ("2023", sub.loc["2023"]), ("2024", sub.loc["2024"]),
                       ("2025", sub.loc["2025"]), ("2026", sub.loc["2026"])):
        seg = seg.dropna(axis=1, thresh=int(0.8 * len(seg))).dropna()
        if len(seg) < 60 or seg.shape[1] < MIN_NAMES:
            continue
        Z = (seg - seg.mean()) / seg.std()
        cov = np.cov(Z.to_numpy(), rowvar=False)
        w = np.sort(np.linalg.eigvalsh(cov))[::-1]
        w = w / w.sum()
        corr = np.corrcoef(seg.to_numpy(), rowvar=False)
        iu = np.triu_indices(corr.shape[0], 1)
        rows.append({"期间": label, "币数": seg.shape[1], "天数": len(seg),
                     "PC1": w[0], "PC2": w[1], "PC3": w[2],
                     "PC1-3合计": w[:3].sum(),
                     "平均相关": float(np.nanmean(corr[iu])),
                     "有效独立赌注": effective_bets(corr)})
    t = pd.DataFrame(rows)
    o = t.copy()
    for c in ("PC1", "PC2", "PC3", "PC1-3合计", "平均相关"):
        o[c] = o[c].map(lambda v: f"{v:.1%}")
    o["有效独立赌注"] = o["有效独立赌注"].map(lambda v: f"{v:.2f}")
    print("\n" + o.to_string(index=False))

    # what IS pc1
    seg = sub.dropna(axis=1, thresh=int(0.8 * len(sub))).dropna()
    Z = (seg - seg.mean()) / seg.std()
    cov = np.cov(Z.to_numpy(), rowvar=False)
    ev, evec = np.linalg.eigh(cov)
    pc1 = pd.Series(Z.to_numpy() @ evec[:, -1], index=seg.index)
    eqw = seg.mean(axis=1)
    print(f"""
  PC1 和「等权大盘」的相关性: {pc1.corr(eqw):+.3f}
  PC1 和 BTC 的相关性:        {pc1.corr(seg['BTCUSDT']) if 'BTCUSDT' in seg else float('nan'):+.3f}
  PC1 载荷的符号: 正 {int((evec[:, -1] > 0).sum())} / 负 {int((evec[:, -1] < 0).sum())}

  载荷几乎全部同号、且与等权大盘高度相关 —— **PC1 就是「币圈整体」**。
  「有效独立赌注」那一列是这件事的实际含义：名义上持有几十个币，
  实际上只押了 1~2 个独立的注。
""")

    # ------------------------------------------------------------------ ② 残差
    print("=" * 150)
    print("② 把主成分对冲掉，剩下什么")
    print("=" * 150)
    print(f"""
  做法：每天用**过去 {LOOKBACK} 天**（不含当天）做 PCA，取前 k 个主成分，
  把当天的横截面收益对这些载荷回归，残差就是「剥掉共同因子之后」的部分。
  载荷只用过去的数据估计——用全样本估载荷再去交易残差，是最经典的一种假回测。
""")

    Rn = R.copy()
    resid = {k: pd.DataFrame(index=R.index, columns=R.columns, dtype=float)
             for k in (1, 3, 5)}
    dates = [d for d in R.index if R.loc[d].notna().sum() >= MIN_NAMES]
    start = R.index.get_loc(dates[0])
    for i in range(max(start, LOOKBACK), len(R)):
        d = R.index[i]
        hist = Rn.iloc[i - LOOKBACK:i]
        cols = hist.columns[hist.notna().all() & Rn.iloc[i].notna()]
        if len(cols) < MIN_NAMES:
            continue
        H = hist[cols].to_numpy()
        H = H - H.mean(axis=0)
        sd = H.std(axis=0)
        sd[sd == 0] = np.nan
        Hz = H / sd
        cov = np.cov(Hz, rowvar=False)
        if not np.isfinite(cov).all():
            continue
        ev, evec = np.linalg.eigh(cov)
        today = ((Rn.iloc[i][cols].to_numpy() - hist[cols].mean().to_numpy()) / sd)
        for k in resid:
            L = evec[:, -k:]                      # top-k loadings
            beta = L.T @ today                    # today's factor scores
            resid[k].loc[d, cols] = today - L @ beta
    for k in resid:
        resid[k] = resid[k].astype(float)

    rows = []
    for k, Rk in resid.items():
        r = Rk.dropna(how="all")
        if not len(r):
            continue
        common = r.notna().sum(axis=1) >= MIN_NAMES
        rr = r[common]
        corr = rr.corr().to_numpy()
        iu = np.triu_indices(corr.shape[0], 1)
        rows.append({"对冲掉的主成分数": k, "天数": len(rr),
                     "残差平均相关": float(np.nanmean(corr[iu])),
                     "残差有效独立赌注": effective_bets(np.nan_to_num(corr, nan=0.0)),
                     "残差1日自相关": float(rr.apply(
                         lambda c: c.autocorr(1) if c.notna().sum() > 60 else np.nan).mean()),
                     "残差5日自相关": float(rr.apply(
                         lambda c: c.autocorr(5) if c.notna().sum() > 60 else np.nan).mean())})
    # the same two numbers on RAW returns, as the thing to compare against
    raw = R.loc[resid[1].dropna(how="all").index]
    corr = raw.corr().to_numpy()
    iu = np.triu_indices(corr.shape[0], 1)
    rows.insert(0, {"对冲掉的主成分数": 0, "天数": len(raw),
                    "残差平均相关": float(np.nanmean(corr[iu])),
                    "残差有效独立赌注": effective_bets(np.nan_to_num(corr, nan=0.0)),
                    "残差1日自相关": float(raw.apply(
                        lambda c: c.autocorr(1) if c.notna().sum() > 60 else np.nan).mean()),
                    "残差5日自相关": float(raw.apply(
                        lambda c: c.autocorr(5) if c.notna().sum() > 60 else np.nan).mean())})
    t2 = pd.DataFrame(rows)
    o = t2.copy()
    for c in ("残差平均相关", "残差1日自相关", "残差5日自相关"):
        o[c] = o[c].map(lambda v: f"{v:+.3f}")
    o["残差有效独立赌注"] = o["残差有效独立赌注"].map(lambda v: f"{v:.1f}")
    print(o.to_string(index=False))
    print("""
  「自相关」是残差能不能交易的关键：负的自相关 = 均值回复 = 昨天跌多了今天倾向反弹。
  如果对冲之后自相关还是接近 0，那残差就只是噪声，没什么可交易的。
""")

    # ------------------------------------------------------------------ ③ 交易
    print("=" * 150)
    print("③ 交易残差：昨天残差为负的做多，为正的做空（横截面 rank 加权）")
    print("=" * 150)
    print("""
  仓位构造沿用本仓库样本外检验里胜出的那一版：**全截面 rank 加权**、
  美元中性、按前一天的信号在次日开盘成交、成本 6.5bp/边。
  流动性下限 $10M —— 这是之前样本外检验里预注册过的门槛，不是这次挑的。
""")
    liq = V.rolling(30).median()
    ok = (liq >= 10e6)

    def run_resid(sig: pd.DataFrame, label: str, syms: list[str] | None = None,
                  costs: B.Costs | None = None) -> tuple[str, B.Result]:
        s = sig.copy()
        if syms is not None:
            s = s[[c for c in s.columns if c in syms]]
        s = s.where(ok.reindex_like(s).fillna(False))
        # mode="rank" uses EVERY name; the default (long_short, n_side=2) would
        # trade only the two most extreme residuals a day, which on this panel
        # means the two biggest movers -- concentration, not a cross-section
        w = X.cross_sectional_weights(-s, mode="rank", gross=1.0)
        return label, X.run(P[w.columns], w, costs or B.Costs(4.5, 2.0), ANN, label, 1)

    rows = []
    curves = {}
    oos = [s for s in P.columns if s not in IN_SAMPLE]
    for k, Rk in resid.items():
        for label, syms in ((f"残差(对冲{k}个PC) 全部", None),
                            (f"残差(对冲{k}个PC) 样本外", oos)):
            lb, res = run_resid(Rk, label, syms)
            if res.stats.sharpe == res.stats.sharpe:
                rows.append({"策略": lb, "币数": int(res.weights.ne(0).sum(axis=1).mean()),
                             "Sharpe": res.stats.sharpe,
                             "年化": (1 + res.stats.total_return) **
                                     (ANN / len(res.rets)) - 1,
                             "波动": res.stats.ann_vol, "回撤": res.stats.max_dd,
                             "年换手": res.stats.turnover_ann,
                             "年化成本": res.costs.sum() / (len(res.rets) / ANN)})
                curves[lb] = res
    # the control: same construction on RAW returns, no hedge at all
    lb, res = run_resid(R.loc[resid[1].index], "对照:原始收益(不对冲)")
    rows.append({"策略": lb, "币数": int(res.weights.ne(0).sum(axis=1).mean()),
                 "Sharpe": res.stats.sharpe,
                 "年化": (1 + res.stats.total_return) ** (ANN / len(res.rets)) - 1,
                 "波动": res.stats.ann_vol, "回撤": res.stats.max_dd,
                 "年换手": res.stats.turnover_ann,
                 "年化成本": res.costs.sum() / (len(res.rets) / ANN)})
    curves[lb] = res
    d = pd.DataFrame(rows)
    o = d.copy()
    for c in ("年化", "波动", "回撤", "年化成本"):
        o[c] = o[c].map(lambda v: f"{v:+.1%}")
    o["Sharpe"] = o["Sharpe"].map(lambda v: f"{v:+.2f}")
    o["年换手"] = o["年换手"].map(lambda v: f"{v:.0f}")
    print(o.to_string(index=False))

    # ------------------------------------------------------------------ 成本
    print("\n" + "=" * 150)
    print("④ 成本敏感度（这类策略换手极高，成本就是生死线）")
    print("=" * 150 + "\n")
    print(f"  {'成本/边':<12}{'对冲3个PC 全部':>18}{'对冲3个PC 样本外':>20}")
    for fee, slip in ((0, 0), (2, 1), (4.5, 2), (10, 5)):
        c = B.Costs(fee, slip)
        _, a = run_resid(resid[3], "", None, c)
        _, b = run_resid(resid[3], "", oos, c)
        print(f"  {c.per_side*1e4:>5.1f}bp{'':<5}{a.stats.sharpe:>+18.2f}"
              f"{b.stats.sharpe:>+20.2f}")

    # the number that decides this idea: how cheap execution has to be
    print("\n" + "=" * 150)
    print("⑤ 打平成本：这个信号毛收益能撑住多少手续费")
    print("=" * 150 + "\n")
    print(f"  {'策略':<26}{'零成本年化':>12}{'年换手':>9}{'打平成本/边':>14}{'实际成本/边':>13}")
    for k in (1, 3, 5):
        for lbl, syms in (("全部", None), ("样本外", oos)):
            _, r0 = run_resid(resid[k], "", syms, B.Costs(0, 0))
            yrs = len(r0.rets) / ANN
            gross = (1 + r0.stats.total_return) ** (1 / yrs) - 1
            _, r1 = run_resid(resid[k], "", syms, B.Costs(4.5, 2.0))
            turn = r1.stats.turnover_ann
            be = gross / turn if turn else np.nan
            print(f"  {'对冲' + str(k) + 'PC ' + lbl:<26}{gross:>+12.1%}{turn:>9.0f}"
                  f"{be*1e4:>13.2f}bp{6.5:>12.1f}bp")
    print("""
  「打平成本」= 零成本下的年化毛收益 / 年换手。低于这个数才可能赚钱。
  币安 USDT-M 永续 taker 是 4.5bp、maker 是 2bp，本仓库一直用 6.5bp（含滑点）。
  如果打平成本只有零点几个 bp，那这个信号不是「需要优化执行」，
  是**在任何现实的成本下都不存在**。
""")

    best = max(curves, key=lambda k: curves[k].stats.sharpe)
    p5, _, p95 = M.block_bootstrap_sharpe(curves[best].rets, ANN, block=20)
    print(f"""
  最好的一档是「{best}」，Sharpe {curves[best].stats.sharpe:+.2f}，
  block bootstrap 95% 区间 [{p5:+.2f}, {p95:+.2f}]。
""")

    REPORTS.mkdir(exist_ok=True)
    t.to_csv(REPORTS / "pca_variance.csv", index=False)
    t2.to_csv(REPORTS / "pca_residual_stats.csv", index=False)
    d.to_csv(REPORTS / "pca_residual_strategies.csv", index=False)
    print(f"  wrote {REPORTS/'pca_variance.csv'} 等 3 个文件")


if __name__ == "__main__":
    main()
