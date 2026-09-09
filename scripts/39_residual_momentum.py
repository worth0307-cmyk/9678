"""Is the cross-sectional momentum result idiosyncratic alpha, or beta rotation?

This repository's one survivor is 14-day cross-sectional momentum: rank-weighted,
dollar-neutral, 3-day rebalance, out-of-sample Sharpe +1.13 with a measured beta
of -0.02 to the equal-weight market.  That beta is a first-order check and it is
not the same as factor neutrality.

The concern the PCA work raised is about the SIGNAL, not the weights.  PC1 is
58-68% of variance in this market, so a 14-day return is largely "that coin's PC1
loading times what the market did".  Ranking on it in a rising market ranks coins
by beta, and the resulting book is long high-beta, short low-beta -- dollar
neutral, factor exposed.  A full-sample regression averages that exposure across
up and down regimes and can report roughly zero while the strategy is timing beta
throughout.

So the test replaces the ranking variable with residual momentum and changes
nothing else.  Two differences would be confounded by doing only that, because a
residual is also volatility-normalised, so a third arm isolates them:

  raw 14d          close/close.shift(14) - 1          the existing baseline
  vol-norm 14d     14-day sum of z-scored returns     normalisation only
  resid 14d, k=1/3/5   14-day sum of PC-hedged residuals   normalisation + hedge

If raw and residual rank equally well, the strategy is real cross-sectional
alpha.  If residual collapses, the survivor was beta rotation and the repository
has been wrong about its best result.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vibt import backtest as B, data as D, factors as F, metrics as M, xsec as X  # noqa: E402

pd.set_option("display.width", 250)
REPORTS = Path(__file__).resolve().parent.parent / "reports"
ANN = 365.0
LOOKBACK, MIN_NAMES, MOM = 120, 30, 14
REBAL, LIQ = 3, 10e6
COSTS = B.Costs(4.5, 2.0)
NOT_COINS = ("USDCUSDT", "BTCDOMUSDT")
IN_SAMPLE = [
    "1000BONKUSDT", "1000FLOKIUSDT", "1000PEPEUSDT", "1000SHIBUSDT", "APTUSDT",
    "ARBUSDT", "BNBUSDT", "BTCUSDT", "DOGEUSDT", "DYDXUSDT", "ENAUSDT", "ETHUSDT",
    "FETUSDT", "HYPEUSDT", "LDOUSDT", "OPUSDT", "PENDLEUSDT", "RENDERUSDT",
    "SEIUSDT", "SOLUSDT", "SUIUSDT", "TAOUSDT", "TIAUSDT", "WIFUSDT", "WLDUSDT",
    "XRPUSDT",
]


def panels():
    syms = [s for s in D.available_symbols(require=("1d",)) if s not in NOT_COINS]
    op, cl, qv = {}, {}, {}
    for s in syms:
        d = D.load("1d", symbol=s)
        op[s], cl[s] = d["open"], d["close"]
        if "quote_volume" in d:
            qv[s] = d["quote_volume"]
    P = pd.DataFrame(op).sort_index()
    return P, pd.DataFrame(cl).reindex(P.index), pd.DataFrame(qv).reindex(P.index)


def run_one(feat: pd.DataFrame, P: pd.DataFrame, ok: pd.DataFrame,
            syms: list[str] | None, costs: B.Costs = COSTS):
    f = feat if syms is None else feat[[c for c in feat.columns if c in syms]]
    f = f.where(ok.reindex_like(f).fillna(False))
    w = X.cross_sectional_weights(f, mode="rank", gross=1.0)
    return X.run(P[w.columns], w, costs, ANN, "", REBAL)


def describe(name: str, res, bench: pd.Series, load1: pd.DataFrame,
             mkt_fwd: pd.Series) -> dict:
    beta, alpha = X.beta_to(res.rets, bench.reindex(res.rets.index))
    expo = F.portfolio_factor_exposure(res.weights, load1)
    yrs = len(res.rets) / ANN
    t = res.stats.total_return
    # the timing question: is the book long the factor exactly when it pays?
    e, m = expo.align(mkt_fwd, join="inner")
    both = pd.concat([e, m], axis=1).dropna()
    timing = float(both.iloc[:, 0].corr(both.iloc[:, 1])) if len(both) > 30 else np.nan
    return {"信号": name, "Sharpe": res.stats.sharpe,
            "年化": (1 + t) ** (1 / yrs) - 1 if t > -1 else -1.0,
            "波动": res.stats.ann_vol, "回撤": res.stats.max_dd,
            "对大盘beta": beta, "年化alpha": alpha,
            "PC1敞口均值": float(expo.mean()), "PC1敞口标准差": float(expo.std()),
            "PC1择时相关": timing,
            "年换手": res.stats.turnover_ann,
            "年化成本": res.costs.sum() / yrs}


def show(rows: list[dict]) -> None:
    d = pd.DataFrame(rows)
    for c in ("年化", "波动", "回撤", "年化alpha", "年化成本"):
        d[c] = d[c].map(lambda v: f"{v:+.1%}" if pd.notna(v) else "")
    for c in ("Sharpe", "对大盘beta", "PC1敞口均值", "PC1敞口标准差", "PC1择时相关"):
        d[c] = d[c].map(lambda v: f"{v:+.3f}" if pd.notna(v) else "")
    d["年换手"] = d["年换手"].map(lambda v: f"{v:.0f}")
    print(d.to_string(index=False))


def main() -> None:
    P, C, V = panels()
    R = np.log(C).diff().replace([np.inf, -np.inf], np.nan)
    ok = (V.rolling(30).median() >= LIQ)

    print("=" * 165)
    print("横截面动量：是特质 alpha，还是伪装的 beta 轮动？")
    print("=" * 165)
    print(f"""
  仓位构造全程不变：rank 加权、美元中性、每 {REBAL} 天再平衡、
  流动性下限 ${LIQ/1e6:.0f}M、成本 {COSTS.per_side*1e4:.1f}bp/边、次日开盘成交。
  唯一变的是**排序用什么**。

  三条臂：
    raw {MOM}d        close/close.shift({MOM})-1          现有基准
    vol-norm {MOM}d   {MOM} 天 z 分数之和                  只加波动归一化
    resid {MOM}d k=…  {MOM} 天 PCA 残差之和                波动归一化 + 剥主成分
  第二条臂是必须的：残差本身带了波动归一化，不设这条臂就分不清是哪一半在起作用。
""")

    print(f"  滚动 PCA：窗口 {LOOKBACK} 天（不含当天），最少 {MIN_NAMES} 个币 …")
    pca = F.rolling_pca(R, LOOKBACK, ks=(1, 3, 5), min_names=MIN_NAMES)
    print(f"  完成。有效日期 {int(pca.n_names.gt(0).sum())} 天，"
          f"中位截面宽度 {pca.n_names[pca.n_names > 0].median():.0f} 个币，"
          f"PC1 方差占比中位 {pca.var1.median():.1%}\n")

    feats = {f"raw {MOM}d": C / C.shift(MOM) - 1,
             f"vol-norm {MOM}d": pca.zscore.rolling(MOM, min_periods=MOM).sum()}
    for k in (1, 3, 5):
        feats[f"resid {MOM}d k={k}"] = pca.resid[k].rolling(MOM, min_periods=MOM).sum()

    eq_mkt = C.pct_change().mean(axis=1)                 # equal-weight market
    bench = P.pct_change().shift(-1).mean(axis=1)        # what the engine earns
    mkt_fwd = eq_mkt.shift(-1)                           # tomorrow's market move

    every = list(P.columns)
    ins = [s for s in every if s in IN_SAMPLE]
    oos = [s for s in every if s not in IN_SAMPLE]

    results = {}
    for label, syms in (("① 样本内 26 币", ins),
                        (f"② 样本外 {len(oos)} 币（从未参与开发）", oos),
                        ("③ 全部 207 币", None)):
        print("=" * 165)
        print(label)
        print("=" * 165 + "\n")
        rows = []
        for name, f in feats.items():
            res = run_one(f, P, ok, syms)
            if not np.isfinite(res.stats.sharpe):
                continue
            results[(label, name)] = res
            rows.append(describe(name, res, bench, pca.load1, mkt_fwd))
        show(rows)
        print()

    # ------------------------------------------------------------------ 读表
    print("=" * 165)
    print("怎么读「PC1 敞口」这两列")
    print("=" * 165)
    print("""
  「PC1敞口均值」= 每天 Σ(权重 × 该币的 PC1 载荷)。美元中性 ≠ 因子中性：
  多头腿如果系统性地拿载荷更高的币，这个数就不是 0，而净权重检验看不见它。

  「PC1择时相关」= 当天的 PC1 敞口 与 次日大盘涨跌 的相关性。
  显著为正 = 这套策略在大盘要涨的时候恰好偏多高 beta —— 那就是 beta 择时，
  不是横截面选股。接近 0 = 敞口和大盘走势无关，赚的不是这个。
""")

    # -------------------------------------------------------- 直接对比：alpha
    print("=" * 165)
    print("直接回答：把 raw 换成 resid，样本外结果变了多少")
    print("=" * 165 + "\n")
    lab = f"② 样本外 {len(oos)} 币（从未参与开发）"
    base = results.get((lab, f"raw {MOM}d"))
    rows = []
    for name in feats:
        r = results.get((lab, name))
        if r is None or base is None:
            continue
        p5, _, p95 = M.block_bootstrap_sharpe(r.rets, ANN, block=20)
        rows.append({"信号": name, "Sharpe": r.stats.sharpe,
                     "bootstrap 95%": f"[{p5:+.2f}, {p95:+.2f}]",
                     "对raw的差": r.stats.sharpe - base.stats.sharpe,
                     "与raw的收益相关": float(r.rets.corr(base.rets))})
    d = pd.DataFrame(rows)
    d["Sharpe"] = d["Sharpe"].map(lambda v: f"{v:+.2f}")
    d["对raw的差"] = d["对raw的差"].map(lambda v: f"{v:+.2f}")
    d["与raw的收益相关"] = d["与raw的收益相关"].map(lambda v: f"{v:+.3f}")
    print(d.to_string(index=False))

    # --------------------------------------------------------- ④ 流动性
    print("\n" + "=" * 165)
    print("④ 样本外为什么这么差：还是流动性吗")
    print("=" * 165)
    print(f"""
  开发用的 26 个币日均成交额中位 ${V[ins].rolling(30).median().median().median()/1e6:,.0f}M，
  样本外 181 个币是 ${V[oos].rolling(30).median().median().median()/1e6:,.0f}M。
  「成交额越大越好」这个方向在 35 号脚本里被独立验证过（用的是那 26 个币，
  不是这批样本外币），所以这里扫门槛不是事后挑参数，是检验一个已有方向。
  但门槛是连续扫的、全部列出，不挑最好的那个说事。
""")
    rows = []
    for thr in (10e6, 30e6, 100e6, 300e6):
        okt = (V.rolling(30).median() >= thr)
        n_ok = int(okt[oos].any().sum())
        for name in (f"raw {MOM}d", f"vol-norm {MOM}d", f"resid {MOM}d k=1"):
            res = run_one(feats[name], P, okt, oos)
            if not np.isfinite(res.stats.sharpe):
                continue
            yrs = len(res.rets) / ANN
            t = res.stats.total_return
            rows.append({"门槛": f"${thr/1e6:,.0f}M", "可用币数": n_ok, "信号": name,
                         "Sharpe": res.stats.sharpe,
                         "年化": (1 + t) ** (1 / yrs) - 1 if t > -1 else -1.0,
                         "回撤": res.stats.max_dd,
                         "平均持仓数": float(res.weights.ne(0).sum(axis=1).mean())})
    d = pd.DataFrame(rows)
    o = d.copy()
    for c in ("年化", "回撤"):
        o[c] = o[c].map(lambda v: f"{v:+.1%}")
    o["Sharpe"] = o["Sharpe"].map(lambda v: f"{v:+.2f}")
    o["平均持仓数"] = o["平均持仓数"].map(lambda v: f"{v:.0f}")
    print(o.to_string(index=False))

    # --------------------------------------------------------- ⑤ 打平成本
    print("\n" + "=" * 165)
    print("⑤ 打平成本（和 PCA 残差那套对照着看）")
    print("=" * 165 + "\n")
    print(f"  {'面板':<14}{'信号':<18}{'零成本年化':>12}{'年换手':>9}{'打平成本/边':>14}")
    for lbl, syms in (("样本内26", ins), ("样本外181", oos), ("全部207", None)):
        for name in (f"raw {MOM}d", f"vol-norm {MOM}d", f"resid {MOM}d k=1"):
            r0 = run_one(feats[name], P, ok, syms, B.Costs(0, 0))
            r1 = run_one(feats[name], P, ok, syms)
            yrs = len(r0.rets) / ANN
            g = (1 + r0.stats.total_return) ** (1 / yrs) - 1
            turn = r1.stats.turnover_ann
            be = g / turn if turn else np.nan
            print(f"  {lbl:<14}{name:<18}{g:>+12.1%}{turn:>9.0f}{be*1e4:>13.2f}bp")
    print(f"""
  实际成本 {COSTS.per_side*1e4:.1f}bp/边。换手只有 ~70 倍/年（PCA 残差那套是 440 倍），
  所以这套策略对成本的容忍度高一个量级——它的问题不在成本，在信号本身。
""")

    REPORTS.mkdir(exist_ok=True)
    pd.DataFrame([describe(n, r, bench, pca.load1, mkt_fwd) | {"面板": l}
                  for (l, n), r in results.items()]).to_csv(
        REPORTS / "residual_momentum.csv", index=False)
    d.to_csv(REPORTS / "residual_momentum_liquidity.csv", index=False)
    print(f"  wrote {REPORTS/'residual_momentum.csv'} 等 2 个文件")


if __name__ == "__main__":
    main()
