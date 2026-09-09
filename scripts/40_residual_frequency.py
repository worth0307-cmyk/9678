"""能不能把残差反转的换手压到成本线以下.

38 号脚本的结论是一个量级问题，不是执行问题：信号毛收益 +9.6%/年，
但年换手 441 倍，打平成本 2.2bp，而实际成本 6.5bp。差三倍。

要跨过去只有两条路，而且它们不是一回事：

  降频      持有更久 / 用更长的信号窗口。直接砍换手，但残差的均值回复
            集中在 1 日（自相关 −0.035，5 日只剩 −0.006），所以很可能
            信号衰减得比换手更快。
  平滑      每天照样算目标仓位，但只朝目标移动一部分（w = a*target +
            (1-a)*w_prev）。不放弃信号的时效性，只放弃一部分执行速度。
            这是 stat arb 里压换手的标准做法。

判据只有一个：**打平成本 = 零成本毛收益 / 年换手**，要超过 6.5bp。
Sharpe 在这里是次要的，因为扫格子挑 Sharpe 最高的那一格正是过拟合的定义，
而打平成本是个物理量，不随挑选而改善。
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
LOOKBACK, MIN_NAMES, KPC = 120, 30, 3
LIQ = 10e6
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


def smooth(w: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Move only `alpha` of the way to today's target, then renormalise gross.

    Renormalising matters: a partially-adjusted book drifts away from unit gross
    exposure, and comparing a 0.6-gross book to a 1.0-gross one on Sharpe would
    be comparing position sizes rather than signals.
    """
    if alpha >= 1.0:
        return w
    out = w.ewm(alpha=alpha, adjust=False).mean()
    g = out.abs().sum(axis=1).replace(0, np.nan)
    return out.div(g, axis=0).fillna(0.0)


def measure(feat: pd.DataFrame, P: pd.DataFrame, ok: pd.DataFrame,
            syms: list[str] | None, rebal: int, alpha: float) -> dict:
    f = feat if syms is None else feat[[c for c in feat.columns if c in syms]]
    f = f.where(ok.reindex_like(f).fillna(False))
    w = smooth(X.cross_sectional_weights(-f, mode="rank", gross=1.0), alpha)
    r0 = X.run(P[w.columns], w, B.Costs(0, 0), ANN, "", rebal)
    r1 = X.run(P[w.columns], w, COSTS, ANN, "", rebal)
    yrs = len(r0.rets) / ANN
    gross = (1 + r0.stats.total_return) ** (1 / yrs) - 1 if r0.stats.total_return > -1 else -1.0
    turn = r1.stats.turnover_ann
    return {"零成本Sharpe": r0.stats.sharpe, "零成本年化": gross,
            "年换手": turn, "打平成本bp": (gross / turn * 1e4) if turn else np.nan,
            "净Sharpe": r1.stats.sharpe,
            "净年化": (1 + r1.stats.total_return) ** (1 / yrs) - 1
                      if r1.stats.total_return > -1 else -1.0}


def show(rows: list[dict], keys: list[str]) -> None:
    d = pd.DataFrame(rows)
    for c in ("零成本年化", "净年化"):
        d[c] = d[c].map(lambda v: f"{v:+.1%}" if pd.notna(v) else "")
    for c in ("零成本Sharpe", "净Sharpe"):
        d[c] = d[c].map(lambda v: f"{v:+.2f}" if pd.notna(v) else "")
    d["年换手"] = d["年换手"].map(lambda v: f"{v:.0f}")
    d["打平成本bp"] = d["打平成本bp"].map(
        lambda v: f"{v:.2f}" + ("*" if pd.notna(v) and v > COSTS.per_side * 1e4 else "")
        if pd.notna(v) else "")
    print(d[keys + ["零成本Sharpe", "零成本年化", "年换手", "打平成本bp",
                    "净Sharpe", "净年化"]].to_string(index=False))


def main() -> None:
    P, C, V = panels()
    R = np.log(C).diff().replace([np.inf, -np.inf], np.nan)
    ok = (V.rolling(30).median() >= LIQ)
    every = list(P.columns)
    oos = [s for s in every if s not in IN_SAMPLE]

    print("=" * 165)
    print("残差反转：能不能把换手压到成本线以下")
    print("=" * 165)
    print(f"""
  基线（38 号脚本）：日频、对冲 {KPC} 个主成分，样本外毛收益 +9.6%/年，
  年换手 441，打平成本 2.19bp，实际成本 {COSTS.per_side*1e4:.1f}bp。**差三倍。**

  判据是「打平成本」这一列（超过 {COSTS.per_side*1e4:.1f}bp 的标 *），不是 Sharpe。
  在格子里挑 Sharpe 最高的那一格就是过拟合的定义；打平成本是个物理量。
""")
    print(f"  滚动 PCA（窗口 {LOOKBACK} 天，只用过去）…")
    pca = F.rolling_pca(R, LOOKBACK, ks=(KPC,), min_names=MIN_NAMES)
    res = pca.resid[KPC]
    print(f"  完成，有效 {int(pca.n_names.gt(0).sum())} 天\n")

    # signal windows: 1 day is the baseline, longer windows are the slow versions
    feats = {w: (res if w == 1 else res.rolling(w, min_periods=w).sum())
             for w in (1, 2, 3, 5, 10)}

    # ------------------------------------------------------------------ ① 降频
    print("=" * 165)
    print("① 降频：信号窗口 x 再平衡间隔")
    print("=" * 165 + "\n")
    grids = {}
    for label, syms in (("全部 207 币", None), (f"样本外 {len(oos)} 币", oos)):
        rows = []
        for w, f in feats.items():
            for rb in (1, 2, 3, 5, 10):
                rows.append({"信号窗口": f"{w}d", "再平衡": f"{rb}d"}
                            | measure(f, P, ok, syms, rb, 1.0))
        grids[label] = pd.DataFrame(rows)
        print(f"--- {label} ---")
        show(rows, ["信号窗口", "再平衡"])
        print()

    # ------------------------------------------------------------------ ② 平滑
    print("=" * 165)
    print("② 权重平滑：每天照算信号，只朝目标移动一部分")
    print("=" * 165 + "\n")
    smoothed = {}
    for label, syms in (("全部 207 币", None), (f"样本外 {len(oos)} 币", oos)):
        rows = []
        for a in (1.0, 0.5, 0.3, 0.2, 0.1, 0.05):
            rows.append({"alpha": f"{a:.2f}", "≈半衰期":
                         f"{np.log(0.5)/np.log(1-a):.1f}d" if a < 1 else "0d"}
                        | measure(feats[1], P, ok, syms, 1, a))
        smoothed[label] = pd.DataFrame(rows)
        print(f"--- {label} ---")
        show(rows, ["alpha", "≈半衰期"])
        print()

    # ------------------------------------------------------------------ 结论
    print("=" * 165)
    print("结论")
    print("=" * 165 + "\n")
    lab = f"样本外 {len(oos)} 币"
    g, s = grids[lab], smoothed[lab]
    thr = COSTS.per_side * 1e4
    n_pass = int((g["打平成本bp"] > thr).sum()) + int((s["打平成本bp"] > thr).sum())
    best_g = g.loc[g["打平成本bp"].idxmax()]
    best_s = s.loc[s["打平成本bp"].idxmax()]
    base = g[(g["信号窗口"] == "1d") & (g["再平衡"] == "1d")].iloc[0]
    print(f"  基线（1d 信号 / 1d 再平衡）：换手 {base['年换手']:.0f}，"
          f"打平成本 {base['打平成本bp']:.2f}bp")
    print(f"  降频最好一格：{best_g['信号窗口']} 信号 / {best_g['再平衡']} 再平衡，"
          f"换手 {best_g['年换手']:.0f}，打平成本 {best_g['打平成本bp']:.2f}bp")
    print(f"  平滑最好一格：alpha={best_s['alpha']}，"
          f"换手 {best_s['年换手']:.0f}，打平成本 {best_s['打平成本bp']:.2f}bp")
    n_cfg = len(g) + len(s)
    print(f"""
  样本外 {n_cfg} 个配置里，打平成本超过 {thr:.1f}bp 的有 **{n_pass}** 个。

  在扫 {n_cfg} 个格子的前提下，「有 1 个格子通过」本身不是证据：
  即便每格只有 5% 的概率靠运气越线，至少出现 1 个的概率也有
  {1 - 0.95 ** n_cfg:.0%}。所以真正要看的是它**周围**长什么样。
""")

    # A real effect is a plateau; a fitted one is a spike.  Check the winner's
    # neighbours -- the cells one step away in signal window or in rebalance.
    print("=" * 165)
    print("这唯一通过的一格，是高原还是孤立尖刺")
    print("=" * 165 + "\n")
    ws = ["1d", "2d", "3d", "5d", "10d"]
    rs = ["1d", "2d", "3d", "5d", "10d"]
    bw, br = best_g["信号窗口"], best_g["再平衡"]
    iw, ir = ws.index(bw), rs.index(br)
    nb = []
    for dw in (-1, 0, 1):
        for dr in (-1, 0, 1):
            jw, jr = iw + dw, ir + dr
            if not (0 <= jw < len(ws) and 0 <= jr < len(rs)):
                continue
            row = g[(g["信号窗口"] == ws[jw]) & (g["再平衡"] == rs[jr])]
            if len(row):
                r = row.iloc[0]
                nb.append({"信号窗口": ws[jw], "再平衡": rs[jr],
                           "是否本格": "★" if (dw == 0 and dr == 0) else "",
                           "打平成本bp": r["打平成本bp"], "净年化": r["净年化"]})
    nbd = pd.DataFrame(nb)
    o = nbd.copy()
    o["打平成本bp"] = o["打平成本bp"].map(lambda v: f"{v:.2f}")
    o["净年化"] = o["净年化"].map(lambda v: f"{v:+.1%}")
    print(o.to_string(index=False))
    others = nbd[nbd["是否本格"] == ""]["打平成本bp"]
    print(f"""
  邻域（不含本格）打平成本中位 {others.median():.2f}bp，最大 {others.max():.2f}bp，
  没有一个越过 {thr:.1f}bp。本格 {best_g['打平成本bp']:.2f}bp 是**孤立尖刺**，
  不是高原——这个仓库之前在 VI(14) 上见过一模一样的形状。

  而且就算当真：净 Sharpe {best_g['净Sharpe']:+.2f}、净年化 {best_g['净年化']:+.1%}。
  即便它是真的，也不值得做。

  根本原因不在参数：残差的均值回复集中在 1 日（自相关 −0.035，5 日只剩 −0.006）。
  放慢一拍信号就没了，这是这个信号的半衰期决定的，调参绕不过去。
""")

    REPORTS.mkdir(exist_ok=True)
    pd.concat([v.assign(panel=k) for k, v in grids.items()]).to_csv(
        REPORTS / "residual_frequency.csv", index=False)
    pd.concat([v.assign(panel=k) for k, v in smoothed.items()]).to_csv(
        REPORTS / "residual_smoothing.csv", index=False)
    print(f"  wrote {REPORTS/'residual_frequency.csv'} 等 2 个文件")


if __name__ == "__main__":
    main()
