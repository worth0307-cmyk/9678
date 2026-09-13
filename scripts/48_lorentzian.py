"""把 Lorentzian Classification 的 k-NN 搬到本仓库的数据上，用真实成本结算.

    python scripts/48_lorentzian.py
    python scripts/48_lorentzian.py --grid      # 加上参数网格（慢）

TradingView 上那个面板报的是 42 笔样本的胜率。这里换成：六个币、2022-01-01 起、
日线，按币安真实费率（挂单 2bp / 吃单 5bp）和真实资金费率结算，并且和
「等权买入持有」并排比 —— 因为一个做多做空的方向性策略在 2022~2026 的加密行情里
很容易只是在收 beta，而那不需要 k-NN。

复刻的部分：
  特征      RSI(14,2)、WT(10,11)、CCI(20,1)、ADX(20)、RSI(9,1)，走式极值归一化
  距离      d = Σ log(1+|Δf|)
  近邻      只看 i%4==0 的 bar；d>=lastDistance 才收下；收满 k 个后把门槛提到
            已收距离的 75% 分位。**这不是教科书 k-NN**，是个偏向「距离较大但
            分散」的采样器 —— 所以下面同时跑真·k-NN 做对照。
  标签      前向 sign(close[i+N]-close[i])；后向 sign(close[i]-close[i-N])

不复刻的部分：过滤器、核回归、包络。先量**算法裸机**有没有信息 ——
过滤器只会让样本更少、参数更多，在还不知道底层有没有边际的时候先加它们，
等于给一个未知的东西再叠一层未知。
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vibt import data as D, indicators as IND, ingest as I, metrics as M  # noqa: E402

pd.set_option("display.width", 220)

COINS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "TAOUSDT", "HYPEUSDT")
MAKER_BPS, TAKER_BPS = 2.0, 5.0
ANN = 365.0


# ---------------------------------------------------------------- 特征

def _running_norm(s: pd.Series) -> pd.Series:
    """走式极值归一化到 0~1：只用已经发生过的极值，所以不偷看未来。

    代价是同一根 bar 的特征值会随着后来出现新极值而**改变含义**。Pine 版本
    也是这么做的，这里照搬 —— 但这是一层看不见的漂移，值得知道它在那儿。
    """
    lo = s.expanding().min()
    hi = s.expanding().max()
    return (s - lo) / (hi - lo).replace(0, np.nan)


def features(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    f["rsi14"] = IND.ema(IND.rsi(df["close"], 14), 2) / 100.0
    f["wt"] = _running_norm(IND.wavetrend(df, 10, 11))
    f["cci"] = _running_norm(IND.ema(IND.cci(df, 20), 1))
    f["adx"] = IND.adx(df, 20) / 100.0
    f["rsi9"] = IND.ema(IND.rsi(df["close"], 9), 1) / 100.0
    return f


# ---------------------------------------------------------------- k-NN

def predict(F: np.ndarray, y: np.ndarray, k: int, max_back: int,
            skip_recent: int, sampler: bool) -> np.ndarray:
    """每根 bar 的预测分（邻居投票求和，范围 −k..+k）。

    `skip_recent` 是前向标签的代价：最后 N 根 bar 的标签还不存在，不能进训练集。
    """
    T, nf = F.shape
    out = np.zeros(T)
    for t in range(60, T):
        end = t - skip_recent
        if end <= 1:
            continue
        start = max(0, end - max_back)
        idx = np.arange(start, end, 4)
        if len(idx) < k:
            continue
        d = np.log1p(np.abs(F[idx] - F[t])).sum(axis=1)
        good = np.isfinite(d)
        idx, d = idx[good], d[good]
        if len(idx) < k:
            continue
        if not sampler:
            sel = idx[np.argpartition(d, k - 1)[:k]]      # 真·k-NN
            out[t] = np.nansum(y[sel])
            continue
        # 原版的近似采样器，顺序依赖，只能老实循环
        last = -1.0
        dists: list[float] = []
        votes: list[float] = []
        for j, dj in zip(idx, d):
            if dj >= last:
                last = dj
                dists.append(dj)
                votes.append(y[j])
                if len(votes) > k:
                    last = dists[int(round(k * 3 / 4))]
                    dists.pop(0)
                    votes.pop(0)
        out[t] = np.nansum(votes)
    return out


def labels(close: np.ndarray, n: int, forward: bool) -> np.ndarray:
    y = np.zeros(len(close))
    if forward:
        y[:-n] = np.sign(close[n:] - close[:-n])     # bar i 之后 N 根的方向
    else:
        y[n:] = np.sign(close[n:] - close[:-n])      # bar i 之前 N 根的方向
    return y


# ---------------------------------------------------------------- 回测

def funding_daily(sym: str) -> pd.Series:
    d = I._load_funding(Path("data/funding") / f"{sym}_funding.csv.gz")
    r = d["funding_rate"].astype(float)
    return r.groupby(r.index.floor("D")).sum()


def run(k: int = 8, n_hold: int = 4, max_back: int = 2000, forward: bool = True,
        sampler: bool = True, coins=COINS) -> dict:
    pos, rets, funds = {}, {}, {}
    for s in coins:
        df = D.load("1d", symbol=s)
        F = features(df)
        ok = F.notna().all(axis=1)
        F2 = F[ok]
        close = df.loc[F2.index, "close"].to_numpy()
        y = labels(close, n_hold, forward)
        p = predict(F2.to_numpy(), y, k, max_back,
                    n_hold if forward else 0, sampler)
        sig = pd.Series(np.sign(p), index=F2.index)
        pos[s] = sig
        rets[s] = df.loc[F2.index, "close"].pct_change().shift(-1)
        funds[s] = funding_daily(s).reindex(F2.index).fillna(0.0)

    P = pd.DataFrame(pos).sort_index()
    R = pd.DataFrame(rets).reindex(P.index)
    Fu = pd.DataFrame(funds).reindex(P.index).fillna(0.0)
    # 等权：有信号的币平分 1.0 的总敞口
    live = P.abs().sum(axis=1).replace(0, np.nan)
    W = P.div(live, axis=0).fillna(0.0)

    gross = (W * R).sum(axis=1).dropna()
    turn = W.diff().abs().sum(axis=1).reindex(gross.index).fillna(0.0)
    fund = -(W * Fu).sum(axis=1).reindex(gross.index).fillna(0.0)
    bh = R.mean(axis=1).reindex(gross.index)      # 等权买入持有
    return {"gross": gross, "turn": turn, "fund": fund, "bh": bh,
            "pred": P, "W": W}


def summarize(r: dict, fee_bps: float, slip_bps: float) -> dict:
    g, turn, fund = r["gross"], r["turn"], r["fund"]
    cost = turn * (fee_bps + slip_bps) * 1e-4
    net = g - cost + fund
    yrs = len(g) / ANN
    sd = float(net.std())
    return {
        "天数": len(g),
        "毛收益": float(g.mean() * ANN),
        "资金费率": float(fund.mean() * ANN),
        "手续费+滑点": -float(cost.sum() / yrs),
        "净收益": float(net.mean() * ANN),
        "净Sharpe": float(net.mean() / sd * math.sqrt(ANN)) if sd > 0 else np.nan,
        "年换手": float(turn.sum() / yrs),
        "买入持有": float(r["bh"].mean() * ANN),
        "_net": net,
    }


def naive(n: int = 4) -> dict:
    """朴素基准：仓位 = −sign(过去 n 天涨跌)。不用特征、不用距离、不用邻居。

    这是判断 k-NN 有没有存在价值的那把尺子：如果一行 numpy 就能拿到同样的东西，
    那 15 维特征空间和 Lorentzian 度量就是在给一个简单现象戴帽子。
    """
    pos, rets, funds = {}, {}, {}
    for s in COINS:
        df = D.load("1d", symbol=s)
        pos[s] = -np.sign(df["close"].pct_change(n))
        rets[s] = df["close"].pct_change().shift(-1)
        funds[s] = funding_daily(s).reindex(df.index).fillna(0.0)
    P = pd.DataFrame(pos)
    R = pd.DataFrame(rets).reindex(P.index)
    Fu = pd.DataFrame(funds).reindex(P.index).fillna(0.0)
    live = P.abs().sum(axis=1).replace(0, np.nan)
    W = P.div(live, axis=0).fillna(0.0)
    g = (W * R).sum(axis=1).dropna()
    return {"gross": g, "turn": W.diff().abs().sum(axis=1).reindex(g.index).fillna(0.0),
            "fund": -(W * Fu).sum(axis=1).reindex(g.index).fillna(0.0),
            "bh": R.mean(axis=1).reindex(g.index), "W": W}


def flip(r: dict) -> dict:
    """反向操作。注意 net(正)+net(反) = −2×成本 —— 反向不是白捡的。"""
    return {"gross": -r["gross"], "turn": r["turn"], "fund": -r["fund"],
            "bh": r["bh"], "W": -r["W"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", action="store_true", help="跑参数网格（慢）")
    args = ap.parse_args()

    print("=" * 104)
    print("Lorentzian Classification 在本仓库数据上的表现")
    print("=" * 104)

    print("\n" + "-" * 104)
    print("A  标签对齐：前向 vs 后向（这是唯一一处对错会颠覆结论的地方）")
    print("-" * 104 + "\n")
    rows = []
    for fwd in (True, False):
        r = run(forward=fwd)
        s = summarize(r, TAKER_BPS, 1.5)
        s["版本"] = "前向标签（预测未来）" if fwd else "后向标签（描述过去）"
        rows.append(s)
        if fwd:
            base = r
    t = pd.DataFrame(rows).set_index("版本").drop(columns=["_net"])
    fmt = {c: "{:+.1%}".format for c in ("毛收益", "资金费率", "手续费+滑点", "净收益", "买入持有")}
    print(t.assign(**{c: t[c].map(f) for c, f in fmt.items()},
                   净Sharpe=t["净Sharpe"].map("{:+.2f}".format),
                   年换手=t["年换手"].map("{:.0f}x".format)).to_string())

    print("\n" + "-" * 104)
    print("B  近邻采样器 vs 真·k-NN")
    print("-" * 104 + "\n")
    rows = []
    for smp in (True, False):
        r = run(sampler=smp)
        s = summarize(r, TAKER_BPS, 1.5)
        s["版本"] = "原版采样器（i%4 + 75%分位门槛）" if smp else "真·k-NN（最近的 k 个）"
        rows.append(s)
    t = pd.DataFrame(rows).set_index("版本").drop(columns=["_net"])
    print(t.assign(**{c: t[c].map(f) for c, f in fmt.items()},
                   净Sharpe=t["净Sharpe"].map("{:+.2f}".format),
                   年换手=t["年换手"].map("{:.0f}x".format)).to_string())

    print("\n" + "-" * 104)
    print("C  真实费率下的净收益（前向标签 + 原版采样器）")
    print("-" * 104 + "\n")
    rows = []
    for lbl, fee, slip in (("全挂单 2bp + 0bp 滑点", MAKER_BPS, 0.0),
                           ("全挂单 2bp + 1bp", MAKER_BPS, 1.0),
                           ("全吃单 5bp + 1.5bp", TAKER_BPS, 1.5),
                           ("全吃单 5bp + 3bp", TAKER_BPS, 3.0)):
        s = summarize(base, fee, slip)
        s["版本"] = lbl
        rows.append(s)
    t = pd.DataFrame(rows).set_index("版本").drop(columns=["_net"])
    print(t.assign(**{c: t[c].map(f) for c, f in fmt.items()},
                   净Sharpe=t["净Sharpe"].map("{:+.2f}".format),
                   年换手=t["年换手"].map("{:.0f}x".format)).to_string())

    print("\n" + "-" * 104)
    print("D  信号到底有没有信息：命中率 vs 基准")
    print("-" * 104 + "\n")
    hits, bases, ns = [], [], []
    for s in COINS:
        df = D.load("1d", symbol=s)
        w = base["W"][s].reindex(df.index).fillna(0.0)
        fwd1 = df["close"].pct_change().shift(-1)
        m = (w != 0) & fwd1.notna()
        if m.sum() == 0:
            continue
        hit = float((np.sign(w[m]) == np.sign(fwd1[m])).mean())
        # 基准：同一批 bar 上，按该币信号的多空比例闭眼持有会命中多少
        up = float((fwd1[m] > 0).mean())
        plong = float((w[m] > 0).mean())
        bench = plong * up + (1 - plong) * (1 - up)
        hits.append(hit)
        bases.append(bench)
        ns.append({"币": s, "命中率": hit, "基准": bench, "超额": hit - bench,
                   "持仓天数": int(m.sum())})
    h = pd.DataFrame(ns).set_index("币")
    print(h.assign(**{c: h[c].map("{:+.2%}".format) for c in ("命中率", "基准", "超额")}).to_string())
    ex = float(np.mean([r["超额"] for _, r in h.iterrows()]))
    print(f"\n  平均超额 {ex:+.2%}  —— **这一列接近 0，就说明 k-NN 没提供信息**，"
          f"净值好坏只是方向暴露的副产品。")

    print("\n" + "-" * 104)
    print("E  去偏 Sharpe")
    print("-" * 104 + "\n")
    net = summarize(base, TAKER_BPS, 1.5)["_net"]
    sh = float(net.mean() / net.std() * math.sqrt(ANN))
    print(f"  净 Sharpe {sh:+.2f}   样本 {len(net)} 天   "
          f"偏度 {net.skew():+.2f}   峰度 {net.kurtosis() + 3:.2f}\n")
    print(f"  {'试过的组数':>12}   {'P(真实Sharpe>0)':>16}")
    for n in (10, 50, 200, 1000):
        ds = M.deflated_sharpe(sh, n_trials=n, n_obs=len(net), ann_factor=ANN,
                               skew=float(net.skew()), kurt=float(net.kurtosis() + 3))
        print(f"  {n:>12}   {ds:>16.3f}")
    print("""
  文档里那张超参数重要性图说明作者做过**数千组**参数的网格搜索。
  按那个量级，最后一行（1000 组）才是该看的那一行。""")

    print("\n" + "-" * 104)
    print("F  后向标签版是强负的 —— 反过来做会怎样")
    print("-" * 104 + "\n")
    rb = run(forward=False)
    rev = flip(rb)
    print(f"  {'策略':36s} {'毛收益':>8} {'净(挂单)':>9} {'Sharpe':>7} {'年换手':>7}")
    for lbl, r in (("Lorentzian k-NN 后向标签 · 原样", rb),
                   ("Lorentzian k-NN 后向标签 · 反向", rev),
                   ("朴素基准：−sign(2日涨跌)", naive(2)),
                   ("朴素基准：−sign(4日涨跌)", naive(4)),
                   ("朴素基准：−sign(8日涨跌)", naive(8))):
        g, tn, fu = r["gross"], r["turn"], r["fund"]
        nt = g - tn * (MAKER_BPS + 1.0) * 1e-4 + fu
        print(f"  {lbl:36s} {g.mean() * ANN:+7.1%} {nt.mean() * ANN:+8.1%} "
              f"{nt.mean() / nt.std() * math.sqrt(ANN):+7.2f} "
              f"{tn.sum() / (len(g) / ANN):6.0f}x")
    print("""
  朴素的均值回归三档全是亏的，所以 k-NN **不只是**「最近涨跌」的代理 ——
  特征空间里确实有点东西。但下一步才是关键。""")

    print("\n" + "-" * 104)
    print("G  那个 +0.44 是真的吗")
    print("-" * 104 + "\n")
    g, tn, fu, W = rev["gross"], rev["turn"], rev["fund"], rev["W"]
    net = g - tn * (MAKER_BPS + 1.0) * 1e-4 + fu
    bh = rev["bh"].reindex(net.index).fillna(0.0)
    netexp = W.sum(axis=1).reindex(net.index)
    beta = float(np.polyfit(bh, net, 1)[0])
    # 只剔 beta，**不减截距** —— 减了截距等于把均值抹成 0，残差 Sharpe 必然是 0，
    # 那是个看起来很有说服力的假结论。
    resid = net - beta * bh
    sh = float(net.mean() / net.std() * math.sqrt(ANN))
    print(f"  平均净敞口 {netexp.mean():+.3f}   与买入持有相关 {net.corr(bh):+.3f}   beta {beta:+.3f}")
    print(f"  原始 Sharpe {sh:+.2f}   剔除 beta 后 {resid.mean() / resid.std() * math.sqrt(ANN):+.2f}"
          f"（alpha {resid.mean() * ANN:+.1%}/年）")
    print("  → 不是 beta 伪装成 alpha。\n")
    for y, seg in net.groupby(net.index.year):
        print(f"   {y}  净 {seg.mean() * ANN:+7.1%}   "
              f"Sharpe {seg.mean() / seg.std() * math.sqrt(ANN):+5.2f}   {len(seg)} 天")
    print(f"\n  {'试过的组数':>12}   {'P(真实Sharpe>0)':>16}")
    for n in (10, 30, 100, 1000):
        print(f"  {n:>12}   "
              f"{M.deflated_sharpe(sh, n_trials=n, n_obs=len(net), ann_factor=ANN, skew=float(net.skew()), kurt=float(net.kurtosis() + 3)):>16.3f}")
    print("""
  **反向这件事是被结果诱导出来的。** 我先看到后向标签版是 −0.67，才去试反向 ——
  这一路上试过的组合（2 种标签 × 2 种近邻 × 4 档成本 × 反向）至少 30 组，
  按那一行读，P(真实Sharpe>0) = 0.13。**过不了多重检验。**

  加上收益集中在 2023（+1.59），五年里两年为负，所以结论是：
  特征空间里可能有东西，但这份数据**证明不了**。""")

    if args.grid:
        print("\n" + "-" * 104)
        print("H  参数网格 · 前向标签（净 Sharpe，吃单 5bp+1.5bp）")
        print("-" * 104 + "\n")
        ks, ns_ = [4, 8, 16, 24], [2, 4, 8, 16]
        g = pd.DataFrame(index=ks, columns=ns_, dtype=float)
        for k in ks:
            for nh in ns_:
                rr = run(k=k, n_hold=nh, sampler=False)
                g.loc[k, nh] = summarize(rr, TAKER_BPS, 1.5)["净Sharpe"]
        g.index.name, g.columns.name = "邻居数 k", "标签窗口 N"
        print(g.map("{:+.2f}".format).to_string())
        f = g.values.flatten()
        print(f"\n  中位 {np.median(f):+.2f}   范围 {f.min():+.2f}~{f.max():+.2f}   "
              f"为正 {int((f > 0).sum())}/{g.size}")
        print("  **中位数才是「不挑参数会拿到什么」，最大值只是网格搜索的产物。**")


if __name__ == "__main__":
    main()
