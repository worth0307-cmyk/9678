"""Hull Suite Strategy 的回测：HMA 斜率反手系统，按真实成本结算.

    python scripts/49_hull_suite.py
    python scripts/49_hull_suite.py --tf 4h

策略本身只有两行：

    HULL = HMA(close, 55)
    HULL[0] > HULL[2]  ->  做多
    HULL[0] < HULL[2]  ->  做空

也就是「55 周期 HMA 在最近 2 根 bar 上是涨还是跌」。永远在场，反手交易，没有止损、
没有止盈、没有过滤。

三件事在跑之前就能看出来：

  1  原脚本写着 `commission_value=0, slippage=0`。**一个反手系统一分钱成本都不收。**
     这不是小疏漏 —— 反手意味着每次换向要平掉旧仓再开等量反向仓，换手是单向
     系统的两倍。
  2  `strat_dir_input` 默认是 "long"，`allow_entry_in` 会挡掉空单。所以默认配置
     跑出来的**不是**多空反手，是「只做多，HMA 转跌时平仓」。图上看到的曲线和
     代码里读到的逻辑是两件事，下面两种都跑。
  3  `HMA(src, 55)` 里 `length/2 = 27.5`，Pine 传给 wma 时截断成 27；
     `round(sqrt(55)) = 7`。复刻必须照做，否则对不上。

要回答的问题不是「它赚不赚钱」，而是**「它比买入持有好吗，比一根普通均线好吗」**。
一个在 2022~2026 加密行情里永远满仓做多的系统，赚钱是默认结果，不是本事。
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

pd.set_option("display.width", 230)

COINS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "TAOUSDT", "HYPEUSDT")
MAKER_BPS, TAKER_BPS = 2.0, 5.0
BARS_PER_YEAR = {"1d": 365.0, "4h": 365.0 * 6, "1h": 365.0 * 24}


# ---------------------------------------------------------------- Hull

def wma(s: pd.Series, n: float) -> pd.Series:
    """加权移动平均。Pine 对非整数周期是**截断**，不是四舍五入 —— 55/2 走的是 27。"""
    n = int(n)
    w = np.arange(1, n + 1, dtype=float)
    w /= w.sum()
    v = np.convolve(s.to_numpy(dtype=float), w[::-1], mode="full")[:len(s)]
    out = pd.Series(v, index=s.index)
    out.iloc[:n - 1] = np.nan
    return out


def hma(s: pd.Series, n: int) -> pd.Series:
    return wma(2 * wma(s, n / 2) - wma(s, n), round(math.sqrt(n)))


def ehma(s: pd.Series, n: int) -> pd.Series:
    return IND.ema(2 * IND.ema(s, int(n / 2)) - IND.ema(s, n), round(math.sqrt(n)))


def thma(s: pd.Series, n: float) -> pd.Series:
    # 原脚本调用时传的是 length/2，函数内部再除 —— 照搬这个嵌套
    return wma(wma(s, n / 3) * 3 - wma(s, n / 2) - wma(s, n), int(n))


def hull(s: pd.Series, n: int, mode: str) -> pd.Series:
    if mode == "Hma":
        return hma(s, n)
    if mode == "Ehma":
        return ehma(s, n)
    if mode == "Thma":
        return thma(s, n / 2)
    raise ValueError(mode)


# ---------------------------------------------------------------- 回测

def funding_bar(sym: str, index: pd.DatetimeIndex, tf: str) -> pd.Series:
    """把资金费率结算并到每根 bar 上。日线按天合计，4h 按 4 小时桶合计。"""
    d = I._load_funding(Path("data/funding") / f"{sym}_funding.csv.gz")
    r = d["funding_rate"].astype(float)
    freq = {"1d": "D", "4h": "4h", "1h": "h"}[tf]
    return r.groupby(r.index.floor(freq)).sum().reindex(index).fillna(0.0)


def signal(df: pd.DataFrame, n: int, mode: str, long_only: bool) -> pd.Series:
    h = hull(df["close"], n, mode)
    up = h > h.shift(2)                      # HULL[0] > HULL[2]
    pos = np.where(up, 1.0, 0.0 if long_only else -1.0)
    return pd.Series(pos, index=df.index).where(h.notna() & h.shift(2).notna())


def run(n: int = 55, mode: str = "Hma", long_only: bool = False,
        tf: str = "1d", coins=COINS) -> dict:
    pos, rets, funds = {}, {}, {}
    for s in coins:
        df = D.load(tf, symbol=s)
        pos[s] = signal(df, n, mode, long_only)
        rets[s] = df["close"].pct_change().shift(-1)
        funds[s] = funding_bar(s, df.index, tf)
    P = pd.DataFrame(pos).sort_index()
    R = pd.DataFrame(rets).reindex(P.index)
    Fu = pd.DataFrame(funds).reindex(P.index).fillna(0.0)
    # 等权：有信号的币平分 1.0 总敞口（原策略是单币满仓，组合化才好比）
    live = P.abs().sum(axis=1).replace(0, np.nan)
    W = P.div(live, axis=0).fillna(0.0)
    g = (W * R).sum(axis=1).dropna()
    return {"gross": g,
            "turn": W.diff().abs().sum(axis=1).reindex(g.index).fillna(0.0),
            "fund": -(W * Fu).sum(axis=1).reindex(g.index).fillna(0.0),
            "bh": R.mean(axis=1).reindex(g.index), "W": W, "tf": tf}


def stats(r: dict, fee: float, slip: float) -> dict:
    ann = BARS_PER_YEAR[r["tf"]]
    g, turn, fund = r["gross"], r["turn"], r["fund"]
    cost = turn * (fee + slip) * 1e-4
    net = g - cost + fund
    yrs = len(g) / ann
    sd = float(net.std())
    eq = (1 + net).cumprod()
    return {"毛收益": float(g.mean() * ann), "资金费率": float(fund.mean() * ann),
            "手续费+滑点": -float(cost.sum() / yrs), "净收益": float(net.mean() * ann),
            "净Sharpe": float(net.mean() / sd * math.sqrt(ann)) if sd > 0 else np.nan,
            "最大回撤": float((eq / eq.cummax() - 1).min()),
            "年换手": float(turn.sum() / yrs),
            "买入持有": float(r["bh"].mean() * ann), "_net": net}


def show(rows: list[dict], cols=("毛收益", "资金费率", "手续费+滑点", "净收益",
                                 "最大回撤", "买入持有")) -> None:
    t = pd.DataFrame(rows).set_index("版本")
    t = t.drop(columns=[c for c in t.columns if c.startswith("_")])
    for c in cols:
        if c in t:
            t[c] = t[c].map("{:+.1%}".format)
    if "净Sharpe" in t:
        t["净Sharpe"] = t["净Sharpe"].map("{:+.2f}".format)
    if "年换手" in t:
        t["年换手"] = t["年换手"].map("{:.0f}x".format)
    print(t.to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", default="1d", choices=["1d", "4h"])
    args = ap.parse_args()
    tf = args.tf
    ann = BARS_PER_YEAR[tf]

    print("=" * 116)
    print(f"Hull Suite Strategy 回测   周期 {tf}   六个币   2022-01-01 起")
    print("=" * 116)

    print("\n" + "-" * 116)
    print("A  原脚本的两种读法（原脚本不收成本；这里按币安真实费率收）")
    print("-" * 116 + "\n")
    rows = []
    for lbl, lo in (("多空反手（代码里写的）", False), ("只做多（默认设置实际跑的）", True)):
        r = run(long_only=lo, tf=tf)
        for cl, fee, slip in (("零成本（原脚本）", 0.0, 0.0),
                              ("挂单 2bp+1bp", MAKER_BPS, 1.0),
                              ("吃单 5bp+1.5bp", TAKER_BPS, 1.5)):
            s = stats(r, fee, slip)
            s["版本"] = f"{lbl} · {cl}"
            rows.append(s)
    show(rows)
    print("""
  第一行和第三行的差，就是「回测里写 commission=0」这一个设置的代价。
  反手系统每次换向要平旧仓再开等量反向仓，换手是单向系统的两倍。""")

    print("\n" + "-" * 116)
    print("B  和基准比：它比买入持有好吗？比一根普通均线好吗？")
    print("-" * 116 + "\n")

    def bench(name, fn, long_only=True):
        pos, rets, funds = {}, {}, {}
        for s in COINS:
            df = D.load(tf, symbol=s)
            sig = fn(df)
            pos[s] = sig if not long_only else sig.clip(lower=0)
            rets[s] = df["close"].pct_change().shift(-1)
            funds[s] = funding_bar(s, df.index, tf)
        P = pd.DataFrame(pos)
        R = pd.DataFrame(rets).reindex(P.index)
        Fu = pd.DataFrame(funds).reindex(P.index).fillna(0.0)
        live = P.abs().sum(axis=1).replace(0, np.nan)
        W = P.div(live, axis=0).fillna(0.0)
        g = (W * R).sum(axis=1).dropna()
        rr = {"gross": g, "turn": W.diff().abs().sum(axis=1).reindex(g.index).fillna(0.0),
              "fund": -(W * Fu).sum(axis=1).reindex(g.index).fillna(0.0),
              "bh": R.mean(axis=1).reindex(g.index), "tf": tf}
        s = stats(rr, TAKER_BPS, 1.5)
        s["版本"] = name
        return s

    rows = []
    r0 = run(tf=tf)
    s0 = stats(r0, TAKER_BPS, 1.5)
    s0["版本"] = "Hull Suite 多空反手"
    rows.append(s0)
    r1 = run(long_only=True, tf=tf)
    s1 = stats(r1, TAKER_BPS, 1.5)
    s1["版本"] = "Hull Suite 只做多"
    rows.append(s1)
    rows.append(bench("对照：close > SMA(55)",
                      lambda d: np.sign(d["close"] - IND.sma(d["close"], 55))))
    rows.append(bench("对照：EMA(55) 在涨",
                      lambda d: np.sign(IND.ema(d["close"], 55).diff(2))))
    rows.append(bench("对照：SMA(55) 在涨",
                      lambda d: np.sign(IND.sma(d["close"], 55).diff(2))))
    bh = float(r0["bh"].mean() * ann)
    rows.append({"版本": "等权买入持有", "毛收益": bh, "资金费率": 0.0,
                 "手续费+滑点": 0.0, "净收益": bh,
                 "净Sharpe": float(r0["bh"].mean() / r0["bh"].std() * math.sqrt(ann)),
                 "最大回撤": float(((1 + r0["bh"]).cumprod() /
                                (1 + r0["bh"]).cumprod().cummax() - 1).min()),
                 "年换手": 0.0, "买入持有": bh})
    show(rows)

    print("\n" + "-" * 116)
    print("C  参数网格：length × Hull 变体（净 Sharpe，吃单 5bp+1.5bp，多空反手）")
    print("-" * 116 + "\n")
    lens = [21, 34, 55, 89, 144, 200]
    modes = ["Hma", "Ehma", "Thma"]
    g = pd.DataFrame(index=lens, columns=modes, dtype=float)
    for n in lens:
        for m in modes:
            g.loc[n, m] = stats(run(n=n, mode=m, tf=tf), TAKER_BPS, 1.5)["净Sharpe"]
    g.index.name, g.columns.name = "length", "变体"
    print(g.map("{:+.2f}".format).to_string())
    f = g.values.flatten()
    print(f"\n  原脚本默认那格 (55, Hma) = {g.loc[55, 'Hma']:+.2f}，"
          f"在 {g.size} 格里排第 {int((f > g.loc[55, 'Hma']).sum()) + 1}")
    print(f"  中位 {np.median(f):+.2f}   范围 {f.min():+.2f}~{f.max():+.2f}   "
          f"为正 {int((f > 0).sum())}/{g.size}")

    print("\n" + "-" * 116)
    print("D  去偏 Sharpe")
    print("-" * 116 + "\n")
    net = s0["_net"]
    sh = float(net.mean() / net.std() * math.sqrt(ann))
    print(f"  多空反手 · 吃单，净 Sharpe {sh:+.2f}，样本 {len(net)} 根\n")
    print(f"  {'试过的组数':>12}   {'P(真实Sharpe>0)':>16}")
    for k in (10, 18, 100, 1000):
        print(f"  {k:>12}   "
              f"{M.deflated_sharpe(sh, n_trials=k, n_obs=len(net), ann_factor=ann, skew=float(net.skew()), kurt=float(net.kurtosis() + 3)):>16.3f}")

    print("\n" + "-" * 116)
    print("E  逐年（**复利实际收益**，不是算术年化）")
    print("-" * 116 + "\n")
    # 单个年度不能用 mean×365：那是算术年化，在大跌的年份会算出「−102.7%」
    # 这种不可能的数 —— 亏损的下限是 −100%。年度要用 ∏(1+r)−1。
    for y, seg in net.groupby(net.index.year):
        b = r0["bh"].reindex(seg.index).fillna(0.0)
        sr = float((1 + seg).prod() - 1)
        br = float((1 + b).prod() - 1)
        print(f"   {y}  策略 {sr:+7.1%}   Sharpe {seg.mean() / seg.std() * math.sqrt(ann):+5.2f}"
              f"   |  买入持有 {br:+7.1%}   {'策略赢' if sr > br else '买入持有赢'}   {len(seg)} 根")


if __name__ == "__main__":
    main()
