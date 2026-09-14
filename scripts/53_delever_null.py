"""趋势过滤 vs 单纯少持仓：把这一路反复出现的那个现象钉死.

    python scripts/53_delever_null.py

49、51、52 三个脚本，四个完全不同的指标（Hull / VIDYA / 经典 Supertrend /
回调规则），都给出同一个形状：**Sharpe 略升，回撤大降，代价是三分之一的时间
不在场**。在这个项目里「换个实现还在」是稳健性的唯一证据，所以这个现象大概率
是真的。

但有一个显而易见的零假设一直没测：**直接少持仓**。

60% 仓位买入持有、40% 现金，回撤自然从 −75% 降到 −45% 左右，而 **Sharpe 完全
不变** —— Sharpe 对杠杆免疫。所以问题是：

    趋势过滤把 Sharpe 从 +0.46 抬到 +0.57，是真的择时，
    还是仅仅等价于「平均只持 60% 仓」？

三种匹配方式，因为「公平」本身有三种读法：

  A  匹配平均敞口   把买入持有降到和过滤器一样的平均仓位，比收益、Sharpe、回撤
  B  匹配最大回撤   买入持有降到多少仓位才有同样的回撤？那时它赚多少？
                    —— 这一条最直观：「如果我只是少持一点直到回撤一样，
                    我赚得更多还是更少？」
  C  Sharpe 差显著吗  分块自助法。降杠杆不改变 Sharpe，所以这等价于问
                    「过滤器的 Sharpe 显著高于买入持有吗」。

口径统一成**每个币固定 1/N 仓位**（52 号脚本修过的那个）。51 号用的是
「有信号的币平分 1.0」，那会在只剩一两个币有信号时变成满仓押注，平均敞口不可比 ——
所以这里的数字和 51 号不同，是故意的。

降杠杆那一侧不收再平衡成本，也就是**把零假设做强**：如果过滤器连一个被优待的
对照都赢不了，那结论就很干净。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module  # noqa: E402

from vibt import data as D, ingest as I  # noqa: E402

_h = import_module("49_hull_suite")
_v = import_module("51_volumatic_vidya")
_t = import_module("52_trend_pullback")

pd.set_option("display.width", 240)
COINS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "TAOUSDT", "HYPEUSDT")
TAKER_BPS, SLIP_BPS = 5.0, 1.5
ANN = 365.0
TF = "1d"
RNG = np.random.default_rng(20260914)


def _panel():
    rets, funds = {}, {}
    for s in COINS:
        df = D.load(TF, symbol=s)
        rets[s] = df["close"].pct_change().shift(-1)
        fr = I._load_funding(Path("data/funding") / f"{s}_funding.csv.gz")["funding_rate"].astype(float)
        funds[s] = fr.groupby(fr.index.floor("D")).sum().reindex(df.index).fillna(0.0)
    return pd.DataFrame(rets), pd.DataFrame(funds)


R_ALL, FU_ALL = _panel()


def net_of(P: pd.DataFrame) -> tuple[pd.Series, float]:
    """固定 1/N 权重，扣手续费与资金费率，返回净收益序列和平均总敞口。"""
    W = P / len(P.columns)
    R = R_ALL.reindex(P.index)[P.columns]
    Fu = FU_ALL.reindex(P.index)[P.columns].fillna(0.0)
    g = (W * R).sum(axis=1)
    turn = W.diff().abs().sum(axis=1).fillna(W.abs().sum(axis=1))
    fund = -(W * Fu).sum(axis=1)
    net = (g - turn * (TAKER_BPS + SLIP_BPS) * 1e-4 + fund).dropna()
    return net, float(W.abs().sum(axis=1).reindex(net.index).mean())


def desc(net: pd.Series, expo: float = np.nan) -> dict:
    sd = float(net.std())
    eq = (1 + net).cumprod()
    return {"年化收益": float(net.mean() * ANN),
            "Sharpe": float(net.mean() / sd * math.sqrt(ANN)) if sd > 0 else np.nan,
            "年化波动": float(sd * math.sqrt(ANN)),
            "最大回撤": float((eq / eq.cummax() - 1).min()),
            "平均敞口": expo}


def max_dd(x: pd.Series) -> float:
    eq = (1 + x).cumprod()
    return float((eq / eq.cummax() - 1).min())


def filters() -> dict[str, pd.DataFrame]:
    out = {}
    out["经典 Supertrend 只做多"] = pd.DataFrame(
        {s: _v.signal(D.load(TF, symbol=s), kind="hl2", long_only=True) for s in COINS})
    out["VIDYA 只做多"] = pd.DataFrame(
        {s: _v.signal(D.load(TF, symbol=s), kind="vidya", long_only=True) for s in COINS})
    out["Hull(55) 只做多"] = pd.DataFrame(
        {s: _h.signal(D.load(TF, symbol=s), 55, "Hma", True) for s in COINS})
    out["回调进场 只做多"] = pd.DataFrame(
        {s: _t.positions(D.load(TF, symbol=s), exit_on_cross=False).clip(lower=0)
         for s in COINS})
    out = {k: v.dropna(how="all") for k, v in out.items()}
    # 四个过滤器的热身长度不同（ATR(200)、VIDYA 爬升、HMA(55)……），各自的
    # 有效起点因此不一样。不统一窗口的话，它们是在**不同的时间段**上互相比较，
    # 而买入持有的基准也会跟着变 —— 第一版就出现了同一个「等权买入持有」
    # 在不同行里是 +0.46 和 +0.88 的情况。
    common = None
    for v in out.values():
        common = v.index if common is None else common.intersection(v.index)
    return {k: v.reindex(common) for k, v in out.items()}


def levered_bh(index: pd.DatetimeIndex, f: float) -> pd.Series:
    """f 倍仓位的等权买入持有，逐日再平衡，**不收成本**（故意优待零假设）。"""
    return (R_ALL.reindex(index).mean(axis=1) * f).dropna()


def f_for_dd(index: pd.DatetimeIndex, target_dd: float) -> float:
    """二分找出「回撤刚好等于 target_dd」的买入持有仓位。"""
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if max_dd(levered_bh(index, mid)) < target_dd:   # 更深
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def block_bootstrap_sharpe_diff(a: pd.Series, b: pd.Series,
                                block: int = 20, draws: int = 2000) -> tuple[float, float]:
    """分块自助法检验 Sharpe(a) − Sharpe(b)。两条序列同步重采样，保留其相关性。"""
    idx = a.index.intersection(b.index)
    x, y = a.reindex(idx).to_numpy(), b.reindex(idx).to_numpy()
    n = len(x)
    nb = int(np.ceil(n / block))
    real = (x.mean() / x.std() - y.mean() / y.std()) * math.sqrt(ANN)
    diffs = np.empty(draws)
    for k in range(draws):
        starts = RNG.integers(0, n, nb)
        sel = np.concatenate([np.arange(s, s + block) % n for s in starts])[:n]
        xs, ys = x[sel], y[sel]
        diffs[k] = (xs.mean() / xs.std() - ys.mean() / ys.std()) * math.sqrt(ANN)
    # 单边 p：自助分布里「差值 <= 0」的比例
    return float(real), float((diffs <= 0).mean())


def main() -> None:
    print("=" * 118)
    print("趋势过滤 vs 单纯少持仓   六个币 · 日线 · 2022 起 · 吃单 5bp+1.5bp + 真实资金费率")
    print("=" * 118)

    fl = filters()
    idx0 = list(fl.values())[0].index
    bh_full = levered_bh(idx0, 1.0)
    print(f"\n  共同窗口 {idx0[0].date()} -> {idx0[-1].date()}   {len(idx0)} 天")
    print("  （四个过滤器的热身长度不同，取交集才能互相比。这段起点在 ATR(200) 之后，")
    print("   2022 上半年的下跌不在窗口内 —— 那对趋势过滤器是**不利**的，先说明。）")
    print(f"\n  等权买入持有（满仓）  年化 {bh_full.mean() * ANN:+.1%}   "
          f"Sharpe {bh_full.mean() / bh_full.std() * math.sqrt(ANN):+.2f}   "
          f"最大回撤 {max_dd(bh_full):.1%}")

    print("\n" + "-" * 118)
    print("A  匹配平均敞口：把买入持有降到和过滤器一样的仓位")
    print("-" * 118 + "\n")
    rows = []
    store = {}
    for name, P in fl.items():
        net, expo = net_of(P)
        store[name] = (net, expo)
        d = desc(net, expo)
        d["版本"] = name
        rows.append(d)
        b = levered_bh(net.index, expo)
        db = desc(b, expo)
        db["版本"] = f"  ↳ 同敞口买入持有（{expo:.0%} 仓）"
        rows.append(db)
    t = pd.DataFrame(rows).set_index("版本")
    for c in ("年化收益", "年化波动", "最大回撤", "平均敞口"):
        t[c] = t[c].map("{:+.1%}".format)
    t["Sharpe"] = t["Sharpe"].map("{:+.2f}".format)
    print(t.to_string())

    print("\n" + "-" * 118)
    print("B  匹配最大回撤：买入持有降到多少仓位才有同样的回撤？那时它赚多少？")
    print("-" * 118 + "\n")
    rows = []
    for name, (net, expo) in store.items():
        dd = max_dd(net)
        f = f_for_dd(net.index, dd)
        b = levered_bh(net.index, f)
        rows.append({"过滤器": name, "过滤器年化": float(net.mean() * ANN),
                     "过滤器回撤": dd, "等回撤仓位": f,
                     "买入持有年化": float(b.mean() * ANN),
                     "买入持有回撤": max_dd(b),
                     "差（过滤器−持有）": float(net.mean() * ANN - b.mean() * ANN)})
    t = pd.DataFrame(rows).set_index("过滤器")
    for c in t.columns:
        t[c] = t[c].map("{:+.1%}".format)
    print(t.to_string())

    print("\n" + "-" * 118)
    print("C  Sharpe 差显著吗（分块自助法，块长 20 天，2000 次）")
    print("-" * 118 + "\n")
    print(f"  {'过滤器':24s} {'Sharpe':>8} {'买入持有':>9} {'差':>7} {'p':>7}")
    for name, (net, expo) in store.items():
        b = levered_bh(net.index, 1.0)          # 降杠杆不改 Sharpe，直接和满仓比
        diff, p = block_bootstrap_sharpe_diff(net, b)
        sn = float(net.mean() / net.std() * math.sqrt(ANN))
        sb = float(b.mean() / b.std() * math.sqrt(ANN))
        flag = "显著" if p < 0.05 else "不显著"
        print(f"  {name:24s} {sn:>+8.2f} {sb:>+9.2f} {diff:>+7.2f} {p:>7.3f}  {flag}")
    print("""
  降杠杆不改变 Sharpe，所以这一栏等价于问「过滤器的 Sharpe 显著高于买入持有吗」。
  p 不显著 = 那 +0.1 的 Sharpe 分不出是择时还是噪声。""")

    print("\n" + "-" * 118)
    print("D  把「2022 上半年不在窗口内」这个偏向补掉")
    print("-" * 118)
    # 上面的共同窗口从 2022-10-28 起，2022 上半年的下跌不在里面 —— 而那正是
    # 趋势过滤器赚钱的地方，所以那个窗口对它们不利。Hull(55) 热身只要 ~62 根，
    # 能覆盖 2022 全年，用它单独检验「把崩盘算进来会不会翻盘」。
    P = pd.DataFrame({s: _h.signal(D.load(TF, symbol=s), 55, "Hma", True)
                      for s in COINS}).dropna(how="all")
    sh = lambda x: float(x.mean() / x.std() * math.sqrt(ANN))  # noqa: E731
    for lbl, idx in (("完整窗口（含 2022 上半年崩盘）", P.index),
                     ("共同窗口（从 2022-10-28 起）", P.index[P.index >= "2022-10-28"])):
        net, expo = net_of(P.reindex(idx))
        bh_f = levered_bh(net.index, 1.0)
        f = f_for_dd(net.index, max_dd(net))
        bh_d = levered_bh(net.index, f)
        diff, p = block_bootstrap_sharpe_diff(net, bh_f)
        print(f"""
  【{lbl}】{len(net)} 天
    Hull 过滤器     年化 {net.mean() * ANN:+6.1%}   Sharpe {sh(net):+.2f}   回撤 {max_dd(net):+.1%}   平均敞口 {expo:.0%}
    买入持有 满仓    年化 {bh_f.mean() * ANN:+6.1%}   Sharpe {sh(bh_f):+.2f}   回撤 {max_dd(bh_f):+.1%}
    买入持有 同回撤   年化 {bh_d.mean() * ANN:+6.1%}   （只需持 {f:.0%} 仓）   回撤 {max_dd(bh_d):+.1%}
    Sharpe 差 {diff:+.2f}   p = {p:.3f}   {'显著' if p < 0.05 else '**不显著**'}""")

    print("""
  完整窗口上，过滤器的 Sharpe 和买入持有**完全相同**（+0.60 对 +0.60，差 +0.01）。
  它做的全部事情就是把平均敞口从 100% 压到 40%。

  **直接持 42% 仓、其余现金**，能拿到同样的最大回撤（−36.9%）和几乎同样的收益
  （+15.8% 对 +18.2%，4.5 年里差 2.4 个百分点，p = 0.52）—— 而且不交一分钱手续费、
  不用盯盘、没有参数可调。

  这一条同时否掉四个工具：Hull、VIDYA、经典 Supertrend、回调规则。
  它们全都塌缩成同一件事 —— **少持仓**。""")


if __name__ == "__main__":
    main()
