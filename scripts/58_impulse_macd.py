"""Impulse MACD [LazyBear]：一个带死区的通道突破，不是 MACD.

    python scripts/58_impulse_macd.py

拆开原脚本：

    hi = SMMA(high, 34)         平滑高价
    lo = SMMA(low,  34)         平滑低价
    mi = ZLEMA(hlc3, 34)        零延迟中线 = 2·EMA − EMA(EMA)

    md = mi > hi ? (mi − hi) : mi < lo ? (mi − lo) : 0
    sb = SMA(md, 9)             信号线
    sh = md − sb                柱状

MACD 是同一序列两条 EMA 之差；这个是「快线跑出慢通道多远」。名字不同于实质。

**唯一和本仓库测过的东西不一样的结构，是 `md == 0` 那个死区。** 通道内输出恒为
零，也就是一个显式的「现在没有信号」状态。53 号脚本已经证明四个趋势过滤器
（Hull / VIDYA / Supertrend / 回调规则）的全部价值就是「有时候空仓」——而这个
指标把空仓做成了一等状态。所以问题很具体：

    显式的死区，能不能赢过「直接少持仓」？

那正是 53 号那套零假设，这里原样再用一遍。

另外两件要查的：

  ZLEMA 会过冲   2·EMA − EMA(EMA) 是外推，减少延迟的代价是转折处冲过头。
                 而「冲过头」恰好就是触发通道突破的东西 —— 所以有一部分信号
                 是滤波器的产物，不是价格的。消融掉它（换成普通 EMA）就能看出
                 有多少。

  三种用法       sign(md) 是通道突破；md > sb 是「MACD 式」的信号线交叉；
                 sh 的符号是柱状翻红翻绿。交易者三种都用，所以三种都测。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vibt import data as D, indicators as IND, ingest as I, metrics as M  # noqa: E402

pd.set_option("display.width", 240)
ROOT = Path(__file__).resolve().parent.parent
COINS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "TAOUSDT", "HYPEUSDT")
TAKER_BPS, SLIP_BPS = 5.0, 1.5
ANN = 365.0
RNG = np.random.default_rng(20260920)


# ---------------------------------------------------------------- 指标

def smma(s: pd.Series, n: int) -> pd.Series:
    """Wilder 平滑，按 Pine 的播种方式：第一个值是 sma(n)，之后递推。

    直接用 ewm(alpha=1/n) 会从第一根就开始递推，前 n 根对不上原脚本。
    """
    x = s.to_numpy(dtype=float)
    seed = s.rolling(n).mean().to_numpy()
    out = np.full(len(x), np.nan)
    prev = np.nan
    for i in range(len(x)):
        if np.isnan(prev):
            if not np.isnan(seed[i]):
                prev = seed[i]
                out[i] = prev
        else:
            prev = (prev * (n - 1) + x[i]) / n
            out[i] = prev
    return pd.Series(out, index=s.index)


def zlema(s: pd.Series, n: int) -> pd.Series:
    """2·EMA − EMA(EMA)。外推，所以在转折处会冲过通道。"""
    e1 = IND.ema(s, n)
    return e1 + (e1 - IND.ema(e1, n))


def impulse(df: pd.DataFrame, n: int = 34, sig: int = 9,
            center: str = "zlema", band: str = "smma") -> pd.DataFrame:
    src = (df["high"] + df["low"] + df["close"]) / 3.0
    if band == "smma":
        hi, lo = smma(df["high"], n), smma(df["low"], n)
    elif band == "ema":                       # 消融：通道换成普通 EMA
        hi, lo = IND.ema(df["high"], n), IND.ema(df["low"], n)
    else:
        raise ValueError(band)
    mi = zlema(src, n) if center == "zlema" else IND.ema(src, n)

    md = pd.Series(np.where(mi > hi, mi - hi,
                            np.where(mi < lo, mi - lo, 0.0)), index=df.index)
    md = md.where(hi.notna() & lo.notna() & mi.notna())
    sb = md.rolling(sig).mean()
    return pd.DataFrame({"md": md, "sb": sb, "sh": md - sb,
                         "hi": hi, "lo": lo, "mi": mi})


# ---------------------------------------------------------------- 回测

_R, _F = None, None


def panel():
    global _R, _F
    if _R is None:
        rets, funds = {}, {}
        for s in COINS:
            df = D.load("1d", symbol=s)
            rets[s] = df["close"].pct_change().shift(-1)
            fr = I._load_funding(ROOT / "data" / "funding" / f"{s}_funding.csv.gz")
            fr = fr["funding_rate"].astype(float)
            funds[s] = fr.groupby(fr.index.floor("D")).sum().reindex(df.index).fillna(0.0)
        _R, _F = pd.DataFrame(rets), pd.DataFrame(funds)
    return _R, _F


def evaluate(P: pd.DataFrame, fee=TAKER_BPS, slip=SLIP_BPS) -> dict:
    R, Fu = panel()
    W = P / len(P.columns)                    # 每个币固定 1/N，不互相牵动
    R = R.reindex(P.index)[P.columns]
    Fu = Fu.reindex(P.index)[P.columns].fillna(0.0)
    g = (W * R).sum(axis=1)
    turn = W.diff().abs().sum(axis=1).fillna(W.abs().sum(axis=1))
    fund = -(W * Fu).sum(axis=1)
    net = (g - turn * (fee + slip) * 1e-4 + fund).dropna()
    yrs = len(net) / ANN
    sd = float(net.std())
    eq = (1 + net).cumprod()
    return {"年化": float(net.sum() / yrs), "Sharpe": float(net.mean() / sd * math.sqrt(ANN)) if sd else np.nan,
            "最大回撤": float((eq / eq.cummax() - 1).min()),
            "年换手": float(turn.sum() / yrs),
            "平均敞口": float(W.abs().sum(axis=1).reindex(net.index).mean()),
            "_net": net}


def signals(mode: str, long_only: bool, **kw) -> pd.DataFrame:
    out = {}
    for s in COINS:
        x = impulse(D.load("1d", symbol=s), **kw)
        if mode == "md":
            v = np.sign(x["md"])              # 通道突破，死区 = 0
        elif mode == "cross":
            v = np.sign(x["md"] - x["sb"])    # 信号线交叉，没有死区
        elif mode == "sh":
            v = np.sign(x["sh"])
        else:
            raise ValueError(mode)
        out[s] = v.clip(lower=0) if long_only else v
    P = pd.DataFrame(out).dropna(how="all")
    return P.fillna(0.0)


def show(rows):
    t = pd.DataFrame(rows).set_index("版本")
    t = t.drop(columns=[c for c in t.columns if c.startswith("_")])
    for c in ("年化", "最大回撤", "平均敞口"):
        t[c] = t[c].map("{:+.1%}".format)
    t["Sharpe"] = t["Sharpe"].map("{:+.2f}".format)
    t["年换手"] = t["年换手"].map("{:.0f}x".format)
    print(t.to_string())


def main() -> None:
    print("=" * 118)
    print("Impulse MACD [LazyBear]   六个币 · 日线 · 2022 起 · 吃单 5bp+1.5bp + 真实资金费率")
    print("=" * 118)

    print("\n" + "-" * 118)
    print("A  死区有多大 —— 这是它唯一的结构性特点")
    print("-" * 118 + "\n")
    rows = []
    for s in COINS:
        x = impulse(D.load("1d", symbol=s)).dropna()
        md = x["md"]
        flips = int((np.sign(md) != np.sign(md).shift()).sum())
        rows.append({"币": s, "根数": len(md),
                     "死区占比": float((md == 0).mean()),
                     "多头占比": float((md > 0).mean()),
                     "空头占比": float((md < 0).mean()),
                     "状态翻转次数": flips,
                     "平均持续(根)": len(md) / max(flips, 1)})
    t = pd.DataFrame(rows).set_index("币")
    for c in ("死区占比", "多头占比", "空头占比"):
        t[c] = t[c].map("{:.1%}".format)
    t["平均持续(根)"] = t["平均持续(根)"].map("{:.1f}".format)
    print(t.to_string())

    print("\n" + "-" * 118)
    print("B  三种用法 × 多空 / 只做多")
    print("-" * 118 + "\n")
    rows = []
    for mode, name in (("md", "sign(md) 通道突破（有死区）"),
                       ("cross", "md > sb 信号线交叉（无死区）"),
                       ("sh", "sh 柱状符号（无死区）")):
        for lo, tag in ((False, "多空"), (True, "只做多")):
            r = evaluate(signals(mode, lo))
            r["版本"] = f"{name} · {tag}"
            rows.append(r)
    R, _ = panel()
    bh = R.mean(axis=1).dropna()
    eqb = (1 + bh).cumprod()
    rows.append({"版本": "等权买入持有", "年化": float(bh.sum() / (len(bh) / ANN)),
                 "Sharpe": float(bh.mean() / bh.std() * math.sqrt(ANN)),
                 "最大回撤": float((eqb / eqb.cummax() - 1).min()),
                 "年换手": 0.0, "平均敞口": 1.0})
    show(rows)

    print("\n" + "-" * 118)
    print("C  消融：ZLEMA 的过冲贡献了多少？通道换成 EMA 呢？")
    print("-" * 118 + "\n")
    rows = []
    for center, band, name in (("zlema", "smma", "原版 ZLEMA 中线 + SMMA 通道"),
                               ("ema", "smma", "中线换成普通 EMA"),
                               ("zlema", "ema", "通道换成 EMA"),
                               ("ema", "ema", "两个都换成 EMA（见下）")):
        r = evaluate(signals("md", True, center=center, band=band))
        r["版本"] = name
        rows.append(r)
    show(rows)
    print("""
  最后一行不是 bug，是**数学必然**：low ≤ hlc3 ≤ high 逐根成立，而 EMA 是
  非负权重的凸组合，所以 EMA(low) ≤ EMA(hlc3) ≤ EMA(high) 逐点成立 ——
  中线永远出不了通道，md 恒为 0。（EMA 13/34/89 各 1600+ 根，越界 0 次。）

  **所以这个指标的信号全部来自两个滤波器的速度差，不来自价格。**
  SMMA(alpha=1/34) 的等效 EMA span 是 67，比 EMA(34) 慢一倍；ZLEMA 还要再
  外推一次。原版出界 86.8% 的时间，换成 EMA 中线降到 72.0% —— 差的那部分
  就是 ZLEMA 过冲造出来的。""")

    print("\n" + "-" * 118)
    print("D  和 53 号同一个零假设：显式死区能赢过「直接少持仓」吗")
    print("-" * 118 + "\n")
    def dd(x):
        e = (1 + x).cumprod()
        return float((e / e.cummax() - 1).min())

    # 两个变体都要测。只测 sign(md) 的话，是在拿**较弱**的那个去对零假设 ——
    # 而 B 表里最好的是信号线交叉（Sharpe +0.78），该被检验的是它。
    nets = {}
    for mode, name in (("md", "通道突破 sign(md)"), ("cross", "信号线交叉 md>sb")):
        r = evaluate(signals(mode, True))
        expo, net = r["平均敞口"], r["_net"]
        bh_s = bh.reindex(net.index).fillna(0.0)
        target = dd(net)
        lo_f, hi_f = 0.0, 1.0
        for _ in range(50):
            mid = (lo_f + hi_f) / 2
            if dd(bh_s * mid) < target:
                hi_f = mid
            else:
                lo_f = mid
        f = (lo_f + hi_f) / 2
        print(f"\n  【{name} · 只做多】")
        print(f"  {'':30s} {'年化':>8} {'Sharpe':>8} {'最大回撤':>9} {'敞口':>7}")
        for lbl, x, e in ((f"  Impulse {name}", net, expo),
                          (f"  同敞口买入持有（{expo:.0%} 仓）", bh_s * expo, expo),
                          (f"  同回撤买入持有（{f:.0%} 仓）", bh_s * f, f),
                          ("  买入持有 满仓", bh_s, 1.0)):
            print(f"  {lbl:30s} {x.sum() / (len(x) / ANN):+8.1%} "
                  f"{x.mean() / x.std() * math.sqrt(ANN):+8.2f} {dd(x):+9.1%} {e:+7.0%}")
        a, b = net.to_numpy(), bh_s.to_numpy()
        nblk, blk = int(np.ceil(len(a) / 20)), 20
        real = (a.mean() / a.std() - b.mean() / b.std()) * math.sqrt(ANN)
        diffs = np.empty(2000)
        for k in range(2000):
            st = RNG.integers(0, len(a), nblk)
            sel = np.concatenate([np.arange(s0, s0 + blk) % len(a) for s0 in st])[:len(a)]
            xa, xb = a[sel], b[sel]
            diffs[k] = (xa.mean() / xa.std() - xb.mean() / xb.std()) * math.sqrt(ANN)
        pv = float((diffs <= 0).mean())
        print(f"    Sharpe 差 {real:+.2f}   分块自助法 p = {pv:.3f}   "
              f"{'显著' if pv < 0.05 else '**不显著**'}")
        nets[mode] = net
    net = nets["cross"]

    print("\n" + "-" * 118)
    print("E  去偏 Sharpe")
    print("-" * 118 + "\n")
    sh = float(net.mean() / net.std() * math.sqrt(ANN))
    print(f"  净 Sharpe {sh:+.2f}，样本 {len(net)} 天\n")
    print(f"  {'试过的组数':>12}   {'P(真实Sharpe>0)':>16}")
    for k in (10, 20, 100, 1000):
        print(f"  {k:>12}   {M.deflated_sharpe(sh, n_trials=k, n_obs=len(net), ann_factor=ANN, skew=float(net.skew()), kurt=float(net.kurtosis() + 3)):>16.3f}")

    print("""
------------------------------------------------------------------------------------------------------------------
读法
------------------------------------------------------------------------------------------------------------------

  1  它不是 MACD。MACD 是同一序列两条 EMA 之差，这个是「快线跑出慢通道多远」，
     一个带死区的通道突破。

  2  两条画在图上的线是**同一个信号**。sh = md − sb，所以 sign(sh) 恒等于
     sign(md − sb) —— B 表里 `sh 柱状符号` 和 `md > sb` 每一位数字都一样。
     柱子翻色和信号线交叉是同一件事，看两遍不会多出信息。

  3  死区只占 12.8%–19.4% 的时间。「通道内 = 没信号」听起来像个大过滤器，
     实际上八成以上的 bar 都在通道外。

  4  信号全部来自滤波器速度差。两个滤波器换成同一族（都 EMA）时 md 恒为 0 ——
     这是凸组合的数学必然，不是参数没调好。所以这个指标度量的是
     「ZLEMA 比 SMMA 快多少」，价格只是载体。

  5  最好的那个变体（信号线交叉，Sharpe +0.78）**没有通过去杠杆零假设**。
     它赢同敞口买入持有 +0.19 Sharpe，p = 0.280；53 号脚本里 Hull / VIDYA /
     Supertrend / 回调规则也是这样倒下的。它确实把回撤从 −71.7% 压到 −35.8%，
     但那件事把仓位砍到 39% 也能做到，不需要指标。

  6  去偏 Sharpe：在这个仓库里已经试过的变体数量级上（几十组），
     P(真实 Sharpe > 0) 只有 0.4–0.5。等于抛硬币。""")


if __name__ == "__main__":
    main()
