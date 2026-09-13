"""为什么 TradingView 上那个 +3199% 看起来那么好.

    python scripts/50_why_it_looks_good.py

用户的两张回测截图本身就是证据：

    起点 2019-09-08  ->  总损益 +651%    最大回撤 72.03%   盈利交易 23/60
    起点 2020-02-10  ->  总损益 +3199%   最大回撤 45.95%   盈利交易 20/52

**起点差 5 个月，结果差 5 倍。** 一个真有边际的系统不会对起点这么敏感。

这个脚本把放大机制一条条量出来。注意本仓库的数据只到 2022-01-01，拿不到
2019~2021 那段（那正好是加密史上最大的一轮牛市，也正是那两张图收益的主要来源）。
所以下面不是复现那两个数字，而是**在能验证的区间上演示同样的机制**。

四个机制：

  1  100% 权益复利   TradingView 默认 default_qty_value=100 —— 每笔压上全部本金。
                     同一串收益率，复利和算术的「总收益」可以差好几倍，而
                     Sharpe 一模一样。总收益是被下注方式放大的，不是被边际。
  2  路径依赖        起点挪一点，结果变很多。这是「几笔大赢主导」的直接后果。
  3  笔数太少        52~60 笔横跨 6.5 年。去掉最好的几笔，看还剩什么。
  4  对照被藏起来    截图里「买入和持有」那条线的眼睛图标是关的。而那正是
                     唯一重要的对照 —— 在一轮 10 倍的牛市里，满仓做多赚钱是
                     默认结果，不是本事。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vibt import data as D, ingest as I  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module  # noqa: E402

_h = import_module("49_hull_suite")
hull, MAKER_BPS, TAKER_BPS = _h.hull, _h.MAKER_BPS, _h.TAKER_BPS

pd.set_option("display.width", 200)
SYM = "BTCUSDT"
ANN = 365.0


def series(sym: str = SYM, n: int = 55, long_only: bool = False):
    df = D.load("1d", symbol=sym)
    h = hull(df["close"], n, "Hma")
    up = h > h.shift(2)
    pos = pd.Series(np.where(up, 1.0, 0.0 if long_only else -1.0), index=df.index)
    pos = pos.where(h.notna() & h.shift(2).notna())
    r = df["close"].pct_change().shift(-1)
    fr = I._load_funding(Path("data/funding") / f"{sym}_funding.csv.gz")["funding_rate"].astype(float)
    fu = fr.groupby(fr.index.floor("D")).sum().reindex(df.index).fillna(0.0)
    m = pos.notna() & r.notna()
    return pos[m], r[m], fu[m]


def equity(pos, r, fu, fee=0.0, slip=0.0, funding=False):
    """TradingView 的口径：每笔压上全部权益，逐 bar 复利。"""
    gross = pos * r
    turn = pos.diff().abs().fillna(pos.abs())
    net = gross - turn * (fee + slip) * 1e-4 + (-(pos * fu) if funding else 0.0)
    return (1 + net).cumprod(), net


def trades(pos, r):
    """按仓位方向切分成「笔」，返回每笔的复利收益。"""
    out = []
    d = pos.to_numpy()
    rr = r.to_numpy()
    i = 0
    while i < len(d):
        j = i
        while j + 1 < len(d) and d[j + 1] == d[i]:
            j += 1
        if d[i] != 0:
            out.append(float(np.prod(1 + d[i] * rr[i:j + 1]) - 1))
        i = j + 1
    return np.array(out)


def main() -> None:
    print("=" * 100)
    print(f"为什么那个总损益看起来那么好   {SYM} 日线 Hull(55)")
    print("=" * 100)
    pos, r, fu = series()
    bh = (1 + r).cumprod()

    print("\n" + "-" * 100)
    print("机制 1  100% 权益复利：同一串收益率，两种下注方式")
    print("-" * 100 + "\n")
    _, net = equity(pos, r, fu, TAKER_BPS, 1.5, funding=True)
    comp = float((1 + net).prod() - 1)
    arith = float(net.sum())
    sh = float(net.mean() / net.std() * math.sqrt(ANN))
    vol = float(net.std() * math.sqrt(ANN))
    yrs = len(net) / ANN
    drag = vol ** 2 / 2
    print(f"  算术（每笔压固定金额）总收益   {arith:+.1%}   年化 {arith / yrs:+.1%}")
    print(f"  复利（每笔压全部本金）总收益   {comp:+.1%}   年化 {(1 + comp) ** (1 / yrs) - 1:+.1%}")
    print(f"  两者的 Sharpe                {sh:+.2f}（**完全相同**）")
    print(f"  年化波动 {vol:.0%}   波动拖累 σ²/2 = {drag:.1%}/年")
    print(f"""
  100% 权益复利是一个**双向**放大器，不是单向的：

      几何年化 ≈ 算术年化 − σ²/2 = {arith / yrs:+.1%} − {drag:.1%} = {arith / yrs - drag:+.1%}

  在这段数据上，{vol:.0%} 的年化波动产生 {drag:.0%}/年 的拖累，把 {arith / yrs:+.1%} 的算术收益
  压成负的。在**顺的**路径上，同一个放大器会把适度的边际堆成三位数 ——
  你那张 +3199% 就是。

  **两种情况下 Sharpe 完全不变。** 所以那个总损益数字说的是「下注方式 × 这段
  行情的路径」，不是「规则有多好」。而截图里没有 Sharpe。""")

    print("\n" + "-" * 100)
    print("机制 2  路径依赖：起点挪一挪")
    print("-" * 100 + "\n")
    ends = net.index[-1]
    rows = []
    for start in pd.date_range(net.index[0], "2025-06-01", freq="MS"):
        seg = net[net.index >= start]
        if len(seg) < 200:
            continue
        b = r[r.index >= start]
        rows.append({"起点": start.date(), "策略总收益": float((1 + seg).prod() - 1),
                     "买入持有": float((1 + b).prod() - 1),
                     "Sharpe": float(seg.mean() / seg.std() * math.sqrt(ANN))})
    t = pd.DataFrame(rows)
    print(f"  终点固定在 {ends.date()}，起点从 {t['起点'].iloc[0]} 逐月推到 {t['起点'].iloc[-1]}"
          f"（{len(t)} 个起点）\n")
    print(f"  策略总收益：最低 {t['策略总收益'].min():+.0%}   中位 {t['策略总收益'].median():+.0%}"
          f"   最高 {t['策略总收益'].max():+.0%}")
    print(f"  对应 Sharpe：最低 {t['Sharpe'].min():+.2f}   中位 {t['Sharpe'].median():+.2f}"
          f"   最高 {t['Sharpe'].max():+.2f}")
    best = t.loc[t["策略总收益"].idxmax()]
    worst = t.loc[t["策略总收益"].idxmin()]
    print(f"\n  最好的起点 {best['起点']}：{best['策略总收益']:+.0%}")
    print(f"  最差的起点 {worst['起点']}：{worst['策略总收益']:+.0%}")
    print(f"""
  同一套参数、同一个币、同一个终点，只是起点不同，总收益从 {worst['策略总收益']:+.0%}
  到 {best['策略总收益']:+.0%}。你那两张截图（+651% vs +3199%，起点差 5 个月）
  就是这件事的实例 —— **它量的是那段行情，不是那套规则。**""")

    print("\n" + "-" * 100)
    print("机制 3  笔数太少：去掉最好的几笔")
    print("-" * 100 + "\n")
    tr = trades(pos, r)
    order = np.argsort(tr)[::-1]
    print(f"  总共 {len(tr)} 笔，胜率 {(tr > 0).mean():.1%}，"
          f"盈利因子 {tr[tr > 0].sum() / abs(tr[tr < 0].sum()):.2f}")
    base = float(np.prod(1 + tr) - 1)
    print(f"  {'':>14} {'总收益':>10}")
    print(f"  {'全部':>14} {base:>+10.1%}")
    for k in (1, 2, 3, 5):
        keep = np.delete(tr, order[:k])
        print(f"  {'去掉最好 ' + str(k) + ' 笔':>14} {float(np.prod(1 + keep) - 1):>+10.1%}")
    top3 = tr[order[:3]].sum()
    print(f"""
  最好的 3 笔贡献了 {top3 / tr[tr > 0].sum():.0%} 的全部盈利。
  你截图里是 60 笔里赢 23 笔 —— **样本量根本不足以区分「有边际」和「运气好」**。""")

    print("\n" + "-" * 100)
    print("机制 4  被关掉的那条对照线")
    print("-" * 100 + "\n")
    print(f"  {SYM} 同期买入持有   {float(bh.iloc[-1] - 1):+.1%}")
    print(f"  Hull Suite 多空反手（含成本+资金费率）  {comp:+.1%}")
    pos_lo, r_lo, fu_lo = series(long_only=True)
    _, net_lo = equity(pos_lo, r_lo, fu_lo, TAKER_BPS, 1.5, funding=True)
    print(f"  Hull Suite 只做多（默认设置）           {float((1 + net_lo).prod() - 1):+.1%}")
    print("""
  截图里「买入和持有」那一行的眼睛图标是**关着的**。那是唯一重要的对照：
  在一轮 10 倍的牛市里，一个大部分时间满仓做多的系统赚钱是默认结果。
  要证明规则有用，得证明它**赢过**那条线，而不是证明它是正的。""")

    print("\n" + "-" * 100)
    print("机制 5  回撤：那个数字意味着什么")
    print("-" * 100 + "\n")
    eq, _ = equity(pos, r, fu, TAKER_BPS, 1.5, funding=True)
    dd = float((eq / eq.cummax() - 1).min())
    print(f"  本仓库区间的最大回撤 {dd:.1%}   你的截图：72.03% 和 45.95%")
    print(f"""
  72% 的回撤意味着账户从 100 万跌到 28 万，需要再涨 257% 才回本。
  而这是**日线收盘价重算**出来的，盘中更深。任何真实资金都撑不到曲线的右端。""")


if __name__ == "__main__":
    main()
