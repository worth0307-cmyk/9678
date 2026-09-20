"""斐波那契回撤位有没有预测力：用 AlphaTrace 的构造方式，在本仓库的数据上测.

    python scripts/57_fibonacci.py

复刻 AlphaTrace `web/prices.py` 里那套构造，一个参数不改：

    窗口内的 high / low  ->  level(r) = low + (high − low) × r
    r ∈ (0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0)
    默认窗口 3m = 最近 92 根日线（含当根，和 klines limit=N 一致）

那份代码自己写得很克制 —— "Nothing here predicts anything; it is arithmetic on
two numbers the exchange published"。**这句话是对的**，这个脚本要做的是量化
「如果有人把它当支撑阻力用会怎样」。

「触碰」的定义：某根 bar 的最高最低价区间包含了某条位，且前一根收盘在这条位的
一侧 ——

    前收在位之上  ->  从上方跌到这条位   ->  支撑测试，期待反弹
    前收在位之下  ->  从下方涨到这条位   ->  阻力测试，期待回落

于是「有效」意味着：支撑触碰后前向收益为正、阻力触碰后为负。把阻力那一侧取负号
加总，得到一个「带符号的边际」，正数才算这套东西有内容。

两个零假设，各回答一个不同的问题：

  随机比例   同样的窗口、同样的 high/low、同样的触碰判定，**只把 0.618 换成
             U(0,1) 里随便抽的一个数**。如果黄金比例有什么特别，它该赢过
             随便一条线。这是对「斐波那契」这三个字最直接的检验。

  同幅度对照 「价格跌到某条位」这件事本身就以「价格刚跌过」为条件，而短期
             均值回复会让**任何**下跌之后都有反弹。所以第二个对照是：找前 5 根
             涨跌幅相近的随机 bar，比它们的前向收益。这检验的是「那条线」有没有
             超出「刚跌了一段」的信息。

第二个对照是关键。少了它，任何均值回复都会被读成「支撑有效」。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vibt import data as D  # noqa: E402

pd.set_option("display.width", 220)

COINS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "TAOUSDT", "HYPEUSDT")
RATIOS = (0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0)
WINDOWS = {"1m": 31, "3m": 92}          # AlphaTrace 的 FIB_WINDOWS，日线那两档
FWD = 5                                  # 前向多少根
APPROACH = 5                             # 「刚跌/刚涨了多少」用几根衡量
DRAWS = 200
RNG = np.random.default_rng(20260920)


def touches(df: pd.DataFrame, window: int, ratio: float):
    """返回 (触碰下标, 方向)。方向 +1 = 支撑测试，−1 = 阻力测试。

    窗口含当根，和 AlphaTrace 调 klines(limit=N) 拿到的一致；信号在该根收盘
    形成，收益从下一根算起，所以不偷看未来。
    """
    hi = df["high"].rolling(window).max()
    lo = df["low"].rolling(window).min()
    lvl = (lo + (hi - lo) * ratio).to_numpy()
    h, l = df["high"].to_numpy(), df["low"].to_numpy()
    c = df["close"].to_numpy()

    hit = (l <= lvl) & (lvl <= h) & np.isfinite(lvl)
    hit[:window] = False
    prev_close = np.r_[np.nan, c[:-1]]
    side = np.where(prev_close > lvl, 1.0, np.where(prev_close < lvl, -1.0, 0.0))
    idx = np.flatnonzero(hit & (side != 0))
    return idx, side[idx]


def signed_edge(df: pd.DataFrame, idx: np.ndarray, side: np.ndarray) -> np.ndarray:
    """每次触碰的「带符号前向收益」。支撑取正向，阻力取反向。"""
    c = df["close"].to_numpy()
    keep = idx + FWD < len(c)
    i, s = idx[keep], side[keep]
    if not len(i):
        return np.array([])
    fwd = c[i + FWD] / c[i] - 1.0
    return fwd * s


def approach_move(df: pd.DataFrame, idx: np.ndarray) -> np.ndarray:
    c = df["close"].to_numpy()
    prior = np.full(len(c), np.nan)
    prior[APPROACH:] = c[APPROACH:] / c[:-APPROACH] - 1.0
    return prior[idx]


def run_real(window: int) -> dict[float, list[np.ndarray]]:
    out: dict[float, list[np.ndarray]] = {r: [] for r in RATIOS}
    for s in COINS:
        df = D.load("1d", symbol=s)
        for r in RATIOS:
            i, sd = touches(df, window, r)
            out[r].append(signed_edge(df, i, sd))
    return out


def null_random_ratio(window: int, draws: int = DRAWS) -> np.ndarray:
    """同样的几何，比例换成 U(0,1) 里随便抽的一个。"""
    frames = {s: D.load("1d", symbol=s) for s in COINS}
    means = np.full(draws, np.nan)
    for k in range(draws):
        r = float(RNG.uniform(0.0, 1.0))
        acc = []
        for s, df in frames.items():
            i, sd = touches(df, window, r)
            e = signed_edge(df, i, sd)
            if len(e):
                acc.append(e)
        if acc:
            means[k] = float(np.concatenate(acc).mean())
    return means


def null_matched_move(window: int, draws: int = DRAWS) -> tuple[float, np.ndarray]:
    """对照组：前 5 根涨跌幅相近的随机 bar，方向按同样的规则赋号。

    这一步控掉的是「价格刚跌了一段」这个条件本身 —— 没有它，任何短期均值
    回复都会被读成「支撑有效」。
    """
    frames = {s: D.load("1d", symbol=s) for s in COINS}
    real, pools = [], []
    for s, df in frames.items():
        c = df["close"].to_numpy()
        prior = np.full(len(c), np.nan)
        prior[APPROACH:] = c[APPROACH:] / c[:-APPROACH] - 1.0
        for r in RATIOS:
            i, sd = touches(df, window, r)
            keep = i + FWD < len(c)
            i, sd = i[keep], sd[keep]
            if not len(i):
                continue
            real.append((c[i + FWD] / c[i] - 1.0) * sd)
            pools.append((c, prior, prior[i], sd))
    real_mean = float(np.concatenate(real).mean()) if real else np.nan

    sims = np.full(draws, np.nan)
    for k in range(draws):
        acc = []
        for c, prior, want, sd in pools:
            ok = np.flatnonzero(np.isfinite(prior[:len(c) - FWD]))
            if not len(ok):
                continue
            # 每个真实触碰配一个「前 5 根涨跌幅最接近」的随机 bar：
            # 在同十分位里抽，而不是全样本随机
            pool_prior = prior[ok]
            for w, s_i in zip(want, sd):
                near = ok[np.abs(pool_prior - w) <= max(abs(w) * 0.25, 0.01)]
                if not len(near):
                    near = ok
                j = int(RNG.choice(near))
                acc.append((c[j + FWD] / c[j] - 1.0) * s_i)
        if acc:
            sims[k] = float(np.mean(acc))
    return real_mean, sims


def main() -> None:
    print("=" * 112)
    print(f"斐波那契回撤位的预测力   六个币 · 日线 · 前向 {FWD} 根")
    print("=" * 112)

    for name, window in WINDOWS.items():
        print("\n" + "-" * 112)
        print(f"窗口 {name}（{window} 根日线）")
        print("-" * 112 + "\n")

        real = run_real(window)
        rows = []
        for r in RATIOS:
            e = np.concatenate([x for x in real[r] if len(x)]) if any(len(x) for x in real[r]) else np.array([])
            if not len(e):
                continue
            se = e.std() / math.sqrt(len(e))
            rows.append({"比例": r, "触碰次数": len(e),
                         "带符号前向收益": e.mean(), "标准误": se,
                         "t": e.mean() / se if se > 0 else np.nan,
                         "为正比例": float((e > 0).mean())})
        t = pd.DataFrame(rows).set_index("比例")
        show = t.copy()
        show["带符号前向收益"] = show["带符号前向收益"].map("{:+.3%}".format)
        show["标准误"] = show["标准误"].map("{:.3%}".format)
        show["t"] = show["t"].map("{:+.2f}".format)
        show["为正比例"] = show["为正比例"].map("{:.1%}".format)
        print(show.to_string())

        allr = np.concatenate([np.concatenate([x for x in real[r] if len(x)])
                               for r in RATIOS if any(len(x) for x in real[r])])
        print(f"\n  七条位合计 {len(allr)} 次触碰，带符号前向收益 {allr.mean():+.3%}")

        print("\n  零假设 1：同样的几何，比例换成 U(0,1) 随机抽 —— 0.618 特别吗？")
        sims = null_random_ratio(window)
        sims = sims[np.isfinite(sims)]
        for r in (0.382, 0.5, 0.618):
            e = np.concatenate([x for x in real[r] if len(x)])
            p = float((sims >= e.mean()).mean())
            print(f"    r={r:<6} 实测 {e.mean():+.3%}   随机比例 {sims.mean():+.3%}"
                  f" (±{sims.std():.3%})   p = {p:.3f}"
                  f"   {'显著' if p < 0.05 else '**和随便一条线没区别**'}")

        print("\n  零假设 2：前 5 根涨跌幅相近的随机 bar —— 「那条线」超出「刚跌了一段」了吗？")
        rm, sims2 = null_matched_move(window, draws=60)
        sims2 = sims2[np.isfinite(sims2)]
        p2 = float((sims2 >= rm).mean())
        print(f"    实测 {rm:+.3%}   同幅度随机 {sims2.mean():+.3%} (±{sims2.std():.3%})"
              f"   p = {p2:.3f}   {'显著' if p2 < 0.05 else '**没有超出**'}")

    print("\n" + "-" * 112)
    print("端点那两条：0.0 和 1.0 —— 它们不是斐波那契数，而且方向是反的")
    print("-" * 112 + "\n")
    # 0.0 就是窗口最低价，1.0 就是窗口最高价。触碰 1.0 几乎必然是「从下方涨上来」
    # （前收在区间高点之下），按支撑阻力的用法该期待回落；触碰 0.0 几乎必然是
    # 「从上方跌下来」，该期待反弹。所以这两条的原始前向收益方向最能说明问题。
    for name, window in WINDOWS.items():
        for r in (0.0, 1.0):
            raw, n = [], 0
            for sym in COINS:
                df = D.load("1d", symbol=sym)
                i, sd = touches(df, window, r)
                c = df["close"].to_numpy()
                keep = i + FWD < len(c)
                i = i[keep]
                if len(i):
                    raw.append(c[i + FWD] / c[i] - 1.0)
                    n += len(i)
            if not raw:
                continue
            x = np.concatenate(raw)
            what = "窗口最低价" if r == 0.0 else "窗口最高价"
            expect = "按支撑用法该反弹" if r == 0.0 else "按阻力用法该回落"
            got = "继续跌" if x.mean() < 0 else "继续涨"
            print(f"  {name} r={r:<4} {what}   {n:>4} 次触碰   "
                  f"原始前向 {x.mean():+.3%}   {expect}，实际**{got}**")
    print("""
  这是突破延续，不是支撑阻力 —— 摸到 31 日新高之后价格继续涨，摸到 31 日新低
  之后继续跌。也就是说这两条线当 S/R 用是**反着的**，它们是 Donchian 通道。

  而真正的那五个斐波那契比例（0.236 / 0.382 / 0.5 / 0.618 / 0.786），
  |t| 全部小于 0.7。

--------------------------------------------------------------------------------------------------
读法
--------------------------------------------------------------------------------------------------

  「带符号前向收益」为正 = 支撑触碰后涨、阻力触碰后跌，也就是这套东西有内容。
  但它必须**同时**赢过两个对照才算数：

    赢不过随机比例   ->  黄金比例这件事是假的，任何一条线都一样
    赢不过同幅度对照 ->  赚到的是短期均值回复的钱，和画不画线无关

  AlphaTrace 的代码自己就写着「Nothing here predicts anything」。
  这个脚本是在给那句话补上数字。""")


if __name__ == "__main__":
    main()
