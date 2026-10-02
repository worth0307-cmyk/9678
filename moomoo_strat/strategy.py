"""把两个指标变成规则，逐只股票算信号，组合回测，和零假设比。

成交口径（股票）：第 t 天收盘算信号，第 t+1 天**开盘**成交；隔夜跳空算在旧仓位上。
每只股票固定 1/N 资金，只做多（「五浪涨完做空」只出逐笔统计，不进组合）。
费用按每边 bps 计，默认 5bp（美股免佣时的滑点 + 平台费量级；港股有印花税，建议 --cost-bps 15）。

规则（全部只做多）：
  S0  基准：收盘 > 200 日均线                     ——最朴素的趋势过滤，拿来当尺子
  S1  SuperTrend(10, 3) 多头                       ——combo_overlay 第 3 节
  S2  推动浪 D⬆ 且 W⬆（「中」档）                ——impulse_wave 的 H/D/W 小标签，股票没有 12 小时
  S2D 只看 D⬆                                      ——诊断：S2 的两半各自有没有用
  S2W 只看 W⬆
  S3  五浪跌完抄底 · 细（日线 ATR×2）           ——一组向下的 1-5 刚确认，次日开盘做多；
  S3M 五浪跌完抄底 · 中（日线 ATR×4）              目标 = 调整目标线（回撤全长 × 0.5），
                                                     收盘跌破浪5 终点 / 同级别新确认 → 次日开盘平
  S4  组合：S1 且 S2 且 收盘 > 200 日均线
"""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pandas as pd

from . import combo, impulse as IW

RULES = {
    "S0": "基准：收盘 > 200 日均线",
    "S1": "SuperTrend 多头",
    "S2": "推动浪 D⬆ 且 W⬆",
    "S2D": "只看 D⬆（诊断）",
    "S2W": "只看 W⬆（诊断）",
    "S3": "五浪跌完抄底 · 细",
    "S3M": "五浪跌完抄底 · 中",
    "S4": "组合 S1+S2+200 日线",
}
POS_RULES = ("S0", "S1", "S2", "S2D", "S2W", "S4")
FADE_RULES = {"S3": "细", "S3M": "中"}


def ann_of(dfs: dict) -> int:
    """一年多少根：有周末数据（加密货币）按 365，否则 252"""
    idx = pd.DatetimeIndex(sorted(set().union(*[d.index for d in dfs.values()])))
    return 365 if (idx.dayofweek >= 5).mean() > 0.05 else 252


# ---------------------------------------------------------------- 逐只股票的指标

ST = (10, 3.0)       # SuperTrend 的 ATR 周期和倍数，combo_overlay 的默认值；run.py 的 --st 可改


def features(df: pd.DataFrame, ann: int) -> dict:
    """一只股票的全部指标，逐根、只用到当根为止"""
    f = pd.DataFrame(index=df.index)
    f["open"], f["high"], f["low"], f["close"] = df["open"], df["high"], df["low"], df["close"]
    f["st"], f["st_line"] = combo.supertrend(df, *ST)
    f["ma200"] = df["close"].rolling(200).mean()
    f["ma50w"] = combo.ma50w(df)
    f["fib_hi"], f["fib_lo"], f["fib_pos"] = combo.fib_pos(df, ann)
    waves = {}
    anyw = {}
    for lv in ("细", "中"):
        r = IW.run(df, IW.AUTO["D"][lv])
        waves[lv], anyw[lv] = r["waves"], r["anyw"]
        if lv == "中":
            f["trD"], f["kD"], f["fD"], f["invD"], f["invUpD"] = r["trend"], r["k"], r["f"], r["inv"], r["inv_up"]
    # 周线：规则只用**走完的周**（历史上每周最后一个交易日收盘才更新），和回测口径一致。
    # 最后一周如果还没走完（股票周五前、加密货币周日前），规则仍用上一周的值；
    # 「本周至今」另存一列 trW_live，就是 TradingView 小标签实时显示的那个。
    w = IW.weekly(df)
    rw = IW.run(w, IW.AUTO["W"]["中"])
    wk = pd.DataFrame({"trW": rw["trend"], "kW": rw["k"], "fW": rw["f"], "invW": rw["inv"],
                       "invUpW": rw["inv_up"]}, index=w.index)
    f["trW_live"] = IW.to_daily(wk["trW"], df.index).fillna(0).astype(int)
    if df.index[-1].dayofweek < (4 if ann == 252 else 6):
        wk = wk.iloc[:-1]
    for col in wk.columns:
        f[col] = IW.to_daily(wk[col], df.index)
    f["trW"] = f["trW"].fillna(0).astype(int)
    return dict(f=f, waves=waves, anyw=anyw)


def positions(f: pd.DataFrame) -> dict[str, np.ndarray]:
    """持仓规则：第 t 天收盘时该不该持有（1 / 0）"""
    above = (f["close"] > f["ma200"]).to_numpy()
    st = (f["st"] == 1).to_numpy()
    d_up, w_up = (f["trD"] == 1).to_numpy(), (f["trW"] == 1).to_numpy()
    return {"S0": above, "S1": st, "S2": d_up & w_up, "S2D": d_up, "S2W": w_up,
            "S4": st & d_up & w_up & above}


# ---------------------------------------------------------------- 收益

def pos_net(pos: np.ndarray, O: np.ndarray, C: np.ndarray, cost: float) -> np.ndarray:
    """pos[t] 是第 t 天收盘定的仓位，第 t+1 天开盘成交。返回每天的净收益。

    第 i 天：昨收 → 今开 拿的是昨天在场的仓位 a，今开 → 今收 拿的是今天开盘换成的仓位 p。
    """
    p = np.r_[0.0, pos[:-1].astype(float)]         # 第 i 天开盘后在场的仓位
    a = np.r_[0.0, p[:-1]]                          # 第 i 天开盘前（隔夜）在场的仓位
    Cp = np.r_[np.nan, C[:-1]]
    with np.errstate(invalid="ignore", divide="ignore"):
        same = p * (C / Cp - 1.0)
        split = a * (O / Cp - 1.0) + p * (C / O - 1.0)
    net = np.where(a == p, same, split) - np.abs(p - a) * cost
    net[0] = 0.0
    return np.nan_to_num(net)


def fade(f: pd.DataFrame, waves: list, cost: float, sides=(1, -1)) -> tuple[np.ndarray, np.ndarray, list, list]:
    """五浪走完反着做。side +1 = 跌完五浪做多，−1 = 涨完五浪做空。

    确认那根收盘出信号，次日开盘进场；目标是限价单，盘中碰到就成交（开盘就越过按开盘价）；
    收盘越过浪5 终点、或同级别又确认了一组 → 次日开盘平。
    返回 (逐日净收益, 逐日是否在场, 已完成和持有中的交易, 今天收盘刚出的待进场信号)。
    """
    O, H, L, C = (f[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    N = len(C)
    net = np.zeros(N)
    held = np.zeros(N)
    trades, pending = [], []
    confs = sorted(w["conf"] for w in waves)
    busy = -1
    for w in waves:
        side = -1 if w["bull"] else 1
        if side not in sides:
            continue
        c0, e = w["conf"], w["conf"] + 1
        tgt, p5 = w["tgt"], w["p5"]
        if (H[c0] >= tgt) if side > 0 else (L[c0] <= tgt):
            continue                                    # 确认那根盘中已经碰过目标：线一出现就断开
        if e >= N:
            pending.append(dict(side=side, tgt=tgt, stop=p5, p0=w["p0"], date=f.index[c0]))
            continue
        if e <= busy:
            continue
        if (O[e] >= tgt) if side > 0 else (O[e] <= tgt):
            continue                                    # 开盘已经跳过目标
        nxt = next((c for c in confs if c > c0), N)
        x, px, why = sim(O, H, L, C, e, side, tgt, p5, N, nxt)
        entry = O[e]
        for i in range(e, x + 1):
            base = entry if i == e else C[i - 1]
            end = px if i == x else C[i]
            net[i] += side * (end / base - 1.0)
            held[i] = 1.0
        net[e] -= cost
        if why != "持有中":
            net[x] -= cost
        busy = x
        trades.append(dict(side=side, entry_i=e, exit_i=x, date=f.index[e], exit_date=f.index[x],
                           entry=entry, exit=px, why=why, tgt=tgt, stop=p5,
                           ret=side * (px / entry - 1.0) - 2 * cost,
                           dt=abs(tgt / entry - 1.0), ds=abs(p5 / entry - 1.0), hold=x - e))
    return net, held, trades, pending


def sim(O, H, L, C, e, side, tgt, stp, N, nxt=None, hmax=None):
    """从第 e 根开盘进场，返回 (离场那根, 成交价, 原因)。真交易和随机零假设共用这一套成交规则。"""
    end = N if hmax is None else min(N, e + hmax + 1)
    flag = None
    for i in range(e, end):
        if flag:
            return i, O[i], flag
        if i > e and ((O[i] >= tgt) if side > 0 else (O[i] <= tgt)):
            return i, O[i], "到目标"
        if (H[i] >= tgt) if side > 0 else (L[i] <= tgt):
            return i, tgt, "到目标"
        if (C[i] < stp) if side > 0 else (C[i] > stp):
            flag = "收盘越过浪5（止损）"
        elif nxt is not None and i == nxt:
            flag = "同级别新确认"
    if hmax is not None and end < N:
        return end - 1, C[end - 1], "持有上限"
    return N - 1, C[N - 1], "持有中"


# ---------------------------------------------------------------- 统计

def sharpe(x: np.ndarray, ann: int) -> float:
    sd = x.std()
    return float(x.mean() / sd * math.sqrt(ann)) if sd > 0 else float("nan")


def max_dd(x: np.ndarray) -> float:
    e = np.cumprod(1 + x)
    return float((e / np.maximum.accumulate(e) - 1).min())


def boot_diff(a: np.ndarray, b: np.ndarray, ann: int, rng, blk: int = 20, n: int = 2000) -> float:
    """分块自助法：Sharpe(a) − Sharpe(b) ≤ 0 的比例（scripts/60 的 boot_p）"""
    nblk = int(np.ceil(len(a) / blk))
    d = np.empty(n)
    for k in range(n):
        st = rng.integers(0, len(a), nblk)
        sel = (st[:, None] + np.arange(blk)[None, :]).ravel()[: len(a)] % len(a)
        xa, xb = a[sel], b[sel]
        d[k] = (xa.mean() / xa.std() - xb.mean() / xb.std()) * math.sqrt(ann) if xa.std() > 0 else -np.inf
    return float((d <= 0).mean())


def deflated(sr_ann: float, n_trials: int, n_obs: int, ann: int, skew: float, kurt: float) -> float:
    """去偏 Sharpe（Bailey & López de Prado 2014），vibt/metrics.py 同一个公式，只是不依赖 scipy"""
    nd = NormalDist()
    sr = sr_ann / math.sqrt(ann)
    if n_trials < 2 or n_obs < 3:
        return float("nan")
    g = 0.5772156649
    e_max = (1 - g) * nd.inv_cdf(1 - 1 / n_trials) + g * nd.inv_cdf(1 - 1 / (n_trials * math.e))
    sr0 = e_max / math.sqrt(n_obs - 1)
    den = 1 - skew * sr + (kurt - 1) / 4 * sr ** 2
    if den <= 0:
        return float("nan")
    return float(nd.cdf((sr - sr0) * math.sqrt(n_obs - 1) / math.sqrt(den)))
