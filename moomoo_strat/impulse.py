"""推动浪指标（pine/impulse_wave_12345.pine）的 Python 版。

逐根重放 ZigZag，每一根只用到这一根为止的数据，给出：

  trend   这一根收盘时的趋势方向 +1 / −1 / 0 —— 就是图上 H/D/W 小标签的那个箭头
          （推动阶段取这组浪的方向，A、B、C 调整阶段取反方向）
  k, f    当前在走第几段（0..7 = 1 2 3 4 5 A B C），f = 这一段按走完处理时的下一段
  inv     失效价：价格越过这里，这套数法就违反艾略特硬规则（图上的「失」）
  waves   每组已确认的 1-5（图上绿/红三角那一根）：方向、确认那根、起点、浪5 终点、调整目标

ZigZag、位置判断、目标、失效价和 pine 里 zigzag() / scValid() / scTgt() / mtfTrend() / 6b
逐行对应；1-5 确认和 scripts/60 的 walk() 一样。tests/test_moomoo_strat.py 拿它和
scripts/59、60 在仓库数据上逐根对过。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# 指标的自动档（ZigZag 阈值 = ATR × 倍数），按周期分档：日线 2/4/8，周线 0.5/1/1.5
AUTO = {"D": {"细": 2.0, "中": 4.0, "粗": 8.0}, "W": {"细": 0.5, "中": 1.0, "粗": 1.5}}
PRM = dict(need5=True, ovTol=0.0, maxSpan=300, ret2=0.618, fib3=1.618, ret4=0.382,
           fib5=1.0, projRetr=0.5, retB=0.618)
NAMES = ["浪1", "浪2", "浪3", "浪4", "浪5", "A", "B", "C", "新浪1"]


def atr_w(h, l, c, n=14):
    pc = np.r_[np.nan, c[:-1]]
    tr = np.nanmax(np.c_[h - l, np.abs(h - pc), np.abs(l - pc)], axis=1)
    o = np.full(len(c), np.nan)
    if len(c) > n:
        o[n] = np.nanmean(tr[1:n + 1])
        for i in range(n + 1, len(c)):
            o[i] = (o[i - 1] * (n - 1) + tr[i]) / n
    return o


def _target(E, x0, xe, j, ext, prm, eps):
    """第 j 段终点的斐波那契目标（归一化坐标，推动方向永远向上）—— pine 的 scTgt()"""
    L1 = (E[0] - x0) if E[0] is not None else None
    if j == 0:
        return xe
    if j == 1:
        return max(E[0] - (0.786 if ext else prm["ret2"]) * L1, x0 + eps)
    if j == 2:
        return max(E[1] + (prm["fib3"] + (1.0 if ext else 0.0)) * L1, E[0] + eps)
    if j == 3:
        return max(E[2] - (0.5 if ext else prm["ret4"]) * (E[2] - E[1]), E[0] - prm["ovTol"] * L1 + eps)
    if j == 4:
        l3 = E[2] - E[1]
        t = E[3] + (prm["fib5"] + (0.618 if ext else 0.0)) * L1
        if t <= E[2]:
            t = E[3] + 0.618 * (E[2] - x0)
        if l3 < L1:
            t = min(t, E[3] + l3)
        if prm["need5"]:
            t = max(t, E[2] + eps)
        return t
    if j == 5:
        a = prm["projRetr"] * (E[4] - x0) / (2 - prm["retB"])
        return max(E[4] - a * (1.618 if ext else 1.0), x0 + eps)
    if j == 6:
        return min(E[5] + (0.786 if ext else prm["retB"]) * (E[4] - E[5]), E[4] - eps)
    if j == 7:
        return max(E[6] - (1.618 if ext else 1.0) * (E[4] - E[5]), x0 + eps)
    raise ValueError(j)


def position(pv, d, ext, ebi, N, prm=PRM, eps=1e-8):
    """最后一根是第 N−1 根时，这组枢轴在走第几段、这一段算不算走完、失效价在哪。

    pv: [(bar, price, isHigh), ...]；d: ZigZag 当前方向；ext / ebi: 当前这一段的极值和它所在的根。
    选位置和 scripts/59 的 scenario() 第 1、2 步一样：k 从 7 往 0 试，取第一个全部合规的。
    """
    n = len(pv)
    if n < 2 or d == 0:
        return None
    for k in range(min(7, n - 1), -1, -1):
        Q = pv[n - 1 - k:]
        bull = not Q[0][2]
        s = 1.0 if bull else -1.0
        if any(Q[i][2] != ((i % 2 == 1) == bull) for i in range(k + 1)):
            continue
        if any(Q[i + 1][0] <= Q[i][0] for i in range(k)):
            continue
        if k >= 1 and prm["maxSpan"] and (N - 1 - Q[0][0]) > prm["maxSpan"]:
            continue
        x = [s * q[1] for q in Q]
        xe = s * ext
        l1 = x[1] - x[0] if k >= 1 else None
        if k >= 2 and not x[2] > x[0]:
            continue
        if k >= 3 and not x[3] > x[1]:
            continue
        if k >= 4 and not x[4] > x[1] - prm["ovTol"] * l1:
            continue
        if k >= 5:
            l3, l5 = x[3] - x[2], x[5] - x[4]
            if l3 < l1 and l3 < l5:
                continue
            if prm["need5"] and not x[5] > x[3]:
                continue
        if k >= 6 and not x[6] > x[0]:
            continue
        if k >= 7 and not x[5] > x[7]:
            continue
        if k == 1 and not xe > x[0]:
            continue
        if k == 3 and not xe > x[1] - prm["ovTol"] * l1:
            continue
        if k in (5, 7) and not xe > x[0]:
            continue
        if k == 6 and not x[5] > xe:
            continue
        break
    else:
        return None
    E = [None] * 10
    for j in range(k):
        E[j] = x[j + 1]
    x0 = x[0]
    done = True                       # k == 0：新一轮浪1 永远按在当前极值走完处理
    if k > 0:
        up = k % 2 == 0
        t = _target(E, x0, xe, k, False, prm, eps)
        beyond = (xe >= t) if up else (xe <= t)
        if beyond and (N - 1 - ebi) <= 1:   # 越过目标、还在走 → 换延长档再比一次
            t = _target(E, x0, xe, k, True, prm, eps)
            beyond = (xe >= t) if up else (xe <= t)
        done = beyond
    E[k] = xe
    f = k + 1 if done else k
    # 失效价（pine 6b）：f=1、2 浪1 起点；f=3、4 浪1 终点；f=5..7 浪5 终点（上限）；f=8 C 终点
    jInv = 7 if f >= 8 else 4 if f >= 5 else 0 if f >= 3 else -1
    xInv = x0 if jInv < 0 else E[jInv]
    if jInv == 0:
        xInv -= prm["ovTol"] * (E[0] - x0)
    corr = 5 <= f <= 7
    return dict(k=k, f=f, done=done, bull=bull, s=s,
                trend=int((-1 if corr else 1) * s),
                inv=s * xInv, inv_up=(corr == bull), p0=Q[0])


def run(df: pd.DataFrame, mult: float, prm=PRM) -> dict:
    """逐根跑 ZigZag → 1-5 确认 → 位置判断。返回逐根数组（长度 = len(df)）和确认列表。"""
    H, L, C = (df[k].to_numpy(dtype=float) for k in ("high", "low", "close"))
    A = atr_w(H, L, C)
    N = len(C)
    trend = np.zeros(N, dtype=int)
    kk = np.full(N, -1, dtype=int)
    ff = np.full(N, -1, dtype=int)
    done = np.zeros(N, dtype=bool)
    inv = np.full(N, np.nan)
    inv_up = np.zeros(N, dtype=bool)
    pv: list = []
    waves: list = []
    d, ext, ebi, e2, e2bi = 0, np.nan, -1, np.nan, -1
    for i in range(N):
        conf = False
        if not np.isnan(A[i]):
            thr = A[i] * mult
            if d == 0:
                if np.isnan(ext):
                    ext, ebi, e2, e2bi = H[i], i, L[i], i
                else:
                    if H[i] > ext:
                        ext, ebi = H[i], i
                    if L[i] < e2:
                        e2, e2bi = L[i], i
                    if C[i] < ext - thr:
                        pv.append((ebi, ext, True)); d, ext, ebi = -1, L[i], i; conf = True
                    elif C[i] > e2 + thr:
                        pv.append((e2bi, e2, False)); d, ext, ebi = 1, H[i], i; conf = True
            elif d == 1:
                if H[i] > ext:
                    ext, ebi = H[i], i
                if C[i] < ext - thr:
                    pv.append((ebi, ext, True)); d, ext, ebi = -1, L[i], i; conf = True
            else:
                if L[i] < ext:
                    ext, ebi = L[i], i
                if C[i] > ext + thr:
                    pv.append((ebi, ext, False)); d, ext, ebi = 1, H[i], i; conf = True
            if len(pv) > 24:            # pine 只留最近 24 个；位置判断最多用 8 个
                pv.pop(0)
        if conf and len(pv) >= 6:       # pine detect()：只在刚确认一个枢轴的那根检查最近 6 个
            p = pv[-6:]
            isH = [x[2] for x in p]; bi = [x[0] for x in p]; pr = [x[1] for x in p]
            bull = isH == [False, True, False, True, False, True]
            bear = isH == [True, False, True, False, True, False]
            l1, l3, l5 = abs(pr[1] - pr[0]), abs(pr[3] - pr[2]), abs(pr[5] - pr[4])
            ok = ((bull or bear) and all(bi[j + 1] > bi[j] for j in range(5))
                  and (bi[5] - bi[0]) <= prm["maxSpan"]
                  and (pr[2] > pr[0] if bull else pr[2] < pr[0]) and not (l3 < l1 and l3 < l5)
                  and (pr[4] > pr[1] - l1 * prm["ovTol"] if bull else pr[4] < pr[1] + l1 * prm["ovTol"])
                  and ((not prm["need5"]) or (pr[5] > pr[3] if bull else pr[5] < pr[3])))
            if ok:
                full = abs(pr[5] - pr[0])
                waves.append(dict(bull=bull, conf=i, p0=pr[0], p5=pr[5], b5=bi[5],
                                  tgt=pr[5] - full * prm["projRetr"] if bull else pr[5] + full * prm["projRetr"]))
        r = position(pv, d, ext, ebi, i + 1, prm) if d != 0 else None
        if r is not None:
            trend[i], kk[i], ff[i], done[i] = r["trend"], r["k"], r["f"], r["done"]
            inv[i], inv_up[i] = r["inv"], r["inv_up"]
    return dict(trend=trend, k=kk, f=ff, done=done, inv=inv, inv_up=inv_up, waves=waves)


def weekly(df: pd.DataFrame) -> pd.DataFrame:
    """日线 → 周线（周一到周日为一周，股票就是周一到周五）。

    每根周线的时间戳是这一周**最后一个交易日**，这样映射回日线时，周线趋势在这一周最后一天
    收盘才更新，周中用的都是上一周走完的值 —— 和 TradingView 历史 K 线上的画法一样，不偷看。
    最后一周可能没走完：它的值等于「本周至今」，和 TradingView 实时那根一样会变。
    """
    wk = df.index.to_period("W-SUN")
    g = df.groupby(wk)
    w = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                      "close": g["close"].last()})
    w.index = pd.DatetimeIndex(g.apply(lambda x: x.index[-1]))
    return w


def to_daily(wser: pd.Series, daily_index: pd.DatetimeIndex) -> pd.Series:
    """周线序列（时间戳 = 那一周最后一个交易日）按时间向前填到日线上"""
    return wser.reindex(daily_index.union(wser.index)).ffill().reindex(daily_index)
