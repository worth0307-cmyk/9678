"""combo_overlay.pine 里拿来做规则的几样东西：SuperTrend、200 日均线、50 周均线、一年斐波那契位置。

均线带（MA/EMA 20-50-100）和背离只用来看图，不进规则 —— 它们在 moomoo 公式指标里画
（见 formula/），或者 moomoo 自带的 MA / EMA 指标就有。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rma(x: np.ndarray, n: int) -> np.ndarray:
    """Pine 的 ta.rma：前 n 根的简单平均起步，之后 (前值·(n−1) + 新值) / n"""
    o = np.full(len(x), np.nan)
    if len(x) >= n:
        o[n - 1] = np.mean(x[:n])
        for i in range(n, len(x)):
            o[i] = (o[i - 1] * (n - 1) + x[i]) / n
    return o


def supertrend(df: pd.DataFrame, period: int = 10, mult: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    """combo_overlay 第 3 节，逐行照搬（Source = hl2，Change ATR Calculation Method = 开 → RMA）。

    返回 (trend, line)：trend = +1 多 / −1 空；line = 多头时的下轨 up、空头时的上轨 dn（图上那条线，
    也就是这一侧的止损位）。
    """
    H, L, C = (df[k].to_numpy(dtype=float) for k in ("high", "low", "close"))
    pc = np.r_[np.nan, C[:-1]]
    tr = np.where(np.isnan(pc), H - L, np.maximum(H - L, np.maximum(np.abs(H - pc), np.abs(L - pc))))
    atr = rma(tr, period)
    src = (H + L) / 2
    N = len(C)
    up = np.full(N, np.nan)
    dn = np.full(N, np.nan)
    trend = np.ones(N, dtype=int)
    line = np.full(N, np.nan)
    t = 1
    for i in range(N):
        if np.isnan(atr[i]):
            trend[i] = t
            continue
        u = src[i] - mult * atr[i]
        u1 = up[i - 1] if i > 0 and not np.isnan(up[i - 1]) else u
        if i > 0 and C[i - 1] > u1:
            u = max(u, u1)
        dd = src[i] + mult * atr[i]
        d1 = dn[i - 1] if i > 0 and not np.isnan(dn[i - 1]) else dd
        if i > 0 and C[i - 1] < d1:
            dd = min(dd, d1)
        up[i], dn[i] = u, dd
        if t == -1 and C[i] > d1:
            t = 1
        elif t == 1 and C[i] < u1:
            t = -1
        trend[i] = t
        line[i] = u if t == 1 else dd
    return trend, line


def ma50w(df: pd.DataFrame, n: int = 50) -> pd.Series:
    """50 周均线在每个交易日收盘时的值：前 49 个走完的周收盘 + 今天的收盘，除以 50。

    一周的最后一天收盘时正好等于官方的 50 周均线；周中就是 TradingView 实时那根显示的值。
    只用到今天为止的数据。
    """
    wk = df.index.to_period("W-SUN")
    wclose = df["close"].groupby(wk).last()
    prev = wclose.rolling(n - 1).sum().shift(1)          # 本周之前 49 周的收盘之和
    return pd.Series((prev.reindex(wk).to_numpy() + df["close"].to_numpy()) / n, index=df.index)


def fib_pos(df: pd.DataFrame, bars: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    """一年窗口（bars 根）的最高、最低、现价在区间里的位置（0 = 窗口低点，1 = 窗口高点）"""
    hh = df["high"].rolling(bars, min_periods=bars // 2).max()
    ll = df["low"].rolling(bars, min_periods=bars // 2).min()
    return hh, ll, (df["close"] - ll) / (hh - ll)
