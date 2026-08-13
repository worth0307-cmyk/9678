"""Indicators.  Everything here is causal: value at bar t uses bars <= t only.

The Vortex Indicator (Botes & Siepman, 2010) is the centrepiece because it is
computable from OHLC alone -- no volume needed, which matches the data we have.

    VM+_t = |High_t - Low_{t-1}|      "upward vortex movement"
    VM-_t = |Low_t  - High_{t-1}|     "downward vortex movement"
    TR_t  = true range
    VI+_n = sum(VM+, n) / sum(TR, n)
    VI-_n = sum(VM-, n) / sum(TR, n)

VI+ > VI- is read as an uptrend; the crossover is the classic signal.  Note
VI+ + VI- is itself informative -- it is high when bars overlap little
(directional, expanding market) and low when the market coils.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- core
def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    a = df["high"] - df["low"]
    b = (df["high"] - prev_close).abs()
    c = (df["low"] - prev_close).abs()
    return pd.concat([a, b, c], axis=1).max(axis=1)


def atr(df: pd.DataFrame, n: int = 14, method: str = "wilder") -> pd.Series:
    tr = true_range(df)
    if method == "wilder":
        return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return tr.rolling(n).mean()


def vortex(df: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    """VI+ / VI- / spread / sum."""
    vm_plus = (df["high"] - df["low"].shift(1)).abs()
    vm_minus = (df["low"] - df["high"].shift(1)).abs()
    tr = true_range(df)
    tr_sum = tr.rolling(n).sum()
    vip = vm_plus.rolling(n).sum() / tr_sum
    vim = vm_minus.rolling(n).sum() / tr_sum
    out = pd.DataFrame({"vi_plus": vip, "vi_minus": vim})
    out["vi_spread"] = vip - vim
    out["vi_sum"] = vip + vim
    return out


def vortex_spread(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return vortex(df, n)["vi_spread"]


# ----------------------------------------------------------------- trend / momentum
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def roc(s: pd.Series, n: int) -> pd.Series:
    return s / s.shift(n) - 1.0


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    up = df["high"].diff()
    dn = -df["low"].diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = true_range(df).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    pdi = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / tr
    mdi = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / tr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def donchian(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Channel from bars strictly before t, so a break of it is testable at t."""
    hi = df["high"].shift(1).rolling(n).max()
    lo = df["low"].shift(1).rolling(n).min()
    return pd.DataFrame({"dc_high": hi, "dc_low": lo, "dc_mid": (hi + lo) / 2})


def keltner(df: pd.DataFrame, n: int = 20, mult: float = 2.0) -> pd.DataFrame:
    mid = ema(df["close"], n)
    band = atr(df, n) * mult
    return pd.DataFrame({"kc_mid": mid, "kc_up": mid + band, "kc_dn": mid - band})


# --------------------------------------------------------------------- volatility
def realized_vol(close: pd.Series, n: int, ann_factor: float) -> pd.Series:
    return np.log(close).diff().rolling(n).std() * np.sqrt(ann_factor)


def parkinson_vol(df: pd.DataFrame, n: int, ann_factor: float) -> pd.Series:
    """High-low range estimator -- ~5x more efficient than close-to-close."""
    hl = np.log(df["high"] / df["low"]) ** 2
    return np.sqrt(hl.rolling(n).mean() / (4 * np.log(2)) * ann_factor)


def efficiency_ratio(close: pd.Series, n: int = 20) -> pd.Series:
    """Kaufman ER: net move / total path.  ~1 = clean trend, ~0 = chop."""
    direction = (close - close.shift(n)).abs()
    volatility = close.diff().abs().rolling(n).sum()
    return direction / volatility.replace(0, np.nan)


def zscore(s: pd.Series, n: int) -> pd.Series:
    m = s.rolling(n).mean()
    sd = s.rolling(n).std()
    return (s - m) / sd.replace(0, np.nan)


def percentile_rank(s: pd.Series, n: int) -> pd.Series:
    """Rolling percentile of the current value within the trailing n-window."""
    return s.rolling(n).apply(lambda w: (w[-1] > w[:-1]).mean(), raw=True)


def crossover(a: pd.Series, b: pd.Series) -> pd.Series:
    """+1 where a crosses above b, -1 where it crosses below, else 0."""
    diff = a - b
    prev = diff.shift(1)
    up = (diff > 0) & (prev <= 0)
    dn = (diff < 0) & (prev >= 0)
    return pd.Series(np.where(up, 1, np.where(dn, -1, 0)), index=a.index)
