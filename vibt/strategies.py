"""Signal generators.

Every function returns a Series of DESIRED POSITION as of that bar's close,
in units of account equity (+1 = full long, -1 = full short, 0.5 = half long).
The backtest engine applies the one-bar execution lag, so nothing here may
shift forward.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as D, indicators as I


# ------------------------------------------------------------------ VI (the user's idea)
def vi_cross(df: pd.DataFrame, n: int = 14, mode: str = "ls") -> pd.Series:
    """Textbook Vortex: long while VI+ > VI-, short (or flat) while VI+ < VI-."""
    v = I.vortex(df, n)
    s = np.sign(v["vi_spread"])
    if mode == "long_only":
        s = s.clip(lower=0)
    elif mode == "short_only":
        s = s.clip(upper=0)
    return s.fillna(0.0)


def vi_cross_band(df: pd.DataFrame, n: int = 14, band: float = 0.05,
                  mode: str = "ls") -> pd.Series:
    """Hysteresis band: only flip once the spread clears +/-band, else hold."""
    v = I.vortex(df, n)["vi_spread"]
    raw = np.where(v > band, 1.0, np.where(v < -band, -1.0, np.nan))
    s = pd.Series(raw, index=df.index).ffill().fillna(0.0)
    if mode == "long_only":
        s = s.clip(lower=0)
    return s


def vi_strength(df: pd.DataFrame, n: int = 14, scale: float = 0.15,
                mode: str = "ls") -> pd.Series:
    """Continuous exposure proportional to the spread, capped at +/-1."""
    v = I.vortex(df, n)["vi_spread"]
    s = (v / scale).clip(-1, 1)
    if mode == "long_only":
        s = s.clip(lower=0)
    return s.fillna(0.0)


def vi_contrarian(df: pd.DataFrame, n: int = 14, q: float = 0.8,
                  lookback: int = 500) -> pd.Series:
    """Fade extreme VI readings -- the 1h diagnostics say short-horizon BTC reverts."""
    v = I.vortex(df, n)["vi_spread"]
    hi = v.rolling(lookback).quantile(q)
    lo = v.rolling(lookback).quantile(1 - q)
    return pd.Series(np.where(v > hi, -1.0, np.where(v < lo, 1.0, 0.0)),
                     index=df.index).fillna(0.0)


def vi_mtf(
    base: pd.DataFrame,
    htf: pd.DataFrame,
    htf_tf: str,
    n_base: int = 14,
    n_htf: int = 14,
    mode: str = "ls",
) -> pd.Series:
    """Higher-timeframe VI sets the permitted direction, base-timeframe VI times it."""
    v_htf = I.vortex(htf, n_htf)
    htf_feat = htf.copy()
    htf_feat["vi_spread"] = v_htf["vi_spread"]
    aligned = D.align_higher_tf(base.index, htf_feat, htf_tf, ["vi_spread"])
    regime = np.sign(aligned[f"vi_spread_{htf_tf}"])
    trigger = np.sign(I.vortex(base, n_base)["vi_spread"])
    s = pd.Series(np.where(regime == trigger, trigger, 0.0), index=base.index)
    if mode == "long_only":
        s = s.clip(lower=0)
    return s.fillna(0.0)


def vi_filtered(df: pd.DataFrame, n: int = 14, er_n: int = 20, er_min: float = 0.30,
                mode: str = "ls") -> pd.Series:
    """VI direction, but only when the market is actually trending (Kaufman ER)."""
    s = vi_cross(df, n, mode)
    er = I.efficiency_ratio(df["close"], er_n)
    return (s * (er >= er_min).astype(float)).fillna(0.0)


# ---------------------------------------------------------------------- classic trend
def ema_cross(df: pd.DataFrame, fast: int = 20, slow: int = 50, mode: str = "ls") -> pd.Series:
    s = np.sign(I.ema(df["close"], fast) - I.ema(df["close"], slow))
    if mode == "long_only":
        s = s.clip(lower=0)
    return s.fillna(0.0)


def donchian_break(df: pd.DataFrame, n_in: int = 20, n_out: int = 10,
                   mode: str = "ls") -> pd.Series:
    """Turtle-style: enter on an n_in break, exit on the opposite n_out break."""
    ch_in = I.donchian(df, n_in)
    ch_out = I.donchian(df, n_out)
    c = df["close"]
    raw = np.where(c > ch_in["dc_high"], 1.0, np.where(c < ch_in["dc_low"], -1.0, np.nan))
    s = pd.Series(raw, index=df.index)
    exit_long = c < ch_out["dc_low"]
    exit_short = c > ch_out["dc_high"]
    s = s.ffill()
    out = s.copy()
    held = 0.0
    vals = s.to_numpy()
    el, es = exit_long.to_numpy(), exit_short.to_numpy()
    res = np.zeros(len(df))
    for i in range(len(df)):
        v = vals[i]
        if not np.isnan(v) and v != 0 and (held == 0 or np.sign(v) != np.sign(held)):
            held = v
        elif held > 0 and el[i]:
            held = 0.0
        elif held < 0 and es[i]:
            held = 0.0
        res[i] = held
    out = pd.Series(res, index=df.index)
    if mode == "long_only":
        out = out.clip(lower=0)
    return out.fillna(0.0)


def tsmom(df: pd.DataFrame, n: int = 30, mode: str = "ls") -> pd.Series:
    s = np.sign(I.roc(df["close"], n))
    if mode == "long_only":
        s = s.clip(lower=0)
    return s.fillna(0.0)


def price_vs_ma(df: pd.DataFrame, n: int = 100, mode: str = "long_only") -> pd.Series:
    s = np.sign(df["close"] - I.sma(df["close"], n))
    if mode == "long_only":
        s = s.clip(lower=0)
    return s.fillna(0.0)


# ------------------------------------------------------------------- mean reversion
def zscore_revert(df: pd.DataFrame, n: int = 24, entry: float = 2.0,
                  exit_z: float = 0.5, mode: str = "ls") -> pd.Series:
    z = I.zscore(df["close"], n)
    res = np.zeros(len(df))
    held = 0.0
    zv = z.to_numpy()
    for i in range(len(df)):
        if np.isnan(zv[i]):
            res[i] = 0.0
            continue
        if held == 0:
            if zv[i] <= -entry:
                held = 1.0
            elif zv[i] >= entry:
                held = -1.0
        elif held > 0 and zv[i] >= -exit_z:
            held = 0.0
        elif held < 0 and zv[i] <= exit_z:
            held = 0.0
        res[i] = held
    s = pd.Series(res, index=df.index)
    if mode == "long_only":
        s = s.clip(lower=0)
    return s


def rsi_revert(df: pd.DataFrame, n: int = 14, lo: float = 30, hi: float = 70,
               mode: str = "ls") -> pd.Series:
    r = I.rsi(df["close"], n)
    raw = np.where(r < lo, 1.0, np.where(r > hi, -1.0, np.nan))
    s = pd.Series(raw, index=df.index)
    mid = (r - 50).abs() < 10
    s[mid] = 0.0
    s = s.ffill().fillna(0.0)
    if mode == "long_only":
        s = s.clip(lower=0)
    return s


# ------------------------------------------------------------------------- overlays
def apply_regime_filter(sig: pd.Series, regime_ok: pd.Series) -> pd.Series:
    return (sig * regime_ok.reindex(sig.index).fillna(0).astype(float)).fillna(0.0)


def apply_vol_target(sig: pd.Series, df: pd.DataFrame, ann_factor: float,
                     n: int = 30, target: float = 0.40, max_lev: float = 1.0) -> pd.Series:
    vol = I.parkinson_vol(df, n, ann_factor)
    lev = (target / vol).clip(0.0, max_lev)
    return (sig * lev).fillna(0.0)


def discretise(sig: pd.Series, step: float = 0.25) -> pd.Series:
    """Round exposure to a grid so vol targeting does not churn on noise."""
    return (sig / step).round() * step
