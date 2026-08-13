"""The production system: a range-based regime allocator for BTCUSDT.

Design principles, each one earned from the study in scripts/01..07:

1.  Trade the DAILY grid.  1h signals lose to fees by a mile (every 1h variant
    tested lost 40-95%); 4h roughly halves the daily edge through turnover.
2.  Never bet on a single lookback.  The best daily lookback in-sample (14) was
    a spike surrounded by mediocre neighbours -- the parameter is noise, the
    family is the signal.  So: vote across many lookbacks.
3.  Prefer RANGE-based direction (Vortex, Directional Movement) over
    close-based averages.  Both range families beat every close-based family
    at the median of their own parameter grids.
4.  Size by volatility, not by conviction.  The vol-target overlay improved the
    median Sharpe AND the median drawdown across all 27 VI lookbacks -- an
    improvement that survives at the median is worth far more than one that
    only shows up in the best cell.
5.  The short book is insurance, not alpha.  It costs Sharpe in aggregate and
    pays it back in the bear.  Default small, and only in a confirmed downtrend.

The output is a target exposure in [-max_short, 1.0] of account equity.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import indicators as I


@dataclass
class Params:
    lookbacks: tuple[int, ...] = (10, 14, 20, 28, 36, 48)
    use_vortex: bool = True
    use_dmi: bool = True
    vol_lookback: int = 30
    target_vol: float = 0.40
    max_leverage: float = 1.0
    exposure_step: float = 0.20     # round target size to this grid to stop churn
    vote_floor: float = 0.0         # votes below this map to zero exposure
    short_size: float = 0.25        # 0 disables the short book
    short_ma: int = 200             # shorts require close below this SMA
    ann_factor: float = 365.0


def direction_votes(df: pd.DataFrame, p: Params) -> pd.DataFrame:
    """One bullish/bearish vote per (indicator, lookback).  All causal."""
    votes = {}
    for n in p.lookbacks:
        if p.use_vortex:
            v = I.vortex(df, n)
            votes[f"vi{n}"] = (v["vi_spread"] > 0).astype(float)
        if p.use_dmi:
            pdm = df["high"].diff().clip(lower=0)
            mdm = (-df["low"].diff()).clip(lower=0)
            votes[f"di{n}"] = (pdm.rolling(n).sum() > mdm.rolling(n).sum()).astype(float)
    out = pd.DataFrame(votes, index=df.index)
    # a vote is only valid once its lookback has filled
    warmup = max(p.lookbacks) + 1
    out.iloc[:warmup] = np.nan
    return out


def target_exposure(df: pd.DataFrame, p: Params | None = None) -> pd.Series:
    """Desired position as of each bar's close, in units of equity."""
    p = p or Params()
    votes = direction_votes(df, p)
    bull = votes.mean(axis=1)                       # 0..1 share of bullish votes

    vol = I.parkinson_vol(df, p.vol_lookback, p.ann_factor)
    lev = (p.target_vol / vol).clip(0.0, p.max_leverage)

    long_part = bull.where(bull >= p.vote_floor, 0.0) * lev

    if p.short_size > 0:
        confirmed_bear = (bull <= 1e-9) & (df["close"] < I.sma(df["close"], p.short_ma))
        short_part = confirmed_bear.astype(float) * lev * p.short_size
    else:
        short_part = 0.0

    raw = long_part - short_part
    return discretise(raw, p.exposure_step).fillna(0.0)


def discretise(s: pd.Series, step: float) -> pd.Series:
    if step <= 0:
        return s
    return (s / step).round() * step


def explain(df: pd.DataFrame, p: Params | None = None, tail: int = 10) -> pd.DataFrame:
    """Human-readable state table -- what the system sees and why."""
    p = p or Params()
    votes = direction_votes(df, p)
    bull = votes.mean(axis=1)
    vol = I.parkinson_vol(df, p.vol_lookback, p.ann_factor)
    lev = (p.target_vol / vol).clip(0.0, p.max_leverage)
    out = pd.DataFrame({
        "close": df["close"],
        "bull_votes": (bull * len(votes.columns)).round(0),
        "n_votes": len(votes.columns),
        "bull_frac": bull.round(3),
        "realised_vol": vol.round(3),
        "vol_scalar": lev.round(3),
        "below_sma200": df["close"] < I.sma(df["close"], p.short_ma),
        "target": target_exposure(df, p),
    })
    return out.tail(tail)
