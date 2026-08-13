"""The VI-Dashboard band strategy, implemented as specified — and its variants.

Rules taken from worth0307-cmyk/vi-dashboard (backend/data_service.py):

    vip > upper -> "above"   |   vip < lower -> "below"   |   else "inside"
    double breakout = 4H state == 1D state, and not "inside"

The trader's rules, as described:
    double-above -> SHORT, double-below -> LONG     ("fade")
    keep adding while VI+ extends further past the band ("pyramid")
    take profit when VI+ falls back                 ("exit on re-entry")

Everything is evaluated on the 4H grid.  The 1D VI+ is carried onto that grid by
CLOSE time, so a 4H bar only ever sees the last daily bar that had already
finished — the same information the dashboard has when it fires an alert.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import data as D, indicators as I


@dataclass
class BandParams:
    vi_len: int = 14
    up_4h: float = 1.19
    dn_4h: float = 0.728
    up_1d: float = 1.155
    dn_1d: float = 0.60

    # "fade"   = short the up-breakout, long the down-breakout (the dashboard rule)
    # "follow" = the opposite sign
    # "regime" = the breakout is only a TRIGGER; the daily trend picks the side
    direction: str = "fade"

    base_size: float = 0.5          # size of the first unit, in equity
    pyramid_steps: int = 2          # how many extra units may be added
    pyramid_extend: float = 0.03    # VI+ must push this much further past the band per add
    max_size: float = 1.0

    # "reenter"  = close when 4H VI+ comes back inside its band
    # "retrace"  = close when VI+ gives back `exit_retrace` of its peak excursion
    exit_mode: str = "reenter"
    exit_retrace: float = 0.5

    stop_atr: float = 0.0           # 0 disables; otherwise stop at N x ATR(14) on the 4H bar
    max_hold_bars: int = 0          # 0 disables a time stop

    trend_filter: bool = False      # only take trades aligned with the daily regime
    trend_lookbacks: tuple[int, ...] = (10, 14, 20, 28, 36, 48)


def percentile_bands(frames: dict, vi_len: int, up_q: float, dn_q: float) -> dict:
    """Bands set by rolling-free sample quantiles of VI+ instead of fixed levels."""
    out = {}
    for tf, key in (("4h", "4h"), ("1d", "1d")):
        v = I.vortex(frames[tf], vi_len)["vi_plus"].dropna()
        out[f"up_{key}"] = float(v.quantile(up_q))
        out[f"dn_{key}"] = float(v.quantile(dn_q))
    return out


def build_frame(frames: dict, p: BandParams) -> pd.DataFrame:
    """4H grid carrying both VI+ series, their states, and the double-breakout flag."""
    f4, f1 = frames["4h"], frames["1d"]
    vi4 = I.vortex(f4, p.vi_len)["vi_plus"]
    vi1 = I.vortex(f1, p.vi_len)["vi_plus"]

    d = f1.copy()
    d["vip"] = vi1
    vi1_on_4h = D.align_higher_tf(f4.index, d, "1d", ["vip"])["vip_1d"]

    df = pd.DataFrame({
        "open": f4["open"], "high": f4["high"], "low": f4["low"], "close": f4["close"],
        "vi4": vi4, "vi1d": vi1_on_4h,
    })
    df["s4"] = np.where(df["vi4"] > p.up_4h, 1, np.where(df["vi4"] < p.dn_4h, -1, 0))
    df["s1d"] = np.where(df["vi1d"] > p.up_1d, 1, np.where(df["vi1d"] < p.dn_1d, -1, 0))
    df["double"] = np.where((df["s4"] == df["s1d"]) & (df["s4"] != 0), df["s4"], 0)
    df["atr"] = I.atr(f4, 14)

    if p.trend_filter or p.direction == "regime":
        votes = []
        for n in p.trend_lookbacks:
            v = I.vortex(f1, n)
            votes.append((v["vi_spread"] > 0).astype(float))
            pdm = f1["high"].diff().clip(lower=0)
            mdm = (-f1["low"].diff()).clip(lower=0)
            votes.append((pdm.rolling(n).sum() > mdm.rolling(n).sum()).astype(float))
        bull = pd.concat(votes, axis=1).mean(axis=1)
        dd = f1.copy()
        dd["bull"] = bull
        df["bull_1d"] = D.align_higher_tf(f4.index, dd, "1d", ["bull"])["bull_1d"]
    else:
        df["bull_1d"] = np.nan
    return df


def target_position(df: pd.DataFrame, p: BandParams) -> pd.Series:
    """Desired position at each 4H bar's close.  Stateful: entries, adds, exits."""
    sign = -1.0 if p.direction == "fade" else 1.0
    n = len(df)
    dbl = df["double"].to_numpy()
    vi4 = df["vi4"].to_numpy()
    close = df["close"].to_numpy()
    high, low = df["high"].to_numpy(), df["low"].to_numpy()
    atr = df["atr"].to_numpy()
    bull = df["bull_1d"].to_numpy()

    pos = np.zeros(n)
    cur = 0.0
    units = 0
    entry_px = np.nan
    band_ref = np.nan      # the band VI+ broke through
    peak_excursion = 0.0   # furthest VI+ got past that band
    held = 0

    for t in range(n):
        d = dbl[t]

        if cur == 0.0:
            if d != 0 and np.isfinite(vi4[t]):
                if p.direction == "regime":
                    # the band event says "now"; the daily regime says "which way"
                    if not np.isfinite(bull[t]):
                        pos[t] = cur
                        continue
                    side = 1.0 if bull[t] > 0.5 else -1.0
                else:
                    side = sign * d
                if p.trend_filter and p.direction != "regime" and np.isfinite(bull[t]):
                    # only trade with the daily regime: longs need a bullish tape
                    if (side > 0 and bull[t] < 0.5) or (side < 0 and bull[t] > 0.5):
                        pos[t] = 0.0
                        continue
                cur = side * p.base_size
                units = 1
                entry_px = close[t]
                band_ref = p.up_4h if d > 0 else p.dn_4h
                peak_excursion = abs(vi4[t] - band_ref)
                held = 0
            pos[t] = cur
            continue

        held += 1
        excursion = abs(vi4[t] - band_ref) if np.isfinite(vi4[t]) else 0.0
        peak_excursion = max(peak_excursion, excursion)

        exit_now = False
        if d == 0:
            # VI+ came back inside the band on at least one timeframe
            if p.exit_mode == "reenter":
                exit_now = True
        if p.exit_mode == "retrace" and peak_excursion > 0:
            if excursion <= peak_excursion * (1 - p.exit_retrace):
                exit_now = True
        if p.stop_atr > 0 and np.isfinite(atr[t]) and np.isfinite(entry_px):
            if cur > 0 and low[t] <= entry_px - p.stop_atr * atr[t]:
                exit_now = True
            if cur < 0 and high[t] >= entry_px + p.stop_atr * atr[t]:
                exit_now = True
        if p.max_hold_bars > 0 and held >= p.max_hold_bars:
            exit_now = True

        if exit_now:
            cur = 0.0
            units = 0
            peak_excursion = 0.0
            pos[t] = cur
            continue

        # pyramid: VI+ has pushed a further `pyramid_extend` past the band
        if units <= p.pyramid_steps and excursion >= p.pyramid_extend * (units + 1):
            step = np.sign(cur) * p.base_size
            if abs(cur + step) <= p.max_size + 1e-9:
                cur += step
                units += 1

        pos[t] = cur

    return pd.Series(pos, index=df.index)
