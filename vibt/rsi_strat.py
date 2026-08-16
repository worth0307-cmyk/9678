"""RSI mean-reversion with pyramiding, as specified against the TradingView RSI.

Rules (BTCUSDT 4h, 1000 USDT, no leverage):

  long   RSI below 30 on three consecutive bars AND the RSI-based MA rising on
         three consecutive bars -> buy 50% of equity at the OPEN of the fourth
         bar.  Every 3 RSI points below the entry reading adds 10% of equity, up
         to five adds, and adds stop once RSI reaches 15.  Close at RSI 70.
  short  the mirror image: RSI above 70, MA falling, sell at the fourth bar's
         open, add every 3 points above entry, adds stop at 85, cover at RSI 30.

Sizing is measured against account equity at the moment of entry, so a fully
pyramided position is 50% + 5x10% = 100% of equity and never exceeds it.  That
is what makes "no leverage" exact rather than approximate.

There is no stop loss, by instruction.  A position is held until its exit level
prints, however long that takes, so the interesting risk number here is not the
closed-trade loss but the worst mark-to-market excursion while holding -- both
are reported.

RSI follows the Pine source exactly, including ta.rma's seeding with a simple
average of the first n values.  pandas' ewm(adjust=False) seeds with the first
observation instead, which leaves a visible difference for the first hundred
bars or so; on a 4h series that is weeks of signals, so it is worth matching.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def rma(s: pd.Series, n: int) -> pd.Series:
    """Wilder's moving average with Pine's seeding (SMA of the first n values)."""
    x = s.to_numpy(dtype=float)
    out = np.full(len(x), np.nan)
    if len(x) < n:
        return pd.Series(out, index=s.index)
    seed = np.nanmean(x[:n])
    out[n - 1] = seed
    alpha = 1.0 / n
    for i in range(n, len(x)):
        prev = out[i - 1]
        out[i] = prev + alpha * (x[i] - prev) if np.isfinite(x[i]) else prev
    return pd.Series(out, index=s.index)


def rsi_pine(close: pd.Series, n: int = 14) -> pd.Series:
    """RSI matching the supplied Pine indicator, including its 0/100 edge cases."""
    change = close.diff()
    up = rma(change.clip(lower=0), n)
    down = rma(-change.clip(upper=0), n)
    out = 100.0 - 100.0 / (1.0 + up / down.replace(0, np.nan))
    out = out.where(down != 0, 100.0)
    out = out.where(up != 0, 0.0)
    return out.where(up.notna() & down.notna())


@dataclass
class RsiParams:
    rsi_len: int = 14
    ma_len: int = 14
    trend_bars: int = 3          # consecutive MA moves required
    rsi_bars: int = 3            # consecutive RSI bars beyond the band
    lower: float = 30.0
    upper: float = 70.0
    add_step: float = 3.0        # RSI points past entry per add
    max_adds: int = 5
    long_add_floor: float = 15.0   # adds stop once RSI reaches this
    short_add_ceil: float = 85.0
    first_frac: float = 0.50     # of equity at entry
    add_frac: float = 0.10
    fee: float = 0.00065         # 4.5bp fee + 2bp slippage, per side


@dataclass
class RsiResult:
    equity: pd.Series
    trades: pd.DataFrame
    rsi: pd.Series
    ma: pd.Series
    params: RsiParams = field(default_factory=RsiParams)


def _streak_up(ma: np.ndarray, t: int, k: int) -> bool:
    """MA rose on each of the last k bars, which needs k+1 finite values."""
    if t - k < 0:
        return False
    seg = ma[t - k:t + 1]
    return bool(np.all(np.isfinite(seg)) and np.all(np.diff(seg) > 0))


def _streak_dn(ma: np.ndarray, t: int, k: int) -> bool:
    if t - k < 0:
        return False
    seg = ma[t - k:t + 1]
    return bool(np.all(np.isfinite(seg)) and np.all(np.diff(seg) < 0))


def run(df: pd.DataFrame, p: RsiParams | None = None,
        equity0: float = 1000.0) -> RsiResult:
    """Signals read bar t's close; every fill happens at bar t+1's open."""
    p = p or RsiParams()
    rsi = rsi_pine(df["close"], p.rsi_len)
    ma = rsi.rolling(p.ma_len).mean()

    r = rsi.to_numpy()
    m = ma.to_numpy()
    op = df["open"].to_numpy(float)
    hi = df["high"].to_numpy(float)
    lo = df["low"].to_numpy(float)
    idx = df.index
    n = len(df)

    equity = equity0
    eq = np.full(n, np.nan)

    side = 0                 # 0 flat, +1 long, -1 short
    legs: list[tuple[float, float]] = []      # (entry price, notional)
    entry_rsi = np.nan
    adds = 0
    entry_i = 0
    worst = 0.0              # worst mark-to-market excursion, fraction of equity
    base = equity0
    trades = []

    def position_value(px: float) -> float:
        return sum(q * side * (px / e - 1) for e, q in legs)

    for t in range(n):
        # ---------- act on the signal formed at t-1, filling at this bar's open
        if t > 0 and np.isfinite(op[t]):
            prev = t - 1
            if side == 0:
                long_ok = (_streak_up(m, prev, p.trend_bars)
                           and np.all(np.isfinite(r[prev - p.rsi_bars + 1:prev + 1]))
                           and np.all(r[prev - p.rsi_bars + 1:prev + 1] < p.lower))
                short_ok = (_streak_dn(m, prev, p.trend_bars)
                            and np.all(np.isfinite(r[prev - p.rsi_bars + 1:prev + 1]))
                            and np.all(r[prev - p.rsi_bars + 1:prev + 1] > p.upper))
                if long_ok or short_ok:
                    side = 1 if long_ok else -1
                    base = equity
                    notional = p.first_frac * base
                    legs = [(op[t], notional)]
                    equity -= notional * p.fee
                    entry_rsi = r[prev]
                    adds = 0
                    entry_i = t
                    worst = 0.0
            else:
                # exit first: an exit signal on the same bar as an add wins
                exit_now = (side > 0 and r[prev] >= p.upper) or \
                           (side < 0 and r[prev] <= p.lower)
                if exit_now:
                    pnl = position_value(op[t])
                    gross = sum(q for _, q in legs)
                    equity += pnl - gross * p.fee
                    trades.append({
                        "direction": "long" if side > 0 else "short",
                        "entry_time": idx[entry_i], "exit_time": idx[t],
                        "bars_held": t - entry_i,
                        "entry_rsi": entry_rsi, "exit_rsi": r[prev],
                        "legs": len(legs), "notional": gross,
                        "pnl": pnl - gross * p.fee,
                        "return_on_equity": (pnl - gross * p.fee) / base,
                        "worst_excursion": worst,
                    })
                    side, legs, adds = 0, [], 0
                elif adds < p.max_adds:
                    if side > 0:
                        want = entry_rsi - p.add_step * (adds + 1)
                        can = r[prev] <= want and r[prev] > p.long_add_floor
                    else:
                        want = entry_rsi + p.add_step * (adds + 1)
                        can = r[prev] >= want and r[prev] < p.short_add_ceil
                    if can:
                        notional = p.add_frac * base
                        legs.append((op[t], notional))
                        equity -= notional * p.fee
                        adds += 1

        # ---------- mark the book, tracking the worst point while holding
        if side != 0 and legs:
            adverse = lo[t] if side > 0 else hi[t]
            if np.isfinite(adverse):
                worst = min(worst, position_value(adverse) / base)
            eq[t] = equity + position_value(df["close"].to_numpy(float)[t])
        else:
            eq[t] = equity

    # settle anything still open at the final close
    if side != 0 and legs:
        px = float(df["close"].iloc[-1])
        pnl = position_value(px)
        gross = sum(q for _, q in legs)
        equity += pnl - gross * p.fee
        trades.append({
            "direction": "long" if side > 0 else "short",
            "entry_time": idx[entry_i], "exit_time": idx[-1],
            "bars_held": n - 1 - entry_i,
            "entry_rsi": entry_rsi, "exit_rsi": r[-1],
            "legs": len(legs), "notional": gross,
            "pnl": pnl - gross * p.fee,
            "return_on_equity": (pnl - gross * p.fee) / base,
            "worst_excursion": worst, "still_open_at_end": True,
        })
        eq[-1] = equity

    return RsiResult(equity=pd.Series(eq, index=idx),
                     trades=pd.DataFrame(trades), rsi=rsi, ma=ma, params=p)
