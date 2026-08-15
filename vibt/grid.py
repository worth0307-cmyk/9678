"""Grid / martingale simulator, for looking at what those equity curves hide.

These bots buy a ladder of levels on the way down, size each level larger than
the last, and close each level individually once price recovers past its own
entry.  Losing levels are never closed, so they never enter realised P&L.  That
single fact produces the whole marketing profile: a 95-99% win rate, a smooth
realised equity curve and a tiny realised drawdown, none of which are evidence
about the strategy -- they are definitional.

So the simulator tracks two account values, and the gap between them is the
entire point:

  realised  -- only closed trades, which is what a dashboard usually shows
  mark      -- realised plus open positions marked to market, which is what
               the account is actually worth and what gets liquidated

Fills use the bar's own high and low, with adverse ordering inside the bar: all
ladder entries that the low reaches are taken before any take-profit the high
would have reached.  Real intrabar sequence is unknowable from OHLC, and this
direction is the one that does not flatter the strategy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class GridParams:
    step: float = 0.01           # add a level each `step` below the last entry
    take_profit: float = 0.01    # close a level once price is tp above ITS entry
    mult: float = 2.0            # each level this many times the previous
    base: float = 0.02           # first level's notional, as a fraction of start equity
    max_levels: int = 7          # ladder depth; beyond this the bot simply waits
    fee: float = 0.0005          # 5bp per side, taker
    direction: int = 1           # +1 long grid, -1 short grid

    @property
    def committed(self) -> float:
        """Total notional if every level fills, as a multiple of starting equity."""
        if self.mult == 1.0:
            return self.base * self.max_levels
        return self.base * (self.mult ** self.max_levels - 1) / (self.mult - 1)


@dataclass
class GridResult:
    realised: pd.Series
    mark: pd.Series
    levels: pd.Series
    closed: int = 0
    wins: int = 0
    liquidated_at: pd.Timestamp | None = None
    max_levels_used: int = 0
    params: GridParams = field(default_factory=GridParams)

    @property
    def win_rate(self) -> float:
        return self.wins / self.closed if self.closed else float("nan")

    @staticmethod
    def _dd(s: pd.Series) -> float:
        return float((s / s.cummax() - 1).min())

    @property
    def realised_dd(self) -> float:
        return self._dd(self.realised)

    @property
    def mark_dd(self) -> float:
        return self._dd(self.mark)


def run_grid(df: pd.DataFrame, p: GridParams, equity0: float = 1.0) -> GridResult:
    """Walk the bars, filling ladder entries and take-profits.

    A level is opened when price trades `step` past the previous entry, and
    closed when price trades `take_profit` back past its own entry.  Nothing
    closes a losing level, which is what the whole design rests on.
    """
    o = df["open"].to_numpy(float)
    hi = df["high"].to_numpy(float)
    lo = df["low"].to_numpy(float)
    n = len(df)

    entries: list[float] = []      # entry price per open level
    sizes: list[float] = []        # notional per open level
    realised = equity0
    d = p.direction

    real_curve = np.empty(n)
    mark_curve = np.empty(n)
    lvl_curve = np.empty(n)
    closed = wins = 0
    max_used = 0
    dead_at = None

    for t in range(n):
        # adverse extreme first: for a long grid that is the low
        adverse = lo[t] if d > 0 else hi[t]
        favour = hi[t] if d > 0 else lo[t]

        if not entries:
            px = o[t]
            entries.append(px)
            sizes.append(p.base * equity0)
            realised -= sizes[-1] * p.fee
        else:
            # fill as many ladder steps as this bar's adverse extreme reaches
            while len(entries) < p.max_levels:
                nxt = entries[-1] * (1 - d * p.step)
                if (d > 0 and adverse <= nxt) or (d < 0 and adverse >= nxt):
                    size = sizes[-1] * p.mult
                    entries.append(nxt)
                    sizes.append(size)
                    realised -= size * p.fee
                else:
                    break
        max_used = max(max_used, len(entries))

        # take profit on any level the favourable extreme reached
        keep_e, keep_s = [], []
        for e, s in zip(entries, sizes):
            tgt = e * (1 + d * p.take_profit)
            if (d > 0 and favour >= tgt) or (d < 0 and favour <= tgt):
                realised += s * p.take_profit - s * p.fee
                closed += 1
                wins += 1
            else:
                keep_e.append(e)
                keep_s.append(s)
        entries, sizes = keep_e, keep_s

        close = df["close"].to_numpy(float)[t]
        floating = sum(s * d * (close / e - 1) for e, s in zip(entries, sizes))
        real_curve[t] = realised
        mark_curve[t] = realised + floating
        lvl_curve[t] = len(entries)

        if mark_curve[t] <= 0 and dead_at is None:
            dead_at = df.index[t]
            real_curve[t:] = 0.0
            mark_curve[t:] = 0.0
            lvl_curve[t:] = len(entries)
            break

    idx = df.index
    return GridResult(
        realised=pd.Series(real_curve, index=idx),
        mark=pd.Series(mark_curve, index=idx),
        levels=pd.Series(lvl_curve, index=idx),
        closed=closed, wins=wins, liquidated_at=dead_at,
        max_levels_used=max_used, params=p,
    )
