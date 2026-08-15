"""Portfolio-level grid bot with the two account rules that actually govern it.

The per-symbol ladder in vibt.grid answers "what does a martingale do to one
coin".  This answers the question that decides whether the product makes money,
because the platform's own settings screens show the account, not the ladder, is
where the outcome is decided:

  asset protection   -- net equity falling to (1 - stop) closes EVERY position
                        and halts trading.  20% is the tightest allowed.
  profit skimming    -- equity reaching (1 + skim) withdraws the profit, so the
                        account never compounds.  5% is the tightest allowed.

Those two turn the strategy into a repeated bet: bank +5%, or lose -20%.  The
arithmetic follows immediately and is the thing worth measuring -- four wins are
needed to pay for one stop, so the strategy is profitable if and only if it
completes more than four skim cycles per stop event.

Positions are hedged, both a long and a short ladder per coin, which is what
"双向持仓" means on the settings screen and is also why the two sides do not
cancel: each side martingales independently against its own entry.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class BotParams:
    step: float = 0.01           # grid spacing
    take_profit: float = 0.01    # per-level target above its own entry
    mult: float = 2.0            # martingale multiplier
    layers: int = 8              # ladder depth (the platform's typical 8-9)
    leverage: float = 20.0       # only caps how much notional the margin allows;
                                 # the order size above is already a notional
    first_order: float = 0.0028  # first level ORDER VALUE / equity (screenshot: 36/13000)
    stop: float = 0.20           # account drawdown that closes everything
    skim: float = 0.05           # profit level that gets withdrawn
    fee: float = 0.0005
    pause_move: float = 0.05     # daily adverse move that suspends new levels
    hedged: bool = True          # run a long ladder and a short ladder


@dataclass
class BotResult:
    equity: pd.Series            # net asset value including open positions
    banked: pd.Series            # cumulative profit withdrawn
    total: pd.Series             # equity + banked, the true account outcome
    skims: int = 0
    stops: int = 0
    stop_dates: list = field(default_factory=list)
    params: BotParams = field(default_factory=BotParams)

    @property
    def wins_per_stop(self) -> float:
        return self.skims / self.stops if self.stops else float("inf")


class _Ladder:
    """One side of one coin."""

    __slots__ = ("d", "entries", "sizes")

    def __init__(self, d: int):
        self.d = d
        self.entries: list[float] = []
        self.sizes: list[float] = []

    def value(self, px: float) -> float:
        return sum(s * self.d * (px / e - 1) for e, s in zip(self.entries, self.sizes))

    def reset(self) -> None:
        self.entries.clear()
        self.sizes.clear()


def run_bot(frames: dict[str, pd.DataFrame], p: BotParams,
            equity0: float = 1.0) -> BotResult:
    """Run the ladders across every coin under one shared account.

    Cash accounting is deliberately simple: realised P&L accrues to `cash`, open
    ladders are marked at each bar's close, and the two account rules act on the
    sum.  That is the quantity the platform itself stops and skims on.
    """
    idx = None
    for f in frames.values():
        idx = f.index if idx is None else idx.union(f.index)
    idx = idx.sort_values()
    cols = {s: f.reindex(idx) for s, f in frames.items()}
    sides = (1, -1) if p.hedged else (1,)
    ladders = {(s, d): _Ladder(d) for s in cols for d in sides}

    cash = equity0
    banked = 0.0
    eq_curve = np.empty(len(idx))
    bank_curve = np.empty(len(idx))
    skims = stops = 0
    stop_dates: list = []
    base_equity = equity0          # the level stop/skim are measured against

    for t, ts in enumerate(idx):
        floating = 0.0
        for s, f in cols.items():
            o, hi, lo, cl = (f["open"].iat[t], f["high"].iat[t],
                             f["low"].iat[t], f["close"].iat[t])
            if not np.isfinite(cl):
                continue
            # a large adverse day suspends NEW levels but holds what is open,
            # which is the platform's anti-wick rule
            day_move = (cl / o - 1) if np.isfinite(o) and o > 0 else 0.0

            for d in sides:
                L = ladders[(s, d)]
                adverse, favour = (lo, hi) if d > 0 else (hi, lo)
                paused = (d > 0 and day_move < -p.pause_move) or \
                         (d < 0 and day_move > p.pause_move)

                if not L.entries:
                    if not paused and np.isfinite(o):
                        L.entries.append(o)
                        L.sizes.append(p.first_order * base_equity)
                        cash -= L.sizes[-1] * p.fee
                elif not paused:
                    while len(L.entries) < p.layers:
                        nxt = L.entries[-1] * (1 - d * p.step)
                        if (d > 0 and adverse <= nxt) or (d < 0 and adverse >= nxt):
                            L.entries.append(nxt)
                            L.sizes.append(L.sizes[-1] * p.mult)
                            cash -= L.sizes[-1] * p.fee
                        else:
                            break

                keep_e, keep_s = [], []
                for e, sz in zip(L.entries, L.sizes):
                    tgt = e * (1 + d * p.take_profit)
                    if (d > 0 and favour >= tgt) or (d < 0 and favour <= tgt):
                        cash += sz * p.take_profit - sz * p.fee
                    else:
                        keep_e.append(e)
                        keep_s.append(sz)
                L.entries, L.sizes = keep_e, keep_s
                floating += L.value(cl)

        equity = cash + floating

        # ---- account rules, in the order the platform applies them
        if equity <= base_equity * (1 - p.stop):
            for L in ladders.values():
                L.reset()
            cash = equity                      # loss is realised on the way out
            stops += 1
            stop_dates.append(ts)
            base_equity = cash
            equity = cash
        elif equity >= base_equity * (1 + p.skim):
            take = equity - base_equity
            for L in ladders.values():
                L.reset()
            cash = equity - take               # profit leaves the account
            banked += take
            skims += 1
            equity = cash

        eq_curve[t] = equity
        bank_curve[t] = banked
        if equity <= 0:
            eq_curve[t:] = 0.0
            bank_curve[t:] = banked
            break

    return BotResult(
        equity=pd.Series(eq_curve, index=idx),
        banked=pd.Series(bank_curve, index=idx),
        total=pd.Series(eq_curve + bank_curve, index=idx),
        skims=skims, stops=stops, stop_dates=stop_dates, params=p,
    )
