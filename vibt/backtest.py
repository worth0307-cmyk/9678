"""Backtest engines.

Timing convention (the thing most backtests get wrong):

    signal s_t is computed from bars up to and including the CLOSE of bar t
    -> it is executed at the OPEN of bar t+1
    -> the position s_t is held over [open_{t+1}, open_{t+2})

So the engine shifts the caller's signal by one bar itself.  Never pass a
pre-shifted signal.  Returns are open-to-open, which is what a live executor
firing a market order on the new bar would actually capture.

Costs are charged on traded notional at every position change:
    cost = |Δposition| * (fee_bps + slippage_bps) / 10000
Binance USDT-M perp taker is 4.5bp at VIP0 (2bp+ with fee discounts); 2bp of
slippage on a market order in BTCUSDT is generous for retail size.  Default
6.5bp per side => 13bp round trip.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import metrics


@dataclass
class Costs:
    fee_bps: float = 4.5
    slip_bps: float = 2.0
    funding_bps_per_day: float = 0.0  # signed cost of holding a long perp; 0 = ignore

    @property
    def per_side(self) -> float:
        return (self.fee_bps + self.slip_bps) / 10_000.0


@dataclass
class Result:
    rets: pd.Series          # net per-bar arithmetic returns
    gross: pd.Series
    position: pd.Series      # position actually held during each bar
    equity: pd.Series
    costs: pd.Series
    trades: pd.DataFrame
    stats: metrics.Stats
    name: str = ""

    def __str__(self) -> str:
        return f"{self.name:<34} {self.stats}"


def _trade_log(pos: pd.Series, price: pd.Series, rets: pd.Series) -> pd.DataFrame:
    """Group consecutive bars of the same sign into round-trip trades."""
    p = pos.to_numpy()
    sign = np.sign(p)
    rows = []
    i = 0
    n = len(p)
    idx = pos.index
    px = price.to_numpy()
    r = rets.to_numpy()
    while i < n:
        if sign[i] == 0:
            i += 1
            continue
        j = i
        while j + 1 < n and sign[j + 1] == sign[i]:
            j += 1
        seg = r[i : j + 1]
        rows.append(
            {
                "entry_time": idx[i],
                "exit_time": idx[min(j + 1, n - 1)],
                "bars": j - i + 1,
                "side": "long" if sign[i] > 0 else "short",
                "entry_px": px[i],
                "exit_px": px[min(j + 1, n - 1)],
                "avg_size": float(np.abs(p[i : j + 1]).mean()),
                "pnl": float(np.prod(1 + seg) - 1),
            }
        )
        i = j + 1
    return pd.DataFrame(rows)


def run(
    df: pd.DataFrame,
    signal: pd.Series,
    costs: Costs | None = None,
    ann_factor: float = 365.0,
    name: str = "",
    bars_per_day: float = 1.0,
) -> Result:
    """Vectorised engine.  `signal` = desired position as of each bar's close."""
    costs = costs or Costs()
    px = df["open"]
    sig = signal.reindex(df.index).fillna(0.0).astype(float)

    pos = sig.shift(1).fillna(0.0)          # held during bar t (entered at open of t)
    bar_ret = px.shift(-1) / px - 1.0       # open-to-open return of bar t
    gross = pos * bar_ret

    turn = pos.diff().abs().fillna(pos.abs())
    cost = turn * costs.per_side
    if costs.funding_bps_per_day:
        cost = cost + pos * (costs.funding_bps_per_day / 10_000.0) / bars_per_day

    net = (gross - cost).iloc[:-1]          # last bar has no forward return
    pos_v = pos.iloc[:-1]
    equity = (1 + net.fillna(0)).cumprod()

    trades = _trade_log(pos_v, px.iloc[:-1], net.fillna(0))
    st = metrics.compute(net, ann_factor, position=pos_v,
                         trade_pnl=trades["pnl"].to_numpy() if len(trades) else None)
    return Result(net, gross.iloc[:-1], pos_v, equity, cost.iloc[:-1], trades, st, name)


def run_with_risk(
    df: pd.DataFrame,
    signal: pd.Series,
    stop_dist: pd.Series | None = None,
    trail_atr: pd.Series | None = None,
    take_dist: pd.Series | None = None,
    costs: Costs | None = None,
    ann_factor: float = 365.0,
    name: str = "",
    bars_per_day: float = 1.0,
    reenter_after_stop: bool = False,
) -> Result:
    """Bar-stepping engine that honours intrabar stops / trailing stops / targets.

    `stop_dist` / `take_dist` / `trail_atr` are absolute price distances known at
    the close of the signal bar (e.g. 2 * ATR).  Intrabar we assume the ADVERSE
    level is hit first whenever both stop and target sit inside one bar -- the
    pessimistic assumption, so results are a floor, not a ceiling.

    After a stop-out the engine stays flat until the signal flips (or, with
    `reenter_after_stop`, until the next bar where the signal is still live).
    """
    costs = costs or Costs()
    idx = df.index
    o = df["open"].to_numpy()
    h = df["high"].to_numpy()
    lo = df["low"].to_numpy()
    n = len(df)

    sig = signal.reindex(idx).fillna(0.0).to_numpy(dtype=float)
    sd = stop_dist.reindex(idx).to_numpy(dtype=float) if stop_dist is not None else None
    td = take_dist.reindex(idx).to_numpy(dtype=float) if take_dist is not None else None
    ta = trail_atr.reindex(idx).to_numpy(dtype=float) if trail_atr is not None else None

    pos = np.zeros(n)
    ret = np.zeros(n)
    cost_arr = np.zeros(n)

    cur = 0.0
    entry_px = np.nan
    stop_px = np.nan
    take_px = np.nan
    best = np.nan
    blocked = False   # stopped out of this side; wait for the signal to flip or go flat
    stop_side = 0.0

    per_side = costs.per_side
    funding = (costs.funding_bps_per_day / 10_000.0) / bars_per_day if costs.funding_bps_per_day else 0.0

    for t in range(n - 1):
        want = sig[t - 1] if t > 0 else 0.0     # signal from previous close
        if blocked:
            if reenter_after_stop or want == 0.0 or np.sign(want) != stop_side:
                blocked = False
            else:
                want = 0.0

        # --- rebalance at this bar's open
        if want != cur:
            traded = abs(want - cur)
            cost_arr[t] += traded * per_side
            if np.sign(want) != np.sign(cur) or cur == 0:
                entry_px = o[t]
                best = o[t]
                k = t - 1 if t > 0 else 0
                stop_px = entry_px - np.sign(want) * sd[k] if sd is not None and np.isfinite(sd[k]) else np.nan
                take_px = entry_px + np.sign(want) * td[k] if td is not None and np.isfinite(td[k]) else np.nan
            cur = want
        pos[t] = cur

        if cur == 0:
            continue

        # A position opened while ATR was still warming up has no stop attached.
        # Arm it as soon as a distance becomes available, measured from the entry
        # price, rather than leaving the risk overlay silently disabled forever.
        k = t - 1 if t > 0 else 0
        if sd is not None and not np.isfinite(stop_px) and np.isfinite(sd[k]):
            stop_px = entry_px - np.sign(cur) * sd[k]
        if td is not None and not np.isfinite(take_px) and np.isfinite(td[k]):
            take_px = entry_px + np.sign(cur) * td[k]

        # --- walk the bar
        exit_px = None
        side = np.sign(cur)
        hit_stop = np.isfinite(stop_px) and ((lo[t] <= stop_px) if side > 0 else (h[t] >= stop_px))
        hit_take = np.isfinite(take_px) and ((h[t] >= take_px) if side > 0 else (lo[t] <= take_px))
        if hit_stop:                      # pessimistic: stop wins ties
            exit_px = stop_px
        elif hit_take:
            exit_px = take_px

        if exit_px is not None:
            ret[t] = cur * (exit_px / o[t] - 1.0) - funding * cur
            cost_arr[t] += abs(cur) * per_side
            stop_side = side
            cur = 0.0
            blocked = hit_stop
            stop_px = take_px = np.nan
            continue

        # --- carry to next open, then update the trailing stop for the next bar
        ret[t] = cur * (o[t + 1] / o[t] - 1.0) - funding * cur
        if ta is not None and np.isfinite(ta[t]):
            if side > 0:
                best = max(best, h[t])
                cand = best - ta[t]
                stop_px = cand if not np.isfinite(stop_px) else max(stop_px, cand)
            else:
                best = min(best, lo[t])
                cand = best + ta[t]
                stop_px = cand if not np.isfinite(stop_px) else min(stop_px, cand)

    net = pd.Series(ret - cost_arr, index=idx).iloc[:-1]
    pos_s = pd.Series(pos, index=idx).iloc[:-1]
    equity = (1 + net).cumprod()
    trades = _trade_log(pos_s, df["open"].iloc[:-1], net)
    st = metrics.compute(net, ann_factor, position=pos_s,
                         trade_pnl=trades["pnl"].to_numpy() if len(trades) else None)
    return Result(net, pd.Series(ret, index=idx).iloc[:-1], pos_s, equity,
                  pd.Series(cost_arr, index=idx).iloc[:-1], trades, st, name)


def buy_and_hold(df: pd.DataFrame, ann_factor: float, costs: Costs | None = None) -> Result:
    sig = pd.Series(1.0, index=df.index)
    return run(df, sig, costs=costs, ann_factor=ann_factor, name="buy & hold")


def vol_target(
    signal: pd.Series,
    vol: pd.Series,
    target_ann_vol: float = 0.35,
    max_leverage: float = 1.0,
    min_leverage: float = 0.0,
) -> pd.Series:
    """Scale a -1..1 signal so that realised vol lands near the target."""
    lev = (target_ann_vol / vol.replace(0, np.nan)).clip(min_leverage, max_leverage)
    return (signal * lev).fillna(0.0)
