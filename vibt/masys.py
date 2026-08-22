"""The six-moving-average system: 均线密集 / 均线发散, two entries, bracket exits.

The method as described: plot MA20/60/120 and EMA20/60/120 together and read two
states off them.

  密集 (cluster)    all six braided, order indistinguishable.  The market's cost
                    basis has converged; nobody has an edge; a move out of the
                    cluster is the start of something.
  发散 (fan)        the six fan out and stack in order -- short above medium
                    above long for a bull, the reverse for a bear.

Two entries follow from that:

  A  均线密集开仓法      find a cluster, trade whichever side price is holding,
                        stop on an effective break back through the cluster.
  B  第一次回踩20均线    after a cluster opens into a fan, take the first pullback
                        that touches MA20 and closes back above it.

and the exits are all fixed-odds brackets: risk the distance to the stop, take
profit at some multiple of it (1:3 and 1:5 are the two quoted).

Everything above is mechanical.  Two things in the source are not, and both are
resolved here by a rule rather than by eye:

  "均线密集"     needs a number.  A fixed percentage cannot work across BTC and a
                 memecoin, or across 4h and daily, so the default is each series'
                 own recent distribution: the six-MA spread in its bottom
                 `q_tight` of the last `window` bars.  Computed on prior bars
                 only -- a cluster you can only identify with hindsight is not a
                 cluster you can trade.
  "有效跌破"     is taken as a bar CLOSING beyond the level, not a wick through
                 it.  That is the more forgiving reading; the stricter one (any
                 touch) is available via `close_only=False` and is worse.

The simulator brackets every trade: entry at the open after the signal bar, one
position at a time, adverse extreme first inside the bar, and a bar that opens
through the stop fills at the open rather than at the stop.  Risk per trade is a
fixed fraction of equity, which is the sizing the method itself prescribes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

MA_LENS = (20, 60, 120)


def six_mas(close: pd.Series, lens: tuple[int, ...] = MA_LENS) -> pd.DataFrame:
    """MA20/60/120 and EMA20/60/120 on one frame."""
    out = {}
    for n in lens:
        out[f"ma{n}"] = close.rolling(n).mean()
        out[f"ema{n}"] = close.ewm(span=n, adjust=False).mean()
    d = pd.DataFrame(out)
    # EMA is defined from bar 0 by construction, which would let a "cluster" be
    # declared out of three near-identical seeds before any of them mean
    # anything.  Hold every column back until its own window has filled.
    warm = max(lens)
    d.iloc[:warm] = np.nan
    return d


def spread(mas: pd.DataFrame, close: pd.Series) -> pd.Series:
    """(widest MA - narrowest MA) / price: 0 = perfectly braided."""
    return (mas.max(axis=1) - mas.min(axis=1)) / close


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Average true range -- the scale a stop distance has to be measured against."""
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(),
                    (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def stack(mas: pd.DataFrame, lens: tuple[int, ...] = MA_LENS) -> pd.Series:
    """+1 when short>medium>long on both families, -1 reversed, 0 mixed."""
    s, m, l = lens
    up = ((mas[f"ma{s}"] > mas[f"ma{m}"]) & (mas[f"ma{m}"] > mas[f"ma{l}"])
          & (mas[f"ema{s}"] > mas[f"ema{m}"]) & (mas[f"ema{m}"] > mas[f"ema{l}"]))
    dn = ((mas[f"ma{s}"] < mas[f"ma{m}"]) & (mas[f"ma{m}"] < mas[f"ma{l}"])
          & (mas[f"ema{s}"] < mas[f"ema{m}"]) & (mas[f"ema{m}"] < mas[f"ema{l}"]))
    return pd.Series(np.where(up, 1.0, np.where(dn, -1.0, 0.0)),
                     index=mas.index).where(mas.notna().all(axis=1))


@dataclass
class MaParams:
    lens: tuple[int, ...] = MA_LENS
    # --- what counts as 密集
    tight_mode: str = "quantile"   # "quantile" | "fixed"
    window: int = 250              # bars of history defining "recent"
    q_tight: float = 0.25          # bottom quarter of the spread distribution
    fixed_tight: float = 0.03      # used when tight_mode == "fixed"
    min_cluster: int = 3           # bars the cluster must persist before it counts

    # --- entries
    close_only: bool = True        # a break needs a CLOSE beyond the level
    min_risk: float = 0.005        # skip signals whose stop is closer than this
    max_risk: float = 0.25         # ... or further than this
    pullback_within: int = 30      # bars after a fan forms to still call it "第一次"

    # --- exits
    target_r: float = 3.0          # 赔率平仓法: 1:3 by default
    max_bars: int = 120            # give up if neither level prints
    risk_frac: float = 0.01        # risk 1% of equity per trade
    fee: float = 0.00065           # 4.5bp fee + 2bp slippage, per side
    allow_short: bool = True
    atr_len: int = 14              # the yardstick the null matches stops against


@dataclass
class Signal:
    i: int                 # bar whose OPEN is the fill
    side: int              # +1 long, -1 short
    stop: float            # price
    kind: str = ""
    target: float = np.nan  # set when the exit rule names a level, not a multiple


@dataclass
class MaResult:
    equity: pd.Series
    trades: pd.DataFrame
    tight: pd.Series
    spread: pd.Series
    stack: pd.Series
    params: MaParams = field(default_factory=MaParams)


def tight_mask(sp: pd.Series, p: MaParams) -> pd.Series:
    """Bars where the six MAs are braided, judged only against prior bars."""
    if p.tight_mode == "fixed":
        return sp < p.fixed_tight
    prior = sp.shift(1)
    thr = prior.rolling(p.window, min_periods=p.window // 2).quantile(p.q_tight)
    return (sp < thr).where(thr.notna(), False)


# --------------------------------------------------------------------- entries
def _context(df: pd.DataFrame, p: MaParams):
    mas = six_mas(df["close"], p.lens)
    sp = spread(mas, df["close"])
    return (mas, tight_mask(sp, p).to_numpy(), stack(mas, p.lens).to_numpy(),
            mas.max(axis=1).to_numpy(), mas.min(axis=1).to_numpy())


def entries_cluster_break(df: pd.DataFrame, p: MaParams) -> list[Signal]:
    """A: price leaves a persistent cluster; stop is the far side of the cluster.

    One entry per cluster, on the bar that first closes outside the band -- not
    on every bar that happens to be outside it.  That distinction is the whole
    rule: 密集开仓法 is an event, and treating it as a state turns it into a
    trend-following filter that fires dozens of times per move.

    The cluster is allowed a grace period after the six MAs stop being tight,
    because the spread widens on the breakout itself; without it the rule would
    disqualify exactly the bar it is supposed to trade.
    """
    _, tight, _, top, bot = _context(df, p)
    cl = df["close"].to_numpy(float)
    op = df["open"].to_numpy(float)

    out: list[Signal] = []
    run = 0
    live = False           # a qualifying cluster is on the table
    grace = 0
    prev_tight = False
    for t in range(len(df) - 1):
        if tight[t]:
            run = run + 1 if prev_tight else 1
            if run == p.min_cluster:
                live, grace = True, 0    # arms once, when the cluster qualifies
        else:
            run = 0
            if live:
                grace += 1
                if grace > p.pullback_within:
                    live = False
        prev_tight = bool(tight[t])
        if not live or not np.isfinite(top[t]):
            continue
        if cl[t] > top[t]:
            side, stop = 1, bot[t]
        elif cl[t] < bot[t] and p.allow_short:
            side, stop = -1, top[t]
        else:
            continue
        live = False                     # this cluster has had its trade
        fill = op[t + 1]
        risk = abs(fill - stop) / fill
        if not (p.min_risk <= risk <= p.max_risk):
            continue
        out.append(Signal(t + 1, side, stop, "密集突破"))
    return out


def entries_ma20_pullback(df: pd.DataFrame, p: MaParams) -> list[Signal]:
    """B: cluster -> fan -> the first pullback to MA20 that closes back above it.

    Three stages, and the fan is a stage of its own: on the bar a cluster ends
    the six MAs are still interleaved, so the stack is almost never ordered yet.
    Requiring the fan on that bar is what makes a literal reading of the rule
    produce two signals in three and a half years.

    "First" is enforced literally -- one signal per fan.  Without that this
    degenerates into "buy every touch of MA20".
    """
    short_len = min(p.lens)
    mas, tight, st, _, _ = _context(df, p)
    ma20 = mas[f"ma{short_len}"].to_numpy()
    cl, op = df["close"].to_numpy(float), df["open"].to_numpy(float)
    hi, lo = df["high"].to_numpy(float), df["low"].to_numpy(float)

    out: list[Signal] = []
    stage = "idle"      # idle -> cluster -> fan_wait -> armed
    armed = 0
    since = 0
    for t in range(len(df) - 1):
        if tight[t]:
            stage, armed, since = "cluster", 0, 0
            continue
        if stage == "cluster":
            stage, since = "fan_wait", 0
        since += 1
        if stage == "fan_wait":
            if st[t] != 0:
                stage, armed, since = "armed", int(st[t]), 0
            elif since > p.pullback_within:
                stage = "idle"
            continue
        if stage != "armed":
            continue
        if since > p.pullback_within or st[t] != armed or not np.isfinite(ma20[t]):
            if since > p.pullback_within or st[t] != armed:
                stage, armed = "idle", 0
            continue
        if armed > 0 and lo[t] <= ma20[t] < cl[t]:
            side, stop = 1, min(lo[t], ma20[t]) * (1 - 1e-9)
        elif armed < 0 and hi[t] >= ma20[t] > cl[t] and p.allow_short:
            side, stop = -1, max(hi[t], ma20[t]) * (1 + 1e-9)
        else:
            continue
        stage, armed = "idle", 0         # 第一次 means once
        fill = op[t + 1]
        risk = abs(fill - stop) / fill
        if not (p.min_risk <= risk <= p.max_risk):
            continue
        out.append(Signal(t + 1, side, stop, "回踩20均线"))
    return out


# ------------------------------------------------------ the stripped controls
# Rules A and B are each a plain price pattern wrapped in the video's cluster
# apparatus.  Removing the apparatus and leaving the pattern is the only way to
# find out which half is doing the work: if these match the full rules, then
# 均线密集/均线发散 is decoration on an effect that needs neither.
def entries_band_break(df: pd.DataFrame, p: MaParams) -> list[Signal]:
    """A minus the cluster: any close out of the six-MA band, once per crossing."""
    _, _, _, top, bot = _context(df, p)
    cl, op = df["close"].to_numpy(float), df["open"].to_numpy(float)
    out, outside = [], 0
    for t in range(len(df) - 1):
        if not np.isfinite(top[t]):
            continue
        now = 1 if cl[t] > top[t] else (-1 if cl[t] < bot[t] else 0)
        if now == 0 or now == outside:
            outside = now
            continue
        outside = now
        if now < 0 and not p.allow_short:
            continue
        stop = bot[t] if now > 0 else top[t]
        fill = op[t + 1]
        risk = abs(fill - stop) / fill
        if p.min_risk <= risk <= p.max_risk:
            out.append(Signal(t + 1, now, stop, "对照:任意突破均线带"))
    return out


def entries_ma20_touch(df: pd.DataFrame, p: MaParams,
                       require_stack: bool = True) -> list[Signal]:
    """B minus the cluster: every dip through MA20 that closes back above it.

    With `require_stack` the six MAs still have to be fanned in trend order, so
    this is "pull-back in an uptrend".  Without it, it is nothing but "price
    poked through its 20-bar average and closed back".
    """
    short_len = min(p.lens)
    mas, _, st, _, _ = _context(df, p)
    ma20 = mas[f"ma{short_len}"].to_numpy()
    cl, op = df["close"].to_numpy(float), df["open"].to_numpy(float)
    hi, lo = df["high"].to_numpy(float), df["low"].to_numpy(float)
    kind = "对照:回踩MA20(带趋势)" if require_stack else "对照:回踩MA20(裸)"
    out = []
    for t in range(len(df) - 1):
        if not np.isfinite(ma20[t]) or (require_stack and st[t] == 0):
            continue
        up_ok = (not require_stack) or st[t] > 0
        dn_ok = ((not require_stack) or st[t] < 0) and p.allow_short
        if up_ok and lo[t] <= ma20[t] < cl[t]:
            side, stop = 1, min(lo[t], ma20[t]) * (1 - 1e-9)
        elif dn_ok and hi[t] >= ma20[t] > cl[t]:
            side, stop = -1, max(hi[t], ma20[t]) * (1 + 1e-9)
        else:
            continue
        fill = op[t + 1]
        risk = abs(fill - stop) / fill
        if p.min_risk <= risk <= p.max_risk:
            out.append(Signal(t + 1, side, stop, kind))
    return out


# --------------------------------------------------------------------- exits
def prior_cluster_levels(df: pd.DataFrame, p: MaParams) -> pd.DataFrame:
    """Every completed cluster's price level, as of each bar.

    上一个均线密集平仓法 targets the last place the market's average cost
    converged.  What is returned is the most recent COMPLETED cluster above the
    current price and the most recent one below it, because which of the two is
    the target depends on the direction of the trade.

    A cluster only becomes a level once it has ended -- while price is still
    inside it, it is not a target, it is where you are.
    """
    mas = six_mas(df["close"], p.lens)
    tight = tight_mask(spread(mas, df["close"]), p).to_numpy()
    mid = ((mas.max(axis=1) + mas.min(axis=1)) / 2).to_numpy()
    cl = df["close"].to_numpy(float)
    n = len(df)

    above = np.full(n, np.nan)
    below = np.full(n, np.nan)
    done: list[float] = []          # levels of clusters that have finished
    run_vals: list[float] = []
    for t in range(n):
        if tight[t]:
            if np.isfinite(mid[t]):
                run_vals.append(mid[t])
        elif run_vals:
            done.append(float(np.mean(run_vals)))
            run_vals = []
        if done and np.isfinite(cl[t]):
            up = [d for d in done if d > cl[t]]
            dn = [d for d in done if d < cl[t]]
            if up:
                above[t] = up[-1]
            if dn:
                below[t] = dn[-1]
    return pd.DataFrame({"above": above, "below": below}, index=df.index)


def swing_extension(df: pd.DataFrame, p: MaParams, i: int, side: int,
                    ratio: float, lookback: int = 60) -> float:
    """Fibonacci extension of the swing that led into bar `i`.

    For a long: the swing runs from the lowest low in the lookback to the
    highest high after it, and the target is swing_low + ratio*(range).  Only
    bars strictly before the entry are read, so a 1.618 target is set from the
    move that has already happened, not the one being predicted.
    """
    lo0 = max(0, i - lookback)
    hi = df["high"].to_numpy(float)[lo0:i]
    lo = df["low"].to_numpy(float)[lo0:i]
    if len(hi) < 5:
        return np.nan
    if side > 0:
        j = int(np.nanargmin(lo))
        top = np.nanmax(hi[j:]) if j < len(hi) else np.nan
        base, rng = lo[j], top - lo[j]
    else:
        j = int(np.nanargmax(hi))
        bot = np.nanmin(lo[j:]) if j < len(lo) else np.nan
        base, rng = hi[j], hi[j] - bot
    if not np.isfinite(rng) or rng <= 0:
        return np.nan
    return float(base + side * ratio * rng)


# ------------------------------------------------------------------- simulator
def _bracket(s: Signal, p: MaParams, op, hi, lo, cl, idx, n, av=None) -> dict | None:
    """Walk one signal's bracket forward.  Adverse extreme first, always."""
    t = s.i
    fill = op[t]
    if not np.isfinite(fill):
        return None
    risk_px = abs(fill - s.stop)
    if risk_px <= 0:
        return None
    tgt = s.target if np.isfinite(s.target) else fill + s.side * p.target_r * risk_px

    exit_i, exit_px, why = None, np.nan, ""
    for u in range(t, min(n, t + p.max_bars + 1)):
        adverse = lo[u] if s.side > 0 else hi[u]
        favour = hi[u] if s.side > 0 else lo[u]
        hit_stop = (adverse <= s.stop) if s.side > 0 else (adverse >= s.stop)
        hit_tgt = (favour >= tgt) if s.side > 0 else (favour <= tgt)
        if hit_stop:                      # a bar holding both fills the stop
            px = s.stop
            if (s.side > 0 and op[u] < s.stop) or (s.side < 0 and op[u] > s.stop):
                px = op[u]               # gapped through: no better than the open
            exit_i, exit_px, why = u, px, "stop"
            break
        if hit_tgt:
            exit_i, exit_px, why = u, tgt, "target"
            break
    if exit_i is None:
        exit_i = min(n - 1, t + p.max_bars)
        exit_px, why = cl[exit_i], "timeout"

    gross_r = s.side * (exit_px - fill) / risk_px
    # two round-trip fees, expressed in R so the whole log is on one scale
    fee_r = 2 * p.fee / (risk_px / fill)
    # how wide the stop is in units of the volatility prevailing BEFORE entry:
    # the number the null has to match, or it is comparing stops to a different
    # market than the one they were sized in
    a = av[t - 1] if av is not None and t > 0 and np.isfinite(av[t - 1]) and av[t - 1] > 0 else np.nan
    return {"kind": s.kind, "side": "long" if s.side > 0 else "short",
            "entry_time": idx[t], "exit_time": idx[exit_i], "entry_i": t,
            "exit_i": exit_i, "bars": exit_i - t, "entry": fill, "stop": s.stop,
            "target": tgt, "exit": exit_px, "why": why, "risk_pct": risk_px / fill,
            "risk_atr": risk_px / a if np.isfinite(a) else np.nan,
            "r_gross": gross_r, "r_multiple": gross_r - fee_r}


def evaluate(df: pd.DataFrame, signals: list[Signal], p: MaParams) -> pd.DataFrame:
    """Every signal bracketed independently -- the measurement, not the portfolio.

    Win rate and reward:risk are properties of the SIGNAL, so they have to be
    measured over all signals.  Running one position at a time throws away
    whichever signals happen to arrive while another trade is open, and on daily
    bars that silently discards most of them; the survivors are then a sample
    selected by timing rather than by the rule.
    """
    op, hi = df["open"].to_numpy(float), df["high"].to_numpy(float)
    lo, cl = df["low"].to_numpy(float), df["close"].to_numpy(float)
    av = atr(df, p.atr_len).to_numpy(float)
    rows = [r for r in (_bracket(s, p, op, hi, lo, cl, df.index, len(df), av)
                        for s in signals) if r is not None]
    return pd.DataFrame(rows)


def simulate(df: pd.DataFrame, signals: list[Signal], p: MaParams,
             equity0: float = 1000.0) -> MaResult:
    """The tradeable version: one position at a time, fixed fractional risk."""
    op, hi = df["open"].to_numpy(float), df["high"].to_numpy(float)
    lo, cl = df["low"].to_numpy(float), df["close"].to_numpy(float)
    idx, n = df.index, len(df)

    by_bar: dict[int, Signal] = {}
    for s in signals:
        by_bar.setdefault(s.i, s)      # first signal on a bar wins

    equity = equity0
    eq = np.full(n, np.nan)
    trades = []
    busy_until = -1
    for t in range(n):
        s = by_bar.get(t)
        if s is not None and t > busy_until:
            row = _bracket(s, p, op, hi, lo, cl, idx, n)
            if row is not None:
                notional = (p.risk_frac * equity) / row["risk_pct"]
                pnl = notional * (row["r_gross"] * row["risk_pct"]) - 2 * notional * p.fee
                trades.append(row | {"pnl": pnl, "equity_before": equity})
                equity += pnl
                busy_until = row["exit_i"]
        eq[t] = equity

    mas = six_mas(df["close"], p.lens)
    sp = spread(mas, df["close"])
    return MaResult(equity=pd.Series(eq, index=idx), trades=pd.DataFrame(trades),
                    tight=tight_mask(sp, p), spread=sp, stack=stack(mas, p.lens),
                    params=p)


def random_entries(df: pd.DataFrame, p: MaParams, n_long: int, n_short: int,
                   rng: np.random.Generator, risk_atrs: np.ndarray,
                   risk_pcts: np.ndarray | None = None,
                   prev_extreme: bool = False) -> list[Signal]:
    """The null: same counts per side, same stop WIDTH, random bars.

    This is the control the method has to beat.  A 1:3 bracket wins near 25% of
    the time on any random walk, so "30-40% at 1:3" is only evidence if the
    entries place the bracket better than chance does.

    Two matches matter and both were wrong in the obvious version of this test:

      volatility  a stop is sized against the volatility at the time it is set.
                  Copying the percentage instead of the ATR multiple drops a 2%
                  stop into whatever regime the random bar lands in, and it gets
                  taken out for reasons that have nothing to do with the entry.
                  So the multiple is what gets resampled; pass `risk_pcts`
                  instead only to reproduce that weaker null deliberately.
      direction   longs and shorts are drawn to their own counts rather than
                  flipped at the pooled ratio, so a sample where everything bled
                  cannot hand the null a different long/short balance than the
                  rule had.

    `prev_extreme` is the strictest form: instead of resampling a width, the stop
    is CONSTRUCTED the way rule B constructs it -- at the previous bar's low for
    a long, its high for a short.  Only the timing is then random, so anything
    the rule still wins by is attributable to when it trades and nothing else.
    """
    warm = max(p.lens) + p.window // 2 + p.atr_len
    lo_i, hi_i = warm, len(df) - 2
    n_sig = n_long + n_short
    if hi_i <= lo_i or n_sig == 0:
        return []
    op = df["open"].to_numpy(float)
    plo, phi = df["low"].to_numpy(float), df["high"].to_numpy(float)
    av = atr(df, p.atr_len).to_numpy(float)
    sides = np.r_[np.ones(n_long), -np.ones(n_short)]
    bars = rng.integers(lo_i, hi_i, size=n_sig)
    widths = mult = None
    if prev_extreme:
        pass
    elif risk_pcts is not None:
        widths = rng.choice(risk_pcts, size=n_sig)
    else:
        mult = rng.choice(risk_atrs[np.isfinite(risk_atrs)], size=n_sig)
    out = []
    for k, b in enumerate(bars):
        fill = op[b]
        if not np.isfinite(fill):
            continue
        side = int(sides[k])
        if prev_extreme:
            edge = plo[b - 1] if side > 0 else phi[b - 1]
            if not np.isfinite(edge):
                continue
            stop = edge * (1 - side * 1e-9)
            r = abs(fill - stop) / fill
        else:
            if widths is not None:
                r = widths[k]
            else:
                if not np.isfinite(av[b - 1]) or av[b - 1] <= 0:
                    continue
                r = mult[k] * av[b - 1] / fill
            stop = fill * (1 - side * r)
        if not (p.min_risk <= r <= p.max_risk):
            continue
        out.append(Signal(int(b), side, stop, "随机"))
    return out
