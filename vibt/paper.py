"""纸面信号生成器：每天输出目标持仓，记录下来，以后和真实成交对账.

The point of this is measurement, not profit.  What it measures, though, is not
what an earlier version of this docstring claimed.  The breakeven cost of the
configuration below is 47.7bp per side against 6.5bp charged (44 号脚本) -- more
than seven times the assumed cost -- so fills are not what decides whether it
works.  The 6.38bp figure that used to be quoted here belongs to the PCA
residual reversion of 39/40 号脚本, a different strategy on 200 coins turning
over 441x a year, and quoting it here pointed the effort at the wrong problem.

What actually decides it is selection: these six coins were picked after seeing
which ones worked, and every parameter below was chosen on their history.  So
the log's first job is pre-registration -- each signal is committed to git
before its outcome is known, which is the one thing that makes a later claim
about this strategy checkable.  Measuring slippage is its second job, and it
bounds the estimate rather than deciding it.

Three properties this module exists to guarantee:

  不看未来   in live mode the only bars used are ones whose close_time has
             already passed.  Binance serves the in-progress candle from the
             same endpoint as settled ones, so a generator that simply takes
             the last row is reading a bar that has not finished -- the single
             most common way live code differs from its own backtest.
  可重放     given the same as-of timestamp the output is identical, so a run
             can be re-created months later and checked against what was sent.
  可对账     every row carries the reference price the decision assumed.  Fills
             get written back beside it, and the difference is the slippage
             number this whole exercise is for.

Rebalance timing follows the backtest: the signal is formed on a completed daily
bar and executed at the next bar's open, which on a 24/7 market is the same
00:00 UTC instant.  The repository already measured what a day's delay costs --
0.3 to 0.6 of Sharpe -- so the log records the intended execution time and the
actual one, and they are not assumed equal.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D, factors as F, xsec as X

PAPER_DIR = Path(__file__).resolve().parent.parent / "paper"
SIGNAL_LOG = PAPER_DIR / "signals.csv"

DEFAULT_COINS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "TAOUSDT", "HYPEUSDT")


@dataclass(frozen=True)
class Params:
    """Frozen so a paper run cannot quietly drift from the tested configuration."""
    coins: tuple[str, ...] = DEFAULT_COINS
    lookback: int = 14           # momentum window, in daily bars
    pca_window: int = 120        # trailing window for the vol normalisation
    rebalance_days: int = 3
    gross: float = 1.0           # total absolute exposure
    min_names: int = 4           # a cross-section narrower than this is not one
    # Recorded rather than applied: the generator does not charge costs, it
    # produces the orders whose real cost is the thing being measured.
    assumed_cost_bps: float = 6.5

    def to_dict(self) -> dict:
        return {"coins": ",".join(self.coins), "lookback": self.lookback,
                "pca_window": self.pca_window, "rebalance_days": self.rebalance_days,
                "gross": self.gross, "assumed_cost_bps": self.assumed_cost_bps}


@dataclass
class Book:
    asof: pd.Timestamp            # the bar whose CLOSE formed this signal
    generated_at: datetime
    execute_at: pd.Timestamp      # the open this is meant to trade at
    weights: pd.Series
    signal: pd.Series
    ref_price: pd.Series          # the close the decision was made on
    is_rebalance: bool
    params: Params = field(default_factory=Params)

    def orders(self, prev: pd.Series | None, equity: float) -> pd.DataFrame:
        """Notional to trade per coin, given the book currently held."""
        prev = (prev if prev is not None
                else pd.Series(0.0, index=self.weights.index)).reindex(
                    self.weights.index).fillna(0.0)
        delta = self.weights - prev
        return pd.DataFrame({
            "target_w": self.weights, "current_w": prev, "delta_w": delta,
            "target_notional": self.weights * equity,
            "trade_notional": delta * equity,
            "ref_price": self.ref_price,
            "trade_qty": (delta * equity / self.ref_price).where(self.ref_price > 0),
        })


def _completed(df: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    """Bars that had FINISHED at `asof` -- never the one still forming."""
    return df[df["close_time"] <= asof]


def _utc_naive(ts) -> pd.Timestamp:
    """Stored bars are UTC-naive, so everything compared against them must be."""
    t = pd.Timestamp(ts)
    return t.tz_convert("UTC").tz_localize(None) if t.tzinfo is not None else t


def latest_complete_bar(p: Params | None = None,
                        now: datetime | None = None) -> pd.Timestamp:
    """The most recent daily bar that has closed across every coin in the book."""
    p = p or Params()
    now = _utc_naive(now if now is not None else datetime.now(timezone.utc))
    ends = []
    for s in p.coins:
        done = _completed(D.load("1d", symbol=s), now)
        if len(done):
            ends.append(done.index[-1])
    if not ends:
        raise RuntimeError("no completed bars for any coin")
    return min(ends)          # the book can only act on what every name has


def generate(asof: pd.Timestamp | None = None, p: Params | None = None,
             prev_rebalance: pd.Timestamp | None = None) -> Book:
    """Target weights formed on the close of `asof`, to trade at the next open.

    `prev_rebalance` comes from the log.  Without it every run would look like a
    rebalance day, which is how a 3-day schedule silently becomes a daily one --
    and turnover is the variable this strategy's viability turns on.
    """
    p = p or Params()
    asof = pd.Timestamp(asof) if asof is not None else latest_complete_bar(p)

    closes, refs = {}, {}
    for s in p.coins:
        done = _completed(D.load("1d", symbol=s), asof)
        if len(done) < p.pca_window + p.lookback:
            continue
        closes[s] = done["close"]
        refs[s] = float(done["close"].iloc[-1])
    if len(closes) < p.min_names:
        raise RuntimeError(f"only {len(closes)} coins have enough history at {asof}")

    C = pd.DataFrame(closes).sort_index()
    R = np.log(C).diff().replace([np.inf, -np.inf], np.nan)
    pca = F.rolling_pca(R, p.pca_window, ks=(1,), min_names=p.min_names)
    feat = pca.zscore.rolling(p.lookback, min_periods=p.lookback).sum()

    row = feat.loc[[asof]] if asof in feat.index else feat.iloc[[-1]]
    w = X.cross_sectional_weights(row, mode="rank", gross=p.gross,
                                  min_names=p.min_names).iloc[0]

    due = (prev_rebalance is None
           or (asof - pd.Timestamp(prev_rebalance)).days >= p.rebalance_days)
    return Book(asof=asof, generated_at=datetime.now(timezone.utc),
                execute_at=asof + pd.Timedelta(days=1),
                weights=w.reindex(list(p.coins)).fillna(0.0),
                signal=row.iloc[0].reindex(list(p.coins)),
                ref_price=pd.Series(refs).reindex(list(p.coins)),
                is_rebalance=bool(due), params=p)


# ------------------------------------------------------------------ 回测
ANN = 365.0


def backtest(p: Params | None = None, rebalance_days: int | None = None,
             coins: tuple[str, ...] | None = None, size: str = "notional") -> dict:
    """Run this configuration over history and return its daily P&L and turnover.

    Lives here rather than in a script because the thing worth backtesting is
    the configuration that is actually running, and it is defined in this file.
    Two scripts reach for it -- 44 号 for the cost economics, 45 号 for how much
    of the result survives being poked -- and a second copy would be free to
    drift away from what `generate` does.

    `size` is a diagnostic, not a setting.  The live book is "notional": the
    signal is vol-normalised but the positions are equal-dollar ranks, so
    whichever name is most volatile dominates the realised P&L whether or not
    the ranking was any good.  "risk" divides the same ranks by trailing vol to
    separate those two effects.  45 号脚本 reports both; `generate` only ever
    produces the first, and changing that is a strategy change, not a tweak.

    No-lookahead: the feature at `t` uses trailing windows only, and the weights
    formed on the close of `t` earn `t+1`'s return, which is the same instant the
    live book is meant to trade at.
    """
    p = p or Params()
    rd = p.rebalance_days if rebalance_days is None else rebalance_days
    names = tuple(coins) if coins is not None else p.coins

    C = pd.DataFrame({s: D.load("1d", symbol=s)["close"] for s in names}).sort_index()
    R = np.log(C).diff().replace([np.inf, -np.inf], np.nan)
    pca = F.rolling_pca(R, p.pca_window, ks=(1,), min_names=p.min_names)
    feat = pca.zscore.rolling(p.lookback, min_periods=p.lookback).sum()
    target = X.cross_sectional_weights(feat, mode="rank", gross=p.gross,
                                       min_names=p.min_names)

    if size == "risk":
        vol = R.rolling(p.pca_window, min_periods=p.pca_window).std()
        target = (target / vol).replace([np.inf, -np.inf], np.nan)
        target = target.div(target.abs().sum(axis=1), axis=0) * p.gross
    elif size != "notional":
        raise ValueError(f"size must be 'notional' or 'risk', got {size!r}")

    # 只在再平衡日换仓，其余持有不动 —— rebalance_days 的全部作用就在这里，
    # 而换手是这个策略的成本账里唯一的变量。
    held = pd.DataFrame(0.0, index=target.index, columns=target.columns)
    cur = pd.Series(0.0, index=target.columns)
    last: pd.Timestamp | None = None
    for t in target.index:
        row = target.loc[t]
        if row.notna().any() and (last is None or (t - last).days >= rd):
            cur = row.fillna(0.0)
            last = t
        held.loc[t] = cur

    fwd = C.pct_change().shift(-1)
    pnl = (held * fwd).sum(axis=1).dropna()
    turn = held.diff().abs().sum(axis=1).fillna(0.0).reindex(pnl.index).fillna(0.0)
    live = held.abs().sum(axis=1).reindex(pnl.index).fillna(0.0) > 0
    return {"pnl": pnl, "turnover": turn, "live": live, "coins": names,
            "held": held.reindex(pnl.index), "fwd": fwd.reindex(pnl.index)}


def stats(pnl: pd.Series, turn: pd.Series | None = None,
          cost_bps: float = 6.5) -> dict:
    """Annualised summary of a P&L series, with the cost arithmetic attached."""
    pnl = pnl.dropna()
    if not len(pnl):
        return {"days": 0, "sharpe": np.nan, "gross_ann": np.nan,
                "turn_ann": np.nan, "be_bps": np.nan, "net_ann": np.nan}
    years = len(pnl) / ANN
    gross_ann = float(pnl.mean() * ANN)
    sd = float(pnl.std())
    sharpe = float(pnl.mean() / sd * np.sqrt(ANN)) if sd > 0 else np.nan
    turn_ann = float(turn.reindex(pnl.index).fillna(0.0).sum() / years) \
        if turn is not None and years > 0 else np.nan
    be = gross_ann / turn_ann if turn_ann and turn_ann > 0 else np.nan
    return {"days": len(pnl), "years": years, "sharpe": sharpe,
            "gross_ann": gross_ann, "turn_ann": turn_ann, "be_bps": be * 1e4,
            "net_ann": gross_ann - (turn_ann * cost_bps * 1e-4 if turn_ann else 0.0)}


# ------------------------------------------------------------------ 日志
COLUMNS = ["asof", "generated_at", "execute_at", "symbol", "signal", "target_w",
           "prev_w", "trade_w", "ref_price", "target_notional", "trade_notional",
           "trade_qty", "is_rebalance", "equity", "params",
           "fill_price", "fill_qty", "filled_at", "fee_paid"]


def load_log(path: Path | None = None) -> pd.DataFrame:
    path = path or SIGNAL_LOG
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(path)
    for c in ("asof", "execute_at", "filled_at"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def last_state(log: pd.DataFrame, coins) -> tuple[pd.Series | None, pd.Timestamp | None]:
    """Weights currently held and the date they were last changed."""
    if not len(log):
        return None, None
    reb = log[log["is_rebalance"].astype(str).str.lower().isin(("true", "1"))]
    if not len(reb):
        return None, None
    last = reb[reb["asof"] == reb["asof"].max()]
    w = last.set_index("symbol")["target_w"].reindex(list(coins)).fillna(0.0)
    return w, pd.Timestamp(last["asof"].iloc[0])


def append(book: Book, orders: pd.DataFrame, equity: float,
           path: Path | None = None) -> Path:
    path = path or SIGNAL_LOG
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for sym in book.weights.index:
        rows.append({
            "asof": book.asof, "generated_at": book.generated_at.isoformat(),
            "execute_at": book.execute_at, "symbol": sym,
            "signal": book.signal.get(sym, np.nan),
            "target_w": book.weights[sym],
            "prev_w": orders.loc[sym, "current_w"],
            "trade_w": orders.loc[sym, "delta_w"],
            "ref_price": book.ref_price.get(sym, np.nan),
            "target_notional": orders.loc[sym, "target_notional"],
            "trade_notional": orders.loc[sym, "trade_notional"],
            "trade_qty": orders.loc[sym, "trade_qty"],
            "is_rebalance": book.is_rebalance, "equity": equity,
            "params": json.dumps(book.params.to_dict(), ensure_ascii=False),
            "fill_price": "", "fill_qty": "", "filled_at": "", "fee_paid": "",
        })
    out = pd.DataFrame(rows)[COLUMNS]
    header = not path.exists()
    out.to_csv(path, mode="a", index=False, header=header, encoding="utf-8-sig")
    return path


def reconcile(log: pd.DataFrame | None = None) -> pd.DataFrame:
    """Realised slippage per filled order -- the number this exercise is for.

    Slippage is signed against the direction traded: paying above the reference
    when buying and receiving below it when selling both count as positive cost,
    so the column is comparable across sides and directly against the 6.5bp the
    backtests assume.
    """
    log = load_log() if log is None else log
    f = log[pd.to_numeric(log.get("fill_price"), errors="coerce").notna()].copy()
    if not len(f):
        return pd.DataFrame()
    f["fill_price"] = pd.to_numeric(f["fill_price"], errors="coerce")
    side = np.sign(pd.to_numeric(f["trade_qty"], errors="coerce"))
    f["slippage_bps"] = side * (f["fill_price"] / f["ref_price"] - 1) * 1e4
    f["delay_h"] = (pd.to_datetime(f["filled_at"]) - f["execute_at"]) \
        .dt.total_seconds() / 3600
    return f[["asof", "symbol", "trade_qty", "ref_price", "fill_price",
              "slippage_bps", "delay_h", "fee_paid"]]
