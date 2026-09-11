"""Performance statistics, plus the honesty tools: t-stats, bootstrap, deflated Sharpe."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Stats:
    n: int
    ann_factor: float
    total_return: float
    cagr: float
    ann_vol: float
    sharpe: float
    sortino: float
    max_dd: float
    calmar: float
    hit_rate: float
    exposure: float
    turnover_ann: float
    trades: int
    profit_factor: float
    avg_win: float
    avg_loss: float
    t_stat: float
    extra: dict = field(default_factory=dict)

    def as_row(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "extra"}
        d.update(self.extra)
        return d

    def __str__(self) -> str:
        return (
            f"ret {self.total_return:+8.1%} | CAGR {self.cagr:+7.1%} | vol {self.ann_vol:6.1%} | "
            f"Sharpe {self.sharpe:+5.2f} | Sortino {self.sortino:+5.2f} | maxDD {self.max_dd:7.1%} | "
            f"Calmar {self.calmar:+5.2f} | expo {self.exposure:5.1%} | trades {self.trades:4d} | "
            f"PF {self.profit_factor:4.2f} | t {self.t_stat:+5.2f}"
        )


def max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    return float((equity / peak - 1.0).min())


def compute(
    rets: pd.Series,
    ann_factor: float,
    position: pd.Series | None = None,
    trade_pnl: np.ndarray | None = None,
) -> Stats:
    """`rets` are per-bar arithmetic net returns of the strategy."""
    r = rets.dropna().to_numpy()
    n = len(r)
    if n == 0:
        raise ValueError("empty return series")
    equity = np.cumprod(1.0 + r)
    years = n / ann_factor
    total = float(equity[-1] - 1.0)
    cagr = float(equity[-1] ** (1 / years) - 1.0) if equity[-1] > 0 else -1.0
    vol = float(r.std(ddof=1) * np.sqrt(ann_factor))
    mean = float(r.mean())
    sharpe = float(mean / r.std(ddof=1) * np.sqrt(ann_factor)) if r.std(ddof=1) > 0 else 0.0
    downside = r[r < 0]
    dstd = downside.std(ddof=1) if len(downside) > 1 else np.nan
    sortino = float(mean / dstd * np.sqrt(ann_factor)) if dstd and dstd > 0 else np.nan
    mdd = max_drawdown(equity)
    calmar = float(cagr / abs(mdd)) if mdd < 0 else np.nan
    t_stat = float(mean / (r.std(ddof=1) / np.sqrt(n))) if r.std(ddof=1) > 0 else 0.0

    if position is not None:
        pos = position.reindex(rets.index).fillna(0.0)
        exposure = float((pos.abs() > 1e-12).mean())
        turnover_ann = float(pos.diff().abs().sum() / years)
        trades = int((pos.diff().abs() > 1e-12).sum())
    else:
        exposure, turnover_ann, trades = np.nan, np.nan, 0

    if trade_pnl is not None and len(trade_pnl):
        wins = trade_pnl[trade_pnl > 0]
        losses = trade_pnl[trade_pnl < 0]
        hit = float(len(wins) / len(trade_pnl))
        pf = float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() != 0 else np.inf
        avg_w = float(wins.mean()) if len(wins) else 0.0
        avg_l = float(losses.mean()) if len(losses) else 0.0
        trades = len(trade_pnl)
    else:
        active = r[np.abs(r) > 1e-12]
        hit = float((active > 0).mean()) if len(active) else np.nan
        wins, losses = r[r > 0], r[r < 0]
        pf = float(wins.sum() / abs(losses.sum())) if losses.sum() != 0 else np.inf
        avg_w = float(wins.mean()) if len(wins) else 0.0
        avg_l = float(losses.mean()) if len(losses) else 0.0

    return Stats(
        n=n, ann_factor=ann_factor, total_return=total, cagr=cagr, ann_vol=vol,
        sharpe=sharpe, sortino=sortino, max_dd=mdd, calmar=calmar, hit_rate=hit,
        exposure=exposure, turnover_ann=turnover_ann, trades=trades,
        profit_factor=pf, avg_win=avg_w, avg_loss=avg_l, t_stat=t_stat,
    )


def deflated_sharpe(sharpe: float, n_trials: int, n_obs: int, ann_factor: float,
                    skew: float = 0.0, kurt: float = 3.0) -> float:
    """Probability the observed Sharpe beats the best expected from `n_trials` of pure noise.

    Bailey & Lopez de Prado (2014).  Sharpe in/out are annualised; converted
    internally to per-observation.
    """
    # scipy only for two normal quantiles, and only here.  Importing it at module
    # scope pulled ~40MB into the daily signal path, which needs a rank and a
    # weighted sum -- the VPS run died on ModuleNotFoundError for a dependency
    # nothing in that path uses.
    from scipy import stats as sps

    sharpe = float(sharpe)
    sr = sharpe / np.sqrt(ann_factor)
    if n_trials < 2 or n_obs < 3:
        return np.nan
    euler = 0.5772156649
    e_max = (1 - euler) * sps.norm.ppf(1 - 1 / n_trials) + euler * sps.norm.ppf(
        1 - 1 / (n_trials * np.e)
    )
    sr0 = e_max / np.sqrt(n_obs - 1) * 1.0  # expected max SR of n_trials null strategies
    denom = np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr**2)
    if denom <= 0:
        return np.nan
    z = (sr - sr0) * np.sqrt(n_obs - 1) / denom
    return float(sps.norm.cdf(z))


def block_bootstrap_sharpe(
    rets: pd.Series, ann_factor: float, block: int = 20, n_boot: int = 2000, seed: int = 7
) -> tuple[float, float, float]:
    """(p5, p50, p95) of the annualised Sharpe under a stationary block bootstrap."""
    r = rets.dropna().to_numpy()
    n = len(r)
    if n < block * 3:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n]
    samples = r[idx]
    mu = samples.mean(axis=1)
    sd = samples.std(axis=1, ddof=1)
    sr = np.where(sd > 0, mu / sd * np.sqrt(ann_factor), 0.0)
    return tuple(float(x) for x in np.percentile(sr, [5, 50, 95]))


def monthly_table(rets: pd.Series) -> pd.DataFrame:
    eq = (1 + rets.fillna(0)).cumprod()
    m = eq.resample("ME").last().pct_change()
    m.iloc[0] = eq.resample("ME").last().iloc[0] - 1
    df = pd.DataFrame({"ret": m})
    df["year"] = df.index.year
    df["month"] = df.index.month
    return df.pivot_table(index="year", columns="month", values="ret")
