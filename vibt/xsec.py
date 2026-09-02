"""Cross-sectional (relative-value) portfolio construction.

Every signal tested so far has been a time-series signal on one asset, which
means it was fighting the thing that dominates this universe: the first
principal component of six crypto assets explains 73% of their variance.  A
dollar-neutral cross-sectional portfolio cancels that component by construction
-- it never expresses a view on whether crypto goes up, only on which names do
better than which others.

Timing matches the rest of the framework: the signal is computed from data
through the close of day t, weights are applied at the open of t+1, and the
return earned is open-to-open.

Availability is handled honestly: a symbol only enters the cross-section on days
it actually has a price and a valid feature, so HYPE (listed 2025-05-30) and TAO
(2024-04-11) simply do not exist in the ranking before they listed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import backtest as B, metrics as M


def price_panel(universe: dict[str, dict[str, pd.DataFrame]], tf: str = "1d",
                field: str = "open") -> pd.DataFrame:
    """Symbols as columns on a shared index."""
    cols = {sym: frames[tf][field] for sym, frames in universe.items()}
    return pd.DataFrame(cols).sort_index()


def feature_panel(universe: dict[str, dict[str, pd.DataFrame]], fn, tf: str = "1d") -> pd.DataFrame:
    """Apply `fn(frame) -> Series` per symbol and align into a panel."""
    return pd.DataFrame({sym: fn(frames[tf]) for sym, frames in universe.items()}).sort_index()


def cross_sectional_weights(
    feature: pd.DataFrame,
    n_side: int = 2,
    mode: str = "long_short",
    min_names: int = 3,
    gross: float = 1.0,
) -> pd.DataFrame:
    """Rank each row and build weights.

    `mode`:
      long_short -- long the top n_side, short the bottom n_side, dollar-neutral
      long_only  -- long the top n_side only
      rank       -- continuous demeaned rank, dollar-neutral (uses every name)

    `gross` is total absolute exposure, so long_short at gross=1.0 puts 0.5 long
    and 0.5 short.
    """
    w = pd.DataFrame(0.0, index=feature.index, columns=feature.columns)
    for ts, row in feature.iterrows():
        valid = row.dropna()
        if len(valid) < min_names:
            continue
        if mode == "rank":
            r = valid.rank()
            z = r - r.mean()
            denom = z.abs().sum()
            if denom > 0:
                w.loc[ts, z.index] = gross * z / denom
            continue
        k = min(n_side, len(valid) // 2 if mode == "long_short" else len(valid))
        if k < 1:
            continue
        order = valid.sort_values(ascending=False)
        longs = order.index[:k]
        if mode == "long_only":
            w.loc[ts, longs] = gross / k
        else:
            shorts = order.index[-k:]
            w.loc[ts, longs] = (gross / 2) / k
            w.loc[ts, shorts] = -(gross / 2) / k
    return w


def risk_parity(weights: pd.DataFrame, vol: pd.DataFrame,
                gross: float = 1.0) -> pd.DataFrame:
    """Re-scale each leg by 1/vol so every name carries similar risk.

    Each leg is renormalised to gross/2 in absolute weight WITH ITS SIGN INTACT
    -- the long leg stays long and the short leg stays short.  Getting that wrong
    silently turns a dollar-neutral book into a long-only one (it shows up as a
    beta near +0.9 instead of ~0, which is the check worth running afterwards).

    A name whose vol is unknown cannot be sized, so it is dropped.  When that
    empties one leg entirely the book stands flat for the day: renormalising the
    survivors would leave gross/2 on one side only, which is a directional bet
    wearing a market-neutral label.  This is not hypothetical -- the feature
    needs 14 bars but the vol estimate needs 30, so any name listed in between
    is selectable and unsizeable at the same time, and a freshly listed pair
    (1000FLOKI + 1000PEPE, May 2023) put the whole short leg in that state.
    """
    iv = (1.0 / vol.reindex(weights.index).reindex(columns=weights.columns))
    iv = iv.replace([np.inf, -np.inf], np.nan)
    x = weights * iv
    x = x.where(np.isfinite(x), 0.0)
    longs, shorts = x.clip(lower=0), (-x).clip(lower=0)
    ls, ss = longs.sum(axis=1), shorts.sum(axis=1)
    out = (longs.div(ls.replace(0, np.nan), axis=0) * gross / 2).fillna(0.0) \
        - (shorts.div(ss.replace(0, np.nan), axis=0) * gross / 2).fillna(0.0)
    return out.where((ls > 0) & (ss > 0), 0.0)


def run(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    costs: B.Costs | None = None,
    ann_factor: float = 365.0,
    name: str = "",
    rebalance: int = 1,
) -> B.Result:
    """Backtest a weight panel.  Weights at row t are executed at the open of t+1."""
    costs = costs or B.Costs()
    px = prices.reindex(columns=weights.columns).sort_index()
    w = weights.reindex(px.index).fillna(0.0)

    if rebalance > 1:
        keep = np.zeros(len(w), dtype=bool)
        keep[::rebalance] = True
        w = w.where(pd.Series(keep, index=w.index), np.nan).ffill().fillna(0.0)

    held = w.shift(1).fillna(0.0)                 # applied at this bar's open
    bar_ret = px.shift(-1) / px - 1.0             # open-to-open
    bar_ret = bar_ret.where(px.notna() & px.shift(-1).notna())

    # A weight on a symbol with no price earns nothing and is not a real position.
    held = held.where(bar_ret.notna(), 0.0)
    gross_ret = (held * bar_ret.fillna(0.0)).sum(axis=1)

    turn = (held - held.shift(1).fillna(0.0)).abs().sum(axis=1)
    cost = turn * costs.per_side
    net = (gross_ret - cost).iloc[:-1]

    equity = (1 + net.fillna(0)).cumprod()
    exposure = held.abs().sum(axis=1).iloc[:-1]
    st = M.compute(net, ann_factor, position=exposure)
    st.extra["gross_exposure"] = float(exposure.mean())
    st.extra["net_exposure"] = float(held.sum(axis=1).iloc[:-1].mean())
    # metrics.compute derives turnover from the position series, which for a
    # panel is GROSS EXPOSURE -- pinned at 1.0 for a dollar-neutral book, so it
    # reports ~0 however violently the names underneath are being rotated.  The
    # cost series was always right; only this number was wrong.  Overwrite it
    # with the name-level turnover the costs were actually charged on.
    years = len(net) / ann_factor if len(net) else np.nan
    st.turnover_ann = float(turn.iloc[:-1].sum() / years) if years else np.nan
    return B.Result(net, gross_ret.iloc[:-1], exposure, equity, cost.iloc[:-1],
                    pd.DataFrame(), st, name, weights=held.iloc[:-1])


def beta_to(rets: pd.Series, bench: pd.Series) -> tuple[float, float]:
    """(beta, annualised alpha) of a return stream against a benchmark."""
    df = pd.concat([rets.rename("r"), bench.rename("b")], axis=1).dropna()
    if len(df) < 30 or df["b"].var() == 0:
        return (np.nan, np.nan)
    beta = float(np.cov(df["r"], df["b"])[0, 1] / df["b"].var())
    alpha = float((df["r"].mean() - beta * df["b"].mean()) * 365)
    return beta, alpha
