"""Rolling PCA on a return panel: loadings, factor scores, residuals.

Everything here is fitted on a TRAILING window that excludes the bar being
decomposed.  Fitting components on the whole sample and then "trading the
residual" is one of the cleanest ways to produce a beautiful backtest of
nothing, because the loadings already know which names were about to diverge.

Two details that decide whether the output means anything:

  standardisation  each coin's return is divided by its own trailing vol before
                   the decomposition, so a memecoin and BTC contribute to the
                   covariance on the same scale.  The residual therefore comes
                   out in units of that coin's own volatility, not in percent.
  sign             an eigenvector's sign is arbitrary.  For crypto PC1 is the
                   market -- every loading the same sign -- so the sign is
                   pinned to make the average loading positive.  Without that
                   the portfolio's "PC1 exposure" flips sign at random dates and
                   any exposure statistic computed from it is noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PcaPanels:
    resid: dict[int, pd.DataFrame]   # k -> residual panel, in trailing-vol units
    zscore: pd.DataFrame             # standardised returns, the un-hedged control
    load1: pd.DataFrame              # each coin's PC1 loading, sign-pinned positive
    var1: pd.Series                  # share of variance PC1 takes, per date
    n_names: pd.Series               # cross-section width per date


def rolling_pca(returns: pd.DataFrame, lookback: int = 120,
                ks: tuple[int, ...] = (1, 3, 5),
                min_names: int = 30) -> PcaPanels:
    """Decompose each row against components fitted on the `lookback` rows before it."""
    idx, cols_all = returns.index, returns.columns
    resid = {k: pd.DataFrame(np.nan, index=idx, columns=cols_all) for k in ks}
    z_out = pd.DataFrame(np.nan, index=idx, columns=cols_all)
    load1 = pd.DataFrame(np.nan, index=idx, columns=cols_all)
    var1 = pd.Series(np.nan, index=idx)
    nn = pd.Series(0, index=idx)

    R = returns.to_numpy(dtype=float)
    kmax = max(ks)
    for i in range(lookback, len(idx)):
        hist = R[i - lookback:i]
        today = R[i]
        ok = np.isfinite(hist).all(axis=0) & np.isfinite(today)
        if ok.sum() < min_names:
            continue
        H = hist[:, ok]
        mu = H.mean(axis=0)
        sd = H.std(axis=0)
        good = sd > 0
        if good.sum() < min_names:
            continue
        sel = np.where(ok)[0][good]
        H = hist[:, sel]
        mu, sd = H.mean(axis=0), H.std(axis=0)
        Hz = (H - mu) / sd
        cov = np.cov(Hz, rowvar=False)
        if not np.isfinite(cov).all():
            continue
        ev, evec = np.linalg.eigh(cov)          # ascending
        if evec.shape[1] < kmax:
            continue
        z = (today[sel] - mu) / sd
        # pin PC1's sign so a positive loading always means "moves with the market"
        if evec[:, -1].mean() < 0:
            evec = evec.copy()
            evec[:, -1] *= -1

        names = cols_all[sel]
        z_out.loc[idx[i], names] = z
        load1.loc[idx[i], names] = evec[:, -1]
        var1.iloc[i] = ev[-1] / ev.sum()
        nn.iloc[i] = len(sel)
        for k in ks:
            L = evec[:, -k:]
            resid[k].loc[idx[i], names] = z - L @ (L.T @ z)
    return PcaPanels(resid=resid, zscore=z_out, load1=load1, var1=var1, n_names=nn)


def portfolio_factor_exposure(weights: pd.DataFrame,
                              loadings: pd.DataFrame) -> pd.Series:
    """Sum of weight x loading -- what the book is betting on the factor.

    A dollar-neutral book is not factor-neutral.  If the long leg happens to
    hold higher-loading names than the short leg, the position is a bet on the
    factor while every dollar nets to zero, which is exactly the exposure a
    net-weight check cannot see.
    """
    w = weights.reindex(index=loadings.index, columns=loadings.columns).fillna(0.0)
    l = loadings.reindex_like(w)
    return (w * l).sum(axis=1).where(l.notna().any(axis=1))
