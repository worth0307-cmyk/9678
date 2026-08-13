"""Charts for the report.  Four figures, each answering one question."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vibt import backtest as B, data as D, metrics as M, system as SYS  # noqa: E402

REPORTS = Path(__file__).resolve().parent.parent / "reports"
COSTS = B.Costs(fee_bps=4.5, slip_bps=2.0)
SPLIT = pd.Timestamp("2025-07-01")
ANN = 365.0

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8983"
GRID = "#e4e3df"
S1 = "#2a78d6"   # system      (categorical slot 1)
S2 = "#eb6834"   # buy & hold  (slot 2)
S3 = "#1baf7a"   # slot 3

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 11,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "legend.frameon": False,
    "figure.dpi": 150,
})


def style(ax) -> None:
    ax.grid(axis="x", visible=False)
    ax.tick_params(length=0)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


def mark_split(ax, frac: float = 0.97, label: bool = True) -> None:
    """Split marker; the label rides at `frac` of the axis height, clear of the data."""
    ax.axvline(SPLIT, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
    if label:
        # blended coords: x in data units (so the date is unit-converted), y in
        # axes fraction -- a raw xaxis_transform would not convert the Timestamp
        ax.annotate("out-of-sample →",
                    xy=(SPLIT, frac), xycoords=("data", "axes fraction"),
                    xytext=(6, 0), textcoords="offset points",
                    color=MUTED, fontsize=8, va="top")


def fig_equity(df, final, bench) -> None:
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(10, 6.6), sharex=True, gridspec_kw={"height_ratios": [2.4, 1]})

    e_s = final.equity
    e_b = bench.equity.reindex(e_s.index).ffill()
    ax.plot(e_s.index, e_s, color=S1, lw=2, zorder=3, solid_capstyle="round")
    ax.plot(e_b.index, e_b, color=S2, lw=2, zorder=2, solid_capstyle="round")
    ax.set_yscale("log")
    ax.set_yticks([0.8, 1.0, 1.4, 2.0, 2.8])
    ax.set_ylim(0.78, 3.15)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2g}x"))
    ax.set_ylabel("growth of 1 USDT (log)")
    ax.set_title("Regime allocator vs buy & hold — BTCUSDT, daily, 6.5bp per side",
                 loc="left", color=INK, fontweight="bold", pad=14)
    for series, color, name in ((e_s, S1, "system"), (e_b, S2, "buy & hold")):
        ax.annotate(f"{name}  {series.iloc[-1]:.2f}x",
                    xy=(series.index[-1], series.iloc[-1]), xytext=(8, 0),
                    textcoords="offset points", color=color, fontsize=9,
                    fontweight="bold", va="center")
    mark_split(ax)
    style(ax)

    pos = final.position
    ax2.fill_between(pos.index, 0, pos, color=S1, alpha=0.28, lw=0, step="post")
    ax2.plot(pos.index, pos, color=S1, lw=1, drawstyle="steps-post")
    ax2.axhline(0, color=MUTED, lw=0.8)
    ax2.set_ylabel("target exposure")
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax2.set_ylim(-0.35, 1.05)
    mark_split(ax2, label=False)
    style(ax2)
    ax2.annotate("shorts only in a confirmed downtrend, capped at 25%",
                 xy=(0.012, 0.06), xycoords="axes fraction", color=MUTED, fontsize=8)

    fig.subplots_adjust(right=0.86)
    fig.savefig(REPORTS / "fig1_equity.png", bbox_inches="tight")
    plt.close(fig)


def fig_drawdown(final, bench) -> None:
    fig, ax = plt.subplots(figsize=(10, 3.6))
    for res, color, name in ((bench, S2, "buy & hold"), (final, S1, "system")):
        e = res.equity
        dd = e / e.cummax() - 1
        ax.fill_between(dd.index, 0, dd, color=color, alpha=0.22, lw=0)
        ax.plot(dd.index, dd, color=color, lw=1.6, label=name, solid_capstyle="round")
        trough = dd.idxmin()
        ax.annotate(f"{dd.min():.0%}", xy=(trough, dd.min()), xytext=(0, -12),
                    textcoords="offset points", color=color, fontsize=9,
                    fontweight="bold", ha="center")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_ylabel("drawdown")
    ax.set_title("The whole point: the drawdown, not the return",
                 loc="left", color=INK, fontweight="bold", pad=12)
    ax.legend(loc="lower left", ncol=2)
    mark_split(ax, frac=0.30)
    style(ax)
    fig.savefig(REPORTS / "fig2_drawdown.png", bbox_inches="tight")
    plt.close(fig)


def fig_families(bench_sharpe: float) -> None:
    g = pd.read_csv(REPORTS / "families.csv")
    order = g.groupby("family")["sharpe"].median().sort_values().index.tolist()
    fig, ax = plt.subplots(figsize=(10, 4.2))
    rng = np.random.default_rng(0)
    for i, fam in enumerate(order):
        v = g.loc[g["family"] == fam, "sharpe"].to_numpy()
        jitter = rng.uniform(-0.13, 0.13, len(v))
        ax.scatter(v, np.full(len(v), i) + jitter, s=26, color=S1, alpha=0.42,
                   edgecolor=SURFACE, linewidth=0.6, zorder=3)
        med = np.median(v)
        ax.plot([med, med], [i - 0.3, i + 0.3], color=INK, lw=2.4, zorder=4,
                solid_capstyle="round")
        ax.annotate(f"{med:+.2f}", xy=(med, i + 0.36), color=INK, fontsize=8.5,
                    ha="center", fontweight="bold")
    ax.axvline(bench_sharpe, color=S2, lw=1.6, zorder=2)
    ax.annotate(f"buy & hold {bench_sharpe:+.2f}", xy=(bench_sharpe, len(order) - 0.35),
                xytext=(6, 0), textcoords="offset points", color=S2, fontsize=8.5,
                fontweight="bold")
    ax.axvline(0, color=MUTED, lw=0.8, zorder=1)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.set_xlabel("Sharpe ratio — every parameter in the family, costs on")
    ax.set_title("Compare families, not cherry-picked parameters\n"
                 "each dot is one lookback; the bar is the family median",
                 loc="left", color=INK, fontweight="bold", pad=12)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.tick_params(length=0)
    ax.set_ylim(-0.6, len(order) - 0.1)
    fig.savefig(REPORTS / "fig3_families.png", bbox_inches="tight")
    plt.close(fig)


def fig_is_oos() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8), sharey=True)
    for ax, tf in zip(axes, ("1d", "4h", "1h")):
        g = pd.read_csv(REPORTS / f"sweep_vi_{tf}.csv").dropna(subset=["sharpe_is", "sharpe_oos"])
        for mode, color, lab in (("long_only", S1, "long only"), ("ls", S3, "long/short")):
            sub = g[g["mode"] == mode]
            ax.scatter(sub["sharpe_is"], sub["sharpe_oos"], s=22, color=color, alpha=0.5,
                       edgecolor=SURFACE, linewidth=0.5, label=lab, zorder=3)
        lim = (-2.6, 2.2)
        ax.plot(lim, lim, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
        ax.axhline(0, color=MUTED, lw=0.8, zorder=1)
        ax.axvline(0, color=MUTED, lw=0.8, zorder=1)
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        ax.set_title(tf, loc="left", color=INK, fontweight="bold")
        ax.set_xlabel("in-sample Sharpe")
        ax.tick_params(length=0)
        ax.grid(True)
    axes[0].set_ylabel("out-of-sample Sharpe")
    axes[0].legend(loc="lower left", fontsize=8)
    axes[1].annotate("points below the diagonal\ndid worse than in-sample",
                     xy=(0.04, 0.06), xycoords="axes fraction", color=MUTED, fontsize=8)
    fig.suptitle("Every Vortex configuration: what it promised vs what it delivered",
                 x=0.007, ha="left", color=INK, fontweight="bold", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(REPORTS / "fig4_is_oos.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = D.load("1d")
    bench = B.buy_and_hold(df, ANN, COSTS)
    final = B.run(df, SYS.target_exposure(df), COSTS, ANN, "system", 1.0)

    fig_equity(df, final, bench)
    fig_drawdown(final, bench)
    fig_families(bench.stats.sharpe)
    fig_is_oos()
    print("wrote:", *sorted(p.name for p in REPORTS.glob("fig*.png")), sep="\n  ")


if __name__ == "__main__":
    main()
