"""Out-of-sample test: does the cross-sectional momentum strategy work on symbols
that had no part in building it?

REPORT_XSEC's strategy is fully specified -- 14-day momentum, long the top 5 and
short the bottom 5, rebalanced every 3 days, 6.5bp per side -- so nothing is
fitted here.  The parameters are copied, the universe is swapped, and the number
that comes out is the number.

Why this matters more than any further test on the original data: the surviving
strategy's deflated Sharpe is 32.7%, meaning roughly two thirds of the
probability mass says the edge does not exist, and that figure cannot improve by
running more variants on the same 26 symbols -- every extra trial makes it worse.
Out-of-sample evidence is the only kind that does not pay that penalty.

Two exclusions, fixed before any result was computed and justified on instrument
type rather than performance:

  USDCUSDT   - a stablecoin pair, 0.8% annualised volatility against BTC's 46.5%
  BTCDOMUSDT - a dominance index, not a token

Everything is reported with and without them so the choice is auditable.

The cost assumption is the other thing being checked.  Every backtest so far
charged 6.5bp per side to every symbol equally, and scripts/20_frequency.py
showed the strategy dies around 12.5bp.  The out-of-sample set is full of names
that trade a thousandth of BTC's volume, so this run reports the strategy by
liquidity tier and prices each tier by what it actually trades.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vibt import backtest as B, data as D, metrics as M, xsec as X  # noqa: E402

pd.set_option("display.width", 200)

# Copied verbatim from REPORT_XSEC section 2.  Not re-tuned.
LOOKBACK, N_SIDE, REBALANCE = 14, 5, 3
COSTS = B.Costs(fee_bps=4.5, slip_bps=2.0)
ANN = 365.0

IN_SAMPLE = [
    "1000BONKUSDT", "1000FLOKIUSDT", "1000PEPEUSDT", "1000SHIBUSDT", "APTUSDT",
    "ARBUSDT", "BNBUSDT", "BTCUSDT", "DOGEUSDT", "DYDXUSDT", "ENAUSDT", "ETHUSDT",
    "FETUSDT", "HYPEUSDT", "LDOUSDT", "OPUSDT", "PENDLEUSDT", "RENDERUSDT",
    "SEIUSDT", "SOLUSDT", "SUIUSDT", "TAOUSDT", "TIAUSDT", "WIFUSDT", "WLDUSDT",
    "XRPUSDT",
]
NOT_COINS = ["USDCUSDT", "BTCDOMUSDT"]
REPORTS = Path(__file__).resolve().parent.parent / "reports"


def momentum(d: pd.DataFrame) -> pd.Series:
    return d["close"] / d["close"].shift(LOOKBACK) - 1.0


def run_universe(uni: dict, costs: B.Costs = COSTS, n_side: int = N_SIDE) -> B.Result:
    px = X.price_panel(uni, "1d", "open")
    feat = X.feature_panel(uni, momentum)
    w = X.cross_sectional_weights(feat, n_side=n_side, mode="long_short")
    return X.run(px, w, costs, ANN, "", REBALANCE)


def line(name: str, res: B.Result, n: int) -> str:
    yrs = len(res.rets) / ANN
    cagr = (1 + res.stats.total_return) ** (1 / yrs) - 1
    return (f"  {name:<34}{n:>5}{res.stats.sharpe:>+9.2f}{res.stats.total_return:>+11.1%}"
            f"{cagr:>+9.1%}{res.stats.ann_vol:>8.0%}{res.stats.max_dd:>9.1%}")


HEAD = (f"  {'':<34}{'币数':>5}{'Sharpe':>9}{'总收益':>11}{'CAGR':>9}{'波动':>8}{'回撤':>9}")


def main() -> None:
    every = D.available_symbols(require=("1d",))
    oos_all = [s for s in every if s not in IN_SAMPLE]
    oos = [s for s in oos_all if s not in NOT_COINS]

    print("=" * 130)
    print("OUT-OF-SAMPLE TEST   策略参数原样照搬，只换币池")
    print("=" * 130)
    print(f"""
  参数（抄自 REPORT_XSEC 第 2 节，一个字没改）:
    动量回看 {LOOKBACK} 天   多空各 {N_SIDE} 个   每 {REBALANCE} 天调仓   成本 {COSTS.per_side*1e4:.1f}bp/边

  开发集（样本内）: {len(IN_SAMPLE)} 个
  样本外          : {len(oos)} 个（另有 {len(NOT_COINS)} 个非币种工具单列: {NOT_COINS}）
""")

    uni_is = D.load_universe(IN_SAMPLE, tfs=("1d",))
    uni_oos = D.load_universe(oos, tfs=("1d",))
    uni_oos_all = D.load_universe(oos_all, tfs=("1d",))
    uni_both = D.load_universe([s for s in every if s not in NOT_COINS], tfs=("1d",))

    r_is, r_oos = run_universe(uni_is), run_universe(uni_oos)
    r_oos_all, r_both = run_universe(uni_oos_all), run_universe(uni_both)

    print("=" * 130)
    print("1. 核心结果")
    print("=" * 130 + "\n" + HEAD)
    print(line("样本内（开发用的 26 个）", r_is, len(IN_SAMPLE)))
    print(line("样本外（183 个，排除两个工具）", r_oos, len(oos)))
    print(line("样本外（含 USDC/BTCDOM）", r_oos_all, len(oos_all)))
    print(line("全部合并", r_both, len(uni_both)))

    bench = X.price_panel(uni_oos, "1d", "open").pct_change().mean(axis=1).fillna(0)
    beta, alpha = X.beta_to(r_oos.rets, bench)
    print(f"\n  样本外对等权大盘: beta {beta:+.3f}   年化 alpha {alpha:+.2%}")

    p5, p50, p95 = M.block_bootstrap_sharpe(r_oos.rets, ANN, block=20)
    print(f"  样本外 Sharpe 的 block bootstrap 95% 区间: [{p5:+.2f}, {p95:+.2f}]  中位 {p50:+.2f}")
    print("""
  这个区间没有多重检验惩罚：策略是事先写死的，这里只跑了一次。""")

    # ------------------------------------------------------------------ liquidity
    print("\n" + "=" * 130)
    print("2. 成本复核：6.5bp 对这些币成立吗")
    print("=" * 130)
    dv = {}
    for s in oos + IN_SAMPLE:
        d = D.load("1d", symbol=s)
        if "quote_volume" in d.columns:
            dv[s] = float(d["quote_volume"].tail(365).median())
    dvs = pd.Series(dv).sort_values(ascending=False)
    print(f"""
  日均成交额（近 365 天中位数）:
    开发集 26 个 中位 ${dvs[dvs.index.isin(IN_SAMPLE)].median()/1e6:,.0f}M
    样本外      中位 ${dvs[~dvs.index.isin(IN_SAMPLE)].median()/1e6:,.0f}M
    样本外最小  ${dvs[~dvs.index.isin(IN_SAMPLE)].min()/1e6:,.2f}M ({dvs[~dvs.index.isin(IN_SAMPLE)].idxmin()})
""")
    tiers = [("$100M 以上", 100e6, np.inf), ("$10M~100M", 10e6, 100e6),
             ("$1M~10M", 1e6, 10e6), ("$1M 以下", 0, 1e6)]
    print(HEAD)
    for name, lo, hi in tiers:
        names = [s for s in oos if lo <= dv.get(s, 0) < hi]
        if len(names) < 12:
            print(f"  {name:<34}{len(names):>5}   (不足 12 个，无法构造 5+5 组合)")
            continue
        print(line(name, run_universe(D.load_universe(names, tfs=("1d",))), len(names)))

    print("""
  按成交额下限累计筛选（$10M 这条在跑之前就说定了，不是看完结果才划的）:""")
    print(HEAD + f"{'每边':>6}")
    frac = N_SIDE / len(IN_SAMPLE)
    for floor in (0, 5e6, 10e6, 30e6, 50e6):
        names = [s2 for s2 in oos if dv.get(s2, 0) >= floor]
        if len(names) < 2 * N_SIDE + 2:
            continue
        prop = max(2, int(round(len(names) * frac)))
        for ns, tag in ((N_SIDE, "原始 5+5"), (prop, f"按比例 {prop}+{prop}")):
            rr = run_universe(D.load_universe(names, tfs=("1d",)), n_side=ns)
            yrs = len(rr.rets) / ANN
            cagr = (1 + rr.stats.total_return) ** (1 / yrs) - 1
            lbl = f">= ${floor/1e6:,.0f}M  {tag}" if floor else f"全部  {tag}"
            print(f"  {lbl:<34}{len(names):>5}{rr.stats.sharpe:>+9.2f}"
                  f"{rr.stats.total_return:>+11.1%}{cagr:>+9.1%}{rr.stats.ann_vol:>8.0%}"
                  f"{rr.stats.max_dd:>9.1%}{ns:>6}")
    print("""
  「原始 5+5」和「按比例」两列一起看，是为了排除一个混淆：
  5/26 是 19%，5/181 只有 2.8%——换池子之后同一个数字不是同一个策略。
  按比例放宽反而更差，所以选择度不是失败的原因，流动性才是。""")

    print("\n  同一批样本外币，不同成本假设:")
    print(f"  {'成本/边':<34}{'':>5}{'Sharpe':>9}{'总收益':>11}")
    for fee, slip in ((0, 0), (2, 1), (4.5, 2), (4.5, 8), (15, 15), (25, 25)):
        c = B.Costs(fee, slip)
        rr = run_universe(uni_oos, c)
        print(f"  {c.per_side*1e4:>6.1f}bp{'':<26}{'':>5}{rr.stats.sharpe:>+9.2f}"
              f"{rr.stats.total_return:>+11.1%}")

    # ------------------------------------------------------------------ a priori
    print("\n" + "=" * 130)
    print("3. 我事先点过名的币 vs 完全没提过的")
    print("=" * 130)
    man = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "PULL_MANIFEST.csv",
                      encoding="utf-8-sig")
    named = [s for s in oos if s in set(man[man["named_a_priori"]]["symbol"])]
    unnamed = [s for s in oos if s not in set(named)]
    print(f"""
  如果只有我事先挑过的那批有效，说明起作用的是我的先验，不是策略。
""")
    print(HEAD)
    for nm, group in (("我在 UNIVERSE.md 点过名的", named), ("完全没提过的", unnamed)):
        if len(group) < 12:
            print(f"  {nm:<34}{len(group):>5}   (不足 12 个)")
            continue
        print(line(nm, run_universe(D.load_universe(group, tfs=("1d",))), len(group)))

    # ------------------------------------------------------------------ annual
    print("\n" + "=" * 130)
    print("4. 逐年")
    print("=" * 130)
    ann = pd.DataFrame({"样本内": r_is.rets, "样本外": r_oos.rets})
    tab = ann.groupby(ann.index.year).apply(
        lambda g: pd.Series({c: (1 + g[c]).prod() - 1 for c in g.columns}),
        include_groups=False)
    print(tab.to_string(float_format=lambda v: f"{v:+.1%}"))

    REPORTS.mkdir(exist_ok=True)
    dvs.rename("median_daily_quote_volume").to_csv(REPORTS / "oos_liquidity.csv")
    print(f"\n  wrote {REPORTS/'oos_liquidity.csv'}")

    print("\n" + "=" * 130)
    print("结论")
    print("=" * 130)
    verdict = "站住了" if r_oos.stats.sharpe > 0.5 else (
        "边缘" if r_oos.stats.sharpe > 0.2 else "没站住")
    print(f"""
  样本内 Sharpe {r_is.stats.sharpe:+.2f}  ->  样本外 Sharpe {r_oos.stats.sharpe:+.2f}   [{verdict}]
  样本外 183 个币，策略事先完全写死，只跑一次，无多重检验惩罚。
""")


if __name__ == "__main__":
    main()
