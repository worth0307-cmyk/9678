"""What the +38.2% buy-and-hold benchmark actually is, and whether the
comparison against it is fair.

The previous script put "4h rule A, +2.3% a year" next to "hold the six majors,
+38.2% a year" and left it there.  That comparison is doing two things at once
and both need unpacking:

  the benchmark   +38.2% is a compound annual rate over one specific window
                  that starts within weeks of the cycle low.  Start dates a few
                  months apart give very different numbers, and one coin
                  supplies close to half of it.
  the comparison  the strategy risks 1% of equity per trade and is therefore
                  flat most of the time, while buy-and-hold is fully exposed
                  every day.  Comparing their raw returns compares position
                  sizes, not edges.  Matching them on volatility is the version
                  that says something.

Volatility matching is not a rescue attempt.  It answers a definite question --
"levered until it hurts as much as holding, what does this pay?" -- and the
leverage it demands is itself the answer to whether that is a real option.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vibt import data as D, masys as MS  # noqa: E402

pd.set_option("display.width", 240)
REPORTS = Path(__file__).resolve().parent.parent / "reports"
EQ0 = 1000.0
MAJORS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT"]
RULES = {"A 密集突破": MS.entries_cluster_break,
         "B 回踩20均线": MS.entries_ma20_pullback}

_CACHE: dict[tuple[str, str], pd.DataFrame] = {}


def load(tf: str, s: str) -> pd.DataFrame:
    k = (tf, s)
    if k not in _CACHE:
        _CACHE[k] = D.load(tf, symbol=s)
    return _CACHE[k]


def hold_curve(symbols: list[str], start=None, end=None) -> pd.Series:
    """Equal-weight, bought once and never rebalanced -- what 'hold' means."""
    px = pd.DataFrame({s: load("1d", s)["close"] for s in symbols}).sort_index()
    if start is not None:
        px = px.loc[px.index >= start]
    if end is not None:
        px = px.loc[px.index <= end]
    px = px.dropna(how="all")
    first = px.ffill().bfill().iloc[0]
    return (px.ffill().div(first) / len(symbols)).sum(axis=1)


def cagr(curve: pd.Series) -> float:
    yrs = (curve.index[-1] - curve.index[0]).days / 365.25
    tot = float(curve.iloc[-1] / curve.iloc[0])
    return tot ** (1 / yrs) - 1 if tot > 0 and yrs > 0 else float("nan")


def stats_of(curve: pd.Series, per_year: float = 365.0) -> dict:
    r = curve.pct_change().dropna()
    dd = float((curve / curve.cummax() - 1).min())
    return {"总收益": float(curve.iloc[-1] / curve.iloc[0] - 1),
            "年化": cagr(curve),
            "年化波动": float(r.std() * np.sqrt(per_year)),
            "最大回撤": dd,
            "Sharpe": float(r.mean() / r.std() * np.sqrt(per_year)) if r.std() else np.nan}


def portfolio_curve(symbols: list[str], tf: str, p: MS.MaParams, fn,
                    risk_frac: float | None = None,
                    mark: bool = True) -> pd.Series:
    """One shared account, every coin at the same risk per trade.

    Two curves are possible and they are not the same thing:

      realised  equity moves only when a trade closes.  Flat between exits, so
                every open position's adverse excursion is invisible and the
                drawdown is understated.  This is the curve a trade log
                produces by default, and it is the same flattering accounting
                the grid-bot analysis in this repository was about.
      mark      open positions are marked to each bar's close.  This is what
                the account is actually worth, and what a margin call would
                read.  It is the default here for that reason.

    Trades compound on the common equity; positions on different coins may
    overlap, and at 1% risk with six coins that overlap stays small.
    """
    rf = p.risk_frac if risk_frac is None else risk_frac
    legs = []
    for s in symbols:
        df = load(tf, s)
        r = MS.simulate(df, fn(df, p), p, EQ0)
        if len(r.trades):
            legs.append((df, r.trades))
    if not legs:
        return pd.Series(dtype=float)

    # everything is accumulated on the strategy's own bar index, then resampled
    # to daily at the end -- reindexing a 4h series straight onto a daily index
    # would silently keep only the 00:00 bars and drop most of the exposure
    base = legs[0][0].index
    allt = pd.concat([t.assign(_sym=i) for i, (_, t) in enumerate(legs)]) \
             .sort_values("exit_time").reset_index(drop=True)
    eq = EQ0
    entry_eq = {}
    realised = []
    for k, row in allt.iterrows():
        entry_eq[k] = eq            # equity before this trade settles, in exit order
        eq *= 1 + rf * row.r_multiple
        realised.append((row.exit_time, eq))
    rs = pd.Series([v for _, v in realised],
                   index=pd.DatetimeIndex([i for i, _ in realised]))
    rs = rs[~rs.index.duplicated(keep="last")]
    real = pd.concat([pd.Series(EQ0, index=base[:1]), rs]).sort_index()
    real = real[~real.index.duplicated(keep="last")].reindex(base).ffill()

    if mark:
        floating = pd.Series(0.0, index=base)
        for k, row in allt.iterrows():
            df = legs[row._sym][0]
            a, b = int(row.entry_i), int(row.exit_i)
            if b <= a:
                continue
            risk_px = abs(row.entry - row.stop)
            if risk_px <= 0:
                continue
            side = 1.0 if row.side == "long" else -1.0
            seg = df["close"].iloc[a:b]
            r_open = side * (seg - row.entry) / risk_px
            floating.iloc[a:b] += (rf * entry_eq[k] * r_open).to_numpy()
        real = real + floating
    return real.resample("1D").last().ffill().dropna()


def main() -> None:
    p = MS.MaParams()
    btc = load("1d", "BTCUSDT")
    start, end = btc.index[0], btc.index[-1]
    yrs = (end - start).days / 365.25

    print("=" * 150)
    print("那个 +38.2% 是什么")
    print("=" * 150)
    print(f"""
  窗口：{start.date()} -> {end.date()}，共 {yrs:.2f} 年（{len(btc)} 根日线）。
  口径：6 个主流币等权买入，一次买进、全程不动、不再平衡。
  总收益 +222.7%，折成年化 = 3.227^(1/{yrs:.2f}) - 1 = +38.2%。

  所以 38.2% 不是「某一年赚了 38%」，是这 {yrs:.2f} 年整体翻了 3.2 倍之后倒推的复合年率。
""")

    # ------------------------------------------------------------------ ① 拆开
    print("=" * 150)
    print("① 这 +222.7% 是谁贡献的")
    print("=" * 150 + "\n")
    rows = []
    for s in MAJORS:
        d = load("1d", s)
        tot = float(d["close"].iloc[-1] / d["close"].iloc[0] - 1)
        rows.append({"币": s, "总收益": tot, "折年化": (1 + tot) ** (1 / yrs) - 1,
                     "占组合贡献": tot / 6})
    d = pd.DataFrame(rows).sort_values("总收益", ascending=False)
    d["占比"] = d["占组合贡献"] / d["占组合贡献"].sum()
    o = d.copy()
    for c in ("总收益", "折年化", "占组合贡献", "占比"):
        o[c] = o[c].map(lambda v: f"{v:+.1%}")
    print(o.to_string(index=False))
    print(f"""
  合计 {d['占组合贡献'].sum():+.1%}。SOL 一个币就占了 {d.iloc[0]['占比']:.0%}，
  前两名占 {d.iloc[:2]['占比'].sum():.0%}。**这个基准本身高度集中。**
""")

    # ------------------------------------------------------------ ② 起点敏感
    print("=" * 150)
    print("② 换个起点，这个数字变多少")
    print("=" * 150 + "\n")
    rows = []
    for st in ("2023-01-01", "2023-07-01", "2024-01-01", "2024-07-01",
               "2025-01-01", "2025-07-01"):
        ts = pd.Timestamp(st, tz=btc.index.tz)
        if ts >= end:
            continue
        c = hold_curve(MAJORS, start=ts)
        if len(c) < 60:
            continue
        rows.append({"起点": st, "年数": (c.index[-1] - c.index[0]).days / 365.25,
                     "总收益": float(c.iloc[-1] / c.iloc[0] - 1), "折年化": cagr(c),
                     "最大回撤": float((c / c.cummax() - 1).min())})
    o = pd.DataFrame(rows)
    for c in ("总收益", "折年化", "最大回撤"):
        o[c] = o[c].map(lambda v: f"{v:+.1%}")
    o["年数"] = o["年数"].map(lambda v: f"{v:.2f}")
    print(o.to_string(index=False))
    print("""
  2023-01-01 距离 2022-11 那个周期低点只有几周。**这个起点是整段数据里最好的一个。**
  晚半年进场，同一套买入持有的年化就完全是另一个数。
""")

    # ------------------------------------------------------------ ③ 滚动一年
    print("=" * 150)
    print("③ 滚动 12 个月的买入持有收益（把「年化」还原成实际经历）")
    print("=" * 150 + "\n")
    c = hold_curve(MAJORS)
    roll = c / c.shift(365) - 1
    roll = roll.dropna()
    print(f"  {'分位':<8}{'12个月收益':>12}")
    for q in (5, 25, 50, 75, 95):
        print(f"  {q:>3}%{'':<4}{roll.quantile(q/100):>+12.1%}")
    print(f"\n  为正的时间占比 {float((roll > 0).mean()):.0%}"
          f"   最好 {roll.max():+.1%}   最差 {roll.min():+.1%}")
    print("""
  年化 +38.2% 的实际体验是这样一条分布，不是每年稳稳 +38%。
""")

    # ------------------------------------------------------------ ④ 风险对比
    print("=" * 150)
    print("④ 真正该比的是风险调整后：策略几乎没上仓位，基准是满仓")
    print("=" * 150 + "\n")
    print("""
  两点先说清楚，否则这张表会被读错：
    · 这里是**一个账户跑 6 个币**，每笔风险 1% 的是整个账户的 1%。
      之前 36 号脚本报的 +2.3% 是「单个币一个独立小账户」的口径，
      同样的规则，同样的仓位比例，但资金基数差 6 倍，两个数不是一回事。
    · 权益曲线**按收盘盯市**，持仓中的浮亏算进回撤。
      只在平仓时记账的那条曲线回撤会明显更小，那正是本仓库批评网格机器人时说的问题。
""")
    bh = stats_of(c)
    rows = [{"对象": "等权买入持有 6 主流币", "周期": "1d"} | bh]
    curves = {}
    for tf in ("1d", "4h"):
        for name, fn in RULES.items():
            eq = portfolio_curve(MAJORS, tf, p, fn)
            if not len(eq) or eq.isna().all():
                continue
            curves[(tf, name)] = eq
            st = stats_of(eq)
            st["只算已平仓的回撤"] = float(
                (lambda q: (q / q.cummax() - 1).min())(
                    portfolio_curve(MAJORS, tf, p, fn, mark=False)))
            rows.append({"对象": f"{name}（每笔风险 {p.risk_frac:.0%}）", "周期": tf} | st)
    t = pd.DataFrame(rows)
    o = t.copy()
    for cc in ("总收益", "年化", "年化波动", "最大回撤", "只算已平仓的回撤"):
        o[cc] = o[cc].map(lambda v: f"{v:+.1%}" if pd.notna(v) else "")
    o["Sharpe"] = o["Sharpe"].map(lambda v: f"{v:+.2f}")
    print(o.to_string(index=False))

    # Sharpe on 45-64 daily trades has enormous error bars; say so with a number
    from vibt import metrics as M
    print("\n  Sharpe 的 block bootstrap 95% 区间（block=20 天）：")
    lo, mid, hi = M.block_bootstrap_sharpe(c.pct_change().dropna(), 365.0, block=20)
    print(f"    {'等权买入持有':<26}[{lo:+.2f}, {hi:+.2f}]")
    for (tf, name), eq in curves.items():
        lo, mid, hi = M.block_bootstrap_sharpe(eq.pct_change().dropna(), 365.0, block=20)
        print(f"    {name + ' ' + tf:<26}[{lo:+.2f}, {hi:+.2f}]")
    print("""
  区间几乎完全重叠 —— 在这段样本里，「这套系统」和「躺着不动」的风险调整后表现
  分不出高下。分不出高下不等于一样好，它等于**样本量不足以区分**。""")

    # ------------------------------------------------------------ ⑤ 波动对齐
    print("\n" + "=" * 150)
    print("⑤ 把策略放大到和买入持有一样的波动，再比一次")
    print("=" * 150 + "\n")
    tgt = bh["年化波动"]
    rows = []
    for (tf, name), eq in curves.items():
        st = stats_of(eq)
        if not st["年化波动"] or not np.isfinite(st["年化波动"]):
            continue
        k = tgt / st["年化波动"]
        r = eq.pct_change().fillna(0) * k
        lev = (1 + r).cumprod()
        rows.append({
            "策略": name, "周期": tf, "放大倍数": k,
            "放大后每笔风险": p.risk_frac * k,
            "放大后年化": cagr(lev), "放大后回撤": float((lev / lev.cummax() - 1).min()),
            "Sharpe": st["Sharpe"]})
    rows.append({"策略": "等权买入持有", "周期": "1d", "放大倍数": 1.0,
                 "放大后每笔风险": np.nan, "放大后年化": bh["年化"],
                 "放大后回撤": bh["最大回撤"], "Sharpe": bh["Sharpe"]})
    s = pd.DataFrame(rows)
    o = s.copy()
    o["放大倍数"] = o["放大倍数"].map(lambda v: f"{v:.1f}x")
    o["放大后每笔风险"] = o["放大后每笔风险"].map(
        lambda v: f"{v:.0%}" if pd.notna(v) else "")
    for cc in ("放大后年化", "放大后回撤"):
        o[cc] = o[cc].map(lambda v: f"{v:+.1%}")
    o["Sharpe"] = o["Sharpe"].map(lambda v: f"{v:+.2f}")
    print(o.to_string(index=False))
    print(f"""
  「放大倍数」是要达到买入持有那 {tgt:.0%} 的年化波动所需要的杠杆，
  「放大后每笔风险」是折算回视频那张仓位表上的数字。

  这个换算才是关键：如果需要把每笔风险从 1% 抬到十几个点，
  那 30% 胜率下连亏 5 笔（概率 {0.7**5:.0%}）就直接把账户打掉一半以上——
  也就是说这个「放大后年化」在算术上成立，在实际操作上不成立。
  Sharpe 那一列不受放大影响，它才是两者可比的部分。
""")

    REPORTS.mkdir(exist_ok=True)
    t.to_csv(REPORTS / "ma_benchmark.csv", index=False)
    print(f"  wrote {REPORTS/'ma_benchmark.csv'}")


if __name__ == "__main__":
    main()
