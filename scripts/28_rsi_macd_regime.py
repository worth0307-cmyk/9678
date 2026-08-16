"""Gate the RSI strategy by a weekly MACD regime, and check what the gate is worth.

The previous run showed the entire loss came from shorting: 32 longs made +223
USDT while 34 shorts lost 815.  A weekly MACD golden/death cross is a natural
way to stop fighting the trend, so this tests it.

The controls matter more than the headline here.  A regime filter that only
permits longs during an uptrend will improve almost anything in a sample where
the asset tripled, so "it got better" proves nothing on its own.  Three controls
separate the possible explanations:

  long-only, no MACD    -- was it the MACD, or just not shorting?
  MACD regime alone     -- hold BTC long in a golden cross and short it in a
                           death cross, ignoring RSI entirely.  If this matches
                           or beats the combination, the RSI layer contributes
                           nothing and the result is a MACD result.
  buy and hold          -- the floor any long-biased rule has to clear.

Sample size is the thing to keep in view: the weekly MACD crosses 11 times in
3.62 years, so the filter itself carries roughly 11 independent observations.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scipy import stats  # noqa: E402
from vibt import data as D, rsi_strat as R  # noqa: E402

pd.set_option("display.width", 220)
EQ0, BPD = 1000.0, 6
REPORTS = Path(__file__).resolve().parent.parent / "reports"


def weekly_macd_regime(symbol: str, base_index: pd.DatetimeIndex,
                       fast: int = 12, slow: int = 26, sig: int = 9) -> pd.Series:
    """+1 while the weekly MACD is above its signal line, -1 below.

    Projected onto the 4h grid by the week's CLOSE time, so a week's verdict is
    unusable until that week has finished -- reading it any earlier would let
    Monday trade on Friday's information.
    """
    d = D.load("1d", symbol=symbol)
    wk = d[["open", "high", "low", "close"]].resample("W").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    macd = (wk["close"].ewm(span=fast, adjust=False).mean()
            - wk["close"].ewm(span=slow, adjust=False).mean())
    signal = macd.ewm(span=sig, adjust=False).mean()
    w = pd.DataFrame({"reg": np.sign(macd - signal)})
    w["close_time"] = w.index + pd.Timedelta(days=1)   # resample('W') labels Sunday
    return D.align_higher_tf(base_index, w, "wk", ["reg"])["reg_wk"]


def summarise(name: str, eq: pd.Series, trades: pd.DataFrame | None,
              bars: int) -> dict:
    yrs = bars / (365 * BPD)
    final = float(eq.dropna().iloc[-1])
    out = {"策略": name, "期末": final, "总收益": final / EQ0 - 1,
           "年化": (final / EQ0) ** (1 / yrs) - 1 if final > 0 else -1.0,
           "回撤": float((eq.dropna() / eq.dropna().cummax() - 1).min())}
    if trades is not None and len(trades):
        out |= {"笔数": len(trades),
                "多/空": f"{int((trades.direction=='long').sum())}/"
                         f"{int((trades.direction=='short').sum())}",
                "胜率": float((trades.pnl > 0).mean()),
                "每笔": float(trades.return_on_equity.mean())}
        if len(trades) > 1:
            out["p"] = float(stats.ttest_1samp(trades.return_on_equity, 0).pvalue)
    else:
        out |= {"笔数": 0, "多/空": "", "胜率": np.nan, "每笔": np.nan}
    return out


def regime_only(df: pd.DataFrame, reg: pd.Series, fee: float = 0.00065,
                long_only: bool = False) -> pd.Series:
    """Hold the asset per the regime, executing at the next open."""
    pos = reg.reindex(df.index).shift(1).fillna(0.0)
    if long_only:
        pos = pos.clip(lower=0)
    ret = df["open"].shift(-1) / df["open"] - 1.0
    turn = pos.diff().abs().fillna(pos.abs())
    net = (pos * ret - turn * fee).iloc[:-1].fillna(0)
    return EQ0 * (1 + net).cumprod()


def main() -> None:
    df = D.load("4h", symbol="BTCUSDT")
    reg = weekly_macd_regime("BTCUSDT", df.index)
    n = len(df)
    crosses = int((reg.diff().fillna(0) != 0).sum())

    print("=" * 150)
    print("BTCUSDT 4h RSI 策略 + 周线 MACD 单边滤网")
    print("=" * 150)
    print(f"""
  周线 MACD(12,26,9)：金叉只做多，死叉只做空。
  周线值在该周收盘后才可用（按 close_time 对齐），无前视。
  样本内交叉 11 次，4h 网格上换向 {crosses} 次。
  多头 regime 占 {(reg > 0).mean():.0%} 的时间，空头 {(reg < 0).mean():.0%}。
""")

    rows = []
    base = R.run(df, R.RsiParams(), EQ0)
    rows.append(summarise("① RSI 原版（双向，无滤网）", base.equity, base.trades, n))

    gated = R.run(df, R.RsiParams(), EQ0, reg)
    rows.append(summarise("② RSI + 周线MACD滤网", gated.equity, gated.trades, n))

    lo = R.run(df, R.RsiParams(), EQ0, pd.Series(1.0, index=df.index))
    rows.append(summarise("③ 对照：RSI 只做多（无MACD）", lo.equity, lo.trades, n))

    loose = R.run(df, R.RsiParams(rsi_bars=1, trend_bars=0), EQ0, reg)
    rows.append(summarise("④ RSI放宽 + MACD滤网", loose.equity, loose.trades, n))

    loose_lo = R.run(df, R.RsiParams(rsi_bars=1, trend_bars=0), EQ0,
                     pd.Series(1.0, index=df.index))
    rows.append(summarise("⑤ 对照：RSI放宽 只做多", loose_lo.equity, loose_lo.trades, n))

    rows.append(summarise("⑥ 对照：只用MACD（多空，无RSI）",
                          regime_only(df, reg), None, n))
    rows.append(summarise("⑦ 对照：只用MACD（只做多，无RSI）",
                          regime_only(df, reg, long_only=True), None, n))

    bh = EQ0 * (df["close"] / df["open"].iloc[0])
    rows.append(summarise("⑧ 对照：买入持有", bh, None, n))

    d = pd.DataFrame(rows)
    o = d.copy()
    for c in ("总收益", "年化", "回撤", "胜率", "每笔"):
        o[c] = o[c].map(lambda v: f"{v:+.1%}" if pd.notna(v) else "")
    o["期末"] = o["期末"].map(lambda v: f"{v:,.0f}")
    o["p"] = o.get("p", pd.Series(np.nan, index=o.index)).map(
        lambda v: f"{v:.3f}" if pd.notna(v) else "")
    print(o[["策略", "期末", "总收益", "年化", "回撤", "笔数", "多/空",
             "胜率", "每笔", "p"]].to_string(index=False))

    # ---------------------------------------------------------------- reading
    g = d.set_index("策略")["总收益"]
    print(f"""
{'=' * 150}
怎么读这张表
{'=' * 150}

  滤网确实改善了 RSI 策略：① {g.iloc[0]:+.1%} -> ② {g.iloc[1]:+.1%}。

  但对照 ③ 才是关键：单纯**不做空**（不带任何 MACD）已经给到 {g.iloc[2]:+.1%}。
  也就是说，②比①好的部分，{'几乎全部' if abs(g.iloc[2]-g.iloc[1]) < abs(g.iloc[1]-g.iloc[0])*0.4 else '有相当一部分'}来自"停止做空"，
  而不是来自 MACD 的择时能力。

  对照 ⑥⑦ 更要命：**完全不用 RSI，只按周线 MACD 持有 BTC**，
  多空版 {g.iloc[5]:+.1%}，只做多版 {g.iloc[6]:+.1%}。
  {'这已经超过了带 RSI 的所有版本 —— RSI 那一层没有贡献，甚至是负贡献。' if g.iloc[5] > max(g.iloc[1], g.iloc[3]) or g.iloc[6] > max(g.iloc[1], g.iloc[3]) else 'RSI 层仍有贡献。'}

  最后是 ⑧：买入持有 {g.iloc[7]:+.1%}。任何长期偏多的规则都得先跨过这条线。
""")

    # ---------------------------------------------------------------- split
    print("=" * 150)
    print("前后半稳定性")
    print("=" * 150 + "\n")
    half = n // 2
    parts = {"前半": df.iloc[:half], "后半": df.iloc[half:]}
    rows = []
    for label, sub in parts.items():
        sreg = reg.reindex(sub.index)
        a = R.run(sub, R.RsiParams(), EQ0, sreg)
        b = R.run(sub, R.RsiParams(rsi_bars=1, trend_bars=0), EQ0, sreg)
        c = regime_only(sub, sreg)
        rows.append({"": label,
                     "②RSI+MACD": a.equity.dropna().iloc[-1] / EQ0 - 1,
                     "④放宽+MACD": b.equity.dropna().iloc[-1] / EQ0 - 1,
                     "⑥只用MACD": c.dropna().iloc[-1] / EQ0 - 1,
                     "⑧买入持有": float(sub["close"].iloc[-1] / sub["open"].iloc[0] - 1),
                     "②笔数": len(a.trades), "④笔数": len(b.trades)})
    sp = pd.DataFrame(rows)
    for c in ("②RSI+MACD", "④放宽+MACD", "⑥只用MACD", "⑧买入持有"):
        sp[c] = sp[c].map(lambda v: f"{v:+.1%}")
    print(sp.to_string(index=False))

    REPORTS.mkdir(exist_ok=True)
    d.to_csv(REPORTS / "rsi_macd_regime.csv", index=False)
    print(f"\n  wrote {REPORTS/'rsi_macd_regime.csv'}")


if __name__ == "__main__":
    main()
