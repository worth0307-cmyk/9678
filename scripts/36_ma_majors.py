"""The six-MA system on the majors -- which is what the video tells you to trade.

Two findings from the earlier scripts point in opposite directions for this
question, and the majors are exactly where they collide:

  liquidity   the excess over a random-entry control grows with volume, and it
              is the only cut whose direction replicated on coins that had not
              been used to find it.  Majors are the top of that scale.
  drawdown    the excess was concentrated in coins that had fallen 90%+.  BTC,
              ETH, SOL and BNB are the opposite of that.

So this is not a formality: the two signals predict different answers, and the
majors decide between them.

Sample size is the obstacle.  On daily bars BTC produces 9 signals in three and
a half years, so a per-coin daily number is meaningless.  Six majors on 1h give
a few hundred, which is why all three timeframes are run -- and 1h is also where
the video puts short-term trading, so it is not a stretch of the method.

Every headline is judged against the same random-entry control used throughout:
same coins, same per-side counts, same stop widths in ATR, random timing.
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
SEED = 20260822
EQ0 = 1000.0

MAJORS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT"]
DEV26 = [
    "1000BONKUSDT", "1000FLOKIUSDT", "1000PEPEUSDT", "1000SHIBUSDT", "APTUSDT",
    "ARBUSDT", "BNBUSDT", "BTCUSDT", "DOGEUSDT", "DYDXUSDT", "ENAUSDT", "ETHUSDT",
    "FETUSDT", "HYPEUSDT", "LDOUSDT", "OPUSDT", "PENDLEUSDT", "RENDERUSDT",
    "SEIUSDT", "SOLUSDT", "SUIUSDT", "TAOUSDT", "TIAUSDT", "WIFUSDT", "WLDUSDT",
    "XRPUSDT",
]
NOT_COINS = ("USDCUSDT", "BTCDOMUSDT")
RULES = {"A 密集突破": MS.entries_cluster_break,
         "B 回踩20均线": MS.entries_ma20_pullback}
TFS = ("1d", "4h", "1h")

_CACHE: dict[tuple[str, str], pd.DataFrame] = {}


def load(tf: str, s: str) -> pd.DataFrame:
    k = (tf, s)
    if k not in _CACHE:
        _CACHE[k] = D.load(tf, symbol=s)
    return _CACHE[k]


def pooled(symbols: list[str], tf: str, p: MS.MaParams, which: str) -> pd.DataFrame:
    fn = RULES[which]
    out = []
    for s in symbols:
        try:
            df = load(tf, s)
        except FileNotFoundError:
            continue
        if len(df) < max(p.lens) + p.window:
            continue
        t = MS.evaluate(df, fn(df, p), p)
        if len(t):
            out.append(t.assign(symbol=s))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def null_for(t: pd.DataFrame, tf: str, p: MS.MaParams,
             rng: np.random.Generator, n_draws: int = 100) -> tuple[float, float]:
    """Null median expectancy and one-sided p, matched per coin and per side."""
    if not len(t):
        return np.nan, np.nan
    cnt = t.groupby(["symbol", "side"]).size().unstack(fill_value=0)
    for c in ("long", "short"):
        if c not in cnt:
            cnt[c] = 0
    atrs = t.risk_atr.to_numpy()
    draws = []
    for _ in range(n_draws):
        parts = []
        for s, row in cnt.iterrows():
            df = load(tf, s)
            z = MS.evaluate(df, MS.random_entries(df, p, int(row["long"]),
                                                  int(row["short"]), rng, atrs), p)
            if len(z):
                parts.append(z)
        if parts:
            draws.append(float(pd.concat(parts, ignore_index=True).r_multiple.mean()))
    dr = np.array(draws)
    obs = float(t.r_multiple.mean())
    return float(np.median(dr)), float((dr >= obs).mean())


def prof(t: pd.DataFrame) -> dict:
    if not len(t):
        return {"笔数": 0}
    return {"笔数": len(t), "胜率": float((t.r_multiple > 0).mean()),
            "期望R": float(t.r_multiple.mean()), "总R": float(t.r_multiple.sum()),
            "止损宽度": float(t.risk_pct.median()),
            "空头占比": float((t.side == "short").mean())}


def main() -> None:
    p = MS.MaParams()
    every = [s for s in D.available_symbols(require=("1d",)) if s not in NOT_COINS]
    keep = [s for s in every if len(load("1d", s)) >= max(p.lens) + p.window]
    others26 = [s for s in DEV26 if s not in MAJORS]
    oos = [s for s in keep if s not in DEV26]
    rng = np.random.default_rng(SEED)

    print("=" * 165)
    print("主流币：BTC / ETH / SOL / BNB / XRP / DOGE")
    print("=" * 165)
    print("""
  前面两条结论在主流币上正好打架：
    「成交额越大，超出随机对照越多」—— 主流币在这条线的最顶端
    「跌得越狠，超出随机对照越多」—— 主流币恰恰是没跌的那批
  所以这不是补充说明，是这两条结论的分歧点。

  日线上 BTC 三年半只有 9 个信号，单币日线数字没有意义，
  所以 1d / 4h / 1h 三个周期都跑。1h 也正是视频里给短线用的周期。
""")

    # ------------------------------------------------------------------ ① 逐币
    for tf in TFS:
        for which in RULES:
            rows = []
            for s in MAJORS:
                t = pooled([s], tf, p, which)
                px = load(tf, s)["close"]
                rows.append({"币": s} | prof(t) |
                            {"期间涨跌": float(px.iloc[-1] / px.iloc[0] - 1)})
            d = pd.DataFrame(rows)
            print("=" * 165)
            print(f"① {tf}   {which}")
            print("=" * 165 + "\n")
            o = d.copy()
            for c in ("胜率", "止损宽度", "空头占比", "期间涨跌"):
                o[c] = o[c].map(lambda v: f"{v:+.0%}" if pd.notna(v) else "")
            for c in ("期望R", "总R"):
                o[c] = o[c].map(lambda v: f"{v:+.2f}" if pd.notna(v) else "")
            print(o.to_string(index=False))
            print()

    # ------------------------------------------------------------- ② 池化+对照
    print("=" * 165)
    print("② 六个主流币合起来，对随机入场")
    print("=" * 165)
    print("""
  单个主流币笔数太少，只能池化。对照仍是逐币逐方向配平的随机入场。
""")
    rows = []
    for tf in TFS:
        for which in RULES:
            t = pooled(MAJORS, tf, p, which)
            if not len(t):
                continue
            nul, pv = null_for(t, tf, p, rng)
            rows.append({"周期": tf, "规则": which} | prof(t) |
                        {"随机对照": nul, "超出对照": t.r_multiple.mean() - nul, "p": pv})
    m = pd.DataFrame(rows)
    o = m.copy()
    for c in ("胜率", "止损宽度", "空头占比"):
        o[c] = o[c].map(lambda v: f"{v:+.0%}")
    for c in ("期望R", "总R", "随机对照", "超出对照"):
        o[c] = o[c].map(lambda v: f"{v:+.3f}")
    o["p"] = o["p"].map(lambda v: f"{v:.3f}")
    print(o.to_string(index=False))

    # ------------------------------------------------------------- ③ 三组对比
    print("\n" + "=" * 165)
    print("③ 主流币 vs 其余开发币 vs 样本外 —— 同一个对照，并排看")
    print("=" * 165 + "\n")
    groups = [("主流6币", MAJORS), ("其余20个开发币", others26),
              (f"样本外{len(oos)}币", oos)]
    rows = []
    for tf in ("1d", "4h"):
        for which in RULES:
            for label, syms in groups:
                if tf == "4h" and label.startswith("样本外"):
                    continue          # no 4h data for the out-of-sample panel
                t = pooled(syms, tf, p, which)
                if not len(t):
                    continue
                nul, pv = null_for(t, tf, p, rng, n_draws=60)
                rows.append({"周期": tf, "规则": which, "组": label,
                             "币数": len(set(t.symbol)), "笔数": len(t),
                             "期望R": float(t.r_multiple.mean()),
                             "随机对照": nul,
                             "超出对照": float(t.r_multiple.mean()) - nul, "p": pv})
    g = pd.DataFrame(rows)
    o = g.copy()
    for c in ("期望R", "随机对照", "超出对照"):
        o[c] = o[c].map(lambda v: f"{v:+.3f}")
    o["p"] = o["p"].map(lambda v: f"{v:.3f}")
    print(o.to_string(index=False))

    # ------------------------------------------------------------------ ④ 多空
    print("\n" + "=" * 165)
    print("④ 主流币的多空拆分（这三年主流币是涨的，空头那边应该是问题所在）")
    print("=" * 165 + "\n")
    rows = []
    for tf in TFS:
        for which in RULES:
            t = pooled(MAJORS, tf, p, which)
            for side in ("long", "short"):
                sub = t[t.side == side]
                if len(sub) >= 10:
                    rows.append({"周期": tf, "规则": which, "方向": side} | prof(sub))
    s = pd.DataFrame(rows).drop(columns=["空头占比"])
    o = s.copy()
    for c in ("胜率", "止损宽度"):
        o[c] = o[c].map(lambda v: f"{v:+.0%}")
    for c in ("期望R", "总R"):
        o[c] = o[c].map(lambda v: f"{v:+.2f}")
    print(o.to_string(index=False))

    # ------------------------------------------------------------------ ⑤ 组合
    print("\n" + "=" * 165)
    print("⑤ 只做这 6 个主流币，按视频的仓位规则（每笔风险 1% 权益）")
    print("=" * 165 + "\n")
    rows = []
    for tf in TFS:
        yrs = len(load(tf, "BTCUSDT")) / D.bars_per_year(tf)
        for which in RULES:
            fn = RULES[which]
            fin, ntr = [], []
            for s in MAJORS:
                df = load(tf, s)
                r = MS.simulate(df, fn(df, p), p, EQ0)
                if len(r.trades):
                    fin.append(float(r.equity.iloc[-1]) / EQ0 - 1)
                    ntr.append(len(r.trades))
            if not fin:
                continue
            f = np.array(fin)
            tot = float(f.mean())
            rows.append({"周期": tf, "规则": which, "币数": len(f),
                         "平均笔数": float(np.mean(ntr)),
                         "6币平均总收益": tot,
                         "折年化": (1 + tot) ** (1 / yrs) - 1 if tot > -1 else -1.0,
                         "为正的币": float((f > 0).mean()),
                         "最好": float(f.max()), "最差": float(f.min())})
    q = pd.DataFrame(rows)
    o = q.copy()
    for c in ("6币平均总收益", "折年化", "为正的币", "最好", "最差"):
        o[c] = o[c].map(lambda v: f"{v:+.1%}")
    print(o.to_string(index=False))
    bh = np.mean([float(load("1d", s)["close"].iloc[-1] / load("1d", s)["open"].iloc[0] - 1)
                  for s in MAJORS])
    yrs = len(load("1d", "BTCUSDT")) / 365.0
    print(f"""
  对照：同期这 6 个币等权买入持有 {bh:+.1%}，折年化 {(1+bh)**(1/yrs)-1:+.1%}。
  每笔只赌 1% 权益，所以这一列不是「策略好不好」，是「按这个仓位能拿到多少」。
""")

    REPORTS.mkdir(exist_ok=True)
    m.to_csv(REPORTS / "ma_majors.csv", index=False)
    g.to_csv(REPORTS / "ma_majors_groups.csv", index=False)
    print(f"  wrote {REPORTS/'ma_majors.csv'}, {REPORTS/'ma_majors_groups.csv'}")


if __name__ == "__main__":
    main()
