"""基差：把资金费率收割这笔交易的**另一半**算出来.

    python scripts/55_basis.py           # 在 VPS 上跑，需要 data/spot/

54 号脚本算的是资金费率流 —— 那是这笔交易的**收入项**，不是交易本身。
真正的持仓是两条腿：

    做空永续 1 份 + 买入现货 1 份（等名义）

价格涨跌在两条腿之间抵消，剩下两样东西：

    资金费率   收（费率为正时）
    基差变动   (永续 − 现货) 的价差自己会动

方向要说准：这个组合**做空永续**，所以永续相对现货**变贵（基差上升）时亏钱**，
变便宜（基差下降）时赚钱。因此在基差偏高时建仓有利，偏低时建仓不利 ——
后者意味着它还要往上均值回复，那段回复就是你的成本。

完整损益是：

    P&L = 现货腿涨跌 − 永续腿涨跌 + 累计资金费率
        = (s_t/s_0 − 1) − (p_t/p_0 − 1) + Σ funding

第一项和第二项之差就是基差的变化。**这才是那个一直没被量化的风险。**

FTX 那两天尤其要看：SOL 的资金费率打到 −200bp 下限，如果基差同时反向走阔，
那就是两条腿一起亏 —— 54 号脚本报的 −36% 就还是低估的。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vibt import data as D, ingest as I  # noqa: E402

pd.set_option("display.width", 235)
ROOT = Path(__file__).resolve().parent.parent
SPOT_DIR = Path(__import__("os").environ.get("SPOT_DIR", ROOT / "data" / "spot"))
COINS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "TAOUSDT")
SPOT_BPS, PERP_BPS = 10.0, 5.0


def pair(sym: str, tf: str) -> pd.DataFrame:
    """把永续和现货对齐到同一根 bar，返回两列收盘价。"""
    p = D.load(tf, symbol=sym)["close"].rename("perp")
    s = D.load(tf, data_dir=SPOT_DIR, symbol=sym)["close"].rename("spot")
    return pd.concat([p, s], axis=1).dropna()


def leg_pnl(df: pd.DataFrame) -> pd.Series:
    """两条腿的逐日损益，占**当期**名义的比例。

    第一版写的是 (s_t/s_0 − p_t/p_0) 再差分。那等价于「期初建一次仓、之后
    再不调整」，而且以**期初**名义为分母 —— 于是价格跌到期初的 8% 之后，
    真实的 25% 基差冲击被缩成 2%。SOL 在 FTX 那天的损益因此被报成 −18.47%，
    真实值是 −42% 量级。

    真实持仓是等名义、持续盯市的，所以逐日损益就是两条腿的**日收益之差**：
    做空永续，永续涨得比现货多就亏。
    """
    return (df["spot"].pct_change() - df["perp"].pct_change()).fillna(0.0)


def basis_bp(df: pd.DataFrame) -> pd.Series:
    return (df["perp"] / df["spot"] - 1.0) * 1e4


def funding_on(index: pd.DatetimeIndex, sym: str, tf: str) -> pd.Series:
    d = I._load_funding(ROOT / "data" / "funding" / f"{sym}_funding.csv.gz")
    r = d["funding_rate"].astype(float)
    freq = {"1d": "D", "1h": "h"}[tf]
    return r.groupby(r.index.floor(freq)).sum().reindex(index).fillna(0.0)


def find_gaps(sym: str, tf: str) -> list:
    """ingest 报了缺口但没说在哪。基差是逐 bar 对齐的，缺口位置要能看见。"""
    idx = D.load(tf, data_dir=SPOT_DIR, symbol=sym).index
    step = pd.Timedelta("1h" if tf == "1h" else "1D")
    d = pd.Series(idx).diff()
    return [(idx[i - 1], idx[i], d.iloc[i]) for i in np.flatnonzero(d > step)]


def main() -> None:
    if not SPOT_DIR.exists():
        print(f"找不到 {SPOT_DIR} —— 先在 VPS 上抓现货：\n"
              "  python scripts/fetch_binance.py --spot --symbols BTCUSDT,ETHUSDT,"
              "SOLUSDT,TAOUSDT --intervals 1h,1d --start 2022-01-01 "
              "--out ~/9678/.spot --no-hints\n"
              "  python -m vibt.ingest ~/9678/.spot --dest ~/9678/data/spot --gzip")
        raise SystemExit(1)

    print("=" * 120)
    print("基差：资金费率收割的另一半")
    print("=" * 120)

    print("\n" + "-" * 120)
    print("0  先定位 ingest 报的那个缺口")
    print("-" * 120 + "\n")
    for s in COINS:
        for tf in ("1h", "1d"):
            g = find_gaps(s, tf)
            if g:
                for a, b, d in g:
                    print(f"  {s:9s} {tf}  缺口 {a} -> {b}   少了 {d - pd.Timedelta(tf.replace('1d', '1D'))}")
            else:
                print(f"  {s:9s} {tf}  无缺口")

    print("\n" + "-" * 120)
    print("A  基差水位（永续 − 现货，单位 bp，1h）")
    print("-" * 120 + "\n")
    rows = []
    store = {}
    for s in COINS:
        df = pair(s, "1h")
        b = basis_bp(df)
        store[s] = (df, b)
        q = b.quantile([0.001, 0.01, 0.25, 0.5, 0.75, 0.99, 0.999])
        rows.append({"币": s, "根数": len(b), "均值": b.mean(), "中位": b.median(),
                     "0.1%": q.iloc[0], "1%": q.iloc[1], "25%": q.iloc[2],
                     "75%": q.iloc[4], "99%": q.iloc[5], "99.9%": q.iloc[6],
                     "最小": b.min(), "最大": b.max(), "为正%": (b > 0).mean() * 100})
    t = pd.DataFrame(rows).set_index("币")
    print(t.round(1).to_string())
    print("""
  正的基差 = 永续比现货贵。做空永续 + 买现货，等于**卖出这个溢价**：
  它收敛你就赚，它走阔你就亏。永续不到期，所以它不保证收敛。""")

    print("\n" + "-" * 120)
    print("B  完整损益：两条腿 + 资金费率，从头持到尾")
    print("-" * 120 + "\n")
    rows = []
    for s in COINS:
        df, b = store[s]
        df = df.resample("D").last().dropna()
        fu = funding_on(df.index, s, "1d")
        basis_pnl = leg_pnl(df)
        total = basis_pnl + fu
        yrs = len(total) / 365.0
        eq = (1 + total).cumprod()
        rows.append({"币": s, "天数": len(total),
                     "资金费率年化": float(fu.sum() / yrs),
                     "基差年化": float(basis_pnl.sum() / yrs),
                     "合计年化": float(total.sum() / yrs),
                     "年化波动": float(total.std() * math.sqrt(365)),
                     "最大回撤": float((eq / eq.cummax() - 1).min()),
                     "单日最差": float(total.min())})
    t = pd.DataFrame(rows).set_index("币")
    for c in ("资金费率年化", "基差年化", "合计年化", "年化波动", "最大回撤", "单日最差"):
        t[c] = t[c].map("{:+.2%}".format)
    print(t.to_string())

    print("\n" + "-" * 120)
    print("C  压力时刻：基差和资金费率是不是同向恶化")
    print("-" * 120 + "\n")
    for s in COINS:
        df, b = store[s]
        d = df.resample("D").last().dropna()
        fu = funding_on(d.index, s, "1d")
        bb = basis_bp(d)
        worst = fu.nsmallest(3)
        print(f"  {s}  资金费率最差的三天：")
        for ts, v in worst.items():
            around = bb.loc[max(bb.index[0], ts - pd.Timedelta("2D")):ts + pd.Timedelta("2D")]
            print(f"    {ts.date()}  费率 {v * 100:+.2f}%   "
                  f"当日基差 {bb.get(ts, np.nan):+.0f}bp   "
                  f"前后 5 天基差区间 [{around.min():+.0f}, {around.max():+.0f}]bp")

    print("\n" + "-" * 120)
    print("D  建仓那一刻的基差 —— 这是实打实先亏/先赚的一笔")
    print("-" * 120 + "\n")
    print(f"  两条腿进出一次的手续费 = (现货{SPOT_BPS:.0f} + 永续{PERP_BPS:.0f})×2 "
          f"= {(SPOT_BPS + PERP_BPS) * 2:.0f}bp\n")
    for s in COINS:
        _, b = store[s]
        print(f"  {s:9s} 随便挑一小时建仓，基差中位 {b.median():+.1f}bp，"
              f"四分位区间 [{b.quantile(0.25):+.1f}, {b.quantile(0.75):+.1f}]bp，"
              f"最差 1% 是 {b.quantile(0.01):+.1f}bp")
    print("""
  基差为正时建仓对空方**有利**（卖在溢价上）。所以真正该等的不是价格，
  是基差 —— 而这和「等回调」不同：基差是可观测的当期状态，不是对未来的预测。""")


    print("\n" + "-" * 120)
    print("E  把 SOL 的 FTX 那几天摊开 —— 尾部就在这里")
    print("-" * 120 + "\n")
    df, _ = store["SOLUSDT"]
    d = df.resample("D").last().dropna()
    fu = funding_on(d.index, "SOLUSDT", "1d")
    bp = leg_pnl(d)
    bb = basis_bp(d)
    win = slice("2022-11-05", "2022-11-16")
    out = pd.DataFrame({"现货": d["spot"][win].round(2), "永续": d["perp"][win].round(2),
                        "基差bp": bb[win].round(0), "基差损益": bp[win],
                        "资金费率": fu[win], "合计": (bp + fu)[win]})
    for c in ("基差损益", "资金费率", "合计"):
        out[c] = out[c].map("{:+.2%}".format)
    print(out.to_string())
    print("""
  这一段是整笔交易全部风险的来源。看三件事：
    基差在 11-09 崩到什么程度、第二天回升多少、以及基差损益和资金费率
    是**互相抵消**还是**同向叠加**。""")


if __name__ == "__main__":
    main()
