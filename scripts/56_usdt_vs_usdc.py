"""BTCUSDT 和 BTCUSDC 两条套利路线的并排比较.

    python scripts/56_usdt_vs_usdc.py        # 在 VPS 上跑，需要 data/spot/

问题有两个：USDC 的手续费更低（永续挂单 0bp），资金费率是不是也更高？

手续费可以直接算，币安普通用户、无 BNB 折扣、全部挂限价单：

    USDT 版   现货 10bp×2 + 永续 2bp×2 = 24bp   一次完整进出
    USDC 版   现货 10bp×2 + 永续 0bp×2 = 20bp

省 4bp。对年化 6.5% 的持仓，那是 0.04% —— **噪声**。成本的大头是现货腿，
而现货 USDC 的挂单费是「标准」，没优惠。所以 USDC 值不值得，取决于资金费率。

资金费率不能直接比：**BTCUSDC 永续 2024-01-03 才上线**，而 BTCUSDT 从 2022 年
就有。跨不同时间段比均值是 53 号脚本犯过的错（同一个「买入持有」在两行里
一个 +0.46 一个 +0.88），所以这里强制对齐到共同窗口。

还有一个数据事实要处理：**BTCUSDC 现货有洞**，日线缺 163 根、1h 缺约 3940 根。
币安在 2022 年底把 USDC 自动转换成 BUSD，USDC 交易对停过一段时间。
这意味着两件事：
  1  跨缺口的 pct_change 会把几个月的价格变动算成一天的损益，必须作废
  2  **这条路线不是一直可用的** —— 那本身就是风险，不是数据瑕疵
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vibt import data as D, ingest as I  # noqa: E402

pd.set_option("display.width", 235)
# DATA_ROOT 只为了让这个脚本能在合成数据上冒烟测试（币安数据只在 VPS 上，
# 开发机既连不上也读不到）。不设的时候就是仓库自己的 data/。
_OVERRIDE = os.environ.get("DATA_ROOT")
DATA = Path(_OVERRIDE) if _OVERRIDE else Path(__file__).resolve().parent.parent / "data"
SPOT = DATA / "spot"
FUNDING = DATA / "funding"

# 币安普通用户，无 BNB 折扣，全部挂限价单（挂单价）
FEES = {  # (现货挂单bp, 永续挂单bp)
    "BTCUSDT": (10.0, 2.0),
    "BTCUSDC": (10.0, 0.0),
}


def carry(sym: str, tf: str = "1d") -> pd.DataFrame:
    """两条腿 + 资金费率的逐 bar 损益。跨缺口的收益作废。"""
    perp = D.load(tf, data_dir=DATA, symbol=sym)["close"].rename("perp")
    spot = D.load(tf, data_dir=SPOT, symbol=sym)["close"].rename("spot")
    df = pd.concat([perp, spot], axis=1, sort=True).dropna()

    # BTCUSDC 现货中间断过 163 天。不作废的话，复牌那一根会把几个月的
    # 价格变动记成「一天的基差损益」—— 一个能凭空造出几十个百分点的数字。
    step = pd.Timedelta("1D" if tf == "1d" else "1h")
    contiguous = df.index.to_series().diff() <= step
    r_spot = df["spot"].pct_change().where(contiguous)
    r_perp = df["perp"].pct_change().where(contiguous)
    basis_pnl = (r_spot - r_perp)

    fp = FUNDING / f"{sym}_funding.csv.gz"
    if not fp.exists():
        fp = FUNDING / f"{sym}_funding.csv"
    f = I._load_funding(fp)
    fr = f["funding_rate"].astype(float)
    freq = {"1d": "D", "1h": "h"}[tf]
    fu = fr.groupby(fr.index.floor(freq)).sum().reindex(df.index).fillna(0.0)

    out = pd.DataFrame({"perp": df["perp"], "spot": df["spot"],
                        "basis_bp": (df["perp"] / df["spot"] - 1) * 1e4,
                        "basis_pnl": basis_pnl, "funding": fu,
                        "ok": contiguous})
    out["total"] = out["basis_pnl"].fillna(0.0) + out["funding"]
    return out


def summarize(x: pd.DataFrame, label: str, fee_rt_bp: float) -> dict:
    n = len(x)
    yrs = n / 365.0
    tot = x["total"]
    eq = (1 + tot).cumprod()
    return {"版本": label, "天数": n,
            "作废的bar": int((~x["ok"]).sum()),
            "资金费率年化": float(x["funding"].sum() / yrs),
            "基差年化": float(x["basis_pnl"].fillna(0.0).sum() / yrs),
            "合计年化": float(tot.sum() / yrs),
            "一次进出费用": -fee_rt_bp * 1e-4,
            "持满一年净额": float(tot.sum() / yrs) - fee_rt_bp * 1e-4,
            "年化波动": float(tot.std() * math.sqrt(365)),
            "最大回撤": float((eq / eq.cummax() - 1).min()),
            "单日最差": float(tot.min())}


def main() -> None:
    missing = [s for s in FEES if not (SPOT / f"{s}_1d.csv.gz").exists()
               and not (SPOT / f"{s}_1d.csv").exists()]
    if missing:
        print(f"缺现货数据：{missing}  —— 先抓：\n"
              "  python scripts/fetch_binance.py --spot --symbols "
              f"{','.join(missing)} --intervals 1h,1d --start 2022-01-01 "
              "--out ~/9678/.sp --no-hints\n"
              "  python -m vibt.ingest ~/9678/.sp --dest ~/9678/data/spot --gzip")
        raise SystemExit(1)

    print("=" * 122)
    print("BTCUSDT vs BTCUSDC：同一笔套利的两条路线")
    print("=" * 122)

    legs = {s: carry(s) for s in FEES}

    print("\n" + "-" * 122)
    print("0  数据可用区间与缺口 —— USDC 那条路线不是一直存在的")
    print("-" * 122 + "\n")
    for s, x in legs.items():
        holes = x.index.to_series().diff()
        big = holes[holes > pd.Timedelta("1D")]
        print(f"  {s}  {x.index[0].date()} -> {x.index[-1].date()}   {len(x)} 天")
        for ts, d in big.items():
            print(f"      缺口 {(ts - d).date()} -> {ts.date()}   断了 {d.days} 天")
        if big.empty:
            print("      无缺口")

    common = None
    for x in legs.values():
        common = x.index if common is None else common.intersection(x.index)
    print(f"\n  共同窗口 {common[0].date()} -> {common[-1].date()}   {len(common)} 天")
    print("  （BTCUSDC 永续 2024-01-03 上线，所以对比只能在这一段上做。）")

    print("\n" + "-" * 122)
    print("A  共同窗口上的完整损益（手续费按全挂单计）")
    print("-" * 122 + "\n")
    rows = []
    for s, x in legs.items():
        spot_bp, perp_bp = FEES[s]
        rt = (spot_bp + perp_bp) * 2
        rows.append(summarize(x.reindex(common), f"{s}（现货{spot_bp:.0f}+永续{perp_bp:.0f}bp）", rt))
    t = pd.DataFrame(rows).set_index("版本")
    for c in ("资金费率年化", "基差年化", "合计年化", "一次进出费用",
              "持满一年净额", "年化波动", "最大回撤", "单日最差"):
        t[c] = t[c].map("{:+.2%}".format)
    print(t.to_string())

    a, b = list(legs)
    da = summarize(legs[a].reindex(common), a, (FEES[a][0] + FEES[a][1]) * 2)
    db = summarize(legs[b].reindex(common), b, (FEES[b][0] + FEES[b][1]) * 2)
    print(f"""
  资金费率之差   {db['资金费率年化'] - da['资金费率年化']:+.2%}/年   （{b} 减 {a}）
  手续费之差     {db['一次进出费用'] - da['一次进出费用']:+.2%}     （一次进出）
  净额之差       {db['持满一年净额'] - da['持满一年净额']:+.2%}/年""")

    print("\n" + "-" * 122)
    print("B  基差水位对比（1h）")
    print("-" * 122 + "\n")
    rows = []
    for s in FEES:
        x = carry(s, "1h")
        bb = x["basis_bp"]
        q = bb.quantile([0.01, 0.25, 0.5, 0.75, 0.99])
        rows.append({"版本": s, "根数": len(bb), "中位": q.iloc[2],
                     "1%": q.iloc[0], "25%": q.iloc[1], "75%": q.iloc[3],
                     "99%": q.iloc[4], "最小": bb.min(), "最大": bb.max(),
                     "为正%": (bb > 0).mean() * 100})
    print(pd.DataFrame(rows).set_index("版本").round(1).to_string())
    print("""
  基差越接近 0 越好建仓：做空永续时，负基差意味着你卖在折价上，
  而它会往上均值回复 —— 那段回复由你付。""")

    print("\n" + "-" * 122)
    print("C  逐年 —— 这一段才是决策的依据")
    print("-" * 122 + "\n")
    # 一律年化。不年化的话 256 天的 2026 和 365 天的 2025 没法并排读，
    # 而这张表唯一的用途就是看**趋势**。
    print(f"  {'':10s} {'年化收益':>9} {'其中资金费率':>13} {'其中基差':>9} {'天数':>6}")
    for s, x in legs.items():
        print(f"  {s}")
        xs = x.reindex(common)
        for y, seg in xs["total"].groupby(common.year):
            n = len(seg)
            ann = (1 + float((1 + seg).prod() - 1)) ** (365.0 / n) - 1
            fu = float(xs["funding"][seg.index].sum()) * 365.0 / n
            ba = float(xs["basis_pnl"][seg.index].fillna(0).sum()) * 365.0 / n
            print(f"  {y:>10}  {ann:+9.2%} {fu:+13.2%} {ba:+9.2%} {n:>6}")
    print("""
  这不是波动，是**趋势**：更多资本涌进来收割，溢价就被压平。
  平均值（共同窗口 +6.9%）是被 2024 年拉起来的，**当下的运行水平才是
  你要面对的那个数**。""")


if __name__ == "__main__":
    main()
