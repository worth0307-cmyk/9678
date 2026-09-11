"""评估任何想法之前先过这一关：胜率 x 赔率 x 成本 x 波动率.

Run it on an idea before building anything:

    python scripts/41_edge_arithmetic.py --win 0.51 --payoff 1
    python scripts/41_edge_arithmetic.py --win 0.31 --payoff 2.7 --cost 6.5
    python scripts/41_edge_arithmetic.py --gross 0.096 --turnover 441

With no arguments it prints the tables the question "is 51% enough" needs, and
then checks itself against the strategies this repository has already measured.
That last section is the point: a screening tool nobody has calibrated is just
another opinion, so it has to reproduce outcomes that are already known.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vibt import data as D, edge as E  # noqa: E402

pd.set_option("display.width", 220)
MAJORS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT")


def daily_sigma() -> float:
    vols = []
    for s in MAJORS:
        try:
            vols.append(float(D.load("1d", symbol=s)["close"].pct_change().std()))
        except FileNotFoundError:
            continue
    return float(np.mean(vols)) if vols else 0.0357


def fmt_verdict(v: dict) -> str:
    return (f"""
  每笔期望         {v['每笔期望R']:+.4f} R
  最优止损宽度     {v['最优止损']:.2%}   （s* = 4c/e，此处净期望恰好是毛期望的一半）
  该宽度下每年     {v['最优处每年笔数']:,.0f} 笔
  该宽度下净期望   {v['最优处净期望R']:+.4f} R
  ------------------------------------------------
  可达最大 Sharpe  {v['可达最大Sharpe']:.2f}
  确认所需笔数     {v['确认所需笔数']:,.0f}
""")


def table_payoff() -> None:
    print("=" * 104)
    print("① 胜率单独没有意义 —— 缺了赔率这道题就无解")
    print("=" * 104 + "\n")
    rows = []
    for p in (0.45, 0.50, 0.51, 0.55, 0.60):
        r = {"胜率": f"{p:.0%}"}
        for b, lab in ((0.8, "1:0.8"), (1.0, "1:1"), (2.0, "1:2"), (3.0, "1:3")):
            r[lab] = E.Edge(p, b).expectancy
        rows.append(r)
    d = pd.DataFrame(rows)
    for c in d.columns[1:]:
        d[c] = d[c].map(lambda v: f"{v:+.3f}")
    print(d.to_string(index=False))
    print("""
  「1:0.8」是滑点与冲击的典型后果：赢的时候少拿一点，输的时候照亏。
  51% 在那一列是 -0.082R，稳定亏钱；45% 配 1:3 是 +0.800R。
  所以本仓库六均线那套 30% 的胜率从来不是问题，1:3 只需要 25% 就打平。""")


def table_sample(sigma: float) -> None:
    print("\n" + "=" * 104)
    print("② 小边际的真正问题不是赚得少，是你永远无法确认它存在")
    print("=" * 104 + "\n")
    rows = []
    for p in (0.51, 0.52, 0.53, 0.55, 0.60):
        e = E.Edge(p, 1.0, sigma_daily=sigma)
        n = e.trades_to_confirm()
        rows.append({"胜率": f"{p:.0%}", "每笔期望R": e.expectancy,
                     "确认所需笔数": n,
                     "@每年250笔": n / 250, "@每年2500笔": n / 2500})
    d = pd.DataFrame(rows)
    d["每笔期望R"] = d["每笔期望R"].map(lambda v: f"{v:+.2f}")
    d["确认所需笔数"] = d["确认所需笔数"].map(lambda v: f"{v:,.0f}")
    for c in ("@每年250笔", "@每年2500笔"):
        d[c] = d[c].map(lambda v: f"{v:.1f}年")
    print(d.to_string(index=False))
    print("""
  单尾检验，alpha=5%、power=80%。51% 需要约 1.5 万笔。
  一年 250 笔要 62 年 —— 回测里看到 51%，它和 50% 在统计上是同一件事。""")


def table_tension(sigma: float, cost: float) -> None:
    print("\n" + "=" * 104)
    print(f"③ 止损越窄笔数越多，但成本按 R 算越贵 —— 两边对着干（51%/1:1，{cost*1e4:.1f}bp）")
    print("=" * 104 + "\n")
    e = E.Edge(0.51, 1.0, cost, sigma)
    rows = []
    for s in (0.005, 0.01, 0.02, 0.04, 0.065, 0.13, 0.25):
        rows.append({"止损宽度": f"{s:.1%}", "每年笔数": e.trades_per_year(s),
                     "成本R/笔": e.cost_r(s), "净期望R": e.net_expectancy(s),
                     "年化Sharpe": e.sharpe(s)})
    d = pd.DataFrame(rows)
    d["每年笔数"] = d["每年笔数"].map(lambda v: f"{v:,.0f}")
    for c in ("成本R/笔", "净期望R"):
        d[c] = d[c].map(lambda v: f"{v:.3f}")
    d["年化Sharpe"] = d["年化Sharpe"].map(lambda v: f"{v:+.2f}")
    print(d.to_string(index=False))
    print(f"""
  没有一行能用，而且这不是调参问题：最优止损 s* = 4c/e = {e.optimal_stop:.1%}，
  在那里也只有 Sharpe {e.max_sharpe:.2f}。
  s* 处净期望恒等于毛期望的一半 —— **无论边际多大，一半交给交易所**。""")


def table_threshold(sigma: float) -> None:
    print("\n" + "=" * 104)
    print("④ 一个百分点值多少：Sharpe 随边际的平方增长")
    print("=" * 104 + "\n")
    rows = []
    for p in (0.51, 0.52, 0.53, 0.55, 0.58, 0.60):
        r = {"胜率": f"{p:.0%}", "每笔期望R": E.Edge(p).expectancy}
        for c, lab in ((0.00065, "6.5bp"), (0.0002, "2bp(maker)"), (0.00005, "0.5bp")):
            ed = E.Edge(p, 1.0, c, sigma)
            r[lab] = ed.max_sharpe
            if c == 0.00065:
                r["最优止损"] = f"{ed.optimal_stop:.1%}"
                r["每年笔数"] = f"{ed.trades_per_year(ed.optimal_stop):,.0f}"
        rows.append(r)
    d = pd.DataFrame(rows)[["胜率", "每笔期望R", "最优止损", "每年笔数",
                            "6.5bp", "2bp(maker)", "0.5bp"]]
    d["每笔期望R"] = d["每笔期望R"].map(lambda v: f"{v:+.2f}")
    for c in ("6.5bp", "2bp(maker)", "0.5bp"):
        d[c] = d[c].map(lambda v: f"{v:.2f}")
    print(d.to_string(index=False))
    print("""
  Sharpe ∝ e² ：边际翻倍，可达 Sharpe 变四倍。
  51% -> 55% 不是「好一点」，是「从不存在到能做」。
  最右一列 0.5bp 是做市商的成本结构 —— 同样的 51% 在那里是另一回事。""")


def calibration(sigma: float) -> None:
    """Does the arithmetic reproduce what the repository already measured?"""
    print("\n" + "=" * 104)
    print("⑤ 自检：这套算术能不能复现本仓库已经测出来的结果")
    print("=" * 104 + "\n")
    print("""  注意口径：交易记录里的胜率和赔率，是在**已经扣掉成本的** R 倍数上统计的，
  所以 Edge(胜率, 赔率).expectancy 直接对应的是**净**期望，不能再减一次成本。
  第一版这里就是这么错的——对止损宽的两行误差小看不出来，
  到 4h 那行（止损只有 1.9%、成本 0.068R）就差出 0.07R，符号都反了。
  毛期望要反过来加回 2c/s。
""")
    rows = [
        # 全部来自 REPORT_MA.md；胜率与赔率均为净口径
        {"来源": "六均线 B 回踩MA20 日线样本外", "胜率": 0.308, "赔率": 2.73,
         "实测净期望R": 0.156, "止损": 0.056},
        {"来源": "六均线 A 密集突破 日线样本外", "胜率": 0.315, "赔率": 2.48,
         "实测净期望R": 0.096, "止损": 0.156},
        {"来源": "六均线 B 回踩MA20 4h样本内", "胜率": 0.279, "赔率": 2.67,
         "实测净期望R": 0.027, "止损": 0.019},
    ]
    out = []
    for r in rows:
        ed = E.Edge(r["胜率"], r["赔率"], 0.00065, sigma)
        cr = ed.cost_r(r["止损"])
        out.append({"来源": r["来源"], "胜率": r["胜率"], "赔率": r["赔率"],
                    "止损": r["止损"],
                    "算式净期望R": ed.expectancy,
                    "实测净期望R": r["实测净期望R"],
                    "差": ed.expectancy - r["实测净期望R"],
                    "成本R": cr,
                    "反推毛期望R": ed.expectancy + cr})
    d = pd.DataFrame(out)
    d["胜率"] = d["胜率"].map(lambda v: f"{v:.1%}")
    d["赔率"] = d["赔率"].map(lambda v: f"1:{v:.2f}")
    d["止损"] = d["止损"].map(lambda v: f"{v:.1%}")
    for c in ("算式净期望R", "实测净期望R", "差", "成本R", "反推毛期望R"):
        d[c] = d[c].map(lambda v: f"{v:+.3f}")
    print(d.to_string(index=False))
    print("""
  三行的「差」都在 ±0.01R 以内，算术无误。
  「成本R」那一列说明为什么 4h 那套做不起来：止损 1.9% 意味着每笔先付 0.068R，
  而它的毛期望只有 0.092R —— **成本吃掉四分之三**。
  日线那两行止损宽（5.6% / 15.6%），成本只占 0.023R / 0.008R，所以活了下来。""")

    print("\n  再平衡型（38/40 号脚本用的是这一支）：")
    for lab, g, t in (("PCA残差 日频 样本外", 0.096, 441),
                      ("PCA残差 平滑a=0.3", 0.090, 279),
                      ("横截面动量 vol-norm 样本外", 0.044, 69)):
        be = E.breakeven_cost(g, t)
        ok = "可做" if be > 0.00065 else "做不了"
        print(f"    {lab:<28} 毛 {g:+.1%} / 换手 {t:>3.0f}x "
              f"-> 打平 {be*1e4:>5.2f}bp  vs 6.5bp  **{ok}**")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--win", type=float, help="win rate, e.g. 0.51")
    ap.add_argument("--payoff", type=float, default=1.0,
                    help="win size as a multiple of the stop (default 1.0)")
    ap.add_argument("--cost", type=float, default=6.5, help="bp per side (default 6.5)")
    ap.add_argument("--sigma", type=float, default=None,
                    help="daily return std; default is this repo's majors")
    ap.add_argument("--gross", type=float,
                    help="rebalancing form: gross annual return, e.g. 0.096")
    ap.add_argument("--turnover", type=float,
                    help="rebalancing form: annual turnover, e.g. 441")
    args = ap.parse_args()

    sigma = args.sigma if args.sigma is not None else daily_sigma()

    if args.gross is not None and args.turnover is not None:
        be = E.breakeven_cost(args.gross, args.turnover)
        print(f"\n  毛收益 {args.gross:+.1%}/年，换手 {args.turnover:,.0f}x/年")
        print(f"  打平成本 {be*1e4:.2f}bp/边   vs 你给的 {args.cost:.1f}bp")
        print(f"  -> {'可做' if be * 1e4 > args.cost else '做不了'}\n")
        return

    if args.win is not None:
        ed = E.Edge(args.win, args.payoff, args.cost / 1e4, sigma)
        print(f"\n  胜率 {args.win:.1%}   赔率 1:{args.payoff:g}   "
              f"成本 {args.cost:.1f}bp/边   日波动 {sigma:.2%}")
        if ed.expectancy <= 0:
            print(f"\n  每笔期望 {ed.expectancy:+.4f} R —— 负期望，后面不用算了\n")
            return
        print(fmt_verdict(ed.verdict()))
        return

    print(f"  （日波动率取本仓库 4 个主流币的平均：{sigma:.2%}）\n")
    table_payoff()
    table_sample(sigma)
    table_tension(sigma, 0.00065)
    table_threshold(sigma)
    calibration(sigma)


if __name__ == "__main__":
    main()
