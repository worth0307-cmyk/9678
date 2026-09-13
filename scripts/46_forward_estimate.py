"""前瞻收益预估：把样本内的 +1.20 折算成一个**能当预期用**的数.

    python scripts/46_forward_estimate.py

45 号脚本把后见之明拆成两层（选币、仓位），并且量出了拿掉之后还剩多少。这个脚本
做下一步：**折算**，然后给出一年期的分布——而不是一个点估计。

三件事决定了这里不能只报一个数字：

  1  样本内的 Sharpe 不是预期值。参数是在这段数据上挑的，币也是。
  2  成本是**确定**的减项。换手 65 次/年 × 6.5bp = 4.2%/年，
     折合 0.16 的 Sharpe —— 前瞻 Sharpe 不到 0.16，净额就是负的。
  3  就算边际是真的，**单独一年仍然接近抛硬币**。Sharpe 0.4 配 25% 的波动，
     一年亏钱的概率是 34%，10% 分位在 −25% 左右。

去偏 Sharpe 给的 P(真实Sharpe>0)≈0.70 在这里当作混合权重用（0.7 的概率有边际，
0.3 的概率什么都没有）。那是个简化：那个概率说的是**符号**，不是幅度。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vibt import paper as PP  # noqa: E402

pd.set_option("display.width", 220)
MAJORS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT")
COST_BPS = 6.5
LBS, WINS = [7, 10, 14, 20, 30], [60, 90, 120, 180, 250]


def _norm_cdf(x: float) -> float:
    """Φ，不引 scipy —— 这条链路在 VPS 上要能跑。"""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def max_dd(pnl: pd.Series) -> float:
    eq = (1.0 + pnl).cumprod()
    return float((eq / eq.cummax() - 1.0).min())


def variant(label: str, **kw) -> dict:
    b = PP.backtest(PP.Params(), **kw)
    s = PP.stats(b["pnl"], b["turnover"], COST_BPS)
    grid = []
    for lb in LBS:
        for w in WINS:
            q = PP.Params(lookback=lb, pca_window=w)
            gb = PP.backtest(q, **kw)
            grid.append(PP.stats(gb["pnl"], gb["turnover"])["sharpe"])
    return {"版本": label, "选中格Sharpe": s["sharpe"],
            "网格中位Sharpe": float(np.median(grid)),
            "年化波动": s["gross_ann"] / s["sharpe"] if s["sharpe"] else np.nan,
            "年换手": s["turn_ann"], "成本拖累": s["turn_ann"] * COST_BPS * 1e-4,
            "样本内净额": s["net_ann"], "最大回撤": max_dd(b["pnl"])}


def main() -> None:
    print("=" * 104)
    print("前瞻收益预估")
    print("=" * 104)

    rows = [variant("在跑的：6币/等名义"),
            variant("6币/等风险", size="risk"),
            variant("4主流/等名义", coins=MAJORS),
            variant("4主流/等风险（后见之明最少）", coins=MAJORS, size="risk")]
    t = pd.DataFrame(rows).set_index("版本")
    show = t.copy()
    for c in ("选中格Sharpe", "网格中位Sharpe"):
        show[c] = show[c].map("{:+.2f}".format)
    for c in ("年化波动", "成本拖累", "样本内净额", "最大回撤"):
        show[c] = show[c].map("{:+.1%}".format)
    show["年换手"] = show["年换手"].map("{:.0f}x".format)
    print("\nA  样本内的四个版本（网格中位 = 不挑参数的话拿到什么）\n")
    print(show.to_string())

    live = t.loc["在跑的：6币/等名义"]
    clean = t.loc["4主流/等风险（后见之明最少）"]
    vol = float(live["年化波动"])
    cost = float(live["成本拖累"])

    print(f"""
  读法：**选中格**那一列是挑过参数的，不能当预期。**网格中位**那一列才是
  「不事后挑参数会拿到什么」。两列的差就是参数搜索这一层的水分。

  最后一行是三层后见之明全部拿掉之后的样子：选中格 {clean['选中格Sharpe']:+.2f}，
  网格中位 {clean['网格中位Sharpe']:+.2f}。**那是零附近的噪声。**
""")

    print("-" * 104)
    print("B  前瞻 Sharpe 该取多少")
    print("-" * 104)
    print(f"""
  往前走，三个成分的命运不一样：

    选币      **不带过去**。TAO/HYPE 过去赢，不代表以后赢 —— 它们正是看过
              结果之后被选中的。这一层前瞻价值按 0 算。
    仓位集中  **部分带过去**，但它不是边际，是暴露。等名义排名会一直重仓
              波动最大的名字；这在 2024~2026 的行情里赚了很多，在反过来的
              行情里会同样快地亏回去。期望按 0 到 +0.3 算，方差极大。
    排序能力  就是最后一行那个 {clean['网格中位Sharpe']:+.2f}。

  所以前瞻 Sharpe 的中枢在 **0.1~0.4**，不是 1.20。下面把 0.0 到 0.8 都列出来。

  成本要先还：换手 {live['年换手']:.0f}x × {COST_BPS}bp = {cost:.1%}/年 = {cost/vol:.2f} 的 Sharpe。
  **前瞻 Sharpe 低于 {cost/vol:.2f}，净额必然为负。**
""")

    print("-" * 104)
    print(f"C  一年期分布（波动按 {vol:.1%} 算，成本 {cost:.1%} 已扣）")
    print("-" * 104 + "\n")
    out = []
    for s in (0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.20):
        mu = s * vol - cost
        out.append({
            "前瞻Sharpe": f"{s:+.2f}", "净额均值": f"{mu:+.1%}",
            "亏钱概率": f"{1 - _norm_cdf(mu / vol):.0%}",
            "10%分位": f"{mu - 1.2816 * vol:+.0%}",
            "50%": f"{mu:+.0%}",
            "90%分位": f"{mu + 1.2816 * vol:+.0%}",
        })
    print(pd.DataFrame(out).set_index("前瞻Sharpe").to_string())

    p_edge = 0.70
    s_if = 0.30
    mu_mix = p_edge * s_if * vol - cost
    print(f"""
  最后一行 +1.20 是样本内的那个数，列在这里只为了对照 —— **它不是预测**。

  把「有 {1-p_edge:.0%} 的概率根本没有边际」也算进去（去偏 Sharpe 的 P(真实>0)≈{p_edge:.2f}，
  有边际时按 {s_if:+.2f} 算）：

      期望 Sharpe {p_edge * s_if:+.2f}   →   毛收益 {p_edge * s_if * vol:+.1%}
      减成本 {cost:.1%}                →   **净额 {mu_mix:+.1%}/年**
      亏钱概率 {1 - _norm_cdf(mu_mix / vol):.0%}     10% 分位 {mu_mix - 1.2816 * vol:+.0%}

  历史最大回撤（样本内、已经是最好看的那个版本）{live['最大回撤']:.1%}。
  前瞻只会更差，因为收益的那一半是选出来的，回撤那一半不是。
""")

    print("-" * 104)
    print("D  一句话")
    print("-" * 104)
    print(f"""
  **预期净收益 {mu_mix:+.0%}/年 上下，一年里有接近一半的时间是亏的，
  10% 的坏情况是 {mu_mix - 1.2816 * vol:+.0%}。**

  它不是 +26.6%。那个数字是「挑过参数 + 挑过币 + 重仓压在最波动的东西上」
  三件事在同一段历史上叠出来的。

  这个预估本身的不确定性也很大 —— 缩小它的唯一办法是纸面日志跑够时间，
  因为未来的时间是这套东西唯一还没被用过的数据。""")


if __name__ == "__main__":
    main()
