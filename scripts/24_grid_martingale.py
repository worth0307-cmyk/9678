"""What a grid/martingale bot's numbers look like on our own data.

Motivated by a specific claim -- 5-30% per 30 days, ~99% win rate -- from a bot
platform.  Rather than argue about whether that is plausible, run the mechanism
on the 1h data already in the repository and read the output.

The interesting column is never the return.  It is the gap between the realised
curve, which only contains closed trades and is what a dashboard usually shows,
and the marked curve, which includes open positions and is what actually gets
liquidated.  A martingale's entire design keeps losers out of the first number.

Compare against what this project established with 209 symbols, 3.6 years and a
full falsification battery: an out-of-sample Sharpe of +0.72 at roughly 13% a
year.  That is the yardstick any claim here has to be read against.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vibt import data as D, grid as G  # noqa: E402

pd.set_option("display.width", 210)
REPORTS = Path(__file__).resolve().parent.parent / "reports"


def annualise(curve: pd.Series, bars_per_year: float) -> float:
    if curve.iloc[-1] <= 0:
        return -1.0
    yrs = len(curve) / bars_per_year
    return float(curve.iloc[-1] ** (1 / yrs) - 1)


def summarise(name: str, r: G.GridResult, bpy: float) -> dict:
    return {
        "配置": name,
        "已实现年化": annualise(r.realised, bpy),
        "实际年化": annualise(r.mark, bpy),
        "已实现回撤": r.realised_dd,
        "**实际最大回撤**": r.mark_dd,
        "胜率": r.win_rate,
        "平仓次数": r.closed,
        "最深层数": r.max_levels_used,
        "爆仓": "是" if r.liquidated_at is not None else "",
        "爆仓时间": str(r.liquidated_at.date()) if r.liquidated_at is not None else "",
    }


def fmt(df: pd.DataFrame) -> str:
    d = df.copy()
    for c in ("已实现年化", "实际年化", "已实现回撤", "**实际最大回撤**", "胜率"):
        d[c] = d[c].map(lambda v: f"{v:+.1%}" if pd.notna(v) else "")
    return d.to_string(index=False)


def main() -> None:
    syms = D.available_symbols(require=("1h",))
    bpy = D.bars_per_year("1h")
    print("=" * 150)
    print(f"GRID / MARTINGALE 模拟   1h K线   可用币种 {len(syms)}")
    print("=" * 150)
    print("""
  规则：价格每跌 step 加一层，每层仓位是上一层的 mult 倍，
  每层各自在涨回自己入场价 +take_profit 时平仓获利。**亏损的层永不平仓。**
  盘中撮合按不利方向优先：先成交所有下跌加仓，再考虑上涨止盈。
""")

    frames = {s: D.load("1h", symbol=s) for s in syms}
    btc = frames["BTCUSDT"]

    # ---------------------------------------------------------------- 1
    print("=" * 150)
    print("1. BTCUSDT 全样本：层数越深，曲线越漂亮")
    print("=" * 150 + "\n")
    rows = []
    for lv in (5, 7, 9, 11, 13):
        p = G.GridParams(max_levels=lv)
        rows.append({**summarise(f"{lv} 层 (满仓需 {p.committed:.1f}x 本金)",
                                 G.run_grid(btc, p), bpy)})
    print(fmt(pd.DataFrame(rows)))
    print("""
  「已实现回撤」几乎为零、胜率 100% —— 这不是业绩，这是马丁的定义：
  亏损的层没有平仓，所以从不进入已实现盈亏。
  真正决定生死的是「实际最大回撤」那一列。""")

    # ---------------------------------------------------------------- 2
    print("\n" + "=" * 150)
    print("2. 同一套参数，跑遍所有币种（7 层，倍数 2.0）")
    print("=" * 150 + "\n")
    rows = []
    for s in syms:
        rows.append({**{"币种": s},
                     **summarise("", G.run_grid(frames[s], G.GridParams()), bpy)})
    t = pd.DataFrame(rows).drop(columns=["配置"])
    t = t.sort_values("**实际最大回撤**")
    print(fmt(t))
    dead = (t["爆仓"] == "是").sum()
    print(f"""
  {len(t)} 个币里 {dead} 个爆仓。
  注意「已实现年化」那一列在爆仓的币上依然很好看——直到归零那一刻为止。""")

    # ---------------------------------------------------------------- 3
    print("\n" + "=" * 150)
    print("3. 单边行情：马丁的死穴")
    print("=" * 150 + "\n")
    windows = [("2024-03-14", "2024-05-01", "2024 春季回调"),
               ("2024-07-29", "2024-08-06", "2024-08-05 日元套息崩盘"),
               ("2025-01-20", "2025-04-10", "2025 一季度下跌"),
               ("2026-01-01", "2026-08-14", "2026 年至今")]
    rows = []
    for a, b, lbl in windows:
        sub = btc.loc[a:b]
        if len(sub) < 50:
            continue
        move = sub["close"].iloc[-1] / sub["open"].iloc[0] - 1
        r = G.run_grid(sub, G.GridParams())
        rows.append({"时段": lbl, "BTC 涨跌": f"{move:+.1%}",
                     "已实现": f"{r.realised.iloc[-1]-1:+.1%}",
                     "**实际**": f"{r.mark.iloc[-1]-1:+.1%}",
                     "实际最深回撤": f"{r.mark_dd:+.1%}",
                     "用到层数": r.max_levels_used,
                     "爆仓": "是" if r.liquidated_at else ""})
    print(pd.DataFrame(rows).to_string(index=False))

    # ---------------------------------------------------------------- 4
    print("\n" + "=" * 150)
    print("4. 网格能扛多大的单边行情")
    print("=" * 150 + "\n")
    print(f"  {'层数':>5}{'覆盖跌幅':>10}{'满仓杠杆':>10}{'跌到底时的浮亏':>16}")
    for lv in (5, 7, 9, 11, 13, 15):
        p = G.GridParams(max_levels=lv)
        cover = 1 - (1 - p.step) ** (lv - 1)
        # average entry sits above the bottom; loss when price reaches the last rung
        w = np.array([p.mult ** i for i in range(lv)])
        px = np.array([(1 - p.step) ** i for i in range(lv)])
        avg = (w * px).sum() / w.sum()
        loss = p.committed * (px[-1] / avg - 1)
        print(f"  {lv:>5}{cover:>10.1%}{p.committed:>10.1f}x{loss:>16.1%}")
    print("""
  「覆盖跌幅」是网格铺满的价格区间，超出之后机器人只能干等着。
  「跌到底时的浮亏」按本金计——超过 −100% 就是爆仓。
  加层数能让曲线更漂亮，代价是杠杆指数增长：每加一层，满仓资金需求翻倍。""")

    # ---------------------------------------------------------------- 5
    print("\n" + "=" * 150)
    print("5. 对照：本项目验证过的横截面策略")
    print("=" * 150)
    print("""
  样本外 37 个从未参与开发的币，全截面 rank 加权：
    年化 +13.2%   波动 20%   最大回撤 −18.4%   Sharpe +0.72
    —— 这个 −18.4% 是**实际**回撤，没有藏在浮亏里的东西。

  网格/马丁的 −18.4% 对应的是「已实现回撤」那一列，通常接近 0%。
  两个数字长得像，含义完全不同：一个是你真实经历过的最大痛苦，
  另一个是把痛苦挪到未平仓头寸里之后剩下的部分。
""")
    REPORTS.mkdir(exist_ok=True)
    t.to_csv(REPORTS / "grid_martingale.csv", index=False)
    print(f"  wrote {REPORTS/'grid_martingale.csv'}")


if __name__ == "__main__":
    main()
