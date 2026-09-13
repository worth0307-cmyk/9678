"""回测**正在纸面跑的那个配置本身** —— vibt.paper.Params()，一个参数都不改.

    python scripts/44_paper_config.py

为什么单独写一个：43 号脚本每天生成持仓，但那个配置从来没有被端到端回测过。
`paper.py` 的文档里写着「打平成本 6.38bp 对收 6.5bp，所以成不成立取决于真实成交」
—— 那个数字是 39/40 号脚本里 **PCA 残差反转** 在 200 个币上的结果，而纸面在跑的
是 `pca.zscore`（**未对冲**的波动率归一化收益）的横截面动量。两件不同的事，
经济学也不同：残差反转一年换手 441 次，这个每 3 天调一次仓。

把别的策略的打平成本写在这个策略的文档里，会让人把力气用错地方：
如果成本根本不是约束，那么攒成交记录就不是当前最该做的事。

口径和 38/40 号脚本一致：换手按单边 sum|Δw| 计，打平成本 = 年化毛收益 / 年换手。

**这六个币是开发集**，所以下面每个数都是样本内的。它回答的是「成本是不是约束」，
不是「这套东西有没有用」。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vibt import paper as PP  # noqa: E402

def run(p: PP.Params, rebalance_days: int) -> dict:
    b = PP.backtest(p, rebalance_days)
    s = PP.stats(b["pnl"], b["turnover"], p.assumed_cost_bps)
    return dict(days=s["days"], years=s["years"], gross_ann=s["gross_ann"],
                sharpe=s["sharpe"], turn_ann=s["turn_ann"], be_bps=s["be_bps"],
                net65=s["net_ann"], first=b["pnl"].index[0].date(),
                last=b["pnl"].index[-1].date())


def main() -> None:
    p = PP.Params()
    print("=" * 96)
    print("纸面配置的成本经济学")
    print("=" * 96)
    print(f"""
  币种        {' '.join(p.coins)}
  动量窗口    {p.lookback} 天（对 z 分数求和）
  归一化窗口  {p.pca_window} 天
  再平衡      每 {p.rebalance_days} 天
  总敞口      {p.gross}
  回测假设    {p.assumed_cost_bps}bp/边
""")

    rows = []
    for rd in (1, 2, 3, 5, 7, 10):
        r = run(p, rd)
        r["rd"] = rd
        rows.append(r)
    t = pd.DataFrame(rows).set_index("rd")

    live = t.loc[p.rebalance_days]
    print(f"  样本 {live['first']} -> {live['last']}"
          f"   {live['days']} 天 / {live['years']:.2f} 年\n")
    out = t[["gross_ann", "sharpe", "turn_ann", "be_bps", "net65"]].copy()
    out.columns = ["年化毛收益", "Sharpe", "年换手(单边)", "打平成本bp", "收6.5bp后净额"]
    out["年化毛收益"] = out["年化毛收益"].map(lambda v: f"{v:+.1%}")
    out["收6.5bp后净额"] = out["收6.5bp后净额"].map(lambda v: f"{v:+.1%}")
    out["Sharpe"] = out["Sharpe"].map(lambda v: f"{v:+.2f}")
    out["年换手(单边)"] = out["年换手(单边)"].map(lambda v: f"{v:.0f}x")
    out["打平成本bp"] = out["打平成本bp"].map(lambda v: f"{v:.1f}")
    out.index.name = "再平衡(天)"
    print(out.to_string())

    print(f"""
  在跑的是 {p.rebalance_days} 天这一行：打平成本 {live['be_bps']:.1f}bp/边，
  而回测假设 {p.assumed_cost_bps}bp。成本要比假设差 {live['be_bps']/p.assumed_cost_bps:.0f} 倍
  才能把这个策略吃掉。

  这和 `paper.py` 文档里写的「6.38bp 对 6.5bp」**不是一回事**。那个数字属于
  PCA 残差反转（39/40 号脚本，200 个币，年换手 441 次），不属于这里在跑的东西。
  两者的约束完全不同：残差反转卡在成本上，这个不卡。

  ⚠️  这六个币是开发集，上面每个数都是样本内的。成本不是约束，不代表它有用 ——
      只说明**下一步该去证伪的是选择偏差，不是滑点**。""")


if __name__ == "__main__":
    main()
