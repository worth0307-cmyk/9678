"""按币安真实费率 + 真实资金费率重算净收益.

    python scripts/47_real_costs.py

在此之前所有净额都用一个 6.5bp/边 的假设，而且**完全没有算资金费率**。这两件事
都要改：

  手续费   币安 U 本位永续，普通用户、无 BNB 折扣：挂单 0.0200%（2.0bp）、
           吃单 0.0500%（5.0bp）。之前那个 6.5bp 等于「全吃单 + 1.5bp 滑点」。
           挂单和吃单差 3bp/边，在 65 次/年的换手下就是 **3.9%/年** —— 比这个
           策略的全部预期收益还大。下单方式不是细节，它是主要变量之一。

  资金费率 这才是漏掉的大项。账本是多空各半、天天持仓，每 8 小时（TAO/HYPE 是
           4 小时）结算一次。多头在费率为正时付钱，空头收钱。而动量策略**做多
           刚涨过的币**，那恰好是费率最容易为正的时候 —— 方向上先天不利。
           六个币的样本内均值：BTC +6.6%/年、ETH +6.0%、HYPE +8.6%、TAO +3.3%，
           SOL −5.0%、BNB −5.2%（年化，按满仓一条腿计）。

对齐方式和回测一致：`held.loc[t]` 是在 t 收盘形成的仓位，持有 (t, t+1]，
所以它承担的是 **t+1 那一天**结算的资金费率。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vibt import ingest as I, paper as PP  # noqa: E402

pd.set_option("display.width", 200)

MAKER_BPS = 2.0      # 币安 U 本位永续，普通用户，无 BNB 折扣
TAKER_BPS = 5.0
MAJORS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT")


def funding_daily(coins) -> pd.DataFrame:
    """每个币每日结算的资金费率合计（多头为正 = 多头付出）。"""
    out = {}
    for s in coins:
        p = Path("data/funding") / f"{s}_funding.csv.gz"
        d = I._load_funding(p)
        r = d["funding_rate"].astype(float)
        out[s] = r.groupby(r.index.floor("D")).sum()
    return pd.DataFrame(out).sort_index()


def evaluate(label: str, fee_bps: float, slip_bps: float, **kw) -> dict:
    b = PP.backtest(PP.Params(), **kw)
    held, pnl, turn = b["held"], b["pnl"], b["turnover"]

    fd = funding_daily(b["coins"]).reindex(columns=list(held.columns))
    fwd_f = fd.reindex(held.index).shift(-1).fillna(0.0)
    # 多头付、空头收：所以 P&L = -(w * rate)
    fund = -(held * fwd_f).sum(axis=1).reindex(pnl.index).fillna(0.0)

    cost = turn * (fee_bps + slip_bps) * 1e-4
    net = pnl - cost + fund
    years = len(pnl) / 365.0
    ann = lambda x: float(x.mean() * 365.0)  # noqa: E731
    sd = float(net.std())
    return {
        "版本": label,
        "毛收益": ann(pnl),
        "资金费率": ann(fund),
        "手续费+滑点": -float(cost.sum() / years),
        "净收益": ann(net),
        "净Sharpe": float(net.mean() / sd * np.sqrt(365.0)) if sd > 0 else np.nan,
    }


def main() -> None:
    print("=" * 104)
    print("按币安真实费率 + 真实资金费率重算")
    print("=" * 104)
    print(f"""
  费率（U 本位永续 / 普通用户 / 无 BNB 折扣）  挂单 {MAKER_BPS}bp   吃单 {TAKER_BPS}bp
""")

    print("-" * 104)
    print("A  资金费率这一项本身有多大（在跑的那本账上）")
    print("-" * 104 + "\n")
    b = PP.backtest(PP.Params())
    fd = funding_daily(b["coins"]).reindex(columns=list(b["held"].columns))
    fwd_f = fd.reindex(b["held"].index).shift(-1).fillna(0.0)
    per = -(b["held"] * fwd_f)
    years = len(b["pnl"]) / 365.0
    rows = []
    for c in per.columns:
        rows.append({"币": c, "资金费率年化": float(per[c].sum() / years),
                     "多头天数": int((b["held"][c] > 0).sum()),
                     "空头天数": int((b["held"][c] < 0).sum())})
    t = pd.DataFrame(rows).set_index("币").sort_values("资金费率年化")
    print(t.assign(资金费率年化=t["资金费率年化"].map("{:+.2%}".format)).to_string())
    tot = float(per.sum(axis=1).sum() / years)
    print(f"\n  合计 {tot:+.2%}/年。"
          f"{'这是净收入' if tot > 0 else '这是净支出'} —— "
          f"{'空头腿收到的多于多头腿付出的' if tot > 0 else '多头腿付出的多于空头腿收到的'}。")

    print("\n" + "-" * 104)
    print("B  下单方式的代价（在跑的那本账，含资金费率）")
    print("-" * 104 + "\n")
    scen = []
    for lbl, fee, slip in (
        ("全挂单 + 0bp 滑点（最好情况）", MAKER_BPS, 0.0),
        ("全挂单 + 1bp 滑点", MAKER_BPS, 1.0),
        ("一半挂单一半吃单 + 1bp", (MAKER_BPS + TAKER_BPS) / 2, 1.0),
        ("全吃单 + 1.5bp（旧的 6.5bp 假设）", TAKER_BPS, 1.5),
        ("全吃单 + 3bp（TAO/HYPE 更现实）", TAKER_BPS, 3.0),
    ):
        scen.append(evaluate(lbl, fee, slip))
    s = pd.DataFrame(scen).set_index("版本")
    print(s.assign(**{c: s[c].map("{:+.1%}".format)
                      for c in ("毛收益", "资金费率", "手续费+滑点", "净收益")},
                   净Sharpe=s["净Sharpe"].map("{:+.2f}".format)).to_string())

    print("\n" + "-" * 104)
    print("C  后见之明最少的那个版本，同样口径")
    print("-" * 104 + "\n")
    scen2 = [evaluate("4主流/等风险 · 全挂单+1bp", MAKER_BPS, 1.0,
                      coins=MAJORS, size="risk"),
             evaluate("4主流/等风险 · 全吃单+1.5bp", TAKER_BPS, 1.5,
                      coins=MAJORS, size="risk")]
    s2 = pd.DataFrame(scen2).set_index("版本")
    print(s2.assign(**{c: s2[c].map("{:+.1%}".format)
                       for c in ("毛收益", "资金费率", "手续费+滑点", "净收益")},
                    净Sharpe=s2["净Sharpe"].map("{:+.2f}".format)).to_string())

    best = s["净收益"].max()
    worst = s["净收益"].min()
    print(f"""
--------------------------------------------------------------------------------------------------------
D  结论
--------------------------------------------------------------------------------------------------------

  **光是下单方式，就在样本内的净收益上造成 {best - worst:.1%} 的差距**
  （{worst:+.1%} 到 {best:+.1%}）。挂单 2bp 对吃单 5bp，一年 65 次换手，
  3bp/边 × 65 = 3.9%/年 —— 比这个策略折算后的全部预期收益还大。

  ⚠️  但挂单不是免费的午餐：限价单会挂不上（信号日却没进场）、会被逆向选择
      （价格朝不利方向动的时候才成交）。这两样都不在上面的数字里，而它们
      恰恰只有真实下单才量得出来 —— **这正是纸面日志现在要回答的问题**。

  注意上面所有数字仍然是**样本内**的：币是挑过的、参数是挑过的、仓位重压在
  最波动的名字上。46 号脚本把那三层折算掉之后的前瞻中枢是 Sharpe 0.1~0.4。
  这里换成真实费率之后，那张表要重算 —— 见下面打印的修正值。""")

    # 把 46 号的混合估计按真实费率重算
    live = s.loc["全挂单 + 1bp 滑点"]
    vol = 0.256
    turn = 64.6
    for lbl, fee, slip in (("挂单 2bp + 1bp 滑点", MAKER_BPS, 1.0),
                           ("吃单 5bp + 1.5bp 滑点", TAKER_BPS, 1.5)):
        c = turn * (fee + slip) * 1e-4
        f = live["资金费率"]
        mu = 0.70 * 0.30 * vol + f - c
        print(f"""
  【{lbl}】 期望 Sharpe +0.21 → 毛 {0.70*0.30*vol:+.1%}
      资金费率 {f:+.1%}   手续费+滑点 {-c:+.1%}   →  **前瞻净额 {mu:+.1%}/年**""")


if __name__ == "__main__":
    main()
