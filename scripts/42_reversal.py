"""胜率低的逻辑，反着做行不行.

The intuition is that a rule which loses must contain information worth having
in the other direction.  The arithmetic says otherwise, and one identity settles
most of it:

    净收益(正向) + 净收益(反向) = -2 x 成本

Whatever the rule is worth, you pay the spread going in and out on both sides,
so the two directions cannot both be positive and typically neither is.  For the
reversal to pay, the original must lose by MORE than the round trip costs on a
GROSS basis -- losing a little is the case where inverting changes nothing.

That gives a dead zone of width 2c/s around zero gross edge.  Most failed rules
land inside it, because most of them fail by paying fees on a signal worth
nothing rather than by being reliably wrong.

Two things this file does not conflate:

  低胜率      is not the same as negative expectancy.  30% wins at 1:3 is a
              PROFITABLE rule; inverting it would be a disaster.  The quantity
              to invert on is expectancy, never the win rate.
  反向        of a bracket strategy is not a mirror image.  Taking the other
              side of the same levels turns 1:b into 1:(1/b) and p into 1-p, so
              reversing a low-win-rate/high-payoff rule gives a high-win-rate
              rule with a small payoff -- and a much wider stop.

This repository has run the experiment once already: the VI band rule was
inverted and Sharpe went from -1.05 to +0.80 (REPORT_FOLLOW.md).  That is the
case where the original lost by enough.  It still did not become tradeable, for
reasons the last section recalls.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vibt import backtest as B, data as D, edge as E, masys as MS  # noqa: E402

pd.set_option("display.width", 230)
REPORTS = Path(__file__).resolve().parent.parent / "reports"
COINS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "TAOUSDT", "HYPEUSDT")
COSTS = B.Costs(4.5, 2.0)
FREE = B.Costs(0, 0)


# ------------------------------------------------------------------ 仓位型
def macd_pos(close: pd.Series) -> pd.Series:
    m = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    return (m > m.ewm(span=9, adjust=False).mean()).astype(float) * 2 - 1


def mom_pos(close: pd.Series, n: int = 14) -> pd.Series:
    return np.sign(close / close.shift(n) - 1).fillna(0.0)


POS_RULES = {"MACD 金叉多/死叉空": macd_pos, "14日动量 顺势": mom_pos}


def run_pos(name: str, fn, tf: str) -> dict:
    """Forward and reversed, at zero cost and at 6.5bp, pooled over the coins."""
    ann = D.bars_per_year(tf)
    acc = {k: [] for k in ("f0", "r0", "f1", "r1", "cost")}
    for s in COINS:
        df = D.load(tf, symbol=s)
        pos = fn(df["close"]).reindex(df.index).fillna(0.0)
        acc["f0"].append(B.run(df, pos, FREE, ann).rets)
        acc["r0"].append(B.run(df, -pos, FREE, ann).rets)
        f1 = B.run(df, pos, COSTS, ann)
        acc["f1"].append(f1.rets)
        acc["r1"].append(B.run(df, -pos, COSTS, ann).rets)
        acc["cost"].append(f1.costs)
    out = {k: pd.concat(v, axis=1).mean(axis=1) for k, v in acc.items()}
    yrs = len(out["f0"]) / ann
    ann_of = lambda r: (1 + r).prod() ** (1 / yrs) - 1        # noqa: E731
    return {"规则": name, "周期": tf,
            "正向零成本": ann_of(out["f0"]), "反向零成本": ann_of(out["r0"]),
            "正向6.5bp": ann_of(out["f1"]), "反向6.5bp": ann_of(out["r1"]),
            "年化成本": out["cost"].sum() / yrs,
            # the identity: the two net series must sum to minus twice the cost
            "恒等式残差": float((out["f1"] + out["r1"] + 2 * out["cost"]).abs().max())}


# ------------------------------------------------------------------ 括号型
def flip(sig: list[MS.Signal], df: pd.DataFrame, p: MS.MaParams) -> list[MS.Signal]:
    """The other side of the same trade: stop and target swap places."""
    op = df["open"].to_numpy(float)
    out = []
    for s in sig:
        fill = op[s.i]
        if not np.isfinite(fill):
            continue
        risk = abs(fill - s.stop)
        tgt = fill + s.side * p.target_r * risk
        # reversed: entry unchanged, stop where the target was, target at the old stop
        out.append(MS.Signal(s.i, -s.side, tgt, "反向", s.stop))
    return out


def run_bracket(name: str, fn, tf: str, p: MS.MaParams) -> dict:
    f, r = [], []
    for s in COINS:
        df = D.load(tf, symbol=s)
        if len(df) < max(p.lens) + p.window:
            continue
        sig = fn(df, p)
        if not sig:
            continue
        f.append(MS.evaluate(df, sig, p))
        r.append(MS.evaluate(df, flip(sig, df, p), p))
    if not f:
        return {}
    F, R = pd.concat(f, ignore_index=True), pd.concat(r, ignore_index=True)
    # cost in R differs between the two because the R unit itself changed
    cf = 2 * p.fee / F.risk_pct.median()
    cr = 2 * p.fee / R.risk_pct.median()
    return {"规则": name, "周期": tf, "笔数": len(F),
            "正向胜率": float((F.r_multiple > 0).mean()),
            "正向净期望R": float(F.r_multiple.mean()),
            "正向毛期望R": float(F.r_multiple.mean()) + cf,
            "反向胜率": float((R.r_multiple > 0).mean()),
            "反向净期望R": float(R.r_multiple.mean()),
            "反向毛期望R": float(R.r_multiple.mean()) + cr,
            "正向成本R": cf, "反向成本R": cr,
            "正向止损": float(F.risk_pct.median()),
            "反向止损": float(R.risk_pct.median())}


def main() -> None:
    print("=" * 130)
    print("胜率低的逻辑，反着做行不行")
    print("=" * 130)
    print("""
  先把话分清楚：**低胜率不等于负期望**。
  30% 胜率配 1:3 是 +0.20R 的赚钱规则，反着做会亏死。
  能不能反，看的是期望，不是胜率。

  然后是那条恒等式：

      净收益(正向) + 净收益(反向) = -2 x 成本

  进出都要付点差，两个方向一起算永远是负的。所以反向要能赚，
  原策略必须在**毛**口径上亏得比手续费还多。亏一点点的那种，反过来还是亏。
""")

    # ------------------------------------------------------------------ ① 恒等式
    print("=" * 130)
    print("① 先验证恒等式（顺便证明模拟没写错）")
    print("=" * 130 + "\n")
    rows = []
    for tf in ("1d", "4h", "1h"):
        for name, fn in POS_RULES.items():
            rows.append(run_pos(name, fn, tf))
    d = pd.DataFrame(rows)
    o = d.copy()
    for c in ("正向零成本", "反向零成本", "正向6.5bp", "反向6.5bp", "年化成本"):
        o[c] = o[c].map(lambda v: f"{v:+.1%}")
    o["恒等式残差"] = o["恒等式残差"].map(lambda v: f"{v:.1e}")
    print(o.to_string(index=False))
    print(f"""
  「恒等式残差」全部是机器精度量级 —— 两个方向的净收益之和确实等于 -2x成本。
  看「正向6.5bp」和「反向6.5bp」两列：**没有任何一行两边都为正**，这是必然的。
""")

    # ------------------------------------------------------------------ ② 死区
    print("=" * 130)
    print("② 关键是「毛」口径亏多少 —— 亏得不够多，反过来也没用")
    print("=" * 130 + "\n")
    rows = []
    for _, r in d.iterrows():
        gross = r["正向零成本"]
        cost = r["年化成本"]
        if r["正向6.5bp"] > 0:
            verdict = "正向本来就赚，不存在反的问题"
        elif r["反向6.5bp"] > 0:
            verdict = "反向可赚"
        elif abs(gross) < cost:
            verdict = "死区：两边都亏"
        else:
            verdict = "反向仍亏"
        rows.append({"规则": r["规则"], "周期": r["周期"],
                     "正向毛年化": gross, "年化成本": cost,
                     "正向净年化": r["正向6.5bp"],
                     "反向净年化": r["反向6.5bp"], "结论": verdict})
    o = pd.DataFrame(rows)
    for c in ("正向毛年化", "年化成本", "正向净年化", "反向净年化"):
        o[c] = o[c].map(lambda v: f"{v:+.1%}")
    print(o.to_string(index=False))
    print("""
  「死区」的定义：|毛收益| < 成本。落在里面的规则不是「反着做就能赚的输家」，
  而是**两个方向都没有东西**的规则——它亏钱是因为在付手续费，不是因为它可靠地错。
  绝大多数失败的策略都落在这里。
""")

    # ------------------------------------------------------------------ ③ 括号
    print("=" * 130)
    print("③ 括号型策略反向：不是镜像，赔率会倒过来")
    print("=" * 130 + "\n")
    p = MS.MaParams()
    rows = []
    for tf in ("1d", "4h"):
        for name, fn in (("六均线A 密集突破", MS.entries_cluster_break),
                         ("六均线B 回踩MA20", MS.entries_ma20_pullback)):
            r = run_bracket(name, fn, tf, p)
            if r:
                rows.append(r)
    b = pd.DataFrame(rows)
    o = b.copy()
    for c in ("正向胜率", "反向胜率", "正向止损", "反向止损"):
        o[c] = o[c].map(lambda v: f"{v:.1%}")
    for c in ("正向净期望R", "正向毛期望R", "反向净期望R", "反向毛期望R",
              "正向成本R", "反向成本R"):
        o[c] = o[c].map(lambda v: f"{v:+.3f}")
    print(o[["规则", "周期", "笔数", "正向胜率", "正向净期望R", "正向成本R",
             "反向胜率", "反向净期望R", "反向成本R", "正向止损", "反向止损"]]
          .to_string(index=False))
    print(f"""
  注意「反向止损」那一列：正向的止盈位变成了反向的止损位，所以止损宽了约
  {p.target_r:g} 倍，反向的每笔成本按 R 算反而**更便宜**。
  但赔率也从 1:{p.target_r:g} 变成 1:{1/p.target_r:.2f}，胜率从三成变成七成——
  **反过来之后是一个「高胜率、小赔率」的策略，不是原策略的镜像。**
""")

    # ------------------------------------------------------------------ ④ 理论
    print("=" * 130)
    print("④ 用算式直接看：什么样的输家值得反")
    print("=" * 130 + "\n")
    print(f"  {'正向胜率':>10}{'正向期望R':>12}{'反向胜率':>10}{'反向赔率':>10}"
          f"{'反向期望R':>12}{'值得反?':>10}")
    for win in (0.20, 0.24, 0.25, 0.30, 0.40):
        fwd = E.Edge(win, 3.0)
        rev = fwd.reversed_edge()
        worth = "是" if rev.expectancy > 0 else "否"
        print(f"  {win:>10.0%}{fwd.expectancy:>+12.3f}{rev.win:>10.0%}"
              f"{'1:' + format(rev.payoff, '.2f'):>10}{rev.expectancy:>+12.3f}{worth:>10}")
    print("""
  1:3 的策略在 25% 胜率处打平。低于 25% 正向亏，反向才开始赚——
  但要在**扣成本后**赚，还得再往下走一截，这段距离就是死区。
""")

    # ------------------------------------------------------------------ ⑤ 前例
    print("=" * 130)
    print("⑤ 本仓库唯一一次「反向成功」，以及它后来怎么了")
    print("=" * 130)
    print("""
  VI 通道那套（REPORT_FOLLOW.md）：把符号改对后 **Sharpe 从 -1.05 变成 +0.80**。
  它符合条件——原策略亏得够多（毛口径也是负的），不在死区里。

  但它仍然没有变成可交易的策略：
    · 打不赢等波动持有（只多 +1.8%/年）
    · 参数族中位 edge +0.03，Wilcoxon p=0.44
    · 逐币只有 12/26 为正
    · deflated Sharpe 21.9%

  **「反过来是正的」只是及格线，不是终点。** 它要过的还是同一套关：
  样本外、成本、参数高原、多重检验校正。
""")

    REPORTS.mkdir(exist_ok=True)
    d.to_csv(REPORTS / "reversal_position.csv", index=False)
    b.to_csv(REPORTS / "reversal_bracket.csv", index=False)
    print(f"  wrote {REPORTS/'reversal_position.csv'} 等 2 个文件")


if __name__ == "__main__":
    main()
