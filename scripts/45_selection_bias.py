"""纸面配置的 +1.20 Sharpe 里，有多少经得起戳.

    python scripts/45_selection_bias.py

44 号脚本证明了成本不是约束（打平 47.7bp 对收 6.5bp，7 倍余量）。那么约束是什么？
**是选择**：这六个币是在看过谁表现好之后挑出来的，`lookback=14 / pca_window=120 /
rebalance_days=3` 也全是在它们的历史上定的。样本内的 +1.20 对此毫无抵抗力。

干净的跨币种样本已经用光了——39/40 号脚本把 181 个从未参与开发的币用在了「波动率
归一化值 +0.68 Sharpe」那个发现上，而那正是这里在跑的东西的来源。所以拿新币种再
检验一次，是在用已经被同一个想法污染过的数据。**未来的时间是唯一还没被污染的资源，
而那要等。**

在等的同时，有四件不需要新数据就能做的事。它们不能证明策略有用，只能**证伪**：

  A  逐年拆开    整段样本的好看，可能全部来自其中一年
  B  留一法      整段样本的好看，可能全部来自一两个事后挑中的币
  C  参数邻域    选中的那组参数是站在高原上，还是站在一根针尖上
  D  去偏 Sharpe 在搜过这么多组参数之后，+1.20 还剩多少是信号

第四项只对**参数**搜索去偏。**选币的那次搜索没有被它覆盖**，而那一次的自由度
大得多——这是下面所有数字共同的、无法从这份数据里消除的上偏。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vibt import metrics as M, paper as PP  # noqa: E402

pd.set_option("display.width", 220)
ANN = 365.0


def by_year(b: dict) -> pd.DataFrame:
    rows = []
    for y, seg in b["pnl"].groupby(b["pnl"].index.year):
        s = PP.stats(seg, b["turnover"].reindex(seg.index))
        rows.append({"年": y, "天数": s["days"], "年化毛收益": s["gross_ann"],
                     "Sharpe": s["sharpe"]})
    return pd.DataFrame(rows).set_index("年")


def leave_one_out(p: PP.Params, base: dict) -> pd.DataFrame:
    rows = []
    for drop in p.coins:
        sub = tuple(c for c in p.coins if c != drop)
        b = PP.backtest(p, coins=sub)
        idx = b["pnl"].index[b["live"].reindex(b["pnl"].index).fillna(False)]
        if not len(idx):
            rows.append({"去掉": drop, "可比天数": 0})
            continue
        # 同一段日期上重算全样本，否则比的是两个不同的时期：去掉任何一个
        # 老币，2022~2024 就只剩 3 个名字，低于 min_names，那段直接空仓。
        s_loo = PP.stats(b["pnl"].loc[idx], b["turnover"].loc[idx])
        base_pnl = base["pnl"].reindex(idx).dropna()
        s_base = PP.stats(base_pnl, base["turnover"].reindex(base_pnl.index))
        rows.append({"去掉": drop, "可比天数": len(idx),
                     "6币Sharpe": s_base["sharpe"], "5币Sharpe": s_loo["sharpe"],
                     "差": s_loo["sharpe"] - s_base["sharpe"]})
    return pd.DataFrame(rows).set_index("去掉")


def contribution(b: dict) -> pd.DataFrame:
    """每个币在这本账里实际赚到/亏掉多少 —— 和留一法互为印证。"""
    held, fwd = b["held"], b["fwd"]
    tot = float((held * fwd).sum(axis=1).sum())
    rows = []
    for c in held.columns:
        pnl_c = (held[c] * fwd[c]).dropna()
        rows.append({"币": c, "累计贡献": float(pnl_c.sum()),
                     "占比": float(pnl_c.sum()) / tot if tot else np.nan,
                     "有仓天数": int((held[c].abs() > 0).sum())})
    return pd.DataFrame(rows).set_index("币").sort_values("累计贡献", ascending=False)


def neighbourhood(p: PP.Params, lookbacks, windows) -> pd.DataFrame:
    out = pd.DataFrame(index=lookbacks, columns=windows, dtype=float)
    for lb in lookbacks:
        for w in windows:
            q = PP.Params(coins=p.coins, lookback=lb, pca_window=w,
                          rebalance_days=p.rebalance_days, gross=p.gross,
                          min_names=p.min_names)
            b = PP.backtest(q)
            out.loc[lb, w] = PP.stats(b["pnl"], b["turnover"])["sharpe"]
    out.index.name = "动量窗口"
    out.columns.name = "归一化窗口"
    return out


def year_row(b: dict) -> str:
    out = []
    for y, seg in b["pnl"].groupby(b["pnl"].index.year):
        out.append(f"{y} {PP.stats(seg)['sharpe']:+.2f}")
    return "  ".join(out)


def variants(p: PP.Params) -> None:
    """两个「少用一点后见之明」的版本，和在跑的那个并排放。"""
    majors = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT")
    rows = []
    for label, kw in (
        ("在跑的：6 币 / 等名义", {}),
        ("只留 4 个主流币（2022 年就都在）", {"coins": majors}),
        ("6 币 / 等风险（按波动率缩放）", {"size": "risk"}),
    ):
        b = PP.backtest(p, **kw)
        s = PP.stats(b["pnl"], b["turnover"])
        rows.append({"版本": label, "Sharpe": s["sharpe"], "年化毛收益": s["gross_ann"],
                     "打平bp": s["be_bps"], "逐年": year_row(b)})
    t = pd.DataFrame(rows).set_index("版本")
    print(t.assign(Sharpe=t["Sharpe"].map("{:+.2f}".format),
                   年化毛收益=t["年化毛收益"].map("{:+.1%}".format),
                   打平bp=t["打平bp"].map("{:.1f}".format)).to_string())


def main() -> None:
    p = PP.Params()
    base = PP.backtest(p)
    s0 = PP.stats(base["pnl"], base["turnover"])

    print("=" * 100)
    print("纸面配置：+1.20 Sharpe 有多少经得起戳")
    print("=" * 100)
    print(f"\n  基准（样本内，2022-01-01 起 {s0['years']:.2f} 年）："
          f"Sharpe {s0['sharpe']:+.2f}   年化毛收益 {s0['gross_ann']:+.1%}"
          f"   打平成本 {s0['be_bps']:.1f}bp\n")

    print("-" * 100)
    print("A  逐年拆开")
    print("-" * 100)
    y = by_year(base)
    print(y.assign(**{"年化毛收益": y["年化毛收益"].map("{:+.1%}".format),
                      "Sharpe": y["Sharpe"].map("{:+.2f}".format)}).to_string())
    pos = int((y["Sharpe"] > 0).sum())
    print(f"\n  {pos}/{len(y)} 年为正。"
          f"最好的一年 {y['Sharpe'].idxmax()}（{y['Sharpe'].max():+.2f}），"
          f"最差 {y['Sharpe'].idxmin()}（{y['Sharpe'].min():+.2f}）")

    print("\n" + "-" * 100)
    print("B  留一法：拿掉任何一个币，还剩多少")
    print("-" * 100)
    loo = leave_one_out(p, base)
    print(loo.assign(**{c: loo[c].map("{:+.2f}".format)
                        for c in ("6币Sharpe", "5币Sharpe", "差")
                        if c in loo}).to_string())

    print("\n  同一本账里各币的累计贡献：")
    con = contribution(base)
    print(con.assign(累计贡献=con["累计贡献"].map("{:+.3f}".format),
                     占比=con["占比"].map("{:+.1%}".format)).to_string())

    print("\n" + "-" * 100)
    print("C  参数邻域（再平衡固定 3 天，格子里是 Sharpe）")
    print("-" * 100)
    lbs, wins = [7, 10, 14, 20, 30], [60, 90, 120, 180, 250]
    grid = neighbourhood(p, lbs, wins)
    print(grid.map("{:+.2f}".format).to_string())
    flat = grid.values.flatten()
    here = grid.loc[p.lookback, p.pca_window]
    rank = int((flat > here).sum()) + 1
    print(f"\n  在跑的那格 (lookback={p.lookback}, window={p.pca_window}) = {here:+.2f}，"
          f"在 {grid.size} 格里排第 {rank}")
    print(f"  全格范围 {flat.min():+.2f} ~ {flat.max():+.2f}，"
          f"中位 {np.median(flat):+.2f}，为正的格子 {int((flat > 0).sum())}/{grid.size}")

    print("\n" + "-" * 100)
    print("D  去偏 Sharpe：搜过这么多组之后，还剩多少是信号")
    print("-" * 100)
    r = base["pnl"]
    skew = float(r.skew())
    kurt = float(r.kurtosis() + 3.0)
    print(f"\n  日收益 偏度 {skew:+.2f}   峰度 {kurt:.2f}   样本 {len(r)} 天\n")
    print(f"  {'试过的组数':>12}   {'P(真实Sharpe>0)':>16}")
    for n in (10, 30, 50, 200, 1000):
        ds = M.deflated_sharpe(s0["sharpe"], n_trials=n, n_obs=len(r),
                               ann_factor=ANN, skew=skew, kurt=kurt)
        print(f"  {n:>12}   {ds:>16.3f}")
    print(f"""
  这张表只对**参数**搜索去偏：上面 C 就搜了 {grid.size} 格，加上 44 号的 6 个再平衡
  周期，光是写下来的就有 {grid.size + 6} 组。

  ⚠️  **选币那次搜索完全没有被覆盖。** 从两百多个永续里挑出六个，自由度比任何
      参数网格都大，而且那次挑选发生在看过结果之后。所以上面每个数字都还带着
      一个无法从这份数据里消掉的上偏 —— 能消掉它的只有未来的时间。""")

    print("\n" + "-" * 100)
    print("E  把后见之明一层层拿掉，还剩多少")
    print("-" * 100 + "\n")
    variants(p)
    print("""
  两条腿各占一半：

  **选币** 四个主流币是 2022 年就在场的，挑它们几乎不需要后见之明；TAO 和 HYPE
  是看过之后加进来的。只留四个主流币，Sharpe 从 +1.20 掉到 +0.57 —— 而且今年
  （2026）从 +2.30 掉到 **+0.06**。当下这一年的强势，全部来自那两个后加的币。

  **仓位** 信号是波动率归一化过的，**但仓位不是**：排名权重是等名义的，所以
  波动最大的那个名字自动主导整本账的盈亏，和排得准不准无关。改成等风险之后
  Sharpe 从 +1.20 掉到 +0.64。

  所以 +1.20 ≈ 「+0.6 的策略」+「+0.6 的『重仓压在最波动的东西上』」，
  而那 +0.6 的策略里又有一大半只存在于 TAO/HYPE 上市之后的时间里。""")


if __name__ == "__main__":
    main()
