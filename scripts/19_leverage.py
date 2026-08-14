"""How much leverage, and what it does to the outcome.

Leverage multiplies the mean linearly and the variance quadratically, so the
growth rate is a downward parabola in leverage: past a point, more leverage
means LESS money, not more risk for more reward.  Where that point sits depends
on the true Sharpe -- which here is only known to within [+0.28, +1.96].

So this script does two things: it shows what each leverage did on the realised
path, and it shows what each leverage does when the true edge is drawn from the
uncertainty band rather than assumed to be the point estimate.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vibt import backtest as B, data as D, metrics as M, xsec as X  # noqa: E402

pd.set_option("display.width", 200)

COSTS = B.Costs(fee_bps=4.5, slip_bps=2.0)
ANN = 365.0
LOOKBACK, N_SIDE, REBAL = 14, 5, 3


def main() -> None:
    syms = D.available_symbols(require=("1d",))
    U = D.load_universe(syms, tfs=("1d",))
    px = X.price_panel(U, "1d", "open")
    feat = X.feature_panel(U, lambda d: d["close"] / d["close"].shift(LOOKBACK) - 1)
    w = X.cross_sectional_weights(feat, n_side=N_SIDE, mode="long_short")
    res = X.run(px, w, COSTS, ANN, "", REBAL)
    r = res.rets.dropna()

    mu_a = float(r.mean() * ANN)            # arithmetic annual mean
    sd_a = float(r.std(ddof=1) * np.sqrt(ANN))
    sharpe = mu_a / sd_a
    kelly = mu_a / sd_a**2
    p5, _, p95 = M.block_bootstrap_sharpe(r, ANN, block=20, n_boot=5000)

    print("=" * 120)
    print("1. 理论最优杠杆（Kelly）")
    print("=" * 120)
    print(f"""
  算术年均收益 μ = {mu_a:.1%}      年化波动 σ = {sd_a:.1%}      Sharpe = {sharpe:.2f}

  增长最优杠杆  f* = μ / σ² = {mu_a:.3f} / {sd_a:.3f}² = {kelly:.2f}x

  但 Kelly 公式假设你**知道**真实的 Sharpe。我们只知道它在 [{p5:+.2f}, {p95:+.2f}] 之间：
""")
    for s, tag in ((p5, "区间下沿"), (sharpe, "点估计"), (p95, "区间上沿")):
        k = (s * sd_a + 0) / sd_a**2
        print(f"    Sharpe {s:+5.2f}（{tag}）  ->  Kelly 杠杆 {k:.2f}x")
    print(f"""
  也就是说 Kelly 杠杆本身在 {p5*sd_a/sd_a**2:.1f}x ~ {p95*sd_a/sd_a**2:.1f}x 之间摇摆。

  关键的非对称性：**超过 2 倍 Kelly，期望增长率变成负数。**
  如果真实 Kelly 是 {p5*sd_a/sd_a**2:.1f}x 而你按点估计上了 {kelly:.1f}x，
  你就在 {kelly/(p5*sd_a/sd_a**2):.1f} 倍 Kelly 上——已经越过归零线。
""")

    print("=" * 120)
    print("2. 各档杠杆在这条真实路径上的结果")
    print("=" * 120)
    print(f"\n  {'杠杆':<6}{'CAGR':>9}{'年化波动':>10}{'最大回撤':>10}{'最差单日':>10}"
          f"{'Calmar':>8}{'10万→':>12}{'爆仓?':>8}")
    rows = []
    for L in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0):
        lr = L * r
        # np.cumprod on a Series keeps the DatetimeIndex, so positional access
        # would be read as a date lookup.  Work in numpy for the path maths.
        eq = np.cumprod(1 + lr.to_numpy())
        wiped = bool((1 + lr).min() <= 0) or bool(eq.min() <= 0.01)
        if wiped:
            first = int(np.argmax((np.cumprod(np.maximum(1 + lr, 1e-9)) <= 0.01)))
            print(f"  {L:<6.1f}{'—':>9}{'—':>10}{'—':>10}{lr.min():>9.1%}"
                  f"{'—':>8}{'归零':>12}{'是':>8}   "
                  f"（第 {first} 天，{r.index[first].date()}）")
            rows.append({"lev": L, "wiped": True})
            continue
        st = M.compute(pd.Series(lr, index=r.index), ANN)
        rows.append({"lev": L, "cagr": st.cagr, "dd": st.max_dd, "wiped": False})
        print(f"  {L:<6.1f}{st.cagr:>+9.1%}{st.ann_vol:>10.1%}{st.max_dd:>10.1%}"
              f"{lr.min():>10.1%}{st.calmar:>8.2f}{100_000*eq[-1]:>12,.0f}{'否':>8}")  # noqa: E501

    print("""
  注意 CAGR 不是线性放大的：2x 的 CAGR 不到 1x 的两倍，
  因为波动拖累（-L²σ²/2）是平方项。杠杆越高，这一项吃得越狠。""")

    print("\n" + "=" * 120)
    print("3. 如果真实边际在区间下沿会怎样")
    print("=" * 120)
    print("""
  把日收益按 (目标Sharpe / 实际Sharpe) 缩放均值，保持波动不变，
  模拟"真实边际只有区间下沿那么强"的路径：
""")
    print(f"  {'假设真实Sharpe':<16}{'杠杆':<8}{'CAGR':>9}{'最大回撤':>10}{'3.6年后10万变成':>16}")
    for target in (p5, 0.5, sharpe):
        # `target` is an ANNUAL Sharpe; r.std() is a DAILY std.  Setting the daily
        # mean to target*daily_std would imply an annual Sharpe of target*sqrt(365),
        # which is where the absurd four-digit CAGRs came from.
        scaled = r - r.mean() + (target / np.sqrt(ANN)) * r.std(ddof=1)
        for L in (1.0, 2.0, 3.0):
            lr = L * scaled
            if (1 + lr).min() <= 0:
                print(f"  {target:<16.2f}{L:<8.1f}{'归零':>9}{'—':>10}{'0':>16}")
                continue
            st = M.compute(pd.Series(lr, index=r.index), ANN)
            eq = 100_000 * float(np.prod(1 + lr.to_numpy()))
            assert abs(st.sharpe - target) < 0.02, \
                f"rescaled series should have Sharpe {target}, got {st.sharpe}"
            print(f"  {target:<16.2f}{L:<8.1f}{st.cagr:>+9.1%}{st.max_dd:>10.1%}{eq:>16,.0f}")

    print("\n" + "=" * 120)
    print("4. 回撤的分布，不是只看历史最差那一次")
    print("=" * 120)
    rng = np.random.default_rng(7)
    arr = r.to_numpy()
    n = len(arr)
    block = 20
    print(f"\n  对每档杠杆做 2000 次分块 bootstrap，看最大回撤的分布：")
    print(f"  {'杠杆':<6}{'回撤中位':>10}{'p90':>9}{'p99':>9}{'超过-50%的概率':>16}{'归零概率':>10}")
    for L in (1.0, 1.5, 2.0, 3.0, 4.0):
        dds, ruin = [], 0
        for _ in range(2000):
            nb = int(np.ceil(n / block))
            starts = rng.integers(0, n - block, size=nb)
            idx = (starts[:, None] + np.arange(block)).ravel()[:n]
            path = L * arr[idx]
            if (1 + path).min() <= 0:
                ruin += 1
                dds.append(-1.0)
                continue
            eq = np.cumprod(1 + path)
            dds.append(float((eq / np.maximum.accumulate(eq) - 1).min()))
        dds = np.array(dds)
        print(f"  {L:<6.1f}{np.median(dds):>10.1%}{np.percentile(dds,10):>9.1%}"
              f"{np.percentile(dds,1):>9.1%}{(dds<-0.5).mean():>16.0%}{ruin/2000:>10.1%}")

    print("\n" + "=" * 120)
    print("5. 杠杆还会放大两件回测没算的成本")
    print("=" * 120)
    years = len(r) / ANN
    cost_yr = res.costs.sum() / years
    print(f"""
  a) 手续费：现在 {cost_yr:.2%}/年，L 倍杠杆就是 {cost_yr:.2%} × L。
     3x 杠杆下光手续费就是 {cost_yr*3:.1%}/年。
  b) 资金费率：回测设成 0。多空对冲会抵消大部分，但不完全，
     而且它同样按 L 倍放大。这是最大的未知数。
  c) 强平：币安按维持保证金算，10 个仓位、总敞口 {1.0:.0%} × L。
     L=3 时总敞口 300%，单个币 30% 净值。某个山寨币一天 +50%（样本里 WLD 真发生过 +55%），
     那一条腿就是 -15% 净值，还没算其他 9 条。
""")

    print("=" * 120)
    print("6. 我的建议")
    print("=" * 120)
    print(f"""
  Kelly 说 {kelly:.1f}x。**我建议 1x，最多 1.5x，起步用 0.5x。** 理由：

  1. Kelly 的输入是"真实 Sharpe"，而我们的估计区间是 [{p5:+.2f}, {p95:+.2f}]。
     按下沿算 Kelly 只有 {p5*sd_a/sd_a**2:.1f}x。上 {kelly:.1f}x 就是 {kelly/(p5*sd_a/sd_a**2):.1f} 倍 Kelly，
     **期望增长率为负**。

  2. Deflated Sharpe 只有 32.7%——大约三分之二的概率这个边际根本不存在。
     对一个可能不存在的边际加杠杆，是把"慢慢亏手续费"变成"快速归零"。

  3. 1x 已经够了。真边际成立的话 1x 给你 {mu_a - sd_a**2/2:.0%}/年；
     边际只有下沿那么强，1x 给你 +9%/年，你还活着；
     边际是零，1x 每年亏 {cost_yr:.1%} 手续费，你有足够时间发现并停手。

  4. 这个策略**已经内含了一层杠杆**：100% 总敞口 = 50% 多 + 50% 空。
     再乘 3x 就是 300% 总敞口。别把"1x 杠杆"理解成"没上杠杆"。

  一句话：**先用 0.5x 跑三到六个月纸面 + 小资金，把资金费率和真实滑点测出来，
  再决定要不要上到 1x。在 deflated Sharpe 过 50% 之前，我不会建议任何超过 1.5x 的数字。**
""")


if __name__ == "__main__":
    main()
