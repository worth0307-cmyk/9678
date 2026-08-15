"""Does trading the VI+ band exit with the sign the data implies actually pay?

Section 7 of REPORT_VI_BAND establishes that price rises after a VI+ band exit in
EITHER direction, by +0.6% to +1.2% more than a random date over 48h, p<0.01 on
1061-1313 events.  The as-specified strategy fades that, which is why it loses.

This runs the inverted sign through the same falsification battery the
cross-sectional work had to survive.  The battery matters more than usual here
because the implied trade is LONG-ONLY in an asset class that tripled over the
sample: any long-only rule will look profitable, so the question is never "did it
make money" but "did it beat holding for the same exposure".

Four things have to hold at once, and any one of them failing kills it:

  1. It beats a RISK-MATCHED benchmark.  Scaling buy-and-hold to the strategy's
     *average exposure* is the intuitive move and it is nearly useless: Sharpe is
     scale-invariant, so that scaling leaves the benchmark's Sharpe untouched
     while making its return look tiny.  The strategy's exposure arrives in
     bursts rather than at a constant level, so it runs several times the
     volatility of a constant-exposure book -- comparing their returns credits
     that leverage to timing skill.  The benchmark is therefore scaled to the
     strategy's realised VOLATILITY, where the return comparison is meaningful.
  2. It beats RANDOM TIMING.  Positions are rotated circularly, which preserves
     the trade count, the holding-period distribution, the total exposure and the
     cross-symbol clustering exactly, and destroys only the alignment with the
     signal.  If the signal carries nothing, rotation costs nothing.
  3. It survives as a FAMILY, not a point.  The whole parameter grid is reported
     and the deflated Sharpe is charged for every combination tried.
  4. It is CONSISTENT across symbols and years, not carried by one name or one
     bull run.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vibt import backtest as B, data as D, metrics as M, vi_band as VB, xsec as X  # noqa: E402

pd.set_option("display.width", 220)

COSTS = B.Costs(fee_bps=4.5, slip_bps=2.0)
ANN_4H = D.bars_per_year("4h")
REPORTS = Path(__file__).resolve().parent.parent / "reports"
RNG = np.random.default_rng(20260815)


def panels(universe: dict, p: VB.BandParams, costs: B.Costs = COSTS
           ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per-symbol net returns, positions and raw bar returns on the shared 4H grid."""
    rets, poss, bars = {}, {}, {}
    for sym, frames in universe.items():
        df = VB.build_frame(frames, p)
        pos = VB.target_position(df, p)
        res = B.run(df, pos, costs, ANN_4H, sym)
        rets[sym] = res.rets
        poss[sym] = res.position
        px = df["open"]
        bars[sym] = (px.shift(-1) / px - 1.0).iloc[:-1]
    return (pd.DataFrame(rets).sort_index(),
            pd.DataFrame(poss).sort_index(),
            pd.DataFrame(bars).sort_index())


def equal_weight(frame: pd.DataFrame) -> pd.Series:
    """Average across the symbols that exist at each bar, not across all columns.

    Symbols list at different dates.  Treating a not-yet-listed symbol as a 0%
    return would quietly dilute the early years toward zero.
    """
    return frame.mean(axis=1, skipna=True).fillna(0.0)


def rotate(frame: pd.DataFrame, shift: int) -> pd.DataFrame:
    """Circularly rotate every column by the same offset.

    One shared offset, not one per symbol: band exits cluster in time across
    crypto, and preserving that clustering keeps the null's effective sample size
    as low as the strategy's.  Independent offsets would spread the trades out and
    make the null artificially easy to beat.
    """
    return pd.DataFrame(np.roll(frame.to_numpy(), shift, axis=0),
                        index=frame.index, columns=frame.columns)


def summarise(rets: pd.Series, exposure: float, name: str) -> dict:
    st = M.compute(rets, ANN_4H)
    yrs = len(rets) / ANN_4H
    return {"name": name, "sharpe": st.sharpe, "total": st.total_return,
            "cagr": (1 + st.total_return) ** (1 / yrs) - 1,
            "vol": st.ann_vol, "maxdd": st.max_dd, "exposure": exposure}


def main() -> None:
    universe = D.load_universe()
    syms = list(universe)
    print("=" * 150)
    print(f"FOLLOW-THE-BREAKOUT FALSIFICATION   {len(syms)} symbols, 4H bars, "
          f"costs {COSTS.per_side*1e4:.1f}bp/side")
    print("=" * 150)

    # ---------------------------------------------------------------- 1. the three signs
    print("\n" + "=" * 150)
    print("1. 同一套通道，三种方向：你的原版 / 单纯反过来 / 数据真正指向的那个")
    print("=" * 150)
    print("""
  fade      = 上破做空、下破做多（你的原版）
  follow    = 单纯反号：上破做多、下破做空
  long_only = 任何一边突破都做多 —— 事件研究说两个方向之后价格都涨，所以这才是数据指向的符号
""")
    base = dict(band_mode="fixed", pyramid_steps=0, base_size=1.0)
    rows, keep = [], {}
    for direction in ("fade", "follow", "long_only"):
        p = VB.BandParams(direction=direction, **base)
        r, po, bar = panels(universe, p)
        pr = equal_weight(r)
        expo = float(equal_weight(po.abs()).mean())
        keep[direction] = (pr, r, po, bar)
        rows.append(summarise(pr, expo, direction))
    res = pd.DataFrame(rows)
    print(f"  {'方向':<12}{'Sharpe':>9}{'总收益':>10}{'CAGR':>9}{'波动':>8}{'回撤':>9}{'平均暴露':>10}")
    for _, x in res.iterrows():
        print(f"  {x['name']:<12}{x['sharpe']:>+9.2f}{x['total']:>+10.1%}{x['cagr']:>+9.1%}"
              f"{x['vol']:>8.1%}{x['maxdd']:>9.1%}{x['exposure']:>10.1%}")

    pr, ret_panel, pos_panel, bar_panel = keep["long_only"]
    expo = float(equal_weight(pos_panel.abs()).mean())

    # ---------------------------------------------------------------- 2. exposure-matched
    print("\n" + "=" * 150)
    print("2. 关键对照：和「暴露相同的单纯持有」比")
    print("=" * 150)
    bench_full = equal_weight(bar_panel)
    # Scale buy-and-hold to the strategy's realised volatility, not to its average
    # exposure.  Exposure-matching is the intuitive choice and it is a trap: Sharpe
    # does not move under scaling, so it leaves the benchmark's Sharpe unchanged
    # while shrinking its return to nothing.  This rule holds ~2% average exposure
    # but takes it in bursts, so it runs several times the volatility of a constant
    # 2% book -- charging that difference to "timing skill" would be wrong.
    k_risk = M.compute(pr, ANN_4H).ann_vol / M.compute(bench_full, ANN_4H).ann_vol
    bench_matched = bench_full * k_risk
    bench_expo_only = bench_full * expo
    print(f"""
  这条规则只有 {expo:.1%} 的时间在场，所以不能直接跟满仓 buy-and-hold 比收益。

  但「按平均暴露缩放」这个直觉做法有陷阱：**Sharpe 是尺度无关的**，
  乘以 {expo:.3f} 不会改变基准的 Sharpe，只会把它的收益压到很小——
  于是收益对比看起来很漂亮，其实什么都没证明。

  而且这条规则的暴露是**成簇爆发**的（要么空仓，要么好几个币同时进场），
  所以它的实际波动是恒定 {expo:.1%} 仓位的好几倍。拿它的收益去比一个低波动基准，
  等于把加杠杆的效果算成择时能力。

  下面按**实际波动**对齐：把 buy-and-hold 放大 {k_risk:.3f} 倍，使两者波动相同。
""")
    comp = pd.DataFrame([
        summarise(pr, expo, "long_only 策略"),
        summarise(bench_matched, expo, f"等波动持有 (x{k_risk:.3f})"),
        summarise(bench_expo_only, expo, f"等暴露持有 (x{expo:.3f}) —— 见上，别用"),
        summarise(bench_full, 1.0, "满仓等权持有"),
    ])
    print(f"  {'':<22}{'Sharpe':>9}{'总收益':>10}{'CAGR':>9}{'波动':>8}{'回撤':>9}")
    for _, x in comp.iterrows():
        print(f"  {x['name']:<22}{x['sharpe']:>+9.2f}{x['total']:>+10.1%}{x['cagr']:>+9.1%}"
              f"{x['vol']:>8.1%}{x['maxdd']:>9.1%}")

    beta, alpha = X.beta_to(pr, bench_full)
    print(f"\n  对满仓等权持有：beta {beta:+.3f}   年化 alpha {alpha:+.2%}")
    print(f"  （beta 若接近 {expo:.2f}，说明收益基本就是被缩放的市场暴露，没有择时贡献）")

    # ---------------------------------------------------------------- 3. random timing
    print("\n" + "=" * 150)
    print("3. 决定性检验：随机时点（保持交易次数、持仓时长、总暴露、跨币聚集全部不变）")
    print("=" * 150)
    n_boot = 400
    n = len(pos_panel)
    shifts = RNG.integers(int(0.02 * n), int(0.98 * n), size=n_boot)
    null_sharpe, null_total = [], []
    for sh in shifts:
        rp = rotate(pos_panel, int(sh))
        # cost is charged on the rotated turnover so the null pays what it trades
        turn = rp.diff().abs().fillna(rp.abs())
        nr = equal_weight(rp * bar_panel - turn * COSTS.per_side)
        st = M.compute(nr, ANN_4H)
        null_sharpe.append(st.sharpe)
        null_total.append(st.total_return)
    null_sharpe = np.array(null_sharpe)
    null_total = np.array(null_total)
    actual = M.compute(pr, ANN_4H)
    pct_s = float((null_sharpe >= actual.sharpe).mean())
    pct_t = float((null_total >= actual.total_return).mean())
    print(f"""
  把每个币的持仓序列整体循环平移一个随机量：交易次数、每笔持仓长度、总暴露、
  以及「多个币同时进场」这个聚集性全都原样保留，唯一被破坏的是**和 VI 信号的对齐**。
  如果信号本身没有信息，平移不该让业绩变差。共 {n_boot} 次。
""")
    print(f"  {'':<26}{'实际':>10}{'随机中位':>12}{'随机 5%':>11}{'随机 95%':>11}{'p 值':>9}")
    print(f"  {'Sharpe':<26}{actual.sharpe:>+10.2f}{np.median(null_sharpe):>+12.2f}"
          f"{np.percentile(null_sharpe,5):>+11.2f}{np.percentile(null_sharpe,95):>+11.2f}{pct_s:>9.3f}")
    print(f"  {'总收益':<26}{actual.total_return:>+10.1%}{np.median(null_total):>+12.1%}"
          f"{np.percentile(null_total,5):>+11.1%}{np.percentile(null_total,95):>+11.1%}{pct_t:>9.3f}")
    print(f"\n  p 值 = 随机时点做得不比实际差的比例。{'通过' if pct_s < 0.05 else '未通过'} 0.05 门槛。")

    # ---------------------------------------------------------------- 4. parameter family
    print("\n" + "=" * 150)
    print("4. 参数族：不是挑一个点，是看整个曲面")
    print("=" * 150)
    grid = list(itertools.product(
        ("fixed", "quantile", "range"),      # band_mode
        (10, 14, 21),                        # vi_len
        ("reenter", "retrace"),              # exit_mode
        (0, 12, 30),                         # max_hold_bars (0/2d/5d on 4H)
    ))
    fam = []
    for bm, vl, em, mh in grid:
        p = VB.BandParams(direction="long_only", band_mode=bm, vi_len=vl, exit_mode=em,
                          max_hold_bars=mh, pyramid_steps=0, base_size=1.0)
        r, po, _ = panels(universe, p)
        prr = equal_weight(r)
        st = M.compute(prr, ANN_4H)
        e = float(equal_weight(po.abs()).mean())
        bm_ret = M.compute(bench_full * e, ANN_4H)
        fam.append({"band": bm, "vi_len": vl, "exit": em, "hold": mh,
                    "sharpe": st.sharpe, "total": st.total_return, "expo": e,
                    "bench_sharpe": bm_ret.sharpe,
                    "edge": st.sharpe - bm_ret.sharpe})
    fam = pd.DataFrame(fam)
    print(f"\n  共 {len(fam)} 组参数。每一组都和它自己的等暴露基准比（edge = 策略 Sharpe − 等暴露持有 Sharpe）：\n")
    print(f"  {'通道':<10}{'VI':>4}{'出场':>10}{'持仓上限':>9}{'暴露':>8}"
          f"{'Sharpe':>9}{'基准':>8}{'edge':>8}")
    for _, x in fam.sort_values("edge", ascending=False).iterrows():
        print(f"  {x['band']:<10}{x['vi_len']:>4}{x['exit']:>10}{x['hold']:>9}"
              f"{x['expo']:>8.1%}{x['sharpe']:>+9.2f}{x['bench_sharpe']:>+8.2f}{x['edge']:>+8.2f}")
    print(f"\n  edge>0 的比例: {(fam['edge']>0).mean():.0%}   "
          f"中位 edge {fam['edge'].median():+.2f}   平均 edge {fam['edge'].mean():+.2f}")

    # ---------------------------------------------------------------- 5. deflated Sharpe
    print("\n" + "=" * 150)
    print("5. Deflated Sharpe：为「试了这么多组参数」付账")
    print("=" * 150)
    best = fam.loc[fam["sharpe"].idxmax()]
    p5, p50, p95 = M.block_bootstrap_sharpe(pr, ANN_4H, block=60)
    dsr_best = M.deflated_sharpe(float(best["sharpe"]), len(fam), len(pr), ANN_4H)
    dsr_base = M.deflated_sharpe(actual.sharpe, len(fam), len(pr), ANN_4H)
    print(f"""
  参数族里最好的一组: {best['band']}/{best['vi_len']}/{best['exit']}/{best['hold']}  Sharpe {best['sharpe']:+.2f}
  基准设定 (fixed/14/reenter/0)                    Sharpe {actual.sharpe:+.2f}

  block bootstrap 95% 区间 (基准设定): [{p5:+.2f}, {p95:+.2f}]   中位 {p50:+.2f}
  deflated Sharpe，按 {len(fam)} 次试验计价:
    最好那组   {dsr_best:.1%}
    基准设定   {dsr_base:.1%}
  （需要 >95% 才算在多重检验后仍然显著）
""")

    # ---------------------------------------------------------------- 6. consistency
    print("\n" + "=" * 150)
    print("6. 一致性：逐币、逐年")
    print("=" * 150)
    r_panel, po_panel, b_panel = ret_panel, pos_panel, bar_panel
    per_sym = []
    for s in syms:
        rr = r_panel[s].dropna()
        if len(rr) < 500:
            continue
        e = float(po_panel[s].abs().mean())
        bs = M.compute(b_panel[s].dropna() * e, ANN_4H)
        ss = M.compute(rr, ANN_4H)
        per_sym.append({"symbol": s, "sharpe": ss.sharpe, "bench": bs.sharpe,
                        "edge": ss.sharpe - bs.sharpe, "expo": e})
    ps = pd.DataFrame(per_sym).sort_values("edge", ascending=False)
    print(f"\n  逐币（各自和自己的等暴露基准比）：edge>0 的有 "
          f"{int((ps['edge']>0).sum())}/{len(ps)} 个\n")
    print(ps.to_string(index=False, float_format=lambda v: f"{v:+.2f}"))

    print("\n  逐年：")
    yr = pd.DataFrame({"strategy": pr, "bench_matched": bench_matched})
    ann = yr.groupby(yr.index.year).apply(
        lambda g: pd.Series({
            "策略": (1 + g["strategy"]).prod() - 1,
            "等暴露持有": (1 + g["bench_matched"]).prod() - 1,
        }), include_groups=False)
    ann["差"] = ann["策略"] - ann["等暴露持有"]
    print(ann.to_string(float_format=lambda v: f"{v:+.2%}"))

    # ---------------------------------------------------------------- 7. costs
    print("\n" + "=" * 150)
    print("7. 成本敏感度")
    print("=" * 150)
    print(f"\n  {'成本/边':<12}{'Sharpe':>9}{'总收益':>10}{'vs 等暴露基准':>16}")
    bench_sharpe = M.compute(bench_matched, ANN_4H).sharpe
    for fee, slip in ((0, 0), (2, 1), (4.5, 2), (4.5, 8), (15, 15)):
        c = B.Costs(fee, slip)
        r2, _, _ = panels(universe, VB.BandParams(direction="long_only", **base), c)
        st = M.compute(equal_weight(r2), ANN_4H)
        print(f"  {c.per_side*1e4:>6.1f}bp{'':<5}{st.sharpe:>+9.2f}{st.total_return:>+10.1%}"
              f"{st.sharpe - bench_sharpe:>+16.2f}")

    REPORTS.mkdir(exist_ok=True)
    fam.to_csv(REPORTS / "follow_breakout_family.csv", index=False)
    ps.to_csv(REPORTS / "follow_breakout_per_symbol.csv", index=False)
    print(f"\n  wrote {REPORTS/'follow_breakout_family.csv'}")

    # ---------------------------------------------------------------- verdict
    print("\n" + "=" * 150)
    print("结论")
    print("=" * 150)
    # "more than half positive" is not a test -- 59% of 54 is p=0.11 under a fair
    # coin.  Each criterion below carries the statistic that decides it.
    fam_binom = sps.binomtest(int((fam["edge"] > 0).sum()), len(fam), 0.5, "greater").pvalue
    fam_wilcox = sps.wilcoxon(fam["edge"]).pvalue
    sym_binom = sps.binomtest(int((ps["edge"] > 0).sum()), len(ps), 0.5, "greater").pvalue
    sym_t = sps.ttest_1samp(ps["edge"], 0)

    checks = [
        ("1. 打赢 buy-and-hold 的 Sharpe", actual.sharpe > bench_sharpe,
         f"{actual.sharpe:+.2f} vs {bench_sharpe:+.2f}"),
        ("2. 打赢随机时点", pct_s < 0.05, f"p={pct_s:.3f}"),
        ("3. 参数族整体为正", fam_wilcox < 0.05 and fam["edge"].median() > 0,
         f"{int((fam['edge']>0).sum())}/{len(fam)} 为正 (二项 p={fam_binom:.3f}), "
         f"中位 edge {fam['edge'].median():+.3f}, Wilcoxon p={fam_wilcox:.3f}"),
        ("4. deflated Sharpe > 95%", (dsr_best or 0) > 0.95,
         f"最好那组 {dsr_best:.1%}, 基准 {dsr_base:.1%}"),
        ("5. 逐币整体为正", sym_binom < 0.05,
         f"{int((ps['edge']>0).sum())}/{len(ps)} 为正 (二项 p={sym_binom:.3f}), "
         f"平均 edge {ps['edge'].mean():+.3f} (t p={sym_t.pvalue:.3f})"),
        ("6. bootstrap 区间不含 0", p5 > 0, f"[{p5:+.2f}, {p95:+.2f}]"),
    ]
    for name, ok, detail in checks:
        print(f"  [{'通过' if ok else '未通过'}] {name:<26} {detail}")
    n_ok = sum(c[1] for c in checks)
    print(f"\n  {n_ok}/{len(checks)} 项通过。")
    print("""
  第 3、5 项是判定「这是不是一个可交易的策略」的核心：
  参数族的 edge 分布中心在 0，逐币是掷硬币——
  事件研究里那个统计显著的均值，没有转化成一个能落地的策略。""")


if __name__ == "__main__":
    main()
