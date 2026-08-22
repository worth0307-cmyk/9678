"""The three exits the video names, on the same entries.

  赔率平仓法        a fixed multiple of the stop distance, 1:3 or 1:5
  上一个均线密集平仓法  the last completed cluster is where the market's average
                    cost converged, so it should act as support/resistance
  斐波那契平仓法     1.618 / 2.618 / 3.618 / 4.236 extensions of the prior swing,
                    for instruments at all-time highs with nothing overhead

Two of these turn out to be the same idea in different clothes.  A Fibonacci
extension measured from the same swing the stop sits on IS an odds exit -- 1.618
is a 1:1.618 bracket.  The only way the two can differ is if the swing used for
the extension is not the swing the stop is derived from, which is why this file
measures the extension from the prior swing and the stop from the entry
structure, exactly as described, and then reports the R multiple each target
actually landed at.

The cluster exit is the one genuinely different rule, and it has a property the
video does not mention: the target is wherever the last cluster happens to be,
so the reward:risk is not chosen, it is inherited.  That number is reported.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vibt import data as D, masys as MS  # noqa: E402

pd.set_option("display.width", 230)
REPORTS = Path(__file__).resolve().parent.parent / "reports"
IN_SAMPLE = [
    "1000BONKUSDT", "1000FLOKIUSDT", "1000PEPEUSDT", "1000SHIBUSDT", "APTUSDT",
    "ARBUSDT", "BNBUSDT", "BTCUSDT", "DOGEUSDT", "DYDXUSDT", "ENAUSDT", "ETHUSDT",
    "FETUSDT", "HYPEUSDT", "LDOUSDT", "OPUSDT", "PENDLEUSDT", "RENDERUSDT",
    "SEIUSDT", "SOLUSDT", "SUIUSDT", "TAOUSDT", "TIAUSDT", "WIFUSDT", "WLDUSDT",
    "XRPUSDT",
]
NOT_COINS = ("USDCUSDT", "BTCDOMUSDT")
ENTRIES = {"A 密集突破": MS.entries_cluster_break,
           "B 回踩20均线": MS.entries_ma20_pullback}
FIBS = (1.618, 2.618, 3.618, 4.236)


def with_cluster_target(df: pd.DataFrame, sig: list[MS.Signal],
                        p: MS.MaParams) -> tuple[list[MS.Signal], int]:
    """Retarget each signal at the last completed cluster in its direction."""
    lv = MS.prior_cluster_levels(df, p)
    up, dn = lv["above"].to_numpy(), lv["below"].to_numpy()
    op = df["open"].to_numpy(float)
    out, hit = [], 0
    for s in sig:
        tgt = up[s.i - 1] if s.side > 0 else dn[s.i - 1]
        fill = op[s.i]
        # a target already behind price is not a target
        if np.isfinite(tgt) and ((s.side > 0 and tgt > fill) or (s.side < 0 and tgt < fill)):
            hit += 1
            out.append(MS.Signal(s.i, s.side, s.stop, s.kind, tgt))
        else:
            out.append(MS.Signal(s.i, s.side, s.stop, s.kind))   # falls back to R
    return out, hit


def with_fib_target(df: pd.DataFrame, sig: list[MS.Signal], p: MS.MaParams,
                    ratio: float) -> tuple[list[MS.Signal], int]:
    op = df["open"].to_numpy(float)
    out, hit = [], 0
    for s in sig:
        tgt = MS.swing_extension(df, p, s.i, s.side, ratio)
        fill = op[s.i]
        if np.isfinite(tgt) and ((s.side > 0 and tgt > fill) or (s.side < 0 and tgt < fill)):
            hit += 1
            out.append(MS.Signal(s.i, s.side, s.stop, s.kind, tgt))
        else:
            out.append(MS.Signal(s.i, s.side, s.stop, s.kind))
    return out, hit


def run(symbols: list[str], tf: str, p: MS.MaParams, which: str) -> pd.DataFrame:
    fn = ENTRIES[which]
    rows = []
    variants = ([("赔率平仓 1:2", "r", 2.0), ("赔率平仓 1:3", "r", 3.0),
                 ("赔率平仓 1:5", "r", 5.0), ("上一个均线密集", "cluster", 0.0)]
                + [(f"斐波那契 {f}", "fib", f) for f in FIBS])
    acc: dict[str, list[pd.DataFrame]] = {v[0]: [] for v in variants}
    cover: dict[str, list[float]] = {v[0]: [] for v in variants}
    for s in symbols:
        df = D.load(tf, symbol=s)
        base = fn(df, p)
        if not base:
            continue
        for label, mode, val in variants:
            if mode == "r":
                q = MS.MaParams(**{**p.__dict__, "target_r": val})
                sig, n_ok = base, len(base)
            elif mode == "cluster":
                q = p
                sig, n_ok = with_cluster_target(df, base, p)
            else:
                q = p
                sig, n_ok = with_fib_target(df, base, p, val)
            t = MS.evaluate(df, sig, q)
            if len(t):
                acc[label].append(t)
                cover[label].append(n_ok / len(base))
    for label, _, _ in variants:
        if not acc[label]:
            continue
        t = pd.concat(acc[label], ignore_index=True)
        win = t.r_multiple > 0
        r = np.sort(t.r_multiple.to_numpy())
        cut = max(1, int(round(0.01 * len(r))))
        gains = r[r > 0].sum()
        rows.append({
            "平仓方式": label, "笔数": len(t),
            "目标可用率": float(np.mean(cover[label])),
            "目标平均R": float(((t.target - t.entry).abs()
                                / (t.entry - t.stop).abs()).mean()),
            "胜率": float(win.mean()),
            "期望R": float(t.r_multiple.mean()),
            # a mean carried by a handful of trades is not a mean you can plan on
            "去掉最好1%": float(r[:-cut].mean()),
            "最大单笔R": float(r[-1]),
            "最好1%占总盈利": float(r[-cut:][r[-cut:] > 0].sum() / gains) if gains > 0 else np.nan,
            "中位持仓": float(t.bars.median()),
            "超时": float((t.why == "timeout").mean()),
        })
    return pd.DataFrame(rows)


def show(d: pd.DataFrame) -> None:
    o = d.copy()
    for c in ("目标可用率", "胜率", "超时", "最好1%占总盈利"):
        o[c] = o[c].map(lambda v: f"{v:.1%}" if pd.notna(v) else "")
    for c in ("目标平均R", "期望R", "去掉最好1%", "最大单笔R", "中位持仓"):
        o[c] = o[c].map(lambda v: f"{v:+.2f}")
    print(o.to_string(index=False))


def main() -> None:
    p = MS.MaParams()
    every = [s for s in D.available_symbols(require=("1d",)) if s not in NOT_COINS]
    oos = [s for s in every if s not in IN_SAMPLE
           and len(D.load("1d", symbol=s)) >= max(p.lens) + p.window]
    fast = [s for s in D.available_symbols(require=("4h", "1d")) if s not in NOT_COINS]

    print("=" * 150)
    print("三种平仓方式，同样的入场")
    print("=" * 150)
    print("""
  「目标可用率」= 该方式能给出一个还在价格前方的目标的比例；给不出时退回 1:3。
  「目标平均R」= 那个目标实际相当于几倍止损距离 —— 赔率平仓法是自己定的，
                 另外两种是**捡到什么算什么**，这是它们真正的区别。
""")
    out = []
    for label, tf, syms in ((f"样本外 {len(oos)} 币", "1d", oos),
                            (f"样本内 {len(fast)} 币", "4h", fast)):
        for which in ENTRIES:
            print("=" * 150)
            print(f"{label}   {tf}   {which}")
            print("=" * 150 + "\n")
            d = run(syms, tf, p, which)
            show(d)
            out.append(d.assign(panel=label, tf=tf, rule=which))
            print()

    print("=" * 150)
    print("怎么读")
    print("=" * 150)
    print("""
  1. 斐波那契那几行的「目标平均R」如果和 1.618/2.618 差不多，
     那斐波那契平仓法就等于换了个名字的赔率平仓法 —— 它只是在选 R。
     如果差得远（比如 B 的止损只有几个百分点，而前一段行情有几十个百分点），
     那 1.618 这个数字就跟风险无关了，它实际是「拿到翻倍或者归零」。
  2. 「上一个均线密集」的目标平均R 是被市场给定的，不是被你选的。
     它的期望R 好不好，取决于上一个密集区恰好离得多远，而这跟你的止损无关。
     「目标可用率」还告诉你：有相当一部分时候，上一个密集区根本不在价格前方，
     这时候这条规则没有定义，只能退回赔率法。
  3. 「去掉最好1%」和「最好1%占总盈利」是用来判断这个期望值能不能指望的。
     如果去掉最好的 1% 之后期望就翻负，那这套平仓法的收益来自极少数几笔，
     样本外能不能再遇到，没人知道 —— 这跟「有正期望」是两回事。
  4. 三种方式如果期望R 都差不多，那就说明**平仓方式不是这套系统的关键变量**，
     真正决定结果的是入场点有没有信息（这在 32_ma_system.py 里已经查过）。
""")
    REPORTS.mkdir(exist_ok=True)
    pd.concat(out, ignore_index=True).to_csv(REPORTS / "ma_exits.csv", index=False)
    print(f"  wrote {REPORTS/'ma_exits.csv'}")


if __name__ == "__main__":
    main()
