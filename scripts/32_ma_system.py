"""The six-MA 双均线系统 from the video, made mechanical and tested.

The method's own claim is unusually easy to check, because it is stated as two
numbers: a win rate of 30-40% (50% at best) and a reward:risk of at least 1:3.
Those are not vague -- they are exactly what a trade log reports.

But they are also almost exactly what CHANCE reports.  A 1:3 bracket resolves at
the target roughly a quarter of the time on any random walk, so "30-40% at 1:3"
is a description of the bracket geometry before it is a description of an edge.
Any entry rule at all, applied to any market, produces that profile.  So the
number to beat is not 25% and not zero -- it is what the SAME brackets, the same
count, the same long/short mix and the same stop distances produce from random
bars in the same sample.  That control is the point of this file; everything
before it is setup.

Run order:
  ① what 均线密集 looks like once it has a definition
  ② the two entries, 4h and daily, in and out of sample
  ③ the same brackets from random bars -- the null
  ④ reward:risk sweep
  ⑤ the tradeable version: one position at a time, 1% risk per trade
  ⑥ costs
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
EQ0 = 1000.0
SEED = 20260822

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
# the same two patterns with the video's cluster apparatus stripped off
CONTROLS = {
    "A0 任意突破均线带": MS.entries_band_break,
    "B0 回踩MA20(要趋势)": lambda df, p: MS.entries_ma20_touch(df, p, True),
    "B00 回踩MA20(裸)": lambda df, p: MS.entries_ma20_touch(df, p, False),
}
ALL_RULES = {**ENTRIES, **CONTROLS}

_CACHE: dict[tuple[str, str], pd.DataFrame] = {}


def load(tf: str, symbol: str) -> pd.DataFrame:
    key = (tf, symbol)
    if key not in _CACHE:
        _CACHE[key] = D.load(tf, symbol=symbol)
    return _CACHE[key]


def usable(symbols: list[str], tf: str, p: MS.MaParams) -> list[str]:
    out = []
    for s in symbols:
        try:
            df = load(tf, s)
        except FileNotFoundError:
            continue
        if len(df) >= max(p.lens) + p.window:
            out.append(s)
    return out


def collect(symbols: list[str], tf: str, p: MS.MaParams, which: str) -> pd.DataFrame:
    """Pooled trade log across symbols: every signal bracketed independently."""
    fn = ALL_RULES[which]
    logs = []
    for s in symbols:
        df = load(tf, s)
        t = MS.evaluate(df, fn(df, p), p)
        if len(t):
            logs.append(t.assign(symbol=s))
    return pd.concat(logs, ignore_index=True) if logs else pd.DataFrame()


def profile(t: pd.DataFrame) -> dict:
    """Win rate, realised reward:risk, expectancy -- the video's own numbers."""
    if not len(t):
        return {}
    win = t.r_multiple > 0
    wins, losses = t.r_multiple[win], t.r_multiple[~win]
    return {
        "笔数": len(t),
        "胜率": float(win.mean()),
        "平均盈利R": float(wins.mean()) if len(wins) else np.nan,
        "平均亏损R": float(losses.mean()) if len(losses) else np.nan,
        "实际赔率": float(wins.mean() / -losses.mean()) if len(losses) and losses.mean() else np.nan,
        "期望R": float(t.r_multiple.mean()),
        "止损宽度": float(t.risk_pct.median()),
        "中位持仓": float(t.bars.median()),
        "触目标": float((t.why == "target").mean()),
        "超时": float((t.why == "timeout").mean()),
    }


def fmt(rows) -> str:
    d = pd.DataFrame(rows)
    for c in ("胜率", "触目标", "超时", "止损宽度"):
        if c in d:
            d[c] = d[c].map(lambda v: f"{v:.1%}" if pd.notna(v) else "")
    for c in ("平均盈利R", "平均亏损R", "实际赔率", "期望R", "中位持仓"):
        if c in d:
            d[c] = d[c].map(lambda v: f"{v:+.2f}" if pd.notna(v) else "")
    return d.to_string(index=False)


# --------------------------------------------------------------------------- ①
def section_shape(symbols: list[str], tf: str, p: MS.MaParams) -> None:
    print("=" * 150)
    print(f"① 先把「均线密集」变成一个数：六条均线的极差 / 价格   （{tf}）")
    print("=" * 150 + "\n")
    rows = []
    for s in symbols[:12]:
        df = load(tf, s)
        mas = MS.six_mas(df["close"], p.lens)
        sp = MS.spread(mas, df["close"])
        tight = MS.tight_mask(sp, p)
        runs = (tight != tight.shift()).cumsum()[tight]
        lens = tight.groupby(runs).size() if len(runs) else pd.Series(dtype=float)
        rows.append({"币": s, "样本": len(df), "价差中位": float(sp.median()),
                     "密集时价差上限": float(sp[tight].max()) if tight.any() else np.nan,
                     "密集占比": float(tight.mean()), "密集段数": int(len(lens)),
                     "中位持续": float(lens.median()) if len(lens) else np.nan})
    o = pd.DataFrame(rows)
    for c in ("价差中位", "密集时价差上限", "密集占比"):
        o[c] = o[c].map(lambda v: f"{v:.1%}" if pd.notna(v) else "")
    print(o.to_string(index=False))
    print(f"""
  阈值是各币自己近 {p.window} 根价差分布的第 {p.q_tight:.0%} 分位，且只看过去的 K 线。
  为什么不能用固定百分比：上表「价差中位」在各币之间就差了好几倍，
  同一个 3% 在 BTC 上是「很密集」，在 memecoin 上是「一直密集」。
""")


# --------------------------------------------------------------------------- ②
def section_rules(symbols: list[str], tf: str, p: MS.MaParams,
                  label: str) -> dict[str, pd.DataFrame]:
    print("=" * 150)
    print(f"② {label}   {tf}   {len(symbols)} 个币   固定赔率 1:{p.target_r:g}")
    print("=" * 150 + "\n")
    logs, rows = {}, []
    for name in ENTRIES:
        t = collect(symbols, tf, p, name)
        logs[name] = t
        if not len(t):
            continue
        rows.append({"规则": name} | profile(t))
        for side in ("long", "short"):
            sub = t[t.side == side]
            if len(sub) >= 20:
                rows.append({"规则": f"  └ 只看{side}"} | profile(sub))
    print(fmt(rows))
    return logs


# --------------------------------------------------------------------------- ③
def section_null(symbols: list[str], tf: str, p: MS.MaParams,
                 logs: dict[str, pd.DataFrame], n_draws: int, label: str,
                 num: str = "③") -> pd.DataFrame:
    print("=" * 150)
    print(f"{num} 关键对照：同样的括号，随机的入场点   （{label}）")
    print("=" * 150)
    print(f"""
  1:{p.target_r:g} 的括号在随机游走上本来就有约 {1/(1+p.target_r):.0%} 的命中率。
  「胜率 30%、赔率 1:3」这句话，先是括号几何的描述，之后才可能是策略的描述。
  下面把每个规则的**每币多头笔数、空头笔数、止损宽度分布**原样搬到随机的 K 线上，
  重抽 {n_draws} 次。规则要有东西，就得跑赢这条分布，而不是跑赢 0。

  三种随机对照，一种比一种严：
    百分比对照    照抄止损的百分比宽度。缺陷：2% 的止损被丢到高波动时段会被无谓打掉，
                  这会把对照压低、把策略显得好看。
    波动对照      照抄止损相对当时 ATR 的**倍数**——同一个止损放到别处的正确含义。
    止损构造对照  不抄宽度，直接照抄**做法**：止损放在上一根 K 线的极值上（B 就是这么放的）。
                  只有入场时机是随机的，所以规则再赢，就只能是赢在「什么时候做」。
""")
    rng = np.random.default_rng(SEED)
    out = []
    for name, t in logs.items():
        if not len(t):
            continue
        cnt = t.groupby(["symbol", "side"]).size().unstack(fill_value=0)
        for col in ("long", "short"):
            if col not in cnt:
                cnt[col] = 0
        real = profile(t)
        # The prev-extreme control only means anything for a rule that builds its
        # stop that way.  Rule A's stop is the far edge of the MA band -- three
        # times wider -- so running it against one-bar stops would hand A a "win"
        # decided entirely by the cost-in-R of a tighter bracket.
        modes = ["波动对照(ATR倍数)", "百分比对照"]
        if name.startswith("B"):
            modes.append("止损构造对照(上根极值)")
        for mode in modes:
            pcts = t.risk_pct.to_numpy() if mode == "百分比对照" else None
            prev = mode.startswith("止损构造")
            atrs = t.risk_atr.to_numpy()
            draws = []
            for _ in range(n_draws):
                parts = []
                for s, row in cnt.iterrows():
                    df = load(tf, s)
                    sig = MS.random_entries(df, p, int(row["long"]), int(row["short"]),
                                            rng, atrs, pcts, prev)
                    z = MS.evaluate(df, sig, p)
                    if len(z):
                        parts.append(z)
                if parts:
                    z = pd.concat(parts, ignore_index=True)
                    draws.append({"win": float((z.r_multiple > 0).mean()),
                                  "exp": float(z.r_multiple.mean()),
                                  "risk": float(z.risk_pct.median())})
            d = pd.DataFrame(draws)
            out.append({
                "规则": name, "对照": mode, "实际止损宽度": real["止损宽度"],
                "随机止损宽度": float(d["risk"].median()), "实际胜率": real["胜率"],
                "随机胜率中位": float(d.win.median()),
                "随机胜率95%": float(d.win.quantile(0.95)),
                "p(胜率)": float((d.win >= real["胜率"]).mean()),
                "实际期望R": real["期望R"], "随机期望中位": float(d["exp"].median()),
                "随机期望95%": float(d["exp"].quantile(0.95)),
                "p(期望)": float((d["exp"] >= real["期望R"]).mean())})
    r = pd.DataFrame(out)
    o = r.copy()
    for c in ("实际胜率", "随机胜率中位", "随机胜率95%", "实际止损宽度", "随机止损宽度"):
        o[c] = o[c].map(lambda v: f"{v:.1%}")
    for c in ("实际期望R", "随机期望中位", "随机期望95%"):
        o[c] = o[c].map(lambda v: f"{v:+.3f}")
    for c in ("p(胜率)", "p(期望)"):
        o[c] = o[c].map(lambda v: f"{v:.3f}")
    print(o.to_string(index=False))
    print("""
  「实际止损宽度」和「随机止损宽度」必须接近，这张表才是在比入场点。
  差得远就说明在比别的东西——多半是在比成本，因为一笔的成本按 R 算是
  2 x 手续费 / 止损宽度，止损窄一倍，成本就贵一倍。
""")
    return r


# --------------------------------------------------------------------------- ④
def section_rr(symbols: list[str], tf: str, base: MS.MaParams, label: str,
               rs=(1.0, 1.5, 2.0, 3.0, 5.0)) -> pd.DataFrame:
    print("\n" + "=" * 150)
    print(f"④ 赔率扫描（{label}，{tf}）：视频说赔率越高越好，那就拉开看")
    print("=" * 150 + "\n")
    rows = []
    for name in ENTRIES:
        for r in rs:
            p = MS.MaParams(**{**base.__dict__, "target_r": r})
            t = collect(symbols, tf, p, name)
            if len(t):
                rows.append({"规则": name, "赔率": f"1:{r:g}",
                             "打平需胜率": 1 / (1 + r)} | profile(t))
    d = pd.DataFrame(rows)
    o = d.copy()
    o["打平需胜率"] = o["打平需胜率"].map(lambda v: f"{v:.0%}")
    print(fmt(o.to_dict("records")))
    return d


# --------------------------------------------------------------------------- ⑤
def section_controls(symbols: list[str], tf: str, p: MS.MaParams,
                     label: str, n_perm: int = 4000) -> pd.DataFrame:
    """Does 均线密集 add anything over the bare price pattern underneath it?"""
    print("\n" + "=" * 150)
    print(f"⑥ 拆开看：把「均线密集」这层拿掉，剩下的裸形态能打多少   （{label}，{tf}）")
    print("=" * 150)
    print("""
  A  = 密集之后的突破          A0  = 任意一次突破均线带（不要求先有密集）
  B  = 密集->发散后第一次回踩   B0  = 任意一次回踩 MA20（要求均线发散）
                                B00 = 任意一次回踩 MA20（连趋势都不要求）
  如果 A≈A0、B≈B0，那视频真正的内容——密集、发散、第一次——就没有承担任何工作。
""")
    rows, logs = [], {}
    for name in ALL_RULES:
        t = collect(symbols, tf, p, name)
        logs[name] = t
        if len(t):
            rows.append({"规则": name} | profile(t))
    print(fmt(rows))

    # The full rule's signals are a subset of the stripped rule's pool, so the
    # question "does the extra machinery pick better trades" is answered by
    # drawing the same number of trades from that pool at random.  No
    # resimulation needed, and it isolates the selection from everything else.
    print(f"""
  子集检验：完整规则的信号本来就是裸形态信号池里的一部分。
  那就从裸形态的池子里随机抽同样多笔，抽 {n_perm} 次，看完整规则挑出来的那一批
  是不是真的更好。这就是「密集/发散/第一次」这层机器的全部价值。
""")
    rng = np.random.default_rng(SEED)
    out = []
    pairs = [("A 密集突破", "A0 任意突破均线带"),
             ("B 回踩20均线", "B0 回踩MA20(要趋势)"),
             ("B 回踩20均线", "B00 回踩MA20(裸)")]
    for full, bare in pairs:
        tf_, tb = logs.get(full), logs.get(bare)
        if tf_ is None or tb is None or not len(tf_) or not len(tb):
            continue
        pool = tb.r_multiple.to_numpy()
        obs = float(tf_.r_multiple.mean())
        k = min(len(tf_), len(pool))
        draws = np.array([rng.choice(pool, size=k, replace=False).mean()
                          for _ in range(n_perm)])
        out.append({"完整规则": full, "裸形态池": bare, "池子笔数": len(pool),
                    "抽取笔数": k, "完整规则期望R": obs,
                    "随机子集中位": float(np.median(draws)),
                    "随机子集95%": float(np.quantile(draws, 0.95)),
                    "p": float((draws >= obs).mean())})
    r = pd.DataFrame(out)
    o = r.copy()
    for c in ("完整规则期望R", "随机子集中位", "随机子集95%"):
        o[c] = o[c].map(lambda v: f"{v:+.3f}")
    o["p"] = o["p"].map(lambda v: f"{v:.4f}")
    print(o.to_string(index=False))
    return r


def section_portfolio(symbols: list[str], tf: str, p: MS.MaParams, label: str) -> None:
    print("\n" + "=" * 150)
    print(f"⑤ 可交易版本（{label}，{tf}）：单币单笔、每笔风险 {p.risk_frac:.0%} 权益")
    print("=" * 150 + "\n")
    rows = []
    for name, fn in ENTRIES.items():
        finals, ntr = [], []
        for s in symbols:
            df = load(tf, s)
            r = MS.simulate(df, fn(df, p), p, EQ0)
            if len(r.trades):
                finals.append(float(r.equity.iloc[-1]) / EQ0 - 1)
                ntr.append(len(r.trades))
        if finals:
            f = np.array(finals)
            rows.append({"规则": name, "有信号的币": len(f), "中位笔数": float(np.median(ntr)),
                         "中位总收益": float(np.median(f)), "平均总收益": float(f.mean()),
                         "为正的比例": float((f > 0).mean()),
                         "最好": float(f.max()), "最差": float(f.min())})
    o = pd.DataFrame(rows)
    for c in ("中位总收益", "平均总收益", "为正的比例", "最好", "最差"):
        o[c] = o[c].map(lambda v: f"{v:+.1%}")
    print(o.to_string(index=False))
    print(f"""
  每笔只赌 {p.risk_frac:.0%}，所以即便规则有效，总收益也是「笔数 x 每笔期望」的量级——
  这正是视频里的仓位表想表达的意思，也说明了为什么它同时在卖交易所返佣。
""")


def main() -> None:
    p = MS.MaParams()
    every = [s for s in D.available_symbols(require=("1d",)) if s not in NOT_COINS]
    ins = usable([s for s in every if s in IN_SAMPLE], "1d", p)
    oos = usable([s for s in every if s not in IN_SAMPLE], "1d", p)
    fast = usable([s for s in D.available_symbols(require=("4h", "1d"))
                   if s not in NOT_COINS], "4h", p)

    print("=" * 150)
    print("六均线系统（MA20/60/120 + EMA20/60/120）：均线密集 / 均线发散")
    print("=" * 150)
    print(f"""
  视频里的规则，全部机械化后是这样：
    密集    六条均线的极差/价格，低于该品种自己近 {p.window} 根的第 {p.q_tight:.0%} 分位，
            且连续 {p.min_cluster} 根成立
    发散    短>中>长（MA 和 EMA 都要），且不在密集状态
    A 开仓  密集区被**收盘价**突破，顺突破方向入场，止损放密集区另一边；每个密集区只做一次
    B 开仓  密集 -> 发散之后，**第一次**回踩 MA20 且收盘站回去，止损放这根的极值
    平仓    固定赔率括号，默认 1:{p.target_r:g}；{p.max_bars} 根还没结果就按收盘价了结
    仓位    每笔风险 = 权益的 {p.risk_frac:.0%}
    成本    {p.fee*1e4:.1f}bp/边，已折算成 R 计入每一笔

  信号在 K 线收盘确认，下一根开盘成交。
  ②③④ 里每个信号独立计算（胜率和赔率是**信号**的属性）；⑤ 才是单币单笔的可交易版。

  视频没说清、我按最有利于策略的方式定的两处：
    「有效跌破」  取**收盘**穿越，而不是插针穿越
    「均线密集」  取相对分位而不是固定百分比，否则跨品种跨周期没法比
""")

    section_shape(fast, "4h", p)
    logs4 = section_rules(fast, "4h", p, "样本内 26 币")
    logs_i = section_rules(ins, "1d", p, "样本内 26 币")
    logs_o = section_rules(oos, "1d", p, f"样本外 {len(oos)} 币（从未参与开发）")
    print("""
  视频给的画像是「胜率 30-40%，最多 50%；赔率至少 1:3」。
  上面三张表的胜率就落在这个区间里 —— 规则复现是对的。
  但这恰恰是问题所在，下一节。
""")

    null4 = section_null(fast, "4h", p, logs4, 100, "样本内 26 币 4h")
    nullo = section_null(oos, "1d", p, logs_o, 60, f"样本外 {len(oos)} 币 日线")

    rr_i = section_rr(ins, "1d", p, "样本内")
    rr_o = section_rr(oos, "1d", p, "样本外")
    print("""
  打平需胜率 = 1/(1+赔率)。如果「期望R」在所有赔率上都贴着 0，
  说明括号放在哪里都一样——入场点没有信息，只是在重新分配同一份随机性。
""")

    section_portfolio(ins, "1d", p, "样本内")
    section_portfolio(oos, "1d", p, "样本外")

    ctrl_o = section_controls(oos, "1d", p, "样本外")
    ctrl_4 = section_controls(fast, "4h", p, "样本内 4h")

    # ------------------------------------------------------------------ ⑦ costs
    print("\n" + "=" * 150)
    print(f"⑦ 成本敏感度（样本外 {len(oos)} 币，日线）")
    print("=" * 150 + "\n")
    print(f"  {'成本/边':<11}{'规则':<16}{'笔数':>7}{'胜率':>8}{'期望R':>10}")
    for fee in (0.0, 0.00065, 0.0015):
        q = MS.MaParams(**{**p.__dict__, "fee": fee})
        for name in ENTRIES:
            t = collect(oos, "1d", q, name)
            if len(t):
                pr = profile(t)
                print(f"  {fee*1e4:>4.1f}bp{'':<5}{name:<16}{pr['笔数']:>7}"
                      f"{pr['胜率']:>8.1%}{pr['期望R']:>+10.3f}")
    print(f"""
  成本按 R 计入：一笔的成本 = 2 x {p.fee*1e4:.1f}bp / 止损宽度。
  止损越窄，同样的手续费吃掉的 R 越多——B 的止损中位只有几个百分点，
  这就是为什么它对成本远比 A 敏感。""")

    # ----------------------------------------------------------------- verdict
    print("\n" + "=" * 150)
    print("结论（下面每个数字都是上面算出来的，没有写死）")
    print("=" * 150)
    for label, nul in (("4h 样本内", null4), ("日线 样本外", nullo)):
        for name, g in nul.groupby("规则", sort=False):
            print(f"\n  {label} / {name}   实际 胜率 {g['实际胜率'].iloc[0]:.1%}"
                  f"  期望 {g['实际期望R'].iloc[0]:+.3f}R")
            for _, row in g.iterrows():
                print(f"    对 {row['对照']:<16} 随机胜率中位 {row['随机胜率中位']:.1%}"
                      f"（p={row['p(胜率)']:.3f}）   随机期望中位 {row['随机期望中位']:+.3f}R"
                      f"（p={row['p(期望)']:.3f}）"
                      f"  -> {'胜出' if row['p(期望)'] < 0.05 else '无法区分'}")
            hard = g[g["对照"] == "波动对照(ATR倍数)"]
            ok = bool((hard["p(期望)"] < 0.05).all())
            print(f"    以严格的波动对照为准：{'显著优于随机入场' if ok else '与随机入场无法区分'}")
    strict = pd.concat([null4, nullo])
    strict = strict[strict["对照"] == "波动对照(ATR倍数)"]
    sig = int((strict["p(期望)"] < 0.05).sum())
    print(f"""
  按波动对照，{sig}/{len(strict)} 个「规则 x 样本」组合显著优于随机入场（p<0.05，单尾）。
""")

    # --------------------------------------------------- multiple testing
    print("=" * 150)
    print("把「一共试了多少次」算进去")
    print("=" * 150)
    tests = []
    for panel, nul in (("4h", null4), ("1d", nullo)):
        for _, r in nul.iterrows():
            tests.append((f"{panel} {r['规则']} vs {r['对照']}", float(r["p(期望)"])))
    for panel, c in (("1d", ctrl_o), ("4h", ctrl_4)):
        for _, r in c.iterrows():
            tests.append((f"{panel} {r['完整规则']} vs {r['裸形态池']}", float(r["p"])))
    t = pd.DataFrame(tests, columns=["检验", "p"]).sort_values("p").reset_index(drop=True)
    m = len(t)
    t["Bonferroni"] = (t["p"] * m).clip(upper=1.0)
    t["BH临界"] = (t.index + 1) / m * 0.05
    t["过BH"] = t["p"] <= t["BH临界"]
    # BH is a step-up procedure: everything at or below the largest passing rank
    # is rejected, so a raw row-by-row comparison overstates how many fail
    passed = t.index[t["过BH"]].max() if t["过BH"].any() else -1
    t["过BH"] = t.index <= passed
    o = t.copy()
    for c in ("p", "Bonferroni", "BH临界"):
        o[c] = o[c].map(lambda v: f"{v:.4f}")
    print("\n" + o.to_string(index=False))
    print(f"""
  一共 {m} 个检验。原始 p<0.05 的有 {int((t['p'] < 0.05).sum())} 个，
  Bonferroni 之后剩 {int((t['Bonferroni'] < 0.05).sum())} 个，
  Benjamini-Hochberg（FDR 5%）之后剩 {int(t['过BH'].sum())} 个。

  这一步不是走形式。这套系统的两条开仓法、两个周期、三种随机对照、
  五档赔率，本身就是一个搜索空间；不做校正，「找到一个 p=0.02」是必然事件。
""")

    # --------------------------------------------------- how big is it, in money
    print("=" * 150)
    print("就算显著，值多少钱？")
    print("=" * 150 + "\n")
    print(f"  {'样本':<14}{'规则':<22}{'笔数':>7}{'每币每年':>10}{'期望R':>9}"
          f"{'年化收益(风险{:.0%}/笔)'.format(p.risk_frac):>22}")
    for label, tf, syms in (("4h 样本内", "4h", fast), ("日线 样本外", "1d", oos)):
        years = np.median([len(load(tf, s)) for s in syms]) / D.bars_per_year(tf)
        for name in ALL_RULES:
            t = collect(syms, tf, p, name)
            if not len(t):
                continue
            per = len(t) / len(syms) / years
            e = float(t.r_multiple.mean())
            print(f"  {label:<14}{name:<22}{len(t):>7}{per:>10.1f}{e:>+9.3f}"
                  f"{per * e * p.risk_frac:>+22.2%}")
    print(f"""
  最后一列是**单个币**上按视频的仓位规则能拿到的年化。要靠它活下去，
  只能靠同时跑很多个币、或者把每笔风险从 {p.risk_frac:.0%} 抬到十几个点——
  而后者会让「胜率 30%」意味着连亏 5 笔的概率有 {0.7**5:.0%}，那已经不是仓位管理问题了。
""")

    REPORTS.mkdir(exist_ok=True)
    pd.concat([rr_i.assign(panel="ins"), rr_o.assign(panel="oos")]).to_csv(
        REPORTS / "ma_system_rr.csv", index=False)
    pd.concat([null4.assign(panel="4h_ins"), nullo.assign(panel="1d_oos")]).to_csv(
        REPORTS / "ma_system_null.csv", index=False)
    print(f"  wrote {REPORTS/'ma_system_rr.csv'}, {REPORTS/'ma_system_null.csv'}")


if __name__ == "__main__":
    main()
