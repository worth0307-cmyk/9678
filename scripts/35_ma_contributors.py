"""Which coins carry the six-MA result, and whether their commonality is usable.

"Who contributed most" is a safe question.  "What do they have in common" is
not: with 181 coins and a median of 5 trades each, ANY set of top contributors
will share several attributes by construction, and reading those attributes back
as a filter is the single most reliable way to manufacture a strategy that only
works on the sample it was read from.

So the file is built in two halves that must not be confused:

  descriptive   who they are, and how much of each coin's total comes from its
                one best trade.  Total R decomposes as trades x mean R, and a
                coin can top the table purely by trading more, so both are shown.
  testable      the same attributes turned into a filter that is FIXED on the
                26 in-sample coins and then applied unchanged to the 181
                out-of-sample coins, each side judged against its own
                random-entry control.  That is the only version of "what do they
                have in common" that could be acted on.

If the descriptive half finds a strong pattern and the testable half finds
nothing, the pattern is hindsight -- which is exactly what the split-half
persistence in 34_ma_by_symbol.py already implies should happen.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scipy import stats  # noqa: E402
from vibt import data as D, masys as MS  # noqa: E402

pd.set_option("display.width", 250)
REPORTS = Path(__file__).resolve().parent.parent / "reports"
SEED = 20260822

IN_SAMPLE = [
    "1000BONKUSDT", "1000FLOKIUSDT", "1000PEPEUSDT", "1000SHIBUSDT", "APTUSDT",
    "ARBUSDT", "BNBUSDT", "BTCUSDT", "DOGEUSDT", "DYDXUSDT", "ENAUSDT", "ETHUSDT",
    "FETUSDT", "HYPEUSDT", "LDOUSDT", "OPUSDT", "PENDLEUSDT", "RENDERUSDT",
    "SEIUSDT", "SOLUSDT", "SUIUSDT", "TAOUSDT", "TIAUSDT", "WIFUSDT", "WLDUSDT",
    "XRPUSDT",
]
NOT_COINS = ("USDCUSDT", "BTCDOMUSDT")
RULES = {"A 密集突破": MS.entries_cluster_break,
         "B 回踩20均线": MS.entries_ma20_pullback}

_CACHE: dict[tuple[str, str], pd.DataFrame] = {}


def load(tf: str, s: str) -> pd.DataFrame:
    k = (tf, s)
    if k not in _CACHE:
        _CACHE[k] = D.load(tf, symbol=s)
    return _CACHE[k]


def trades_of(symbols: list[str], tf: str, p: MS.MaParams, which: str) -> pd.DataFrame:
    fn = RULES[which]
    out = []
    for s in symbols:
        df = load(tf, s)
        t = MS.evaluate(df, fn(df, p), p)
        if len(t):
            out.append(t.assign(symbol=s))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def attrs(symbols: list[str], tf: str) -> pd.DataFrame:
    """Everything about a coin that is knowable without running the strategy."""
    rows = []
    for s in symbols:
        df = load(tf, s)
        px = df["close"]
        ret = px.pct_change()
        peak = px.cummax()
        rows.append({
            "symbol": s,
            "历史长度": len(df),
            "日均成交额": float(df["quote_volume"].median()) if "quote_volume" in df else np.nan,
            "年化波动": float(ret.std() * np.sqrt(D.bars_per_year(tf))),
            "全程涨跌": float(px.iloc[-1] / px.iloc[0] - 1),
            "最大回撤": float((px / peak - 1).min()),
            "距高点": float(px.iloc[-1] / px.max() - 1),
            "偏度": float(ret.skew()),
        })
    return pd.DataFrame(rows)


def contribution(t: pd.DataFrame, a: pd.DataFrame) -> pd.DataFrame:
    """Per-coin totals plus how much rides on that coin's single best trade."""
    g = t.groupby("symbol")
    d = pd.DataFrame({
        "笔数": g.size(),
        "期望R": g.r_multiple.mean(),
        "总R": g.r_multiple.sum(),
        "最好一笔R": g.r_multiple.max(),
        "空头占比": g.apply(lambda z: float((z.side == "short").mean()),
                            include_groups=False),
    }).reset_index()
    d["去掉最好一笔后总R"] = d["总R"] - d["最好一笔R"].clip(lower=0)
    d["最好一笔占比"] = np.where(d["总R"] > 0,
                                 d["最好一笔R"].clip(lower=0) / d["总R"], np.nan)
    return d.merge(a, on="symbol", how="left")


def fmt(d: pd.DataFrame) -> pd.DataFrame:
    o = d.copy()
    for c in ("空头占比", "年化波动", "全程涨跌", "最大回撤", "距高点", "最好一笔占比"):
        if c in o:
            o[c] = o[c].map(lambda v: f"{v:+.0%}" if pd.notna(v) else "")
    for c in ("期望R", "总R", "最好一笔R", "去掉最好一笔后总R", "偏度"):
        if c in o:
            o[c] = o[c].map(lambda v: f"{v:+.2f}" if pd.notna(v) else "")
    if "日均成交额" in o:
        o["日均成交额"] = o["日均成交额"].map(
            lambda v: f"{v/1e6:,.0f}M" if pd.notna(v) else "")
    return o


# --------------------------------------------------------------------------- ③
def trade_concentration(t: pd.DataFrame) -> dict:
    r = np.sort(t.r_multiple.to_numpy())
    pos = r[r > 0].sum()
    out = {"总笔数": len(r), "总R": float(r.sum())}
    for q in (0.01, 0.05, 0.10):
        k = max(1, int(round(q * len(r))))
        out[f"最好{q:.0%}笔占正贡献"] = float(r[-k:].sum() / pos) if pos > 0 else np.nan
    # how many coins would change sign if their single best trade vanished
    g = t.groupby("symbol").r_multiple
    tot, best = g.sum(), g.max().clip(lower=0)
    flip = ((tot > 0) & ((tot - best) < 0)).mean()
    out["靠单笔才转正的币"] = float(flip)
    return out


# --------------------------------------------------------------------------- ④
def commonality(d: pd.DataFrame, k: int = 20) -> pd.DataFrame:
    """Top-k contributors vs the rest, on attributes knowable in advance."""
    d = d.sort_values("总R", ascending=False)
    top, rest = d.head(k), d.iloc[k:]
    cols = ["笔数", "空头占比", "日均成交额", "年化波动", "全程涨跌",
            "最大回撤", "距高点", "历史长度", "偏度"]
    rows = []
    for c in cols:
        a = top[c].dropna().to_numpy()
        b = rest[c].dropna().to_numpy()
        if len(a) < 5 or len(b) < 5:
            continue
        u, pv = stats.mannwhitneyu(a, b, alternative="two-sided")
        # rank-biserial: +1 means every top coin exceeds every other coin
        eff = 2 * u / (len(a) * len(b)) - 1
        rows.append({"属性": c, f"前{k}名中位": float(np.median(a)),
                     "其余中位": float(np.median(b)), "效应量": eff, "p": pv})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- ⑤
def null_for(symbols: list[str], tf: str, p: MS.MaParams, which: str,
             rng: np.random.Generator, n_draws: int = 60) -> tuple[float, float, float]:
    """Observed pooled expectancy, null median, and one-sided p for a coin set."""
    fn = RULES[which]
    real = []
    for s in symbols:
        df = load(tf, s)
        t = MS.evaluate(df, fn(df, p), p)
        if len(t):
            real.append(t.assign(symbol=s))
    if not real:
        return np.nan, np.nan, np.nan
    R = pd.concat(real, ignore_index=True)
    cnt = R.groupby(["symbol", "side"]).size().unstack(fill_value=0)
    for c in ("long", "short"):
        if c not in cnt:
            cnt[c] = 0
    atrs = R.risk_atr.to_numpy()
    draws = []
    for _ in range(n_draws):
        parts = []
        for s, row in cnt.iterrows():
            df = load(tf, s)
            z = MS.evaluate(df, MS.random_entries(df, p, int(row["long"]),
                                                  int(row["short"]), rng, atrs), p)
            if len(z):
                parts.append(z)
        if parts:
            draws.append(float(pd.concat(parts, ignore_index=True).r_multiple.mean()))
    obs = float(R.r_multiple.mean())
    dr = np.array(draws)
    return obs, float(np.median(dr)), float((dr >= obs).mean())


def main() -> None:
    p = MS.MaParams()
    every = [s for s in D.available_symbols(require=("1d",)) if s not in NOT_COINS]
    keep = [s for s in every if len(load("1d", s)) >= max(p.lens) + p.window]
    ins = [s for s in keep if s in IN_SAMPLE]
    oos = [s for s in keep if s not in IN_SAMPLE]

    print("=" * 165)
    print("哪几个币贡献最大，这些币有什么共性")
    print("=" * 165)
    print(f"""
  样本外 {len(oos)} 个币，日线。总R = 笔数 x 每笔期望，所以一个币可以纯靠「做得多」
  排到前面，两项都列出来。「最好一笔占比」是这个币的正收益里有多少来自它单独最好的那一笔。
""")

    tabs = {}
    for which in RULES:
        t = trades_of(oos, "1d", p, which)
        d = contribution(t, attrs(oos, "1d")).sort_values("总R", ascending=False)
        tabs[which] = (t, d)

        print("=" * 165)
        print(f"① {which}   前 20 名贡献者")
        print("=" * 165 + "\n")
        cols = ["symbol", "笔数", "期望R", "总R", "最好一笔R", "去掉最好一笔后总R",
                "最好一笔占比", "空头占比", "日均成交额", "年化波动", "全程涨跌", "最大回撤"]
        print(fmt(d[cols].head(20)).to_string(index=False))
        head = d.head(20)
        print(f"""
  前 20 名合计 {head['总R'].sum():+.1f}R，占全部正贡献的 {head['总R'].sum()/d['总R'].clip(lower=0).sum():.0%}；
  全部 {len(d)} 个币合计 {d['总R'].sum():+.1f}R。
  这 20 个币里，有 {int((head['最好一笔占比'] > 0.5).sum())} 个的正收益一半以上来自**单独一笔**。
""")

    # ------------------------------------------------------------------ ② 分解
    print("=" * 165)
    print("② 排名靠前是因为「做得多」还是「做得准」")
    print("=" * 165 + "\n")
    rows = []
    for which, (t, d) in tabs.items():
        rows.append({
            "规则": which,
            "corr(总R, 笔数)": float(d["总R"].corr(d["笔数"])),
            "corr(总R, 每笔期望)": float(d["总R"].corr(d["期望R"])),
            "前20名中位笔数": float(d.head(20)["笔数"].median()),
            "其余中位笔数": float(d.iloc[20:]["笔数"].median()),
            "前20名中位每笔": float(d.head(20)["期望R"].median()),
            "其余中位每笔": float(d.iloc[20:]["期望R"].median()),
        })
    o = pd.DataFrame(rows)
    for c in ("corr(总R, 笔数)", "corr(总R, 每笔期望)", "前20名中位每笔", "其余中位每笔"):
        o[c] = o[c].map(lambda v: f"{v:+.3f}")
    print(o.to_string(index=False))

    # ------------------------------------------------------------------ ③ 集中
    print("\n" + "=" * 165)
    print("③ 换成按「笔」看：贡献集中在几笔上")
    print("=" * 165 + "\n")
    rows = [{"规则": w} | trade_concentration(t) for w, (t, _) in tabs.items()]
    o = pd.DataFrame(rows)
    for c in o.columns:
        if "占" in c or "币" in c:
            o[c] = o[c].map(lambda v: f"{v:.0%}" if pd.notna(v) else "")
    o["总R"] = o["总R"].map(lambda v: f"{v:+.1f}")
    print(o.to_string(index=False))

    # ------------------------------------------------------------------ ④ 共性
    for which, (t, d) in tabs.items():
        print("\n" + "=" * 165)
        print(f"④ {which}   前 20 名 vs 其余，逐项比较")
        print("=" * 165 + "\n")
        c = commonality(d)
        o = c.copy()
        for col in (f"前20名中位", "其余中位"):
            o[col] = [f"{v/1e6:,.0f}M" if r == "日均成交额" and pd.notna(v)
                      else (f"{v:+.0%}" if r in ("空头占比", "年化波动", "全程涨跌",
                                                 "最大回撤", "距高点") else f"{v:+.2f}")
                      for r, v in zip(c["属性"], c[col])]
        o["效应量"] = o["效应量"].map(lambda v: f"{v:+.2f}")
        o["p"] = o["p"].map(lambda v: f"{v:.4f}")
        print(o.to_string(index=False))
    print("""
  效应量是 rank-biserial：+1 表示前 20 名在这一项上全面高于其余币，0 表示没有差别。
  注意这一节**不是证据**——20 个币在 9 个属性上比较，出现一两个 p<0.05 是必然的，
  而且这 20 个币本来就是按结果挑出来的。下一节才是能不能用的检验。
""")

    # ------------------------------------------------------------------ ⑤ 检验
    print("=" * 165)
    print("⑤ 把共性变成一个事前筛子：在样本内 26 币上定死，原样套到样本外")
    print("=" * 165)
    rng = np.random.default_rng(SEED)
    ia, oa = attrs(ins, "1d"), attrs(oos, "1d")
    print(f"""
  筛子用两个属性：日均成交额（越大越好）和最大回撤（跌得越狠越好），
  方向来自 34 号脚本在样本内看到的单调分组。

  阈值不能照抄样本内的绝对值：样本内 26 个币的成交额中位是
  {ia['日均成交额'].median()/1e6:,.0f}M，而样本外 181 个币的中位只有
  {oa['日均成交额'].median()/1e6:,.0f}M —— 直接套过去只会选出 2 个币，什么都测不了。
  所以固定的是**用哪两个属性、往哪个方向切**，切点取样本外自己的中位数。
""")
    tv, td = float(oa["日均成交额"].median()), float(oa["最大回撤"].median())
    hi_v = set(oa[oa["日均成交额"] >= tv]["symbol"])
    deep = set(oa[oa["最大回撤"] <= td]["symbol"])
    variants = {
        "两项都要": sorted(hi_v & deep),
        "只要成交额高": sorted(hi_v),
        "只要回撤深": sorted(deep),
        "全部": oos,
    }
    print(f"  切点：成交额 >= {tv/1e6:,.0f}M，最大回撤 <= {td:.0%}")
    for k, v in variants.items():
        print(f"    {k}: {len(v)} 个币")
    print()
    rows = []
    for which in RULES:
        for label, syms in variants.items():
            if len(syms) < 5:
                continue
            obs, nul, pv = null_for(syms, "1d", p, which, rng)
            rows.append({"规则": which, "组": label, "币数": len(syms),
                         "实际期望R": obs, "随机对照": nul,
                         "超出对照": obs - nul, "p": pv})
    o = pd.DataFrame(rows)
    for c in ("实际期望R", "随机对照", "超出对照"):
        o[c] = o[c].map(lambda v: f"{v:+.3f}")
    o["p"] = o["p"].map(lambda v: f"{v:.3f}")
    print(o.to_string(index=False))
    print("""
  「筛子选中」的超出对照如果没有明显高于「筛子排除」，
  那第④节看到的共性就只是这批币被挑出来时自带的属性，不能事前用。
""")

    # The direction of the two cuts was read off the OUT-OF-SAMPLE buckets in
    # 34_ma_by_symbol.py, so the test above is in-sample with respect to the
    # filter even though it is out-of-sample with respect to the strategy.  The
    # only independent evidence for the direction is the 26 coins that panel
    # never touched.
    print("=" * 165)
    print("⑤b 但方向是从哪儿来的？在样本内 26 个币上独立验一次")
    print("=" * 165)
    print("""
  上面筛子的两个方向，是在 34 号脚本里**从样本外那 181 个币的分组**看出来的。
  也就是说：对策略而言那是样本外，对筛子而言那是样本内。
  真正没被用过的是样本内那 26 个币——如果方向在它们身上也成立，筛子才算独立成立。
""")
    rows = []
    for tf in ("1d", "4h"):
        syms = [s for s in ins if (tf == "1d" or s in
                                   D.available_symbols(require=("4h", "1d")))]
        a = attrs(syms, tf)
        for which in RULES:
            t = trades_of(syms, tf, p, which)
            if not len(t):
                continue
            d = contribution(t, a)
            for col, lab in (("日均成交额", "成交额"), ("最大回撤", "最大回撤")):
                if d[col].isna().all():
                    continue
                med = d[col].median()
                hi = d[d[col] >= med]
                lo = d[d[col] < med]
                if hi["笔数"].sum() < 20 or lo["笔数"].sum() < 20:
                    continue
                rows.append({
                    "周期": tf, "规则": which, "属性": lab,
                    "高的一半期望R": hi["总R"].sum() / hi["笔数"].sum(),
                    "低的一半期望R": lo["总R"].sum() / lo["笔数"].sum(),
                })
    if rows:
        s = pd.DataFrame(rows)
        s["差"] = s["高的一半期望R"] - s["低的一半期望R"]
        o = s.copy()
        for c in ("高的一半期望R", "低的一半期望R", "差"):
            o[c] = o[c].map(lambda v: f"{v:+.3f}")
        print(o.to_string(index=False))
        print(f"""
  样本外看到的方向是：成交额高的更好、回撤深的（数值更小）更好。
  上表里「成交额」一行的「差」为正、「最大回撤」一行的「差」为负，才算方向一致。
  4h 只有 26 个币但笔数多，日线笔数少，两个都看。
""")

    # ------------------------------------------------------------------ ⑥ 一致
    print("=" * 165)
    print("⑥ 内部一致性：A 觉得好的币，B 也觉得好吗")
    print("=" * 165 + "\n")
    da = tabs["A 密集突破"][1][["symbol", "期望R", "总R"]].rename(
        columns={"期望R": "A每笔", "总R": "A总R"})
    db = tabs["B 回踩20均线"][1][["symbol", "期望R", "总R"]].rename(
        columns={"期望R": "B每笔", "总R": "B总R"})
    m = da.merge(db, on="symbol")
    r1, p1 = stats.spearmanr(m["A每笔"], m["B每笔"])
    r2, p2 = stats.spearmanr(m["A总R"], m["B总R"])
    topa = set(m.nlargest(20, "A总R").symbol)
    topb = set(m.nlargest(20, "B总R").symbol)
    print(f"  两条规则都有交易的币: {len(m)}")
    print(f"  每笔期望的秩相关: {r1:+.3f} (p={p1:.3f})")
    print(f"  总R  的秩相关:     {r2:+.3f} (p={p2:.3f})")
    print(f"  各自前 20 名的重叠: {len(topa & topb)}/20"
          f"（随机预期约 {20*20/len(m):.1f}）")
    print("""
  如果一个币真的「适合均线系统」，两条开仓法应该同时受益。
  重叠接近随机预期，说明「贡献大」是这条规则在这个币上的运气，不是这个币的属性。
""")

    REPORTS.mkdir(exist_ok=True)
    pd.concat([d.assign(rule=w) for w, (_, d) in tabs.items()]).to_csv(
        REPORTS / "ma_contributors.csv", index=False)
    print(f"  wrote {REPORTS/'ma_contributors.csv'}")


if __name__ == "__main__":
    main()
