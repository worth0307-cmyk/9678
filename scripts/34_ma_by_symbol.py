"""The six-MA rules broken out per symbol.

The pooled number (+0.156R for rule B out of sample) is an average over 181
coins, and an average is exactly the wrong statistic to trust here.  Two very
different worlds produce it:

  broad    most coins slightly positive -- an effect you could expect to meet
           again on a coin you have not traded yet
  narrow   most coins near zero and a handful enormous -- an average you cannot
           plan around, because reproducing it means drawing the same handful

So this file asks three questions the pooled table cannot answer:

  1. how concentrated is it?  Drop the best N coins and see what is left.
  2. does a coin that worked keep working?  Split each coin's own trades in
     half by time and correlate.  This is the same test that killed per-coin
     funding and momentum fitting earlier in this repository.
  3. does it sort on anything you can see in advance -- liquidity, volatility,
     or whether the coin went up or down?

Per-coin samples are tiny on daily bars (a median of about 5 trades), so no
single coin's number means anything.  Everything here is read at the level of
the distribution, and the per-coin table is printed mostly to make that
concrete rather than to rank coins.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scipy import stats  # noqa: E402
from vibt import data as D, masys as MS  # noqa: E402

pd.set_option("display.width", 240)
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


def per_symbol(symbols: list[str], tf: str, p: MS.MaParams,
               which: str) -> pd.DataFrame:
    fn = RULES[which]
    rows = []
    for s in symbols:
        df = load(tf, s)
        t = MS.evaluate(df, fn(df, p), p)
        if not len(t):
            continue
        px = df["close"]
        qv = df["quote_volume"].median() if "quote_volume" in df else np.nan
        rows.append({
            "symbol": s, "笔数": len(t),
            "胜率": float((t.r_multiple > 0).mean()),
            "期望R": float(t.r_multiple.mean()),
            "总R": float(t.r_multiple.sum()),
            "多/空": f"{int((t.side=='long').sum())}/{int((t.side=='short').sum())}",
            "止损宽度": float(t.risk_pct.median()),
            "币价涨跌": float(px.iloc[-1] / px.iloc[0] - 1),
            "日均成交额": float(qv),
            "年化波动": float(px.pct_change().std() * np.sqrt(D.bars_per_year(tf))),
        })
    return pd.DataFrame(rows)


def show(d: pd.DataFrame, n: int | None = None) -> None:
    o = d.copy()
    if n is not None:
        o = pd.concat([o.head(n), o.tail(n)])
    for c in ("胜率", "止损宽度", "币价涨跌", "年化波动"):
        if c in o:
            o[c] = o[c].map(lambda v: f"{v:+.0%}" if pd.notna(v) else "")
    for c in ("期望R", "总R"):
        o[c] = o[c].map(lambda v: f"{v:+.2f}")
    if "日均成交额" in o:
        o["日均成交额"] = o["日均成交额"].map(
            lambda v: f"{v/1e6:,.0f}M" if pd.notna(v) else "")
    print(o.to_string(index=False))


# --------------------------------------------------------------------------- ②
def concentration(d: pd.DataFrame, label: str) -> dict:
    """How much of the pooled edge survives dropping the best few coins?"""
    tot = d["总R"].sum()
    n_tr = d["笔数"].sum()
    srt = d.sort_values("总R", ascending=False)
    out = {"面板": label, "币数": len(d), "总笔数": int(n_tr),
           "整体期望R": tot / n_tr,
           "为正的币": float((d["期望R"] > 0).mean())}
    for k in (1, 3, 5, 10):
        rest = srt.iloc[k:]
        out[f"去掉最好{k}个"] = rest["总R"].sum() / rest["笔数"].sum()
    # what share of all the positive total-R comes from the top 10% of coins
    pos = srt["总R"].clip(lower=0).sum()
    top = srt["总R"].head(max(1, len(d) // 10)).clip(lower=0).sum()
    out["前10%币占正贡献"] = top / pos if pos > 0 else np.nan
    return out


# --------------------------------------------------------------------------- ③
def persistence(symbols: list[str], tf: str, p: MS.MaParams, which: str,
                min_each: int = 3) -> dict:
    """Split each coin's own trades in half by time; does the first half predict?"""
    fn = RULES[which]
    a, b, names = [], [], []
    for s in symbols:
        df = load(tf, s)
        t = MS.evaluate(df, fn(df, p), p)
        if len(t) < 2 * min_each:
            continue
        h = len(t) // 2
        a.append(float(t.r_multiple.iloc[:h].mean()))
        b.append(float(t.r_multiple.iloc[h:].mean()))
        names.append(s)
    if len(a) < 5:
        return {"规则": which, "可用币数": len(a)}
    a, b = np.array(a), np.array(b)
    r, pv = stats.pearsonr(a, b)
    rs, ps = stats.spearmanr(a, b)
    # the tradeable version of the question: pick the good half, keep trading it
    good = a > np.median(a)
    return {"规则": which, "可用币数": len(a),
            "前后半相关r": r, "p": pv, "秩相关": rs, "秩p": ps,
            "符号一致率": float(np.mean(np.sign(a) == np.sign(b))),
            "前半好的币后半期望R": float(b[good].mean()),
            "前半差的币后半期望R": float(b[~good].mean())}


# --------------------------------------------------------------------------- ④
def buckets(d: pd.DataFrame, col: str, label: str, k: int = 4) -> pd.DataFrame:
    """Sort coins by something knowable in advance, then look at each bucket."""
    x = d.dropna(subset=[col]).copy()
    if len(x) < k * 3:
        return pd.DataFrame()
    x["_b"] = pd.qcut(x[col].rank(method="first"), k,
                      labels=[f"{label} {i+1}/{k}" for i in range(k)])
    g = x.groupby("_b", observed=True)
    out = g.apply(lambda z: pd.Series({
        "币数": len(z), "笔数": int(z["笔数"].sum()),
        "期望R": z["总R"].sum() / z["笔数"].sum(),
        "为正的币": float((z["期望R"] > 0).mean()),
        f"{label}中位": float(z[col].median()),
    }), include_groups=False)
    return out.reset_index().rename(columns={"_b": "分组"})


# --------------------------------------------------------------------------- ⑥
def bucket_null(d: pd.DataFrame, tf: str, p: MS.MaParams, which: str, col: str,
                label: str, k: int = 4, n_draws: int = 60) -> pd.DataFrame:
    """Re-run the random-entry control inside each bucket.

    A bucket that looks good is not evidence until the same bucket's random
    control has been run.  Coins that fell 97% hand any short a profit, so the
    only way to tell an entry rule from that drift is to draw the same number of
    shorts on the same coins at random times -- which is what this does, bucket
    by bucket.
    """
    fn = RULES[which]
    x = d.dropna(subset=[col]).copy()
    if len(x) < k * 3:
        return pd.DataFrame()
    x["_b"] = pd.qcut(x[col].rank(method="first"), k,
                      labels=[f"{label} {i+1}/{k}" for i in range(k)])
    rng = np.random.default_rng(SEED)
    rows = []
    for b, grp in x.groupby("_b", observed=True):
        syms = grp["symbol"].tolist()
        real = []
        for s in syms:
            df = load(tf, s)
            t = MS.evaluate(df, fn(df, p), p)
            if len(t):
                real.append(t.assign(symbol=s))
        if not real:
            continue
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
                sig = MS.random_entries(df, p, int(row["long"]), int(row["short"]),
                                        rng, atrs)
                z = MS.evaluate(df, sig, p)
                if len(z):
                    parts.append(z)
            if parts:
                draws.append(float(pd.concat(parts, ignore_index=True).r_multiple.mean()))
        obs = float(R.r_multiple.mean())
        dr = np.array(draws)
        rows.append({"分组": str(b), "币数": len(syms), "笔数": len(R),
                     f"{label}中位": float(grp[col].median()),
                     "实际期望R": obs, "随机对照中位": float(np.median(dr)),
                     "超出对照": obs - float(np.median(dr)),
                     "p": float((dr >= obs).mean())})
    return pd.DataFrame(rows)


def main() -> None:
    p = MS.MaParams()
    every = [s for s in D.available_symbols(require=("1d",)) if s not in NOT_COINS]
    keep = [s for s in every if len(load("1d", s)) >= max(p.lens) + p.window]
    ins = [s for s in keep if s in IN_SAMPLE]
    oos = [s for s in keep if s not in IN_SAMPLE]
    fast = [s for s in D.available_symbols(require=("4h", "1d")) if s not in NOT_COINS]

    print("=" * 160)
    print("六均线系统 —— 分币种")
    print("=" * 160)
    print(f"""
  汇总数字（样本外 181 币日线，B 规则 +0.156R）是 181 个币的平均。
  平均值在这里恰恰是最不该信的统计量：中位数 5 笔/币，单个币的数字没有意义。
  所以下面只看**分布**——集中度、前后半持续性、以及能不能按事先看得见的东西分组。
""")

    # ------------------------------------------------------------------ ① table
    for tf, syms, label in (("1d", ins, "样本内 26 币 日线"),
                            ("4h", fast, "样本内 26 币 4h")):
        for which in RULES:
            d = per_symbol(syms, tf, p, which).sort_values("期望R", ascending=False)
            print("=" * 160)
            print(f"① {label}   {which}")
            print("=" * 160 + "\n")
            show(d)
            print(f"\n  为正的币 {int((d['期望R']>0).sum())}/{len(d)}"
                  f"   中位笔数 {d['笔数'].median():.0f}"
                  f"   整体期望R {d['总R'].sum()/d['笔数'].sum():+.3f}\n")

    # ------------------------------------------------------------------ ② conc
    print("=" * 160)
    print("② 集中度：把最好的几个币去掉，还剩多少")
    print("=" * 160 + "\n")
    rows, oos_tabs = [], {}
    for which in RULES:
        for tf, syms, label in (("1d", oos, f"样本外{len(oos)}币 日线"),
                                ("1d", ins, "样本内26币 日线"),
                                ("4h", fast, "样本内26币 4h")):
            d = per_symbol(syms, tf, p, which)
            if tf == "1d" and syms is oos:
                oos_tabs[which] = d
            rows.append({"规则": which} | concentration(d, label))
    c = pd.DataFrame(rows)
    o = c.copy()
    for col in ("整体期望R", "去掉最好1个", "去掉最好3个", "去掉最好5个", "去掉最好10个"):
        o[col] = o[col].map(lambda v: f"{v:+.3f}")
    for col in ("为正的币", "前10%币占正贡献"):
        o[col] = o[col].map(lambda v: f"{v:.0%}" if pd.notna(v) else "")
    print(o.to_string(index=False))
    print("""
  「去掉最好N个」如果掉得很快，说明汇总数字是被少数几个币扛起来的。
  「为正的币」如果只在 50% 附近，说明大多数币上这个规则和抛硬币没区别。
""")

    # --------------------------------------------------------- ③ OOS extremes
    for which, d in oos_tabs.items():
        d = d.sort_values("期望R", ascending=False)
        print("=" * 160)
        print(f"③ 样本外 {len(d)} 币，{which} —— 最好和最差各 12 个")
        print("=" * 160 + "\n")
        show(d[["symbol", "笔数", "胜率", "期望R", "总R", "多/空",
                "币价涨跌", "日均成交额", "年化波动"]], n=12)
        print()

    # ------------------------------------------------------------------ ④ pers
    print("=" * 160)
    print("④ 前后半持续性：某个币上「好用」，下半段还好用吗")
    print("=" * 160 + "\n")
    rows = []
    for tf, syms, label in (("1d", oos, f"样本外{len(oos)}币 日线"),
                            ("4h", fast, "样本内26币 4h")):
        for which in RULES:
            r = persistence(syms, tf, p, which)
            if "前后半相关r" in r:
                rows.append({"面板": label} | r)
    pr = pd.DataFrame(rows)
    o = pr.copy()
    for col in ("前后半相关r", "秩相关", "前半好的币后半期望R", "前半差的币后半期望R"):
        o[col] = o[col].map(lambda v: f"{v:+.3f}")
    for col in ("p", "秩p"):
        o[col] = o[col].map(lambda v: f"{v:.3f}")
    o["符号一致率"] = o["符号一致率"].map(lambda v: f"{v:.0%}")
    print(o.to_string(index=False))
    print("""
  这一栏决定「选币」有没有意义。相关性接近 0、符号一致率接近 50%，
  就说明「这个币适合这套系统」是个事后叙述，不能拿来做事前选择——
  本仓库前面对资金费率和动量做同样的检验时，得到的也是同一个结论。
""")

    # ------------------------------------------------------------------ ⑤ cuts
    print("=" * 160)
    print("⑤ 按事先看得见的东西分组（样本外 181 币，日线）")
    print("=" * 160)
    for which, d in oos_tabs.items():
        print(f"\n  --- {which} ---")
        for col, label in (("日均成交额", "成交额"), ("年化波动", "波动"),
                           ("币价涨跌", "期间涨跌")):
            b = buckets(d, col, label)
            if not len(b):
                continue
            o = b.copy()
            o["期望R"] = o["期望R"].map(lambda v: f"{v:+.3f}")
            o["为正的币"] = o["为正的币"].map(lambda v: f"{v:.0%}")
            fmtc = f"{label}中位"
            o[fmtc] = o[fmtc].map(
                lambda v: f"{v/1e6:,.0f}M" if col == "日均成交额" else f"{v:+.0%}")
            print()
            print(o.to_string(index=False))
    print("""
  上面这些分组本身还不算证据。「期间涨跌」尤其：跌了 97% 的币，
  随便什么时候做空都赚，所以这一组期望高不代表入场点有信息。
  下一节把随机对照按同样的分组各跑一遍——这才是能不能区分的判据。
""")

    # ---------------------------------------------------------- ⑥ bucket nulls
    print("=" * 160)
    print("⑥ 分组内部再做一次随机对照（样本外，日线）")
    print("=" * 160)
    print("""
  同一组的币、同样的每币多空笔数、同样的止损 ATR 倍数，只把入场时点打散。
  「超出对照」才是这一组真正的增量；「实际期望R」高但「超出对照」接近 0，
  说明那一组赚的是那批币自己的漂移，不是规则。
""")
    bn = []
    for which, d in oos_tabs.items():
        for col, label in (("币价涨跌", "期间涨跌"), ("日均成交额", "成交额")):
            t = bucket_null(d, "1d", p, which, col, label)
            if not len(t):
                continue
            bn.append(t.assign(rule=which, cut=label))
            o = t.copy()
            fmtc = f"{label}中位"
            o[fmtc] = o[fmtc].map(
                lambda v: f"{v/1e6:,.0f}M" if col == "日均成交额" else f"{v:+.0%}")
            for c in ("实际期望R", "随机对照中位", "超出对照"):
                o[c] = o[c].map(lambda v: f"{v:+.3f}")
            o["p"] = o["p"].map(lambda v: f"{v:.3f}")
            print(f"\n  --- {which} / 按{label} ---")
            print(o.to_string(index=False))
    print()

    REPORTS.mkdir(exist_ok=True)
    if bn:
        pd.concat(bn, ignore_index=True).to_csv(REPORTS / "ma_bucket_null.csv",
                                                index=False)
    pd.concat([v.assign(rule=k) for k, v in oos_tabs.items()]).to_csv(
        REPORTS / "ma_by_symbol.csv", index=False)
    print(f"  wrote {REPORTS/'ma_by_symbol.csv'}")


if __name__ == "__main__":
    main()
