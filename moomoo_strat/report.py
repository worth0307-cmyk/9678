"""回测报告和每日筛选。"""

from __future__ import annotations

import datetime as dt
import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import impulse as IW, strategy as S

OUT = Path(__file__).resolve().parent / "out"
WARM = 260   # 每只股票前 260 根（约一年）：200 日线、周线 ZigZag 都还没热起来，规则和买入持有都从这之后才算


def prepare(dfs: dict, names: dict, ann: int, log=print) -> dict:
    """每只股票算一遍指标。不到 WARM + 60 根的跳过（预热完只剩不到三个月，没法评估）"""
    feats = {}
    for n, (code, df) in enumerate(dfs.items(), 1):
        if len(df) < WARM + 60:
            log(f"  {code} 只有 {len(df)} 根，跳过")
            continue
        feats[code] = S.features(df, ann)
        if n % 10 == 0:
            log(f"  指标 {n}/{len(dfs)}")
    return feats


# ---------------------------------------------------------------- 回测

def backtest(feats: dict, names: dict, ann: int, cost: float, tag: str, null_reps: int = 300,
             seed: int = 20261001) -> str:
    rng = np.random.default_rng(seed)
    codes = list(feats)
    allidx = pd.DatetimeIndex(sorted(set().union(*[feats[c]["f"].index for c in codes])))
    idx = allidx[allidx >= min(feats[c]["f"].index[WARM] for c in codes)]
    T, NS = len(idx), len(codes)
    loc = {c: idx.get_indexer(feats[c]["f"].index) for c in codes}

    def port(per: dict) -> np.ndarray:
        """每只 1/N 固定仓位：没上市、还在预热的日子算 0"""
        acc = np.zeros(T)
        for c, x in per.items():
            m = loc[c] >= 0
            acc[loc[c][m]] += x[m]
        return acc / NS

    def roll(p: np.ndarray, o: int) -> np.ndarray:
        q = p.copy()
        q[WARM:] = np.roll(p[WARM:], o % max(1, len(p) - WARM))
        return q

    OC = {c: (feats[c]["f"]["open"].to_numpy(float), feats[c]["f"]["close"].to_numpy(float)) for c in codes}
    bh_each = {}
    for c in codes:
        x = np.r_[0.0, OC[c][1][1:] / OC[c][1][:-1] - 1.0]
        x[:WARM + 1] = 0.0                  # 规则最早第 WARM 根收盘进场、第 WARM+1 根才有收益，买入持有也从那根算
        bh_each[c] = x
    bh = port(bh_each)
    pos = {}
    for c in codes:
        live = np.arange(len(OC[c][1])) >= WARM
        pos[c] = {r: p & live for r, p in S.positions(feats[c]["f"]).items()}

    nets, expo, per_stock, trades_all, nulls = {}, {}, {}, [], {}
    for r in S.POS_RULES:
        each = {c: S.pos_net(pos[c][r], *OC[c], cost) for c in codes}
        nets[r] = port(each)
        expo[r] = float(port({c: np.r_[0.0, pos[c][r][:-1].astype(float)] for c in codes}).mean())
        per_stock[r] = {c: S.sharpe(each[c][WARM:], ann) for c in codes}
        # 随机择时：每只股票的持仓序列整体平移同一个随机天数（首尾相接），仓位比例、持有时长、
        # 换手全都不变，只是时机错开 —— 规则要是没有择时能力，真的那条不该明显好过这些
        sh = np.empty(null_reps)
        for k in range(null_reps):
            o = int(rng.integers(20, max(21, T - 20)))
            sh[k] = S.sharpe(port({c: S.pos_net(roll(pos[c][r], o), *OC[c], cost) for c in codes}), ann)
        nulls[r] = sh
    fade_null = {}
    for r, lv in S.FADE_RULES.items():
        each, held, tr_all = {}, {}, []
        for c in codes:
            f = feats[c]["f"]
            wv = [w for w in feats[c]["waves"][lv] if w["conf"] >= WARM]
            _, _, tr, _ = S.fade(f, wv, cost, sides=(1, -1))
            # 组合只做多：重算一遍只做多的净值；做空的逐笔照样统计
            each[c], held[c], _, _ = S.fade(f, wv, cost, sides=(1,))
            for t in tr:
                t.update(code=c, name=names.get(c, ""), rule=r)
            tr_all += tr
        nets[r] = port(each)
        expo[r] = float(port(held).mean())
        per_stock[r] = {c: S.sharpe(each[c][WARM:], ann) for c in codes}
        trades_all += tr_all
        fade_null[r] = (bracket_null(feats, tr_all, cost, rng), shift_null(feats, tr_all, cost, rng))
    per_stock["买入持有"] = {c: S.sharpe(bh_each[c][WARM:], ann) for c in codes}

    # ---- 汇总表
    rows = []
    half = T // 2
    for r in S.RULES:
        x, e = nets[r], expo[r]
        if not x.std() > 0:
            rows.append(dict(规则=r, 说明=S.RULES[r], 结论="没有交易"))
            continue
        bx = bh * e
        ex = x - bx
        sr, sb = S.sharpe(x, ann), S.sharpe(bh, ann)
        p_boot = S.boot_diff(x, bh, ann, rng)
        p_shift = float((nulls[r] >= sr).mean()) if r in nulls else np.nan
        dsr = S.deflated(S.sharpe(ex, ann), len(S.RULES), T, ann,
                         float(pd.Series(ex).skew()), float(pd.Series(ex).kurtosis() + 3))
        h1 = S.sharpe(x[:half], ann) - S.sharpe(bh[:half], ann)
        h2 = S.sharpe(x[half:], ann) - S.sharpe(bh[half:], ann)
        row = dict(规则=r, 说明=S.RULES[r], 年化=x.mean() * ann, 复利总收益=np.prod(1 + x) - 1,
                   Sharpe=sr, 最大回撤=S.max_dd(x), 平均仓位=e,
                   同仓位买入持有年化=bx.mean() * ann, 同仓位买入持有回撤=S.max_dd(bx),
                   Sharpe差=sr - sb, 前半段Sharpe差=h1, 后半段Sharpe差=h2,
                   p_自助法=p_boot, p_随机择时=p_shift, 去偏Sharpe=dsr)
        if r in S.FADE_RULES:
            tl = [t for t in trades_all if t["rule"] == r and t["side"] > 0]
            real = float(np.mean([t["ret"] for t in tl])) if tl else np.nan
            nb, ns = (fade_null[r][0].get(1), fade_null[r][1].get(1))
            row.update(笔数=len(tl), 独立时段=episodes(tl), 胜率=np.mean([t["ret"] > 0 for t in tl]) if tl else np.nan,
                       平均每笔=real, p_随机括号=float((nb >= real).mean()) if nb is not None else np.nan,
                       p_整体平移=float((ns >= real).mean()) if ns is not None else np.nan)
        row["结论"] = verdict_fade(row) if r in S.FADE_RULES else verdict(row)
        rows.append(row)
    rows.append(dict(规则="B&H", 说明="等权买入持有（不计费用）", 年化=bh.mean() * ann,
                     复利总收益=np.prod(1 + bh) - 1, Sharpe=S.sharpe(bh, ann), 最大回撤=S.max_dd(bh), 平均仓位=1.0))
    tab = pd.DataFrame(rows)

    # ---- 落盘
    OUT.mkdir(exist_ok=True)
    stem = OUT / f"backtest_{tag}_{dt.date.today():%Y%m%d}"
    tab.to_csv(f"{stem}_rules.csv", index=False, encoding="utf-8-sig")
    ps = pd.DataFrame(per_stock)
    ps.insert(0, "名称", [names.get(c, "") for c in ps.index])
    ps.to_csv(f"{stem}_stocks.csv", encoding="utf-8-sig")
    if trades_all:
        pd.DataFrame(trades_all).drop(columns=["entry_i", "exit_i"]).to_csv(f"{stem}_trades.csv", index=False,
                                                                            encoding="utf-8-sig")
    try:
        equity_png(idx, nets, bh, ann, Path(f"{stem}_equity.png"), tag)
        png = f"{stem}_equity.png"
    except Exception as e:                       # 没装 matplotlib 或没中文字体，不影响表
        png = f"（没出图：{e}）"
    md = render(tab, ps, trades_all, fade_null, idx, NS, ann, cost, tag, png,
                fade_detail(feats, trades_all, idx, cost, rng, bh, ann))
    Path(f"{stem}.md").write_text(md, encoding="utf-8")
    return md + f"\n\n文件：{stem}.md / _rules.csv / _stocks.csv / _trades.csv / _equity.png\n"


def bracket_null(feats: dict, trades: list, cost: float, rng, reps: int = 200) -> dict:
    """五浪规则的零假设：同一只股票、同方向、随便哪天开盘进场，目标和止损离进场价的百分比、
    持有上限都和真交易一样。返回 {方向: 每轮随机交易平均收益的数组}"""
    out = {}
    for side in (1, -1):
        tl = [t for t in trades if t["side"] == side]
        if not tl:
            continue
        acc = np.zeros((reps, len(tl)))
        for j, t in enumerate(tl):
            f = feats[t["code"]]["f"]
            O, H, L, C = (f[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
            N = len(C)
            for k, e in enumerate(rng.integers(WARM, max(WARM + 1, N - 2), reps)):
                en = O[e]
                tgt = en * (1 + side * t["dt"])
                stp = en * (1 - side * t["ds"])
                x, px, _ = S.sim(O, H, L, C, int(e), side, tgt, stp, N, None, max(1, t["hold"]))
                acc[k, j] = side * (px / en - 1.0) - 2 * cost
        out[side] = acc.mean(axis=1)
    return out


def shift_null(feats: dict, trades: list, cost: float, rng, reps: int = 500) -> dict:
    """更严的零假设：同一方向的全部交易一起往后挪同一个随机天数（每只股票在自己的区间里首尾相接），
    方向、目标和止损距离、持有上限都不变。

    几只股票同一天一起进场的「扎堆」原样保留。自选股里的大盘科技股同涨同跌，一次大跌往往让好几只
    同一周一起出信号；逐笔独立随机会把它们当成好几次独立的机会，p 值就偏乐观，这里不会。"""
    cache = {}
    out = {}
    for side in (1, -1):
        tl = [t for t in trades if t["side"] == side]
        if not tl:
            continue
        acc = np.zeros(reps)
        for k in range(reps):
            o = int(rng.integers(20, 10_000))
            rets = []
            for t in tl:
                if t["code"] not in cache:
                    f = feats[t["code"]]["f"]
                    cache[t["code"]] = tuple(f[c].to_numpy(dtype=float) for c in ("open", "high", "low", "close"))
                O, H, L, C = cache[t["code"]]
                N = len(C)
                e = WARM + (t["entry_i"] - WARM + o) % max(1, N - 2 - WARM)
                en = O[e]
                x, px, _ = S.sim(O, H, L, C, int(e), side, en * (1 + side * t["dt"]), en * (1 - side * t["ds"]),
                                 N, None, max(1, t["hold"]))
                rets.append(side * (px / en - 1.0) - 2 * cost)
            acc[k] = np.mean(rets)
        out[side] = acc
    return out


def episodes(trades: list, gap_days: int = 14) -> int:
    """进场日期相隔不到两周的算同一个时段：同一次大跌里好几只一起出信号，只算一次"""
    ds = sorted(t["date"] for t in trades)
    return int(sum(1 for i, d in enumerate(ds) if i == 0 or (d - ds[i - 1]).days > gap_days))


def verdict_fade(r: dict) -> str:
    """五浪规则平均仓位只有百分之几，大部分时间空仓，和买入持有比 Sharpe 没有意义；看单笔有没有赢随机进场"""
    n = r.get("笔数", 0)
    if not n:
        return "没有做多交易"
    p = max(r.get("p_随机括号", np.nan), r.get("p_整体平移", np.nan))
    ep = r.get("独立时段", n)
    if p < 0.05 and ep >= 20:
        return "单笔显著好于随机进场（仍需样本外再看）"
    if p < 0.05:
        return f"单笔好于随机进场，但只有 {n} 笔 / {ep} 个独立时段，不够下结论"
    if p < 0.10:
        return "单笔偏好于随机进场，证据不够"
    return "单笔不比随机进场好"


def verdict(r: dict) -> str:
    d, pb, psh = r["Sharpe差"], r["p_自助法"], r.get("p_随机择时", np.nan)
    pbr = r.get("p_随机括号", np.nan)
    p_main = psh if not np.isnan(psh) else pbr
    if d > 0 and pb < 0.05 and p_main < 0.05 and r["去偏Sharpe"] > 0.9:
        return "显著好于同仓位买入持有（仍需样本外再看）"
    if d > 0 and (pb < 0.10 or p_main < 0.10):
        return "偏好，但证据不够"
    if d > 0:
        return "略好于同仓位买入持有，不显著（像运气）"
    return "不如同仓位买入持有"


def pct(v, s=True):
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else (f"{v:+.1%}" if s else f"{v:.0%}")


def num(v, fmt="+.2f"):
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else format(v, fmt)


SIZING = ((0.05, 20), (0.10, 10), (0.20, 5))      # (每笔占当时资金的比例, 最多同时几笔)：满仓都是 100%，不加杠杆


def sized(feats: dict, trades: list, idx, cost: float, frac: float, kmax: int) -> tuple[np.ndarray, np.ndarray, int, int]:
    """按笔分配资金：每个信号拿当时资金的 frac，最多同时 kmax 笔，满了就跳过新信号（按进场日期先到先得）。
    仓位每天按 frac 再平衡（近似），空闲资金不计利息。返回 (每日净收益, 每日在场比例, 做了几笔, 跳过几笔)。"""
    T = len(idx)
    net = np.zeros(T)
    expo = np.zeros(T)
    open_until: list[int] = []
    took = skipped = 0
    for t in sorted(trades, key=lambda t: (t["date"], t["code"])):
        f = feats[t["code"]]["f"]
        e = idx.get_loc(f.index[t["entry_i"]])
        x = idx.get_loc(f.index[t["exit_i"]])
        open_until = [u for u in open_until if u >= e]      # 当天开盘前已经平掉的才腾出位置
        if len(open_until) >= kmax:
            skipped += 1
            continue
        open_until.append(x)
        took += 1
        C = f["close"].to_numpy(dtype=float)
        i0, i1 = t["entry_i"], t["exit_i"]
        prices = np.r_[t["entry"], C[i0:i1], t["exit"]] if i1 > i0 else np.r_[t["entry"], t["exit"]]
        r = prices[1:] / prices[:-1] - 1.0                  # 进场那天开盘 → 收盘，…，最后一天 → 离场价
        days = [idx.get_loc(d) for d in f.index[i0:i1 + 1]]
        net[days] += frac * r[: len(days)]
        expo[days] += frac
        net[e] -= frac * cost
        if t["why"] != "持有中":
            net[x] -= frac * cost
    return net, expo, took, skipped


def fade_detail(feats: dict, trades: list, idx, cost: float, rng, bh=None, ann: int = 252) -> list[str]:
    """五浪做多拆开看：前后半段、逐年、持有多久、括号多大，以及消融 —— 艾略特那几条规则到底有没有用"""
    L = []
    mid = idx[len(idx) // 2]
    for r, lv in S.FADE_RULES.items():
        tl = [t for t in trades if t["rule"] == r and t["side"] > 0]
        if not tl:
            continue
        tt = pd.DataFrame(tl)
        tt["date"] = pd.to_datetime(tt["date"])
        a, b = tt[tt["date"] < mid], tt[tt["date"] >= mid]
        half = lambda g: f"{len(g)} 笔，胜率 {(g['ret'] > 0).mean():.0%}，平均 {g['ret'].mean():+.1%}" if len(g) else "0 笔"  # noqa: E731
        yr = tt.groupby(tt["date"].dt.year)["ret"].agg(["count", "mean"])
        L += [f"**{r} {S.RULES[r]}**（{len(tt)} 笔）", "",
              f"- 前半段（~ {mid.date()}）{half(a)}；后半段 {half(b)}",
              "- 逐年：" + "　".join(f"{y} {int(c)} 笔 {m:+.1%}" for y, (c, m) in yr.iterrows()),
              f"- 持有中位 {tt['hold'].median():.0f} 个交易日；目标平均在进场价上方 {tt['dt'].mean():.1%}，"
              f"止损（收盘跌破浪5）平均在下方 {tt['ds'].mean():.1%}；最好一笔 {tt['ret'].max():+.1%}，最差一笔 {tt['ret'].min():+.1%}"]
        # 消融：同样的进出场，用在任何一次「净跌五段、刚确认反弹」上；比满足艾略特规则的和不满足的
        ab = []
        for c, F in feats.items():
            ew_confs = {w["conf"] for w in F["waves"][lv]}
            cand = [w for w in F["anyw"][lv] if w["conf"] >= WARM and not w["bull"]]
            _, _, tr, _ = S.fade(F["f"], cand, cost, sides=(1,))
            for t in tr:
                t["ew"] = (t["entry_i"] - 1) in ew_confs
            ab += tr
        ew = np.array([t["ret"] for t in ab if t["ew"]])
        rest = np.array([t["ret"] for t in ab if not t["ew"]])
        if len(ew) and len(rest):
            allr = np.r_[ew, rest]
            perm = np.array([allr[rng.choice(len(allr), len(ew), replace=False)].mean() for _ in range(5000)])
            p = float((perm >= ew.mean()).mean())
            L.append(f"- 消融：同样的进出场用在**任何一次净跌五段之后的反弹**上（不管艾略特规则）。"
                     f"满足艾略特规则的 {len(ew)} 笔平均 {ew.mean():+.1%}、胜率 {(ew > 0).mean():.0%}；"
                     f"不满足的 {len(rest)} 笔平均 {rest.mean():+.1%}、胜率 {(rest > 0).mean():.0%}；置换 p = {p:.3f}。"
                     + ("艾略特规则有增量。" if p < 0.05 else
                        "**看不出艾略特规则有增量**：起作用的主要是「跌了一大段、开始反弹就买」本身。"))
        # 实际怎么用：只做这一条，每个信号分一份固定比例的资金
        if bh is not None:
            L += ["", f"按笔分配资金（只做 {r} 做多，别的都不做；满仓 100%、不加杠杆；空闲资金不计利息）：", "",
                  "| 每笔资金 | 最多同时 | 年化 | 总收益 | Sharpe | 最大回撤 | 平均仓位 | 做了 / 满了跳过 | 同仓位买入持有 年化 / 回撤 |",
                  "|---|---|---|---|---|---|---|---|---|"]
            for frac, kmax in SIZING:
                x, ex, took, skip = sized(feats, tl, idx, cost, frac, kmax)
                e = float(ex.mean())
                bx = bh * e
                L.append(f"| {frac:.0%} | {kmax} 笔 | {pct(x.mean() * ann)} | {pct(np.prod(1 + x) - 1)} | "
                         f"{num(S.sharpe(x, ann))} | {pct(S.max_dd(x))} | {pct(e, False)} | {took} / {skip} | "
                         f"{pct(bx.mean() * ann)} / {pct(S.max_dd(bx))} |")
        L.append("")
    if L:
        L = ["", "## 五浪做多：拆开看", "",
             "前后两段、每一年都赚，才像真的；消融回答「是数浪有用，还是只是跌多了反弹」。", ""] + L
    return L


def render(tab, ps, trades, fade_null, idx, NS, ann, cost, tag, png, extra=()) -> str:
    L = [f"# 回测：{tag}", "",
         f"{NS} 只 · {idx[0].date()} ~ {idx[-1].date()}（{len(idx)} 个交易日）· 每只固定 1/{NS} 仓 · 只做多 · "
         f"收盘出信号、次日开盘成交 · 每边 {cost * 1e4:.1f}bp · 每只前 {WARM} 根是指标预热期，不计入", "",
         "## 总表", "",
         "| 规则 | 说明 | 年化 | 总收益 | Sharpe | 回撤 | 平均仓位 | 同仓位买入持有 年化 / 回撤 | Sharpe 差 | p 自助法 | p 随机择时/括号 | 去偏 | 结论 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in tab.iterrows():
        if r["规则"] == "B&H":
            continue
        if "年化" not in r or pd.isna(r.get("年化")):
            L.append(f"| {r['规则']} | {r['说明']} | | | | | | | | | | | {r['结论']} |")
            continue
        p2 = r.get("p_随机择时") if not pd.isna(r.get("p_随机择时", np.nan)) else \
            max(r.get("p_随机括号", np.nan), r.get("p_整体平移", np.nan))
        L.append(f"| {r['规则']} | {r['说明']} | {pct(r['年化'])} | {pct(r['复利总收益'])} | {num(r['Sharpe'])} | "
                 f"{pct(r['最大回撤'])} | {pct(r['平均仓位'], False)} | {pct(r['同仓位买入持有年化'])} / "
                 f"{pct(r['同仓位买入持有回撤'])} | {num(r['Sharpe差'])} | {num(r['p_自助法'], '.3f')} | "
                 f"{num(p2, '.3f')} | {num(r['去偏Sharpe'], '.2f')} | {r['结论']} |")
    b = tab[tab["规则"] == "B&H"].iloc[0]
    L += ["", f"等权买入持有（不计费用）：年化 {pct(b['年化'])}，总收益 {pct(b['复利总收益'])}，"
              f"Sharpe {num(b['Sharpe'])}，最大回撤 {pct(b['最大回撤'])}。", "",
          "怎么读：", "",
          "- **Sharpe 差**：规则的 Sharpe 减买入持有的 Sharpe。只做多的规则一半时间空仓，回撤自然小，"
          "所以不能拿它和满仓买入持有比收益或回撤，要和**同样仓位**的买入持有比；Sharpe 不随仓位缩放，差就是择时带来的部分。",
          "- **p 自助法**：分块自助法下 Sharpe 差 ≤ 0 的概率。",
          "- **p 随机择时**：把每只股票的持仓序列整体平移一个随机天数，仓位比例、每段持有多久、换手都不变，"
          "只是时机错开；真规则的 Sharpe 不如这些随机版本的比例。",
          "- **五浪规则（S3、S3M）另看**：平均仓位只有百分之几，大部分时间空仓，它的 Sharpe 差没有意义，"
          "结论按**单笔**下：和**随机括号**（同方向、同目标和止损距离、同持有上限，随便哪天进场）、"
          "**整体平移**（全部交易一起挪同一个随机天数，保留几只股票同一天一起进场的扎堆）比，表里取两个 p 里大的那个。",
          f"- **去偏**：一共试了 {len(S.RULES)} 套规则，挑最好的那套天然占便宜；这是扣掉这层运气之后，"
          "「超额（规则 − 同仓位买入持有）的真实 Sharpe > 0」的概率。",
          "- **前/后半段 Sharpe 差**：两段方向一致才像真的。", "",
          "| 规则 | 前半段 Sharpe 差 | 后半段 Sharpe 差 |", "|---|---|---|"]
    for _, r in tab.iterrows():
        if "前半段Sharpe差" in r and not pd.isna(r.get("前半段Sharpe差", np.nan)):
            L.append(f"| {r['规则']} | {num(r['前半段Sharpe差'])} | {num(r['后半段Sharpe差'])} |")

    # 五浪逐笔
    if trades:
        tt = pd.DataFrame(trades)
        L += ["", "## 五浪反着做：逐笔", "",
              "| 规则 | 方向 | 笔数 | 独立时段 | 胜率 | 平均每笔 | 中位 | 随机括号平均 | p 随机括号 | p 整体平移 | 离场原因 |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for (r, side), g in tt.groupby(["rule", "side"]):
            nb, ns = (fade_null.get(r, ({}, {}))[0].get(side), fade_null.get(r, ({}, {}))[1].get(side))
            real = g["ret"].mean()
            why = "，".join(f"{k} {v}" for k, v in g["why"].value_counts().items())
            L.append(f"| {r} | {'跌完五浪做多' if side > 0 else '涨完五浪做空（不进组合）'} | {len(g)} | "
                     f"{episodes(g.to_dict('records'))} | "
                     f"{(g['ret'] > 0).mean():.0%} | {pct(real)} | {pct(g['ret'].median())} | "
                     f"{pct(nb.mean()) if nb is not None else ''} | "
                     f"{num(float((nb >= real).mean()), '.3f') if nb is not None else ''} | "
                     f"{num(float((ns >= real).mean()), '.3f') if ns is not None else ''} | {why} |")
        L += ["", "独立时段：进场日期相隔不到两周的算同一次（一次大跌里好几只一起出信号）。"
                  "笔数再多，独立时段少，能下的结论也少。"]
        L += list(extra)

    # 逐只
    L += ["", "## 逐只：Sharpe 高于这只股票自己的买入持有的比例", ""]
    for r in ("S0", "S1", "S2", "S4"):
        if r in ps:
            ok = (ps[r] > ps["买入持有"]).sum()
            L.append(f"- {r} {S.RULES[r]}：{ok} / {len(ps)} 只")
    L += ["", "逐只明细在 _stocks.csv。**别从里面挑「这只股票上有效」的规则** —— 几十只里总有几只碰巧对得上，"
              "那是挑出来的，不是规则的本事。", "", f"净值图：{png}", ""]
    return "\n".join(L)


def equity_png(idx, nets, bh, ann, path: Path, tag: str) -> None:
    import logging
    import warnings

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, NullLocator

    warnings.filterwarnings("ignore", module="matplotlib")
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "PingFang SC", "Heiti SC", "SimHei",
                                       "WenQuanYi Zen Hei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.plot(idx, np.cumprod(1 + bh), color="#9e9e9e", lw=1.6,
            label=f"等权买入持有  Sharpe {S.sharpe(bh, ann):+.2f}")
    cols = {"S0": "#607d8b", "S1": "#1976d2", "S2": "#e53935", "S2D": "#ef9a9a", "S2W": "#b71c1c",
            "S3": "#43a047", "S3M": "#a5d6a7", "S4": "#8e24aa"}
    for r, x in nets.items():
        if x.std() > 0:
            ax.plot(idx, np.cumprod(1 + x), color=cols.get(r), lw=1.3, ls="--" if r in ("S2D", "S2W", "S3M") else "-",
                    label=f"{r} {S.RULES[r].replace('⬆', '↑')}  Sharpe {S.sharpe(x, ann):+.2f}")
    ax.set_yscale("log")
    lo, hi = ax.get_ylim()
    ticks = [t for t in (0.2, 0.3, 0.5, 0.7, 1, 1.5, 2, 3, 5, 7, 10, 15, 20, 30, 50) if lo <= t <= hi]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_yticklabels([f"{t:g}" for t in ticks])
    ax.set_ylabel("净值（起点 = 1）")
    ax.axhline(1, color="#bdbdbd", lw=0.8)
    ax.set_title(f"{tag}：各规则净值（每只 1/N 仓，只做多，次日开盘成交，含费用）", loc="left")
    ax.legend(fontsize=8.5, frameon=False, loc="upper left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------- 每日筛选

def _wave(k, done, f):
    if k is None or (isinstance(k, float) and np.isnan(k)) or int(k) < 0:
        return ""
    k, f = int(k), int(f)
    return f"{IW.NAMES[k]}走完→{IW.NAMES[min(f, 8)]}" if done else f"{IW.NAMES[k]}进行中"


def _arrow(t):
    return {1: "⬆", -1: "⬇"}.get(int(t) if not pd.isna(t) else 0, "·")


def screen(feats: dict, names: dict, cost: float, tag: str) -> tuple[pd.DataFrame, str]:
    rows = []
    for c, F in feats.items():
        f = F["f"]
        last, prev = f.iloc[-1], f.iloc[-2]
        pos = S.positions(f)
        st = f["st"].to_numpy()
        flip = int(len(st) - 1 - np.flatnonzero(st != st[-1])[-1]) if (st != st[-1]).any() else len(st)
        chg = [f"{r}{'进' if pos[r][-1] else '出'}" for r in ("S0", "S1", "S2", "S4") if pos[r][-1] != pos[r][-2]]
        # 五浪只出做多：「涨完五浪做空」在 7姐妹 和 80 只大盘股上都比随机进场还差
        fade_s, s3 = [], False
        for r, lv in S.FADE_RULES.items():
            _, _, tr, pend = S.fade(f, F["waves"][lv], cost, sides=(1,))
            for p in pend:
                fade_s.append(f"{r} 今天确认跌完五浪 → 明天开盘买，目标 {p['tgt']:.2f}，收盘跌破 {p['stop']:.2f} 就卖")
            for t in tr:
                if t["why"] == "持有中":
                    fade_s.append(f"{r} 多单持有中（{t['date'].date()} 开盘 {t['entry']:.2f} 进），"
                                  f"目标 {t['tgt']:.2f}，收盘跌破 {t['stop']:.2f} 就卖")
            if r == "S3" and (pend or any(t["why"] == "持有中" for t in tr)):
                s3 = True
        invD = f"{'涨过' if last['invUpD'] else '跌破'} {last['invD']:.2f}" if not pd.isna(last["invD"]) else ""
        invW = f"{'涨过' if last['invUpW'] else '跌破'} {last['invW']:.2f}" if not pd.isna(last["invW"]) else ""
        rows.append({
            "代码": c, "名称": names.get(c, ""), "日期": f.index[-1].date(), "收盘": round(last["close"], 3),
            "涨跌": f"{last['close'] / prev['close'] - 1:+.2%}",
            "D": _arrow(last["trD"]), "W": _arrow(last["trW"]), "W本周至今": _arrow(last["trW_live"]),
            "ST": "多" if last["st"] == 1 else "空", "ST线(止损)": round(last["st_line"], 3), "ST已持续(天)": flip,
            "D位置": _wave(last["kD"], bool(last["fD"] > last["kD"]), last["fD"]), "D失效价": invD,
            "W位置": _wave(last["kW"], bool(last["fW"] > last["kW"]) if not pd.isna(last["fW"]) else False,
                          last["fW"] if not pd.isna(last["fW"]) else -1), "W失效价": invW,
            "200日线": round(last["ma200"], 3) if not pd.isna(last["ma200"]) else None,
            "50周线": round(last["ma50w"], 3) if not pd.isna(last["ma50w"]) else None,
            "一年区间位置": f"{last['fib_pos']:.0%}" if not pd.isna(last["fib_pos"]) else "",
            **{r: "✓" if pos[r][-1] else "" for r in ("S0", "S1", "S2", "S4")}, "S3": "✓" if s3 else "",
            "今日变化": " ".join(chg), "五浪": "；".join(fade_s),
        })
    t = pd.DataFrame(rows).sort_values(["S3", "S4", "S2", "S1", "代码"], ascending=[False, False, False, False, True])
    OUT.mkdir(exist_ok=True)
    path = OUT / f"screen_{tag}_{dt.date.today():%Y%m%d}.csv"
    t.to_csv(path, index=False, encoding="utf-8-sig")
    return t, str(path)
