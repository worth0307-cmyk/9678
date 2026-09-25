"""推动浪指标（pine/impulse_wave.pine）拿来交易，能赚钱吗.

    python scripts/60_impulse_wave_backtest.py

指标自己不给买卖信号，它在图上画三样东西。按这三样东西各定一种最直接的用法：

  A  做调整        一组 1-5 刚确认（绿/红三角那一根），反着做，目标就是那条
                   「已确认推动浪的调整目标」横线（浪5 终点回撤全长 × 0.5），
                   止损就是让那条线作废的条件：收盘越过浪5 终点。
                   **进出场和图上那条线的起止完全一致**：线出现就开仓，线断开就平仓
                   （碰到目标 / 收盘越过浪5 / 同级别又确认了一组新的 1-5）。
                   这是艾略特理论本身的说法：五浪走完，接下来是 ABC 调整。

  B  跟推演方向    每根收盘时看推演的下一段往哪走，就持那个方向。
                   B1 全部腿多空   B2 只做多   B3 只做推动方向的腿（浪 1/3/5）
                   B4 只做浪3 —— 艾略特交易者最爱的那一段

口径和 53、58 号脚本一致：六个币、日线为主（4 小时作稳健性）、每币固定 1/6 仓、
吃单 5bp + 滑点 1.5bp、真实资金费率、信号在收盘算出、收盘成交。每一根都只用
到那一根为止的数据（ZigZag、确认、推演全部逐根重算），没有未来函数。

指标默认只开「细」「中」两个级别；「粗」也一并测了。日线的阈值按指标的自动档：
ATR×2 / ×4 / ×8。

零假设：
  A 用「随机入场、同样的括号」：同一个币、同一个方向、随便哪一根入场，目标和
    止损离入场的百分比和真交易一样，持有上限也一样。数浪如果有用，真交易的平均
    收益应该明显好过这些随机括号。
  B 的只做多版本和 53、58 号一样：拿「同敞口买入持有」比，分块自助法看 Sharpe 差。
"""

from __future__ import annotations

import math
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from vibt import data as D, ingest as I, metrics as M  # noqa: E402

SC = import_module("59_impulse_scenario_check")     # 推演引擎的逐行移植，59 号已经对过

pd.set_option("display.width", 240)
ROOT = Path(__file__).resolve().parent.parent
COINS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "TAOUSDT", "HYPEUSDT")
LEVELS = {"细": 2.0, "中": 4.0, "粗": 8.0}          # 日线 / 4h 的自动档
COST = (5.0 + 1.5) * 1e-4                            # 每边
ANN = {"1d": 365.0, "4h": 365.0 * 6}
RNG = np.random.default_rng(20260925)
PRM = dict(need5=True, ovTol=0.0, maxSpan=300, projRetr=0.5)


# ---------------------------------------------------------------- 逐根重放指标

def walk(df: pd.DataFrame, mult: float, scen: bool = True) -> dict:
    """逐根跑 ZigZag → 检测已确认 1-5 → 推演，和 pine 的执行顺序一样。

    返回
      waves   每组已确认 1-5：方向、确认那根、浪5 终点、调整目标
      legdir  每根收盘时推演下一段的方向（+1 / −1 / 0 = 没有推演）
      leg     每根收盘时推演下一段是周期里的第几段（0..9 = 1 2 3 4 5 A B C 1′ 2′，−1 = 无）
    """
    H, L, C = (df[k].to_numpy(dtype=float) for k in ("high", "low", "close"))
    A = SC.atr_w(H, L, C)
    N = len(C)
    pv: list = []
    d, ext, ebi, e2, e2bi = 0, np.nan, -1, np.nan, -1
    waves, anyw = [], []
    legdir = np.zeros(N)
    leg = np.full(N, -1)
    for i in range(N):
        conf = False
        if not np.isnan(A[i]):
            thr = A[i] * mult
            if d == 0:
                if np.isnan(ext):
                    ext, ebi, e2, e2bi = H[i], i, L[i], i
                else:
                    if H[i] > ext:
                        ext, ebi = H[i], i
                    if L[i] < e2:
                        e2, e2bi = L[i], i
                    if C[i] < ext - thr:
                        pv.append((ebi, ext, True)); d, ext, ebi = -1, L[i], i; conf = True
                    elif C[i] > e2 + thr:
                        pv.append((e2bi, e2, False)); d, ext, ebi = 1, H[i], i; conf = True
            elif d == 1:
                if H[i] > ext:
                    ext, ebi = H[i], i
                if C[i] < ext - thr:
                    pv.append((ebi, ext, True)); d, ext, ebi = -1, L[i], i; conf = True
            else:
                if L[i] < ext:
                    ext, ebi = L[i], i
                if C[i] > ext + thr:
                    pv.append((ebi, ext, False)); d, ext, ebi = 1, H[i], i; conf = True
        # detect()：和 pine 一样，只在刚确认一个枢轴的那根检查最近 6 个
        if conf and len(pv) >= 6:
            p = pv[-6:]
            isH = [x[2] for x in p]; bi = [x[0] for x in p]; pr = [x[1] for x in p]
            bull = isH == [False, True, False, True, False, True]
            bear = isH == [True, False, True, False, True, False]
            l1, l3, l5 = abs(pr[1] - pr[0]), abs(pr[3] - pr[2]), abs(pr[5] - pr[4])
            ok = ((bull or bear) and all(bi[k + 1] > bi[k] for k in range(5))
                  and (bi[5] - bi[0]) <= PRM["maxSpan"]
                  and (pr[2] > pr[0] if bull else pr[2] < pr[0]) and not (l3 < l1 and l3 < l5)
                  and (pr[4] > pr[1] - l1 * PRM["ovTol"] if bull else pr[4] < pr[1] + l1 * PRM["ovTol"])
                  and ((not PRM["need5"]) or (pr[5] > pr[3] if bull else pr[5] < pr[3])))
            if ok:
                full = abs(pr[5] - pr[0])
                waves.append(dict(bull=bull, conf=i, p5=pr[5], p0=pr[0],
                                  tgt=pr[5] - full * PRM["projRetr"] if bull else pr[5] + full * PRM["projRetr"]))
            # 消融：不管艾略特规则，任何一个刚确认的转折，只要最近五段净方向和它一致，都当成「五浪走完」
            up = isH[5]
            if (pr[5] > pr[0]) if up else (pr[5] < pr[0]):
                full = abs(pr[5] - pr[0])
                anyw.append(dict(bull=up, conf=i, p5=pr[5], p0=pr[0], ew=bool(ok),
                                 tgt=pr[5] - full * PRM["projRetr"] if up else pr[5] + full * PRM["projRetr"]))
        # 推演：用到这一根为止的枢轴和当前极值
        if scen and d != 0 and len(pv) >= 2:
            r = SC.scenario(dict(pv=pv, d=d, ext=ext, ebi=ebi, N=i + 1, close=C[i]))
            if r is not None:
                f = r["k"] + 1 if r["done"] else r["k"]
                if f <= 9:
                    leg[i] = f
                    legdir[i] = r["s"] * (1.0 if f % 2 == 0 else -1.0)
    return dict(waves=waves, anyw=anyw, legdir=legdir, leg=leg)


# ---------------------------------------------------------------- 资金费率、收益

_FUND: dict = {}


def funding(sym: str, idx: pd.DatetimeIndex, tf: str) -> np.ndarray:
    """每根K线区间内发生的资金费率之和（持有这一根的多头要付的比例）"""
    key = (sym, tf)
    if key not in _FUND:
        fr = I._load_funding(ROOT / "data" / "funding" / f"{sym}_funding.csv.gz")["funding_rate"].astype(float)
        _FUND[key] = fr.groupby(fr.index.floor("D" if tf == "1d" else "4h")).sum()
    return _FUND[key].reindex(idx).fillna(0.0).to_numpy()


def bar_positions(pos: np.ndarray, C: np.ndarray, fu: np.ndarray) -> np.ndarray:
    """收盘定仓、收盘成交：第 t 根收盘定下的仓位，吃第 t+1 根的收盘到收盘"""
    N = len(C)
    net = np.zeros(N)
    prev = 0.0
    for t in range(N - 1):
        p = pos[t]
        net[t + 1] = p * (C[t + 1] / C[t] - 1.0) - abs(p - prev) * COST - p * fu[t + 1]
        prev = p
    return net


def fade_trades(df: pd.DataFrame, waves: list, fu: np.ndarray, expire: bool = True) -> tuple[np.ndarray, list]:
    """用法 A：确认那根收盘反向开仓，线断开就平。逐根给出这个币的净收益，和每笔交易。

    expire=False：不因为「同级别又确认了一组」而平仓，只看目标和止损 —— 消融对比用，
    那时各笔交易会重叠，只看逐笔统计，不看净值。"""
    O, H, L, C = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    N = len(C)
    net = np.zeros(N)
    trades = []
    confs = sorted(w["conf"] for w in waves)
    for w in waves:
        c0 = w["conf"]
        side = -1.0 if w["bull"] else 1.0
        tgt, p5 = w["tgt"], w["p5"]
        # 确认那一根盘中已经碰过目标 → 图上那条线一出现就断开，不交易；最后一根确认的也没法交易
        if ((L[c0] <= tgt) if w["bull"] else (H[c0] >= tgt)) or c0 >= N - 1:
            continue
        nxt = next((c for c in confs if c > c0), N) if expire else N   # 同级别下一组确认会让这条线作废
        entry = C[c0]
        net[c0] -= COST
        i, why, px = c0, "数据结束", C[-1]
        for i in range(c0 + 1, N):
            gap = (O[i] <= tgt) if w["bull"] else (O[i] >= tgt)
            touch = (L[i] <= tgt) if w["bull"] else (H[i] >= tgt)
            stop = (C[i] > p5) if w["bull"] else (C[i] < p5)
            if gap:
                px, why = O[i], "到目标"
            elif touch:
                px, why = tgt, "到目标"
            elif stop:
                px, why = C[i], "收盘越过浪5"
            elif i == nxt:
                px, why = C[i], "新的一组确认"
            else:
                net[i] += side * (C[i] / C[i - 1] - 1.0) - side * fu[i]
                continue
            net[i] += side * (px / C[i - 1] - 1.0) - side * fu[i] - COST
            break
        else:
            px = C[-1]
        trades.append(dict(side=side, entry_i=c0, exit_i=i, entry=entry, exit=px, why=why,
                           ret=side * (px / entry - 1.0) - 2 * COST,
                           dt=abs(tgt / entry - 1.0), ds=abs(p5 / entry - 1.0), hold=i - c0,
                           t_entry=df.index[c0]))
    return net, trades


def bracket_null(df: pd.DataFrame, trades: list, reps: int = 200) -> np.ndarray:
    """随机入场、同样的括号：每笔真交易配 reps 笔随机入场（同方向、同目标距离、同止损距离、
    同持有上限），返回每轮随机交易的平均收益"""
    O, H, L, C = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    N = len(C)
    out = np.zeros((reps, len(trades)))
    for j, tr in enumerate(trades):
        side, dt, ds, hmax = tr["side"], tr["dt"], tr["ds"], max(1, tr["hold"])
        starts = RNG.integers(30, max(31, N - 2), reps)
        for r, s0 in enumerate(starts):
            e = C[s0]
            tgt = e * (1 + dt) if side > 0 else e * (1 - dt)
            stp = e * (1 - ds) if side > 0 else e * (1 + ds)
            px = C[min(N - 1, s0 + hmax)]
            for i in range(s0 + 1, min(N, s0 + hmax + 1)):
                if (O[i] >= tgt) if side > 0 else (O[i] <= tgt):
                    px = O[i]; break
                if (H[i] >= tgt) if side > 0 else (L[i] <= tgt):
                    px = tgt; break
                if (C[i] < stp) if side > 0 else (C[i] > stp):
                    px = C[i]; break
            out[r, j] = side * (px / e - 1.0) - 2 * COST
    return out.mean(axis=1) if len(trades) else np.zeros(reps)


def shift_null(dfs: dict, trades: list, reps: int = 1000) -> np.ndarray:
    """整体平移：所有交易一起往前 / 往后挪同一个随机根数（首尾相接），每笔的方向、括号、
    持有上限都不变。同一天好几个币一起做空的那种「扎堆」原样保留 —— 比逐笔独立随机更严。"""
    out = np.zeros(reps)
    if not trades:
        return out
    for r in range(reps):
        o = int(RNG.integers(20, 10_000))
        acc = []
        for tr in trades:
            df = dfs[tr["coin"]]
            O, H, L, C = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
            N = len(C)
            s0 = 30 + (tr["entry_i"] - 30 + o) % max(1, N - 32)
            side, dt, ds, hmax = tr["side"], tr["dt"], tr["ds"], max(1, tr["hold"])
            e = C[s0]
            tgt = e * (1 + dt) if side > 0 else e * (1 - dt)
            stp = e * (1 - ds) if side > 0 else e * (1 + ds)
            px = C[min(N - 1, s0 + hmax)]
            for i in range(s0 + 1, min(N, s0 + hmax + 1)):
                if (O[i] >= tgt) if side > 0 else (O[i] <= tgt):
                    px = O[i]; break
                if (H[i] >= tgt) if side > 0 else (L[i] <= tgt):
                    px = tgt; break
                if (C[i] < stp) if side > 0 else (C[i] > stp):
                    px = C[i]; break
            acc.append(side * (px / e - 1.0) - 2 * COST)
        out[r] = float(np.mean(acc))
    return out


def fade_table(tf: str, mults, retrs) -> pd.DataFrame:
    """A 在不同阈值 × 不同目标比例下的表现，看是高原还是一根针"""
    data = {s: D.load(tf, symbol=s) for s in COINS}
    fu = {s: funding(s, data[s].index, tf) for s in COINS}
    idx = pd.DatetimeIndex(sorted(set().union(*[d.index for d in data.values()])))
    rows = []
    old = PRM["projRetr"]
    for m in mults:
        for rr in retrs:
            PRM["projRetr"] = rr
            nets, trs = {}, []
            for s in COINS:
                W = walk(data[s], m, scen=False)
                n_, tr = fade_trades(data[s], W["waves"], fu[s])
                nets[s] = pd.Series(n_, index=data[s].index)
                trs += tr
            net = pd.DataFrame(nets).reindex(idx).fillna(0.0).mean(axis=1)
            tt = pd.DataFrame(trs)
            sh = float(net.mean() / net.std() * math.sqrt(ANN[tf])) if net.std() > 0 else np.nan
            rows.append({"ATR×": m, "目标比例": rr, "笔数": len(tt),
                         "平均每笔": float(tt["ret"].mean()) if len(tt) else np.nan,
                         "胜率": float((tt["ret"] > 0).mean()) if len(tt) else np.nan, "Sharpe": sh})
    PRM["projRetr"] = old
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 汇总

def stats(net: pd.Series, ann: float, expo: float | None = None, turn: float | None = None) -> dict:
    net = net.dropna()
    yrs = len(net) / ann
    sd = float(net.std())
    eq = (1 + net).cumprod()
    d = {"年化": float(net.sum() / yrs), "复利总收益": float(eq.iloc[-1] - 1),
         "Sharpe": float(net.mean() / sd * math.sqrt(ann)) if sd else np.nan,
         "最大回撤": float((eq / eq.cummax() - 1).min())}
    if expo is not None:
        d["平均敞口"] = expo
    if turn is not None:
        d["年换手"] = turn
    return d


def show(rows: list[dict]) -> None:
    t = pd.DataFrame(rows).set_index("版本")
    t = t.drop(columns=[c for c in t.columns if c.startswith("_")])
    for c in ("年化", "复利总收益", "最大回撤", "平均敞口", "胜率", "平均每笔"):
        if c in t:
            t[c] = t[c].map(lambda v: "" if pd.isna(v) else f"{v:+.1%}")
    if "Sharpe" in t:
        t["Sharpe"] = t["Sharpe"].map(lambda v: "" if pd.isna(v) else f"{v:+.2f}")
    if "年换手" in t:
        t["年换手"] = t["年换手"].map(lambda v: "" if pd.isna(v) else f"{v:.0f}x")
    print(t.to_string())


def dd(x: pd.Series) -> float:
    e = (1 + x).cumprod()
    return float((e / e.cummax() - 1).min())


def boot_p1(a: np.ndarray, ann: float, blk: int = 20, n: int = 2000) -> tuple[float, float]:
    """Sharpe(a) 和分块自助法 p 值（Sharpe ≤ 0 的比例）"""
    real = a.mean() / a.std() * math.sqrt(ann)
    nblk = int(np.ceil(len(a) / blk))
    sh = np.empty(n)
    for k in range(n):
        st = RNG.integers(0, len(a), nblk)
        x = a[np.concatenate([np.arange(s0, s0 + blk) % len(a) for s0 in st])[: len(a)]]
        sh[k] = x.mean() / x.std() * math.sqrt(ann)
    return float(real), float((sh <= 0).mean())


def boot_p(a: np.ndarray, b: np.ndarray, ann: float, blk: int = 20, n: int = 2000) -> tuple[float, float]:
    """Sharpe(a) − Sharpe(b) 和分块自助法 p 值（差 ≤ 0 的比例）"""
    real = (a.mean() / a.std() - b.mean() / b.std()) * math.sqrt(ann)
    nblk = int(np.ceil(len(a) / blk))
    diffs = np.empty(n)
    for k in range(n):
        st = RNG.integers(0, len(a), nblk)
        sel = np.concatenate([np.arange(s0, s0 + blk) % len(a) for s0 in st])[: len(a)]
        xa, xb = a[sel], b[sel]
        diffs[k] = (xa.mean() / xa.std() - xb.mean() / xb.std()) * math.sqrt(ann)
    return float(real), float((diffs <= 0).mean())


def run_tf(tf: str, levels=("细", "中", "粗"), null_reps: int = 200, verbose: bool = True) -> dict:
    ann = ANN[tf]
    data = {s: D.load(tf, symbol=s) for s in COINS}
    idx = sorted(set().union(*[d.index for d in data.values()]))
    idx = pd.DatetimeIndex(idx)
    fu = {s: funding(s, data[s].index, tf) for s in COINS}
    bh = pd.DataFrame({s: data[s]["close"].pct_change() for s in COINS}).reindex(idx).mean(axis=1).fillna(0.0)

    out = {"data": data, "bh": bh, "rows": [], "nets": {}, "expo": {}, "trades": {}, "null": {}, "legs": {}}
    for lv in levels:
        W = {s: walk(data[s], LEVELS[lv]) for s in COINS}
        anytr = []
        for s in COINS:
            _, tr = fade_trades(data[s], W[s]["anyw"], fu[s], expire=False)
            ew_conf = {w["conf"] for w in W[s]["waves"]}
            for t in tr:
                t["coin"] = s
                t["ew"] = t["entry_i"] in ew_conf
            anytr += tr
        out.setdefault("any", {})[lv] = pd.DataFrame(anytr)
        out["legs"][lv] = {s: (W[s]["leg"], W[s]["legdir"]) for s in COINS}

        # ---- A 做调整
        nets, trades, nulls = {}, [], []
        for s in COINS:
            n_, tr = fade_trades(data[s], W[s]["waves"], fu[s])
            nets[s] = pd.Series(n_, index=data[s].index)
            for t in tr:
                t["coin"] = s
            trades += tr
            if null_reps and tr:
                nulls.append((len(tr), bracket_null(data[s], tr, null_reps)))
        net = pd.DataFrame(nets).reindex(idx).fillna(0.0).mean(axis=1)
        held = pd.DataFrame({s: pd.Series(0.0, index=data[s].index) for s in COINS})
        for t in trades:
            held.loc[data[t["coin"]].index[t["entry_i"] + 1]: data[t["coin"]].index[t["exit_i"]], t["coin"]] = 1.0
        expo = float(held.reindex(idx).fillna(0.0).mean(axis=1).mean())
        tt = pd.DataFrame(trades)
        row = {"版本": f"A 做调整 · {lv}", **stats(net, ann, expo)}
        if len(tt):
            row.update({"笔数": len(tt), "胜率": float((tt["ret"] > 0).mean()),
                        "平均每笔": float(tt["ret"].mean()), "持有中位(根)": float(tt["hold"].median())})
        out["rows"].append(row)
        out["nets"][f"A·{lv}"] = net
        out["expo"][f"A·{lv}"] = expo
        out["trades"][lv] = tt
        if nulls:
            tot = sum(k for k, _ in nulls)
            null_mean = sum(k * v for k, v in nulls) / tot     # 每轮随机交易的平均收益（按笔数加权）
            out["null"][lv] = (float(tt["ret"].mean()), null_mean)

        # ---- B 跟推演方向
        for key, name, fn in (
            ("B1", "B1 推演方向·多空", lambda leg, d: d),
            ("B2", "B2 推演方向·只做多", lambda leg, d: np.clip(d, 0, None)),
            ("B3", "B3 只做浪1/3/5", lambda leg, d: np.where(np.isin(leg, (0, 2, 4)), d, 0.0)),
            ("B4", "B4 只做浪3", lambda leg, d: np.where(leg == 2, d, 0.0)),
        ):
            nets, expos, turns = {}, [], []
            for s in COINS:
                leg, dr = W[s]["leg"], W[s]["legdir"]
                pos = fn(leg, dr).astype(float)
                C = data[s]["close"].to_numpy(dtype=float)
                nets[s] = pd.Series(bar_positions(pos, C, fu[s]), index=data[s].index)
                expos.append(pd.Series(np.abs(pos), index=data[s].index))
                turns.append(pd.Series(np.abs(np.diff(pos, prepend=0.0)), index=data[s].index))
            net = pd.DataFrame(nets).reindex(idx).fillna(0.0).mean(axis=1)
            expo = float(pd.DataFrame(expos).T.reindex(idx).fillna(0.0).mean(axis=1).mean())
            turn = float(pd.DataFrame(turns).T.reindex(idx).fillna(0.0).mean(axis=1).sum() / (len(idx) / ann))
            out["rows"].append({"版本": f"{name} · {lv}", **stats(net, ann, expo, turn)})
            out["nets"][f"{key}·{lv}"] = net
            out["expo"][f"{key}·{lv}"] = expo
    out["rows"].append({"版本": "等权买入持有（不计费用和资金费）", **stats(bh, ann, 1.0, 0.0)})
    return out


def main() -> None:
    print("=" * 120)
    print("推动浪指标 impulse_wave 的三种用法   六个币 · 2022 起 · 吃单 5bp+1.5bp + 真实资金费率 · 每币固定 1/6 仓")
    print("=" * 120)

    R = run_tf("1d")
    ann = ANN["1d"]

    print("\n" + "-" * 120)
    print("1  日线，全部版本")
    print("-" * 120 + "\n")
    show(R["rows"])

    print("\n" + "-" * 120)
    print("2  A 做调整：逐笔看 —— 离场原因、多空两边")
    print("-" * 120 + "\n")
    for lv, tt in R["trades"].items():
        if not len(tt):
            print(f"  {lv}：没有交易\n")
            continue
        g = tt.groupby("why")["ret"].agg(["count", "mean"])
        side = tt.groupby("side")["ret"].agg(["count", "mean", lambda x: (x > 0).mean()])
        print(f"  【{lv}】{len(tt)} 笔   平均每笔 {tt['ret'].mean():+.2%}   中位 {tt['ret'].median():+.2%}   "
              f"目标平均离入场 {tt['dt'].mean():.1%}，止损平均离入场 {tt['ds'].mean():.1%}")
        for why, r in g.iterrows():
            print(f"      {why:10s} {int(r['count']):4d} 笔   平均 {r['mean']:+.2%}")
        for sd_, r in side.iterrows():
            print(f"      {'做多（跌完五浪后）' if sd_ > 0 else '做空（涨完五浪后）':14s} {int(r['count']):4d} 笔   "
                  f"平均 {r['mean']:+.2%}   胜率 {r.iloc[2]:.0%}")
        print()

    print("-" * 120)
    print("3  A 的零假设：随机入场配同样的括号（同币、同方向、同目标/止损距离、同持有上限，每笔配 200 次）")
    print("-" * 120 + "\n")
    for lv, (real, null) in R["null"].items():
        p = float((null >= real).mean())
        print(f"  {lv}：真交易平均 {real:+.2%}   随机括号平均 {null.mean():+.2%}"
              f"（5%–95%：{np.quantile(null, .05):+.2%} ~ {np.quantile(null, .95):+.2%}）   "
              f"p = {p:.3f}   {'显著' if p < 0.05 else '**不显著**'}")

    print("\n" + "-" * 120)
    print("3b A 的零假设（更严）：所有交易一起整体平移同一个随机根数，保留「几个币同一天一起做」的扎堆")
    print("-" * 120 + "\n")
    for lv, tt in R["trades"].items():
        if not len(tt):
            continue
        null = shift_null(R["data"], tt.to_dict("records"), 1000)
        real = float(tt["ret"].mean())
        print(f"  {lv}：真交易平均 {real:+.2%}   整体平移后平均 {null.mean():+.2%}"
              f"（5%–95%：{np.quantile(null, .05):+.2%} ~ {np.quantile(null, .95):+.2%}）   "
              f"p = {float((null >= real).mean()):.3f}")

    print("\n" + "-" * 120)
    print("3c 消融：艾略特规则有没有用？同样的做法用在「任何一个转折」上（只要求最近五段净方向一致）")
    print("    都不因新确认平仓、只看目标和止损，所以和上面的 A 数字不完全一样；比的是同一框架里两组的差")
    print("-" * 120 + "\n")
    for lv, at in R["any"].items():
        if not len(at):
            continue
        ew, rest = at[at["ew"]], at[~at["ew"]]
        diff = float(ew["ret"].mean() - rest["ret"].mean()) if len(ew) else np.nan
        # 置换检验：从全部转折里随机抽和「合规」一样多的笔数，平均收益 ≥ 真的那组的比例
        allr = at["ret"].to_numpy()
        perm = np.array([allr[RNG.choice(len(allr), len(ew), replace=False)].mean() for _ in range(5000)]) if len(ew) else np.array([np.nan])
        pv = float((perm >= ew["ret"].mean()).mean()) if len(ew) else np.nan
        print(f"  {lv}：合规 1-5 {len(ew):3d} 笔 平均 {ew['ret'].mean():+.2%} 胜率 {(ew['ret'] > 0).mean():.0%}   "
              f"其余转折 {len(rest):4d} 笔 平均 {rest['ret'].mean():+.2%} 胜率 {(rest['ret'] > 0).mean():.0%}   "
              f"差 {diff:+.2%}   置换 p = {pv:.3f}")

    print("\n" + "-" * 120)
    print("3d A 的参数曲面（日线）：ZigZag 阈值 × 目标比例。高原才算数，一根针不算")
    print("-" * 120 + "\n")
    ft = fade_table("1d", (1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0), (0.382, 0.5, 0.618))
    pv_ = ft.pivot(index="ATR×", columns="目标比例", values="Sharpe").round(2)
    nv_ = ft.pivot(index="ATR×", columns="目标比例", values="笔数")
    av_ = ft.pivot(index="ATR×", columns="目标比例", values="平均每笔")
    print("  Sharpe"); print(pv_.to_string())
    print("\n  笔数"); print(nv_.to_string())
    print("\n  平均每笔"); print(av_.map(lambda v: "" if pd.isna(v) else f"{v:+.1%}").to_string())
    print(f"\n  {len(ft)} 格里 Sharpe > 0 的 {int((ft['Sharpe'] > 0).sum())} 格，中位 {ft['Sharpe'].median():+.2f}")

    print("\n" + "-" * 120)
    print("4  显著性：多空版本看 Sharpe 是否 > 0；只做多版本和同敞口买入持有比（53、58 号同一个零假设）")
    print("-" * 120 + "\n")
    bh = R["bh"]
    for key, net in R["nets"].items():
        a_ = net.to_numpy()
        if not a_.std() > 0:
            print(f"  {key:8s} 没有交易")
            continue
        if key.startswith("B2"):
            expo = R["expo"][key]
            b_ = bh.reindex(net.index).fillna(0.0)
            real, p = boot_p(a_, b_.to_numpy(), ann)
            print(f"  {key:8s} 只做多  Sharpe {a_.mean() / a_.std() * math.sqrt(ann):+.2f}   "
                  f"同敞口({expo:.0%})买入持有 {b_.mean() / b_.std() * math.sqrt(ann):+.2f}   "
                  f"回撤 {dd(net):+.1%} vs {dd(b_ * expo):+.1%}   Sharpe 差 {real:+.2f}，p = {p:.3f}")
        else:
            real, p = boot_p1(a_, ann)
            print(f"  {key:8s} 多空    Sharpe {real:+.2f}   P(Sharpe ≤ 0) = {p:.3f}")

    print("\n" + "-" * 120)
    print("5  去偏 Sharpe：日线试了 15 个版本（5 种用法 × 3 个级别）；算上 4 小时和参数曲面其实更多，这里是往宽了算")
    print("-" * 120 + "\n")
    best = max((k for k in R["nets"] if R["nets"][k].std() > 0), key=lambda k: R["nets"][k].mean() / R["nets"][k].std())
    net = R["nets"][best]
    sh = float(net.mean() / net.std() * math.sqrt(ann))
    ps = M.deflated_sharpe(sh, n_trials=15, n_obs=len(net), ann_factor=ann,
                           skew=float(net.skew()), kurt=float(net.kurtosis() + 3))
    print(f"  最好的是 {best}：Sharpe {sh:+.2f}   P(真实 Sharpe > 0 | 试了 15 组) = {ps:.3f}")

    print("\n" + "-" * 120)
    print("6  4 小时（稳健性）")
    print("-" * 120 + "\n")
    R4 = run_tf("4h", null_reps=100)
    show(R4["rows"])
    ann4 = ANN["4h"]
    print()
    for lv, tt in R4["trades"].items():
        if not len(tt):
            continue
        real, null = R4["null"][lv]
        sh_ = shift_null(R4["data"], tt.to_dict("records"), 300)
        at = R4["any"][lv]
        ew, rest = at[at["ew"]], at[~at["ew"]]
        allr = at["ret"].to_numpy()
        perm = np.array([allr[RNG.choice(len(allr), len(ew), replace=False)].mean() for _ in range(5000)])
        print(f"  A·{lv}：{len(tt)} 笔 平均 {real:+.2%}   随机括号 p = {float((null >= real).mean()):.3f}   "
              f"整体平移 p = {float((sh_ >= real).mean()):.3f}   "
              f"消融：合规 {ew['ret'].mean():+.2%} vs 其余转折 {rest['ret'].mean():+.2%}（{len(rest)} 笔），"
              f"置换 p = {float((perm >= ew['ret'].mean()).mean()):.3f}")
    bh4 = R4["bh"]
    for key, net in R4["nets"].items():
        if key.startswith("B2"):
            b_ = bh4.reindex(net.index).fillna(0.0)
            real, p = boot_p(net.to_numpy(), b_.to_numpy(), ann4, blk=120)
            print(f"  {key}（只做多）Sharpe {net.mean() / net.std() * math.sqrt(ann4):+.2f} vs 买入持有 "
                  f"{b_.mean() / b_.std() * math.sqrt(ann4):+.2f}，Sharpe 差 {real:+.2f}，p = {p:.3f}")
    ft4 = fade_table("4h", (1.5, 2.0, 2.5, 3.0, 4.0), (0.382, 0.5, 0.618))
    print("\n  A 的参数曲面（4 小时）Sharpe")
    print(ft4.pivot(index="ATR×", columns="目标比例", values="Sharpe").round(2).to_string())
    print("  笔数")
    print(ft4.pivot(index="ATR×", columns="目标比例", values="笔数").to_string())
    print(f"  {len(ft4)} 格里 Sharpe > 0 的 {int((ft4['Sharpe'] > 0).sum())} 格，中位 {ft4['Sharpe'].median():+.2f}")
    return R, R4


def figure(R: dict, R4: dict, path: Path) -> None:
    """净值曲线（对数坐标）：日线、4 小时各一张，外加日线 A 的逐笔收益"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["WenQuanYi Zen Hei", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(3, 1, figsize=(11, 12), gridspec_kw={"height_ratios": [3, 3, 1.6]})
    lines = (("A·细", "A 做调整 · 细", "#1976d2", "-"), ("A·中", "A 做调整 · 中", "#8e24aa", "-"),
             ("B1·细", "B1 推演方向多空 · 细", "#e53935", "--"), ("B2·中", "B2 推演方向只做多 · 中", "#fb8c00", "--"))
    for ax, RR, title in ((axes[0], R, "日线"), (axes[1], R4, "4 小时")):
        bh_ = RR["bh"]
        ax.plot((1 + bh_).cumprod(), color="#9e9e9e", lw=1.4,
                label=f"等权买入持有（不计费用）   Sharpe {bh_.mean() / bh_.std() * math.sqrt(ANN['1d' if RR is R else '4h']):+.2f}")
        for key, lbl, col, ls in lines:
            if key in RR["nets"]:
                net = RR["nets"][key]
                ax.plot((1 + net).cumprod(), color=col, lw=1.6, ls=ls,
                        label=f"{lbl}   Sharpe {net.mean() / net.std() * math.sqrt(ANN['1d' if RR is R else '4h']):+.2f}")
        ax.set_yscale("log")
        ticks = [0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{t:g}" for t in ticks])
        ax.minorticks_off()
        ax.set_ylabel("净值（起点 = 1）")
        ax.axhline(1, color="#bdbdbd", lw=0.8)
        ax.set_title(f"{title}：六币等权、每币 1/6 仓，含 6.5bp/边成本和资金费率", fontsize=12, loc="left")
        ax.legend(fontsize=9, loc="upper left", frameon=False)
        ax.grid(alpha=0.25)
    ax = axes[2]
    for lv, col in (("细", "#1976d2"), ("中", "#8e24aa")):
        tt = R["trades"].get(lv)
        if tt is not None and len(tt):
            ax.bar(tt["t_entry"], tt["ret"] * 100, width=12, color=col, alpha=0.8, label=f"日线 A · {lv}（{len(tt)} 笔）")
    ax.axhline(0, color="#757575", lw=0.8)
    ax.set_ylabel("每笔收益 %")
    ax.set_title("日线 A 做调整的每一笔：四年多一共 21 笔", fontsize=12, loc="left")
    ax.legend(fontsize=9, frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=110)


if __name__ == "__main__":
    R_, R4_ = main()
    rows = pd.DataFrame(R_["rows"]).assign(周期="1d")
    rows = pd.concat([rows, pd.DataFrame(R4_["rows"]).assign(周期="4h")])
    rows.drop(columns=[c for c in rows.columns if c.startswith("_")]).to_csv(ROOT / "reports" / "impulse_wave_bt.csv", index=False)
    try:
        figure(R_, R4_, ROOT / "reports" / "impulse_wave_bt.png")
    except ImportError:
        print("（没装 matplotlib，跳过出图）")
