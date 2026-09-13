"""趋势过滤 + 回调进场：把「等回调」这件事单独拿出来检验.

    python scripts/52_trend_pullback.py

规则（按用户描述翻译成可执行的形式）：

    趋势     中线 ± ATR(200)×2 的迟滞状态机（和 51 号脚本同一套）
    下跌趋势 价格反弹到快线之上（回调），再跌回快线之下时 -> 做空
    上涨趋势 价格回落到快线之下（回调），再涨回快线之上时 -> 做多
    出场     趋势翻转，或价格反向穿越快线
    绝不逆势

这里要回答的**不是**「趋势跟随有没有用」—— 51 号脚本已经量过了（只做多趋势过滤
把回撤从 −75% 降到 −49%，Sharpe 从 +0.46 到 +0.57）。

要回答的是那句话里真正的主张：**「等回调再进场」比「趋势一转就进场」好吗？**
耐心是有代价的 —— 等待期间的行情你没参与。所以对照必须是：

    基准 A  趋势一转就满仓，一直持到趋势翻转（不择时）
    基准 B  同样的笔数、同样的持仓长度，但**进场点随机**落在同向趋势段里
            （匹配的随机零假设 —— 这才能分开「趋势的钱」和「择时的钱」）

以及那句「时间框架要短」：1h / 4h / 1d 三个周期并排跑。成本随交易次数线性增长，
而边际不会，所以这条建议本身也是可证伪的。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module  # noqa: E402

from vibt import data as D, indicators as IND, ingest as I  # noqa: E402

_v = import_module("51_volumatic_vidya")
trend_state, MAKER_BPS, TAKER_BPS = _v.trend_state, _v.MAKER_BPS, _v.TAKER_BPS

pd.set_option("display.width", 240)
COINS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "TAOUSDT", "HYPEUSDT")
BPY = {"1d": 365.0, "4h": 365.0 * 6, "1h": 365.0 * 24}
RNG = np.random.default_rng(20260913)


def trend(df: pd.DataFrame, atr_len: int = 200, band: float = 2.0) -> np.ndarray:
    c = (df["high"] + df["low"]) / 2          # hl2：51 号脚本里表现最好的中线
    a = IND.atr(df, atr_len)
    return trend_state(df["close"].to_numpy(dtype=float),
                       (c + a * band).to_numpy(), (c - a * band).to_numpy())


def positions(df: pd.DataFrame, fast: int = 20, exit_on_cross: bool = True,
              **tkw) -> pd.Series:
    """趋势 + 回调进场，返回每根 bar 的仓位（+1/0/−1）。"""
    t = trend(df, **tkw)
    f = IND.ema(df["close"], fast).to_numpy()
    c = df["close"].to_numpy(dtype=float)
    n = len(c)
    pos = np.zeros(n)
    state = 0
    pulled = False          # 本段趋势里是否已经出现过回调
    for i in range(1, n):
        if not np.isfinite(f[i]) or not np.isfinite(f[i - 1]):
            continue
        if t[i] != t[i - 1]:            # 趋势翻转：清空、重新等回调
            state, pulled = 0, False
        if t[i] > 0:                                     # 上涨趋势
            if c[i] < f[i]:
                pulled = True                            # 回调发生
            if state == 0 and pulled and c[i] > f[i] and c[i - 1] <= f[i - 1]:
                state, pulled = 1, False                 # 回调后重新站上 -> 做多
            elif state == 1 and exit_on_cross and c[i] < f[i] and c[i - 1] >= f[i - 1]:
                state = 0
        else:                                            # 下跌趋势
            if c[i] > f[i]:
                pulled = True
            if state == 0 and pulled and c[i] < f[i] and c[i - 1] >= f[i - 1]:
                state, pulled = -1, False
            elif state == -1 and exit_on_cross and c[i] > f[i] and c[i - 1] <= f[i - 1]:
                state = 0
        pos[i] = state
    return pd.Series(pos, index=df.index)


def always_in(df: pd.DataFrame, **tkw) -> pd.Series:
    """基准 A：趋势一转就满仓，不等回调。"""
    return pd.Series(trend(df, **tkw), index=df.index)


def evaluate(posmap: dict, tf: str, fee=TAKER_BPS, slip=1.5) -> dict:
    P = pd.DataFrame(posmap).sort_index()
    rets, funds = {}, {}
    for s in P.columns:
        df = D.load(tf, symbol=s)
        rets[s] = df["close"].pct_change().shift(-1)
        fr = I._load_funding(Path("data/funding") / f"{s}_funding.csv.gz")["funding_rate"].astype(float)
        freq = {"1d": "D", "4h": "4h", "1h": "h"}[tf]
        funds[s] = fr.groupby(fr.index.floor(freq)).sum().reindex(df.index).fillna(0.0)
    R = pd.DataFrame(rets).reindex(P.index)
    Fu = pd.DataFrame(funds).reindex(P.index).fillna(0.0)
    # 每个币固定 1/N 仓位，**不是**「有信号的币平分 1.0」。
    # 后者会让任何一个币的进出场都改变其余所有币的权重，凭空制造换手 ——
    # 在 1h 上那会把换手算成 3266x/年，成本 −212%，完全是口径造出来的。
    # 固定权重也更接近真实做法：每个标的一份钱，互不影响。
    W = P / len(P.columns)
    ann = BPY[tf]
    g = (W * R).sum(axis=1).dropna()
    turn = W.diff().abs().sum(axis=1).reindex(g.index).fillna(0.0)
    fund = -(W * Fu).sum(axis=1).reindex(g.index).fillna(0.0)
    cost = turn * (fee + slip) * 1e-4
    net = g - cost + fund
    yrs = len(g) / ann
    sd = float(net.std())
    eq = (1 + net).cumprod()
    return {"毛收益": float(g.mean() * ann), "资金费率": float(fund.mean() * ann),
            "手续费+滑点": -float(cost.sum() / yrs), "净收益": float(net.mean() * ann),
            "净Sharpe": float(net.mean() / sd * math.sqrt(ann)) if sd > 0 else np.nan,
            "最大回撤": float((eq / eq.cummax() - 1).min()),
            "年换手": float(turn.sum() / yrs),
            "在场时间": float((P.abs().sum(axis=1) > 0).mean()), "_net": net}


def show(rows):
    t = pd.DataFrame(rows).set_index("版本")
    t = t.drop(columns=[c for c in t.columns if c.startswith("_")])
    for c in ("毛收益", "资金费率", "手续费+滑点", "净收益", "最大回撤", "在场时间"):
        t[c] = t[c].map("{:+.1%}".format)
    t["净Sharpe"] = t["净Sharpe"].map("{:+.2f}".format)
    t["年换手"] = t["年换手"].map("{:.0f}x".format)
    print(t.to_string())


def trades_of(pos: np.ndarray) -> list[tuple[int, int, int]]:
    """(进场下标, 持有根数, 方向)"""
    out, i, n = [], 1, len(pos)
    while i < n:
        if pos[i] != 0 and pos[i - 1] != pos[i]:
            d = pos[i]
            j = i
            while j + 1 < n and pos[j + 1] == d:
                j += 1
            out.append((i, j - i + 1, int(d)))
            i = j + 1
        else:
            i += 1
    return out


def random_null(tf: str, draws: int = 200, **kw) -> tuple[float, np.ndarray]:
    """匹配的随机零假设：笔数、持仓长度、方向、所处趋势段都一样，只有进场时点随机。

    这才能把「趋势的钱」和「择时的钱」分开 —— 如果随机进场拿到同样的结果，
    那耐心就没有被回报。
    """
    real, sims = [], np.zeros(draws)
    per_coin = []
    for s in COINS:
        df = D.load(tf, symbol=s)
        p = positions(df, **kw).to_numpy()
        t = trend(df)
        r = df["close"].pct_change().shift(-1).to_numpy()
        tr = trades_of(p)
        real += [float(np.nansum(r[i:i + h] * d)) for i, h, d in tr]
        per_coin.append((t, r, tr))
    real_mean = float(np.mean(real)) if real else np.nan

    for k in range(draws):
        acc = []
        for t, r, tr in per_coin:
            for _, h, d in tr:
                # 只在同向趋势段里随机取起点，长度一致
                ok = np.flatnonzero((t == d) & np.isfinite(np.r_[r[:len(t)]]))
                ok = ok[(ok + h) < len(r)]
                if len(ok) == 0:
                    continue
                i0 = int(RNG.choice(ok))
                acc.append(float(np.nansum(r[i0:i0 + h] * d)))
        sims[k] = float(np.mean(acc)) if acc else np.nan
    return real_mean, sims


def main() -> None:
    print("=" * 122)
    print("趋势过滤 + 回调进场：「等回调」到底值不值")
    print("=" * 122)

    for tf in ("1h", "4h", "1d"):
        print("\n" + "-" * 122)
        print(f"周期 {tf}")
        print("-" * 122 + "\n")
        rows = []
        pb = {s: positions(D.load(tf, symbol=s)) for s in COINS}
        pbh = {s: positions(D.load(tf, symbol=s), exit_on_cross=False) for s in COINS}
        ai = {s: always_in(D.load(tf, symbol=s)) for s in COINS}
        lo = lambda m: {s: v.clip(lower=0) for s, v in m.items()}  # noqa: E731
        for lbl, pm in (("回调进场 · 穿回快线就出 · 多空", pb),
                        ("回调进场 · 持到趋势翻转 · 多空", pbh),
                        ("趋势满仓 · 多空（不等回调）", ai),
                        ("回调进场 · 穿回快线就出 · 只做多", lo(pb)),
                        ("回调进场 · 持到趋势翻转 · 只做多", lo(pbh)),
                        ("趋势满仓 · 只做多", lo(ai))):
            s = evaluate(pm, tf)
            s["版本"] = lbl
            rows.append(s)
        df0 = D.load(tf, symbol=COINS[0])
        bh = pd.DataFrame({s: D.load(tf, symbol=s)["close"].pct_change().shift(-1)
                           for s in COINS}).mean(axis=1).dropna()
        ann = BPY[tf]
        rows.append({"版本": "等权买入持有", "毛收益": float(bh.mean() * ann),
                     "资金费率": 0.0, "手续费+滑点": 0.0, "净收益": float(bh.mean() * ann),
                     "净Sharpe": float(bh.mean() / bh.std() * math.sqrt(ann)),
                     "最大回撤": float(((1 + bh).cumprod() / (1 + bh).cumprod().cummax() - 1).min()),
                     "年换手": 0.0, "在场时间": 1.0})
        show(rows)

    print("\n" + "-" * 122)
    print("匹配的随机零假设：笔数 / 持仓长度 / 方向 / 趋势段全部一致，只有进场时点随机")
    print("-" * 122 + "\n")
    for tf in ("1h", "4h", "1d"):
        real, sims = random_null(tf, draws=200)
        p = float((sims >= real).mean())
        print(f"  {tf:3s}  回调进场每笔均值 {real:+.4%}   "
              f"随机进场 {np.nanmean(sims):+.4%} (±{np.nanstd(sims):.4%})   "
              f"p = {p:.3f}  {'择时有信息' if p < 0.05 else '**和随机进场没有区别**'}")
    print("""
  这一行才是那句经验的判据。如果 p 不显著，说明赚到的是**趋势本身**的钱，
  「等回调」只是让你少在场、少交易 —— 那是省成本，不是择时能力。""")


if __name__ == "__main__":
    main()
