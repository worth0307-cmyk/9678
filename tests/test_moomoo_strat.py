"""moomoo_strat：和已验证过的脚本一致、没有未来函数、成交口径对。"""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from moomoo_strat import combo, impulse as IW, strategy as S  # noqa: E402
from vibt import data as D  # noqa: E402

BT = import_module("60_impulse_wave_backtest")


@pytest.fixture(scope="module")
def btc() -> pd.DataFrame:
    return D.load("1d", symbol="BTCUSDT")


@pytest.mark.parametrize("sym", ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
@pytest.mark.parametrize("mult", [2.0, 4.0])
def test_impulse_matches_script_60(sym, mult):
    """1-5 确认和每根的「下一段」与 scripts/60 的 walk() 逐根一致（60 又和 59、pine 对过）"""
    df = D.load("1d", symbol=sym)
    r, w = IW.run(df, mult), BT.walk(df, mult)
    key = lambda ws: [(x["conf"], x["bull"], round(x["tgt"], 8)) for x in ws]  # noqa: E731
    assert key(r["waves"]) == key(w["waves"])
    assert np.array_equal(np.where(r["f"] >= 0, r["f"], -1), w["leg"])


def test_trend_is_leg_direction_of_impulse_phase(btc):
    """趋势 = 推动阶段取这组浪方向、调整阶段取反：和 60 的「下一段方向」在推动腿上同号"""
    r, w = IW.run(btc, 4.0), BT.walk(btc, 4.0)
    imp = np.isin(r["f"], (0, 2, 4, 8))           # 下一段是浪1/3/5/新浪1 → 趋势就是下一段方向
    assert imp.sum() > 100
    assert np.array_equal(r["trend"][imp], w["legdir"][imp].astype(int))


def test_features_are_causal(btc):
    """把未来砍掉，过去每一根的指标值和规则持仓都不能变"""
    full = S.features(btc, 365)["f"]
    cols = ["st", "st_line", "ma200", "ma50w", "fib_pos", "trD", "kD", "invD", "trW", "kW", "invW"]
    pos_full = S.positions(full)
    for cut in (400, 777, 1001, 1300, len(btc) - 3):
        part = S.features(btc.iloc[:cut], 365)["f"]
        a, b = part[cols], full[cols].iloc[:cut]
        pd.testing.assert_frame_equal(a, b, check_dtype=False, obj=f"cut={cut}")
        pos_part = S.positions(part)
        for r in pos_full:
            assert np.array_equal(pos_part[r], pos_full[r][:cut]), (cut, r)


def test_weekly_uses_completed_weeks_only(btc):
    """周中每一天的周线趋势都等于上一周走完时的值"""
    f = S.features(btc, 365)["f"]
    w = IW.weekly(btc)
    rw = IW.run(w, IW.AUTO["W"]["中"])
    wt = pd.Series(rw["trend"], index=w.index)
    for day in f.index[400:1600:37]:
        done = wt[wt.index <= day]
        assert f.loc[day, "trW"] == done.iloc[-1]


def test_ma50w_equals_weekly_sma_at_week_end(btc):
    m = combo.ma50w(btc)
    w = IW.weekly(btc)["close"].rolling(50).mean().dropna()
    np.testing.assert_allclose(m.reindex(w.index).to_numpy(), w.to_numpy(), rtol=1e-12)


def test_supertrend_flips_only_on_cross(btc):
    t, line = combo.supertrend(btc)
    C = btc["close"].to_numpy()
    for i in np.flatnonzero(np.diff(t) != 0) + 1:
        prev_line = line[i - 1]                    # 翻转那根收盘越过上一根的线
        assert (C[i] < prev_line) if t[i] == -1 else (C[i] > prev_line)
    assert set(np.unique(t)) == {-1, 1}


def test_pos_net_next_open():
    O = np.array([10.0, 11.0, 12.0, 12.0, 13.0])
    C = np.array([10.0, 11.5, 12.5, 12.0, 14.0])
    pos = np.array([1, 1, 0, 0, 0], dtype=bool)        # 第 0 天收盘买、第 1 天开盘成交；第 2 天收盘卖
    net = S.pos_net(pos, O, C, cost=0.001)
    assert net[0] == 0
    assert net[1] == pytest.approx(11.5 / 11.0 - 1 - 0.001)   # 开盘进场：只拿开盘到收盘
    assert net[2] == pytest.approx(12.5 / 11.5 - 1)
    assert net[3] == pytest.approx(12.0 / 12.5 - 1 - 0.001)   # 隔夜跳空算在旧仓位上，开盘卖出
    assert net[4] == 0


def test_fade_long_hits_target():
    idx = pd.date_range("2024-01-01", periods=6, freq="B")
    f = pd.DataFrame({"open": [10, 10, 10.5, 11, 11.5, 12], "high": [10, 10.6, 11, 11.6, 12.4, 12],
                      "low": [9.5, 9.8, 10.2, 10.8, 11.4, 11.8], "close": [10, 10.5, 10.9, 11.5, 12.2, 12]},
                     index=idx, dtype=float)
    waves = [dict(bull=False, conf=0, p0=14.0, p5=9.0, b5=0, tgt=12.0)]
    net, held, tr, pend = S.fade(f, waves, cost=0.0, sides=(1,))
    assert len(tr) == 1 and tr[0]["why"] == "到目标" and tr[0]["exit_i"] == 4
    assert tr[0]["ret"] == pytest.approx(12.0 / 10.0 - 1)
    assert np.prod(1 + net) - 1 == pytest.approx(12.0 / 10.0 - 1)
