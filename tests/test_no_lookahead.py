"""The tests that decide whether any of the numbers mean anything.

If the engine can see the future, every backtest in this repo is fiction.
These check that it cannot.
"""

from __future__ import annotations

import sys
from pathlib import Path

import shutil

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vibt import backtest as B, data as D, indicators as I, system as SYS  # noqa: E402


@pytest.fixture(scope="module")
def daily() -> pd.DataFrame:
    return D.load("1d")


def test_indicators_are_causal(daily):
    """Truncating the future must not change any past indicator value."""
    cut = len(daily) - 40
    head = daily.iloc[:cut]
    for name, fn in {
        "vortex": lambda d: I.vortex(d, 14)["vi_spread"],
        "atr": lambda d: I.atr(d, 14),
        "parkinson": lambda d: I.parkinson_vol(d, 30, 365),
        "efficiency": lambda d: I.efficiency_ratio(d["close"], 20),
        "donchian": lambda d: I.donchian(d, 20)["dc_high"],
        "rsi": lambda d: I.rsi(d["close"], 14),
        "adx": lambda d: I.adx(d, 14),
    }.items():
        full = fn(daily).iloc[:cut]
        part = fn(head)
        pd.testing.assert_series_equal(full, part, check_names=False,
                                       obj=f"{name} leaks future data")


def test_system_target_is_causal(daily):
    cut = len(daily) - 60
    full = SYS.target_exposure(daily).iloc[:cut]
    part = SYS.target_exposure(daily.iloc[:cut])
    pd.testing.assert_series_equal(full, part, check_names=False)


def test_engine_applies_execution_lag(daily):
    """A signal that only fires on the last bar must earn nothing."""
    sig = pd.Series(0.0, index=daily.index)
    sig.iloc[-1] = 1.0
    res = B.run(daily, sig, B.Costs(0, 0), 365.0)
    assert res.rets.abs().sum() == pytest.approx(0.0, abs=1e-12)


def test_perfect_foresight_is_detected(daily):
    """Sanity check on the test itself: a peeking signal must look absurdly good.

    The engine holds signal[t-1] over bar t, and bar t earns open[t+1]/open[t]-1,
    so the cheating signal at bar s is the sign of open[s+2]/open[s+1]-1.
    """
    cheat = np.sign(daily["open"].shift(-2) / daily["open"].shift(-1) - 1).fillna(0.0)
    res = B.run(daily, cheat, B.Costs(0, 0), 365.0)
    assert res.stats.sharpe > 10, "engine is not paying the signal it was given"
    assert (res.gross.dropna() >= -1e-12).all(), "a perfect signal should never lose"


def test_costs_reduce_returns(daily):
    sig = SYS.target_exposure(daily)
    free = B.run(daily, sig, B.Costs(0, 0), 365.0)
    paid = B.run(daily, sig, B.Costs(4.5, 2.0), 365.0)
    assert paid.stats.total_return < free.stats.total_return
    assert paid.costs.sum() > 0


def test_flat_signal_is_flat(daily):
    res = B.run(daily, pd.Series(0.0, index=daily.index), B.Costs(4.5, 2.0), 365.0)
    assert res.stats.total_return == pytest.approx(0.0, abs=1e-12)
    assert res.stats.exposure == pytest.approx(0.0)


def test_buy_and_hold_matches_price(daily):
    res = B.buy_and_hold(daily, 365.0, B.Costs(0, 0))
    # open-to-open over the held window, minus the final unpriced bar
    expected = daily["open"].iloc[-1] / daily["open"].iloc[1] - 1
    assert res.stats.total_return == pytest.approx(expected, rel=1e-6)


def test_higher_tf_alignment_has_no_lookahead():
    frames = D.load_all()
    base, htf = frames["4h"], frames["1d"]
    aligned = D.align_higher_tf(base.index, htf, "1d", ["close"])
    joined = pd.DataFrame({"base_time": base.index, "htf_close": aligned["close_1d"].to_numpy()})
    for _, row in joined.dropna().sample(200, random_state=0).iterrows():
        t = row["base_time"]
        eligible = htf[htf["close_time"] <= t]
        assert row["htf_close"] == pytest.approx(eligible["close"].iloc[-1]), \
            f"HTF value at {t} is not the last CLOSED daily bar"


def test_risk_engine_stop_is_pessimistic(daily):
    """When stop and target both sit in one bar, the stop must win."""
    sig = pd.Series(1.0, index=daily.index)
    atr = I.atr(daily, 14)
    tight = B.run_with_risk(daily, sig, stop_dist=atr * 0.1, take_dist=atr * 0.1,
                            costs=B.Costs(0, 0), ann_factor=365.0)
    loose = B.run_with_risk(daily, sig, stop_dist=atr * 5.0, take_dist=atr * 5.0,
                            costs=B.Costs(0, 0), ann_factor=365.0)
    assert tight.stats.total_return < loose.stats.total_return


def test_vortex_matches_definition(daily):
    n = 14
    v = I.vortex(daily, n)
    vm_p = (daily["high"] - daily["low"].shift(1)).abs().rolling(n).sum()
    tr = I.true_range(daily).rolling(n).sum()
    pd.testing.assert_series_equal(v["vi_plus"].dropna(), (vm_p / tr).dropna(),
                                   check_names=False)
    assert (v[["vi_plus", "vi_minus"]].dropna() > 0).all().all()


def test_rolling_bands_are_causal():
    """Adaptive rails must be built from prior bars only.

    This is the bug the fixed-band version cannot have and the adaptive one can:
    a rail computed from the whole sample would quietly leak the future into
    every historical signal.
    """
    from vibt import vi_band as VB

    frames = D.load_all()
    cut = len(frames["4h"]) - 200
    for mode in ("quantile", "range"):
        p = VB.BandParams(band_mode=mode)
        full = VB.build_frame(frames, p).iloc[:cut]
        truncated = {"4h": frames["4h"].iloc[:cut], "1d": frames["1d"], "1h": frames["1h"]}
        part = VB.build_frame(truncated, p)
        for col in ("up4", "dn4", "s4"):
            pd.testing.assert_series_equal(full[col], part[col], check_names=False,
                                           obj=f"{mode}/{col} leaks future data")


def test_band_states_match_dashboard_rule():
    """vip > upper -> 1, vip < lower -> -1, double = both timeframes agree."""
    from vibt import vi_band as VB

    frames = D.load_all()
    p = VB.BandParams()
    df = VB.build_frame(frames, p)
    live = df.dropna(subset=["vi4", "s1d"])
    assert (live.loc[live["vi4"] > p.up_4h, "s4"] == 1).all()
    assert (live.loc[live["vi4"] < p.dn_4h, "s4"] == -1).all()
    inside = live[(live["vi4"] <= p.up_4h) & (live["vi4"] >= p.dn_4h)]
    assert (inside["s4"] == 0).all()
    dbl = live[live["double"] != 0]
    assert (dbl["s4"] == dbl["s1d"]).all()
    assert (live.loc[live["s4"] != live["s1d"], "double"] == 0).all()


def test_vortex_matches_dashboard_implementation(daily):
    """Our Vortex must equal the dashboard's backend/vortex.py, bar for bar."""
    n = 14
    rows = list(zip(daily["open"], daily["high"], daily["low"], daily["close"]))
    vm_p, vm_m, trs, vip = [], [], [], []
    for i, (_, h, lo, c) in enumerate(rows):
        prev = rows[i - 1] if i > 0 else None
        pc = prev[3] if prev else None
        tr = (h - lo) if pc is None else max(h - lo, abs(h - pc), abs(lo - pc))
        trs.append(tr)
        vm_p.append(abs(h - prev[2]) if prev else None)
        vm_m.append(abs(lo - prev[1]) if prev else None)
        if i >= n and all(v is not None for v in vm_p[i - n + 1: i + 1]):
            s = sum(trs[i - n + 1: i + 1])
            vip.append(sum(vm_p[i - n + 1: i + 1]) / s if s else None)
        else:
            vip.append(None)
    ours = I.vortex(daily, n)["vi_plus"].to_numpy()
    for i, ref in enumerate(vip):
        if ref is None or np.isnan(ours[i]):
            continue
        assert abs(ours[i] - ref) < 1e-9, f"bar {i}: {ours[i]} vs dashboard {ref}"


def test_multi_symbol_loader_roundtrip(tmp_path):
    """A second symbol dropped into the data dir must load like the first."""
    from vibt import ingest as ING

    src = tmp_path / "exports"
    src.mkdir()
    original = D.DATA_DIR / "BTCUSDT_1d.csv"
    for tf in ("1h", "4h", "1d"):
        shutil.copyfile(D.DATA_DIR / f"BTCUSDT_{tf}.csv",
                        src / f"FAKEUSDT_{tf}_2024-01-01_to_now.csv")
    assert ING.parse_name(Path("FAKEUSDT_4h_2024-01-01_to_now.csv")) == ("FAKEUSDT", "4h")

    dest = tmp_path / "data"
    report = ING.ingest(src, dest)
    assert set(report["symbol"]) == {"FAKEUSDT"}
    assert set(report["tf"]) == {"1h", "4h", "1d"}
    assert (report["gaps"] == 0).all()
    assert (report["bad_ohlc"] == 0).all()

    assert D.available_symbols(dest) == ["FAKEUSDT"]
    loaded = D.load("1d", dest, "FAKEUSDT")
    reference = pd.read_csv(original, encoding="utf-8-sig")
    assert len(loaded) == len(reference)
    universe = D.load_universe(data_dir=dest)
    assert set(universe) == {"FAKEUSDT"}
    assert set(universe["FAKEUSDT"]) == {"1h", "4h", "1d"}


def test_band_study_is_symbol_agnostic(tmp_path):
    """vi_band must run on any symbol's frames, not just the default one."""
    from vibt import vi_band as VB

    frames = D.load_all()
    for mode in ("fixed", "quantile", "range"):
        p = VB.BandParams(band_mode=mode)
        pos = VB.target_position(VB.build_frame(frames, p), p)
        assert len(pos) == len(frames["4h"])
        assert pos.abs().max() <= p.max_size + 1e-9


def test_exposure_within_bounds(daily):
    p = SYS.Params()
    s = SYS.target_exposure(daily, p)
    assert s.max() <= p.max_leverage + 1e-9
    assert s.min() >= -p.max_leverage * p.short_size - 1e-9
