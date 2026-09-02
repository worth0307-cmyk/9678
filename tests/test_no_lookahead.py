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


def _export_as_plain_csv(tf: str, target: Path, symbol: str = "BTCUSDT") -> None:
    """Write a stored data file out as an uncompressed CSV, whatever form it is in.

    The ingest tests simulate a fresh export, and the exporter always emits plain
    CSV.  Copying the bytes would smuggle gzip content into a `.csv` name once the
    repository stores that symbol compressed.
    """
    src = D.resolve(tf, D.DATA_DIR, symbol)
    pd.read_csv(src, encoding="utf-8-sig").to_csv(target, index=False, encoding="utf-8-sig")


def test_multi_symbol_loader_roundtrip(tmp_path):
    """A second symbol dropped into the data dir must load like the first."""
    from vibt import ingest as ING

    src = tmp_path / "exports"
    src.mkdir()
    original = D.resolve("1d", D.DATA_DIR, "BTCUSDT")
    for tf in ("1h", "4h", "1d"):
        _export_as_plain_csv(tf, src / f"FAKEUSDT_{tf}_2024-01-01_to_now.csv")
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


def test_ingest_keeps_the_longest_history(tmp_path):
    """Two exports of the same symbol/tf: the one starting earlier must win.

    Exporting with two different --start values leaves duplicate keys, and glob
    order would otherwise decide which one lands in data/.
    """
    from vibt import ingest as ING

    src = tmp_path / "exports"
    src.mkdir()
    short = pd.read_csv(D.resolve("1d", D.DATA_DIR, "BTCUSDT"), encoding="utf-8-sig")
    ts = pd.to_datetime(short.iloc[:, 0])
    pre = pd.DataFrame({short.columns[0]: [ts.iloc[0] - pd.Timedelta(days=k)
                                           for k in range(200, 0, -1)]})
    for col in short.columns[1:]:
        pre[col] = float(short[col].iloc[0])
    long = pd.concat([pre, short])

    short.to_csv(src / "ZZZUSDT_1d_2024-01-01_to_now.csv", index=False, encoding="utf-8-sig")
    long.to_csv(src / "ZZZUSDT_1d_2023-01-01_to_now.csv", index=False, encoding="utf-8-sig")

    dest = tmp_path / "data"
    report = ING.ingest(src, dest)
    assert len(report) == 1, "one row per (symbol, timeframe), not one per file"
    assert report.iloc[0]["source"] == "ZZZUSDT_1d_2023-01-01_to_now.csv"
    assert report.iloc[0]["bars"] == len(long)
    assert len(D.load("1d", dest, "ZZZUSDT")) == len(long)


def test_band_study_is_symbol_agnostic(tmp_path):
    """vi_band must run on any symbol's frames, not just the default one."""
    from vibt import vi_band as VB

    frames = D.load_all()
    for mode in ("fixed", "quantile", "range"):
        p = VB.BandParams(band_mode=mode)
        pos = VB.target_position(VB.build_frame(frames, p), p)
        assert len(pos) == len(frames["4h"])
        assert pos.abs().max() <= p.max_size + 1e-9


def test_gzipped_data_reads_identically(tmp_path):
    """A .csv.gz must load exactly like the .csv, and be discoverable."""
    from vibt import ingest as ING

    src = tmp_path / "exports"
    src.mkdir()
    for tf in ("1h", "4h", "1d"):
        _export_as_plain_csv(tf, src / f"ZIPUSDT_{tf}_2023-01-01_to_now.csv")
    dest = tmp_path / "data"
    ING.ingest(src, dest, compress=True)

    assert not list(dest.glob("*.csv")), "compress=True should not leave plain CSVs"
    assert len(list(dest.glob("*.csv.gz"))) == 3
    assert D.available_symbols(dest, require=("1h", "4h", "1d")) == ["ZIPUSDT"]
    pd.testing.assert_frame_equal(D.load("1d", dest, "ZIPUSDT"), D.load("1d"))


def test_plain_and_gzip_can_coexist(tmp_path):
    """Mixed layouts resolve per file, and re-ingesting swaps rather than shadows.

    Leaving both a stale .csv and a fresh .csv.gz for one symbol would resolve to
    whichever the lookup prefers, silently serving old data.
    """
    from vibt import ingest as ING

    src = tmp_path / "exports"
    src.mkdir()
    for tf in ("1h", "4h", "1d"):
        _export_as_plain_csv(tf, src / f"MIXUSDT_{tf}_2023-01-01_to_now.csv")
    dest = tmp_path / "data"

    ING.ingest(src, dest, compress=False)
    assert len(list(dest.glob("*.csv"))) == 3
    ING.ingest(src, dest, compress=True)      # re-ingest compressed
    assert not list(dest.glob("MIXUSDT_*.csv")), "stale plain CSV was left behind"
    assert len(list(dest.glob("MIXUSDT_*.csv.gz"))) == 3
    assert len(D.load("1h", dest, "MIXUSDT")) == len(D.load("1h"))


def test_missing_symbol_error_names_both_forms(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"NOPEUSDT_1d\.csv.*NOPEUSDT_1d\.csv\.gz"):
        D.load("1d", tmp_path, "NOPEUSDT")


def test_xsec_weights_are_dollar_neutral_and_causal():
    """Long/short weights must net to zero, and risk parity must not flip a leg."""
    from vibt import indicators as I2, xsec as X

    universe = D.load_universe()
    if len(universe) < 3:
        pytest.skip("needs a multi-symbol panel")
    feat = X.feature_panel(universe, lambda d: I2.vortex(d, 14)["vi_spread"])
    w = X.cross_sectional_weights(feat, n_side=2, mode="long_short")
    active = w[w.abs().sum(axis=1) > 0]
    assert np.allclose(active.sum(axis=1), 0.0, atol=1e-9), "book is not dollar-neutral"
    assert np.allclose(active.abs().sum(axis=1), 1.0, atol=1e-9), "gross exposure drifted"

    vol = X.feature_panel(universe, lambda d: I2.parkinson_vol(d, 30, 365.0))
    rp = X.risk_parity(w, vol)
    act = rp[rp.abs().sum(axis=1) > 0]
    assert np.allclose(act.sum(axis=1), 0.0, atol=1e-9), "risk parity broke neutrality"
    # every name long before must still be long after, and vice versa
    both = (w != 0) & (rp != 0)
    assert (np.sign(w[both].fillna(0)) == np.sign(rp[both].fillna(0))).all().all(), \
        "risk parity flipped the sign of a leg"

    # causality: truncating the panel must not change earlier weights
    cut = len(feat) - 120
    w_short = X.cross_sectional_weights(feat.iloc[:cut], n_side=2, mode="long_short")
    pd.testing.assert_frame_equal(w.iloc[:cut], w_short)


def test_xsec_run_applies_execution_lag():
    """Weights set on the last row must earn nothing."""
    from vibt import xsec as X

    universe = D.load_universe()
    if len(universe) < 3:
        pytest.skip("needs a multi-symbol panel")
    px = X.price_panel(universe, "1d", "open")
    w = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    w.iloc[-1] = [1.0 / len(px.columns)] * len(px.columns)
    res = X.run(px, w, B.Costs(0, 0), 365.0)
    assert res.rets.abs().sum() == pytest.approx(0.0, abs=1e-12)


def test_exposure_within_bounds(daily):
    p = SYS.Params()
    s = SYS.target_exposure(daily, p)
    assert s.max() <= p.max_leverage + 1e-9
    assert s.min() >= -p.max_leverage * p.short_size - 1e-9


def test_risk_parity_stands_flat_when_a_leg_cannot_be_sized():
    """If every name on one side has unknown vol, the book must go flat.

    Dropping the unsizeable names and renormalising the rest leaves gross/2 on a
    single side -- a directional bet labelled market-neutral.  The feature needs
    14 bars and the vol estimate needs 30, so a recently listed name really can
    be selectable and unsizeable at once.
    """
    from vibt import xsec as X

    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    cols = ["AAA", "BBB", "CCC", "DDD"]
    w = pd.DataFrame(0.0, index=idx, columns=cols)
    w[["AAA", "BBB"]] = 0.25          # long leg
    w[["CCC", "DDD"]] = -0.25         # short leg

    vol = pd.DataFrame(0.5, index=idx, columns=cols)
    vol.loc[idx[1], ["CCC", "DDD"]] = np.nan   # whole short leg unsizeable
    vol.loc[idx[2], "CCC"] = np.nan            # only part of the leg

    rp = X.risk_parity(w, vol)
    net, gross = rp.sum(axis=1), rp.abs().sum(axis=1)

    assert abs(net.loc[idx[0]]) < 1e-12 and abs(gross.loc[idx[0]] - 1.0) < 1e-12
    assert (rp.loc[idx[1]] == 0).all(), "one-sided book was not flattened"
    # A partly-sizeable leg still trades: the survivor carries that leg alone.
    assert abs(net.loc[idx[2]]) < 1e-12 and abs(gross.loc[idx[2]] - 1.0) < 1e-12
    assert rp.loc[idx[2], "DDD"] < 0 and rp.loc[idx[2], "CCC"] == 0


def test_cross_check_reports_bar_counts_not_just_a_maximum():
    """One corrupted close must read as 1 differing bar, not as a scary maximum."""
    from vibt import ingest as ING

    hourly = D.load("1h", symbol="BTCUSDT").head(400)
    built = D.resample_from(hourly, "4h")
    given = built.copy()

    n, bad = ING.disagreeing_bars(built, given)
    assert n == len(given) and len(bad) == 0

    ts = given.index[5]
    given.loc[ts, "close"] *= 1.02
    n, bad = ING.disagreeing_bars(built, given)
    assert n == len(given), "comparison silently shrank"
    assert list(bad.index) == [ts]
    assert 0.019 < bad.iloc[0] < 0.021


def test_incremental_merge_restores_the_full_history(tmp_path):
    """Splice a recent slice onto truncated history and get the original back.

    An incremental export covers only the last few days.  Without merging it
    would REPLACE the stored file, silently destroying years of history while
    every integrity check still passed -- the short file is perfectly valid.
    """
    from vibt import ingest as ING

    full = D.load("1d", symbol="BTCUSDT")
    dest = tmp_path / "data"
    dest.mkdir()

    # Stored history stops 30 bars short of what the exchange now has.
    ING._write(full.iloc[:-30], dest / "BTCUSDT_1d.csv", compress=False)
    assert len(D.load("1d", dest, "BTCUSDT")) == len(full) - 30

    # A refresh fetches the last 50 bars, overlapping the stored tail by 20.
    src = tmp_path / "exports"
    src.mkdir()
    ING._write(full.iloc[-50:], src / "BTCUSDT_1d_refresh.csv", compress=False)

    report = ING.ingest(src, dest, merge=True)
    assert report.iloc[0]["added"] == 30
    assert report.iloc[0]["overlap"] == 20
    assert report.iloc[0]["restated"] == 0

    got = D.load("1d", dest, "BTCUSDT")
    pd.testing.assert_frame_equal(got, full)


def test_merge_without_it_would_have_truncated(tmp_path):
    """The same refresh without --merge replaces history: that is the hazard."""
    from vibt import ingest as ING

    full = D.load("1d", symbol="BTCUSDT")
    dest = tmp_path / "data"
    dest.mkdir()
    ING._write(full.iloc[:-30], dest / "BTCUSDT_1d.csv", compress=False)

    src = tmp_path / "exports"
    src.mkdir()
    ING._write(full.iloc[-50:], src / "BTCUSDT_1d_refresh.csv", compress=False)

    ING.ingest(src, dest, merge=False)
    assert len(D.load("1d", dest, "BTCUSDT")) == 50, "non-merge should replace"


def test_merge_separates_a_restatement_from_a_forming_last_bar():
    """A changed final bar is normal; a changed settled bar is not."""
    from vibt import ingest as ING

    full = D.load("1d", symbol="BTCUSDT").iloc[-40:]
    old = full.iloc[:-10].copy()
    new = full.iloc[-20:].copy()

    # The previous export's last bar was still forming, so it reads low.
    old.iloc[-1, old.columns.get_loc("close")] *= 0.97
    merged, info = ING.merge_frames(old, new)
    assert info["restated"] == [], "a forming final bar is not a restatement"
    assert info["tail_revised"] is True
    assert merged["close"].iloc[len(old) - 1] == new["close"].loc[old.index[-1]]

    # A settled bar changing is a real restatement and must be surfaced.
    old2 = full.iloc[:-10].copy()
    victim = old2.index[-5]
    old2.loc[victim, "high"] *= 1.05
    _, info2 = ING.merge_frames(old2, new)
    assert info2["restated"] == [victim]


def test_volume_survives_ingest_and_merge(tmp_path):
    """Volume columns must round-trip, including through an incremental merge.

    The whole point of re-pulling the existing symbols is the volume field: the
    6.5bp cost assumption every backtest rests on has never been checked against
    how much these names actually trade.  Dropping the column anywhere in the
    pipeline would make that pull worthless while every other check still passed.
    """
    from vibt import ingest as ING

    idx = pd.date_range("2024-01-01", periods=200, freq="D")
    base = pd.DataFrame({
        "开盘": np.linspace(100, 300, 200), "最高": np.linspace(101, 303, 200),
        "最低": np.linspace(99, 297, 200), "收盘": np.linspace(100.5, 301, 200),
        "volume": np.arange(200, dtype=float) + 1000,
        "quote_volume": (np.arange(200, dtype=float) + 1000) * 200,
        "trades": np.arange(200, dtype=float) + 50,
        "taker_buy_base": (np.arange(200, dtype=float) + 1000) / 2,
        "taker_buy_quote": (np.arange(200, dtype=float) + 1000) * 100,
    })
    base.insert(0, "日期时间(北京)", (idx + pd.Timedelta(hours=8)).strftime("%Y-%m-%d %H:%M:%S"))

    src, dest = tmp_path / "pull", tmp_path / "data"
    src.mkdir(); dest.mkdir()
    base.iloc[:150].to_csv(src / "VOLUSDT_1d_2024-01-01_to_now.csv",
                           index=False, encoding="utf-8-sig")
    ING.ingest(src, dest, compress=True)

    got = D.load("1d", dest, "VOLUSDT")
    for c in D.EXTRA_COLS:
        assert c in got.columns, f"{c} was dropped by ingest"
    assert got["volume"].iloc[0] == 1000.0
    assert got["trades"].iloc[-1] == 50.0 + 149

    # An incremental refresh must not wipe volume off the bars it is splicing onto.
    src2 = tmp_path / "refresh"; src2.mkdir()
    base.iloc[140:].to_csv(src2 / "VOLUSDT_1d_refresh.csv", index=False, encoding="utf-8-sig")
    rep = ING.ingest(src2, dest, compress=True, merge=True)
    assert rep.iloc[0]["added"] == 50

    merged = D.load("1d", dest, "VOLUSDT")
    assert len(merged) == 200
    assert merged["volume"].notna().all(), "merge left holes in volume"
    assert merged["volume"].iloc[0] == 1000.0        # from the old side
    assert merged["volume"].iloc[-1] == 1000.0 + 199  # from the new side


def test_grid_realised_curve_hides_the_loss(tmp_path):
    """The realised curve must stay smooth while the marked account collapses.

    This is the property the whole simulator exists to expose, so it is pinned:
    on a one-way decline a martingale closes nothing at a loss, so realised P&L
    barely moves while mark-to-market equity falls apart.  If a refactor ever
    made the two curves agree, the simulator would have stopped modelling the
    thing that matters.
    """
    from vibt import grid as G

    # A decline deep enough to fill the ladder and keep going, but not deep
    # enough to liquidate -- liquidation zeroes both curves and would hide the
    # very divergence being tested.
    n = 400
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    trend = 100 * (1 - 0.0007) ** np.arange(n)
    # amplitude must exceed take_profit or no level ever closes and the
    # win-rate half of the claim cannot be demonstrated
    px = pd.Series(trend * (1 + 0.015 * np.sin(np.arange(n) / 3)), index=idx)
    df = pd.DataFrame({"open": px, "high": px * 1.002,
                       "low": px * 0.998, "close": px})

    r = G.run_grid(df, G.GridParams(max_levels=7, fee=0.0))
    assert r.liquidated_at is None, "scenario should stop short of liquidation"
    assert r.closed > 0 and r.win_rate == 1.0, "every closed trade must be a winner"
    assert r.realised_dd > -0.02, f"realised curve should stay flat, got {r.realised_dd:.3f}"
    assert r.mark_dd < -0.20, f"marked curve must collapse, got {r.mark_dd:.3f}"
    assert r.max_levels_used == 7, "a sustained move should fill the whole ladder"


def test_grid_ladder_is_filled_adversely_within_the_bar():
    """Entries the low reaches must fill before take-profits the high reaches.

    Intrabar order is unknowable from OHLC, so the simulator has to assume the
    order that does not flatter the strategy.  A bar that spans several rungs
    and also clears a take-profit must add the rungs first.
    """
    from vibt import grid as G

    idx = pd.date_range("2024-01-01", periods=2, freq="h")
    df = pd.DataFrame({"open": [100.0, 100.0], "high": [100.0, 102.0],
                       "low": [100.0, 96.0], "close": [100.0, 101.0]}, index=idx)
    r = G.run_grid(df, G.GridParams(step=0.01, take_profit=0.01, max_levels=5, fee=0.0))
    assert r.max_levels_used >= 4, "the low should have filled several rungs first"


def test_rsi_strategy_cannot_see_the_bar_it_trades():
    """Perturbing a bar's close must not change any fill at or before that bar.

    Signals are read from bar t's close and filled at bar t+1's open, so
    rewriting bar t's close may move later trades but must leave everything up
    to and including t untouched.  A stateful pyramiding loop is exactly where an
    off-by-one slips in unnoticed.
    """
    from vibt import rsi_strat as R

    df = D.load("4h", symbol="BTCUSDT").iloc[:3000].copy()
    base = R.run(df)
    assert len(base.trades) > 0, "need trades for this test to mean anything"

    cut = df.index[2000]
    bumped = df.copy()
    bumped.loc[cut, "close"] *= 1.15

    after = R.run(bumped)
    a = base.trades[base.trades.entry_time <= cut]
    b = after.trades[after.trades.entry_time <= cut]
    assert len(a) == len(b), "a later close changed an earlier trade count"
    pd.testing.assert_series_equal(a["pnl"].reset_index(drop=True),
                                   b["pnl"].reset_index(drop=True),
                                   check_exact=False, rtol=1e-9)


def test_rsi_pyramid_never_exceeds_full_equity():
    """50% + 5x10% must stay at or under 100% of the equity it was sized from."""
    from vibt import rsi_strat as R

    df = D.load("4h", symbol="BTCUSDT")
    res = R.run(df)
    p = res.params
    cap = p.first_frac + p.max_adds * p.add_frac
    assert abs(cap - 1.0) < 1e-12, "spec says the ladder tops out at 100%"
    assert res.trades["legs"].max() <= 1 + p.max_adds, "more adds than allowed"


def test_rsi_matches_the_pine_definition():
    """RSI must reproduce ta.rma seeding, not pandas' first-value seeding."""
    from vibt import rsi_strat as R

    s = D.load("4h", symbol="BTCUSDT")["close"].iloc[:400]
    got = R.rsi_pine(s, 14)
    chg = s.diff()
    up = R.rma(chg.clip(lower=0), 14)
    dn = R.rma(-chg.clip(upper=0), 14)
    assert abs(up.iloc[13] - chg.clip(lower=0).iloc[:14].mean()) < 1e-12
    assert got.dropna().between(0, 100).all()
    assert got.iloc[:13].isna().all(), "RSI must not exist before its warmup"


# --------------------------------------------------------------- six-MA system
def test_ma_cluster_threshold_cannot_see_the_current_bar():
    """The 'is this a cluster' threshold must be built from prior bars only.

    A tight-MA threshold that includes today's spread in its own quantile is the
    subtlest lookahead available here: it never moves a fill by one bar, it just
    quietly makes clusters identifiable at the moment they matter.
    """
    from vibt import masys as MS

    df = D.load("1d", symbol="BTCUSDT").copy()
    p = MS.MaParams()
    mas = MS.six_mas(df["close"], p.lens)
    sp = MS.spread(mas, df["close"])
    base = MS.tight_mask(sp, p)

    bumped = sp.copy()
    cut = 800
    bumped.iloc[cut] *= 5.0
    after = MS.tight_mask(bumped, p)
    assert not base.iloc[:cut].ne(after.iloc[:cut]).any(), \
        "a later spread value changed an earlier cluster verdict"


def test_ma_signals_cannot_see_the_bar_they_fill_on():
    """Rewriting bar t's close must not alter any signal placed at or before t."""
    from vibt import masys as MS

    df = D.load("4h", symbol="BTCUSDT").iloc[:4000].copy()
    p = MS.MaParams()
    for fn in (MS.entries_cluster_break, MS.entries_ma20_pullback):
        base = fn(df, p)
        assert base, "need signals for this test to mean anything"
        cut = 3000
        bumped = df.copy()
        bumped.iloc[cut, bumped.columns.get_loc("close")] *= 1.2
        after = fn(bumped, p)
        a = [(s.i, s.side, round(s.stop, 8)) for s in base if s.i <= cut]
        b = [(s.i, s.side, round(s.stop, 8)) for s in after if s.i <= cut]
        assert a == b, f"{fn.__name__} moved a signal at or before the changed bar"


def test_ma_bracket_fills_the_stop_when_a_bar_holds_both():
    """One bar spanning stop and target must resolve as the loss, never the win."""
    from vibt import masys as MS

    idx = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    df = pd.DataFrame({"open": [100.0, 100.0, 100.0],
                       "high": [100.0, 140.0, 100.0],
                       "low": [100.0, 80.0, 100.0],
                       "close": [100.0, 100.0, 100.0]}, index=idx)
    p = MS.MaParams(target_r=3.0, fee=0.0)
    sig = [MS.Signal(i=1, side=1, stop=90.0)]      # target 130, stop 90, both hit
    t = MS.evaluate(df, sig, p)
    assert t.loc[0, "why"] == "stop"
    assert t.loc[0, "r_multiple"] == pytest.approx(-1.0)


def test_ma_gap_through_the_stop_fills_at_the_open():
    """A bar opening past the stop cannot fill at the stop price."""
    from vibt import masys as MS

    idx = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    df = pd.DataFrame({"open": [100.0, 100.0, 70.0],
                       "high": [100.0, 101.0, 72.0],
                       "low": [100.0, 99.0, 65.0],
                       "close": [100.0, 100.0, 70.0]}, index=idx)
    p = MS.MaParams(target_r=3.0, fee=0.0)
    t = MS.evaluate(df, [MS.Signal(i=1, side=1, stop=90.0)], p)
    assert t.loc[0, "exit"] == pytest.approx(70.0), "filled better than the gap allowed"
    assert t.loc[0, "r_multiple"] == pytest.approx(-3.0)


def test_ma_cluster_break_fires_once_per_cluster():
    """A: staying outside the band is a state; leaving it is the event traded."""
    from vibt import masys as MS

    df = D.load("1d", symbol="BTCUSDT")
    p = MS.MaParams()
    mas = MS.six_mas(df["close"], p.lens)
    tight = MS.tight_mask(MS.spread(mas, df["close"]), p)
    clusters = int(((tight & ~tight.shift(1).fillna(False)).sum()))
    assert len(MS.entries_cluster_break(df, p)) <= clusters, \
        "more entries than clusters means the rule is firing on a state"


def test_ma_null_matches_the_stop_width_it_resamples():
    """The random control must reproduce the stop widths, not invent its own."""
    from vibt import masys as MS

    df = D.load("1d", symbol="BTCUSDT")
    p = MS.MaParams()
    real = MS.evaluate(df, MS.entries_ma20_pullback(df, p), p)
    assert len(real) >= 3
    rng = np.random.default_rng(0)
    sig = MS.random_entries(df, p, 200, 200, rng, real.risk_atr.to_numpy())
    z = MS.evaluate(df, sig, p)
    assert len(z) > 50
    lo, hi = real.risk_atr.min(), real.risk_atr.max()
    assert z.risk_atr.between(lo * 0.99, hi * 1.01).all(), \
        "the null drew stop widths outside the distribution it was given"
    assert (z.side == "long").sum() > 0 and (z.side == "short").sum() > 0


def test_ma_null_respects_the_requested_side_counts():
    """Longs and shorts are drawn to their own counts, not flipped at a ratio."""
    from vibt import masys as MS

    df = D.load("1d", symbol="BTCUSDT")
    p = MS.MaParams(min_risk=0.0, max_risk=1.0)
    rng = np.random.default_rng(3)
    sig = MS.random_entries(df, p, 40, 10, rng, np.array([1.0, 1.5, 2.0]))
    assert sum(s.side > 0 for s in sig) == 40
    assert sum(s.side < 0 for s in sig) == 10


def test_ma_warmup_holds_every_average_back():
    """EMA is defined from bar 0; a cluster must not be callable before warmup."""
    from vibt import masys as MS

    df = D.load("1d", symbol="BTCUSDT")
    mas = MS.six_mas(df["close"], MS.MA_LENS)
    warm = max(MS.MA_LENS)
    assert mas.iloc[:warm].isna().all().all(), "an average was live before its window"
    assert mas.iloc[warm:].notna().all().all()


def test_panel_turnover_is_name_level_not_gross_exposure():
    """A dollar-neutral book rotating every name must not report ~0 turnover.

    metrics.compute derives turnover from the position series, and for a panel
    that series is GROSS EXPOSURE -- pinned at 1.0 for a dollar-neutral book.
    Left alone it reports near-zero turnover however violently the names
    underneath are rotated, which hides exactly the fact that decides whether a
    high-frequency cross-sectional signal can pay for itself.
    """
    from vibt import xsec as X

    syms = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
    P = pd.DataFrame({s: D.load("1d", symbol=s)["open"] for s in syms}).sort_index()
    w = pd.DataFrame(0.0, index=P.index, columns=P.columns)
    # every day: flip the whole book between two pairs, so 2.0 notional turns over
    w.iloc[::2, 0], w.iloc[::2, 1] = 0.5, -0.5
    w.iloc[1::2, 2], w.iloc[1::2, 3] = 0.5, -0.5

    r = X.run(P, w, B.Costs(4.5, 2.0), 365.0, "")
    # the first bar is flat by construction (weights are shifted before use), so
    # the constant-exposure claim is about every bar after the warmup
    gross = r.weights.abs().sum(axis=1).iloc[1:]
    assert gross.std() < 1e-9 and gross.iloc[0] == pytest.approx(1.0), \
        "gross exposure should be pinned at 1.0 -- that is what hides the rotation"
    assert r.stats.turnover_ann == pytest.approx(730, rel=0.02), \
        "turnover must count name-level rotation, not the constant gross exposure"
    # and the cost series must agree with that turnover
    years = len(r.rets) / 365.0
    implied = r.costs.sum() / years / B.Costs(4.5, 2.0).per_side
    assert implied == pytest.approx(r.stats.turnover_ann, rel=0.02)
