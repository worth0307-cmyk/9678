"""Normalise exported kline CSVs into data/{SYMBOL}_{tf}.csv, and check them.

Written for the output of the dashboard's own exporter:

    python3 backend/tools/export_klines.py --symbol ETHUSDT --market futures \
        --intervals 1h,4h,1d --start 2024-01-01 --out ./exports

which writes e.g. `ETHUSDT_4h_2024-01-01_to_now.csv`.  Filenames from other
sources work too as long as the symbol and interval are the first two
underscore-separated fields.  Run:

    python -m vibt.ingest ./exports
    python -m vibt.ingest ./exports --dry-run     # report only, write nothing
"""

from __future__ import annotations

import argparse
import gzip
import re
import shutil
from pathlib import Path

import pandas as pd

from . import data as D

# The interval must be followed by a separator or end-of-name.  `\b` will not do:
# `_` is a word character, so there is no boundary in "..._4h_2024-01-01".
FNAME = re.compile(r"^(?P<symbol>[A-Z0-9]+)[_-](?P<tf>1h|4h|1d)(?=[_.\-]|$)", re.IGNORECASE)


def parse_name(path: Path) -> tuple[str, str] | None:
    m = FNAME.match(path.stem)
    if not m:
        return None
    return m.group("symbol").upper(), m.group("tf").lower()


def check(df: pd.DataFrame, tf: str) -> dict:
    gaps = D.gap_report(df, tf)
    bad_ohlc = int(
        ((df["high"] < df["low"])
         | (df["high"] < df[["open", "close"]].max(axis=1) - 1e-9)
         | (df["low"] > df[["open", "close"]].min(axis=1) + 1e-9)).sum()
    )
    return {
        "bars": len(df),
        "first": df.index[0],
        "last": df.index[-1],
        "gaps": len(gaps),
        "missing_bars": int(gaps["bars_missing"].sum()) if len(gaps) else 0,
        "bad_ohlc": bad_ohlc,
        "nonpositive": int((df[["open", "high", "low", "close"]] <= 0).any(axis=1).sum()),
    }


def ingest(src: Path, dest: Path | None = None, dry_run: bool = False,
           compress: bool = False, merge: bool = False) -> pd.DataFrame:
    """Copy the exports into data/, one file per (symbol, timeframe).

    Exporting the same symbol twice with different --start leaves two files for
    the same (symbol, tf).  Rather than letting glob order decide, load every
    candidate and keep the one with the longest history; the rest are reported
    as superseded so nothing disappears silently.
    """
    dest = dest or D.DATA_DIR
    dest.mkdir(parents=True, exist_ok=True)

    candidates: dict[tuple[str, str], list[tuple[Path, pd.DataFrame]]] = {}
    for path in sorted(src.glob("*.csv")):
        parsed = parse_name(path)
        if not parsed:
            print(f"  skip {path.name}  (cannot read SYMBOL_tf from the filename)")
            continue
        try:
            df = _load_direct(path, parsed[1])
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"  FAIL {path.name}: {exc}")
            continue
        if df.empty:
            print(f"  skip {path.name}  (no rows)")
            continue
        candidates.setdefault(parsed, []).append((path, df))

    rows = []
    for (symbol, tf), items in sorted(candidates.items()):
        items.sort(key=lambda it: (it[1].index[0], -len(it[1])))
        path, df = items[0]
        for other_path, other_df in items[1:]:
            print(f"  superseded: {other_path.name} ({len(other_df)} bars from "
                  f"{other_df.index[0].date()})  <  keeping {path.name} "
                  f"({len(df)} bars from {df.index[0].date()})")
        target = dest / (f"{symbol}_{tf}.csv.gz" if compress else f"{symbol}_{tf}.csv")

        extra: dict = {}
        if merge:
            try:
                old = D.load(tf, dest, symbol)
            except FileNotFoundError:
                old = None
            if old is not None:
                # Keep whatever columns the stored file has -- subsetting to OHLC
                # here would drop volume from the old side on every refresh.
                df, info = merge_frames(old, df)
                extra = {"added": info["added"], "overlap": info["overlap"],
                         "restated": len(info["restated"])}
                if info["restated"]:
                    shown = ", ".join(str(t) for t in info["restated"][:3])
                    print(f"  RESTATED {symbol} {tf}: the exchange changed "
                          f"{len(info['restated'])} settled bar(s) ({shown})"
                          + (" ..." if len(info["restated"]) > 3 else ""))
                elif info["added"] == 0:
                    print(f"  {symbol} {tf}: already current, nothing new")

        if not dry_run:
            # Drop the other form so a symbol never has both a stale .csv and a
            # fresh .csv.gz, which would silently resolve to the stale one.
            other = dest / (f"{symbol}_{tf}.csv" if compress else f"{symbol}_{tf}.csv.gz")
            other.unlink(missing_ok=True)
            if merge:
                _write(df, target, compress)
            elif compress:
                with open(path, "rb") as fi, gzip.open(target, "wb") as fo:
                    shutil.copyfileobj(fi, fo)
            else:
                shutil.copyfile(path, target)
        rows.append({"symbol": symbol, "tf": tf, "source": path.name,
                     **check(df, tf), **extra})
    return pd.DataFrame(rows)


def _write(df: pd.DataFrame, target: Path, compress: bool) -> None:
    """Write a frame back in the exporter's own format: Beijing time, its headers."""
    out = pd.DataFrame({
        "日期时间(北京)": (df.index + pd.Timedelta(hours=8)).strftime("%Y-%m-%d %H:%M:%S"),
        "开盘": df["open"], "最高": df["high"], "最低": df["low"], "收盘": df["close"],
    })
    for c in D.EXTRA_COLS:                 # volume survives a merge, or it was
        if c in df.columns:                # never worth fetching in the first place
            out[c] = df[c].to_numpy()
    out.to_csv(target, index=False, encoding="utf-8-sig",
               compression="gzip" if compress else None)


def merge_frames(old: pd.DataFrame, new: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Splice an incremental export onto stored history, newest data winning.

    New data wins on overlap because the previous export's final bar was very
    likely still forming when it was taken -- Binance returns the in-progress
    candle -- so that bar legitimately differs and the fresh copy is the
    complete one.  Any disagreement *before* that last bar is a different animal
    entirely: it means the exchange restated settled history, which silently
    invalidates every backtest run against the old copy.  The two are counted
    separately so the second can never hide inside the first.
    """
    cols = ["open", "high", "low", "close"]
    overlap = old.index.intersection(new.index)
    restated = []
    if len(overlap):
        diff = (old.loc[overlap, cols] - new.loc[overlap, cols]).abs()
        scale = old.loc[overlap, cols].abs().clip(lower=1e-12)
        moved = ((diff / scale) > 1e-9).any(axis=1)
        restated = [t for t in overlap[moved] if t != old.index[-1]]

    combined = pd.concat([old[~old.index.isin(new.index)], new]).sort_index()
    return combined, {
        "added": int(len(combined) - len(old)),
        "overlap": int(len(overlap)),
        "restated": restated,
        "tail_revised": bool(len(overlap) and old.index[-1] in overlap
                             and not old.loc[old.index[-1], cols]
                             .equals(new.loc[old.index[-1], cols])),
    }


def _load_direct(path: Path, tf: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig").rename(columns=D._COLUMN_MAP)
    df["ts"] = pd.to_datetime(df["ts"]) - pd.Timedelta(hours=8)
    df = df.set_index("ts").sort_index()
    keep = ["open", "high", "low", "close"] + [c for c in D.EXTRA_COLS if c in df.columns]
    df = df[~df.index.duplicated(keep="last")][keep].astype(float)
    df.index.name = "open_time"
    df["close_time"] = df.index + pd.Timedelta(minutes=D.TF_MINUTES[tf])
    return df


def disagreeing_bars(built: pd.DataFrame, given: pd.DataFrame,
                     tol: float = 1e-9) -> tuple[int, pd.Series]:
    """Bars where a rebuilt higher timeframe differs from the supplied one.

    Returns (bars compared, worst per-bar relative difference for the ones that
    disagree).  The *count* is the number that matters.  A single bad close
    makes the largest relative difference look exactly like wholesale
    corruption, so reporting only a maximum turns a two-bar artefact into a
    scare -- which is precisely what the first version of this check did.
    """
    common = built.index.intersection(given.index)
    if not len(common):
        return 0, pd.Series(dtype=float)
    rel = ((built.loc[common] - given.loc[common]).abs() / given.loc[common]).max(axis=1)
    return len(common), rel[rel > tol]


def cross_check(dest: Path | None = None, show: int = 4) -> None:
    """Rebuild each timeframe from the one below it and confirm the files agree.

    1d is rebuilt from 4h as well as from 1h: daily OHLC only reads the first
    open, the last close and the extremes, so a 4h file can carry a wrong
    intermediate close and still aggregate to a perfect day.
    """
    dest = dest or D.DATA_DIR
    seen: dict[tuple[str, pd.Timestamp], int] = {}
    for symbol in D.available_symbols(dest):
        f = D.load_all(dest, symbol)
        msgs = []
        for src, tf, rule in (("1h", "4h", "4h"), ("1h", "1d", "1D"), ("4h", "1d", "1D")):
            n, bad = disagreeing_bars(D.resample_from(f[src], rule),
                                      f[tf][["open", "high", "low", "close"]])
            if not n:
                msgs.append(f"{src}->{tf}: no overlap")
                continue
            msgs.append(f"{src}->{tf}: {len(bad)}/{n}"
                        + (f" (worst {bad.max():.1e})" if len(bad) else ""))
            for ts in bad.index:
                seen[(f"{src}->{tf}", ts)] = seen.get((f"{src}->{tf}", ts), 0) + 1
        print(f"  {symbol:<14} " + "   ".join(msgs))

    if not seen:
        print("\n  every timeframe reconstructs exactly from the one below it")
        return
    # A timestamp that disagrees across many symbols at once is an exchange-side
    # event, not a per-symbol data error.  Ranking by how many symbols share a
    # timestamp is what separates the two without reading 26 rows by hand.
    ranked = sorted(seen.items(), key=lambda kv: (-kv[1], kv[1]))
    print(f"\n  bars that disagree, most widely shared first "
          f"({len(ranked)} distinct, showing {min(show, len(ranked))}):")
    for (pair, ts), count in ranked[:show]:
        note = "  <- exchange-wide, not a per-symbol fault" if count > 2 else ""
        print(f"    {pair}  {ts}  {count} symbol(s){note}")


FUND_NAME = re.compile(r"^(?P<symbol>[A-Z0-9]+)[_-]funding$", re.IGNORECASE)


def _load_funding(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    ts = df.columns[0]
    df["ts"] = pd.to_datetime(df[ts]) - pd.Timedelta(hours=8)   # Beijing -> UTC
    df = df.drop(columns=[ts]).set_index("ts").sort_index()
    return df[~df.index.duplicated(keep="last")]


def _write_funding(df: pd.DataFrame, target: Path, compress: bool) -> None:
    out = df.copy()
    out.insert(0, "日期时间(北京)",
               (out.index + pd.Timedelta(hours=8)).strftime("%Y-%m-%d %H:%M:%S"))
    out.to_csv(target, index=False, encoding="utf-8-sig",
               compression="gzip" if compress else None)


def funding_interval_hours(idx: pd.DatetimeIndex) -> float:
    """Modal spacing between settlements, in hours.

    Not a formality: Binance settles most pairs every 8 hours but some every 4,
    and switches a pair temporarily during extreme volatility.  A mean rate
    "per period" is therefore not comparable across symbols, and comparing them
    without normalising to a daily rate reverses the ranking outright -- TAO at
    +0.15bp per 4h period is +0.91bp/day, against BTC at +0.60bp per 8h period,
    which is +1.81bp/day.
    """
    if len(idx) < 3:
        return float("nan")
    d = pd.Series(idx).diff().dropna().dt.total_seconds() / 3600.0
    return float(d.mode().iloc[0]) if len(d.mode()) else float(d.median())


def ingest_funding(src: Path, dest: Path | None = None, dry_run: bool = False,
                   compress: bool = True, merge: bool = False) -> pd.DataFrame:
    """Same discipline as the klines: merge onto history, count restatements.

    Funding is settled history -- unlike a kline it is never "still forming" --
    so ANY disagreement on an overlapping timestamp is a restatement, with no
    final-bar exemption to hide behind.
    """
    dest = (dest or D.DATA_DIR) / "funding"
    rows = []
    for path in sorted(src.glob("*_funding.csv")) + sorted(src.glob("*_funding.csv.gz")):
        m = FUND_NAME.match(path.name.split(".")[0])
        if not m:
            continue
        sym = m.group("symbol").upper()
        new = _load_funding(path)
        target = dest / (f"{sym}_funding.csv.gz" if compress else f"{sym}_funding.csv")
        extra = {"added": len(new), "overlap": 0, "restated": 0}
        out = new
        old_path = target if target.exists() else dest / f"{sym}_funding.csv"
        if merge and old_path.exists():
            old = _load_funding(old_path)
            overlap = old.index.intersection(new.index)
            col = "funding_rate"
            diff = (old.loc[overlap, col].astype(float)
                    - new.loc[overlap, col].astype(float)).abs()
            extra = {"added": int(len(new.index.difference(old.index))),
                     "overlap": int(len(overlap)),
                     "restated": int((diff > 1e-12).sum())}
            out = pd.concat([old[~old.index.isin(new.index)], new]).sort_index()
        rate = out["funding_rate"].astype(float)
        hrs = funding_interval_hours(out.index)
        rows.append({"symbol": sym, "rows": len(out),
                     "first": out.index[0], "last": out.index[-1],
                     "interval_h": hrs,
                     "mean_bp_period": round(rate.mean() * 1e4, 4),
                     "mean_bp_day": round(rate.mean() * 1e4 * (24.0 / hrs), 4)
                     if hrs and hrs == hrs else float("nan"),
                     **extra})
        if not dry_run:
            dest.mkdir(parents=True, exist_ok=True)
            _write_funding(out, target, compress)
            if old_path != target:
                old_path.unlink(missing_ok=True)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="normalise exported klines into data/")
    ap.add_argument("src", type=Path, help="folder holding the exported CSVs")
    ap.add_argument("--dest", type=Path, default=None, help="target data dir (default: data/)")
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    ap.add_argument("--gzip", action="store_true",
                    help="store as .csv.gz (~4x smaller; loaders read it transparently)")
    ap.add_argument("--merge", action="store_true",
                    help="splice onto existing history instead of replacing it "
                         "(use for incremental refreshes)")
    args = ap.parse_args()

    report = ingest(args.src, args.dest, args.dry_run, args.gzip, args.merge)
    fund = ingest_funding(args.src, args.dest, args.dry_run, args.gzip, args.merge)
    if not fund.empty:
        print("\nfunding:")
        print(fund.to_string(index=False))
        bad_f = fund[fund["restated"] > 0]
        if len(bad_f):
            print(f"\n  WARNING: {int(bad_f['restated'].sum())} settled funding rows "
                  f"were restated by the exchange")
    if report.empty:
        print("nothing ingested")
        return
    print("\ncoverage:")
    print(report.to_string(index=False))

    bad = report[(report["gaps"] > 0) | (report["bad_ohlc"] > 0) | (report["nonpositive"] > 0)]
    if len(bad):
        print("\nPROBLEMS FOUND -- do not trust backtests on these until resolved:")
        print(bad.to_string(index=False))
    else:
        print("\nno gaps, no impossible OHLC, no non-positive prices")

    if not args.dry_run:
        print("\ncross-check (1h resampled vs the supplied 4h/1d):")
        cross_check(args.dest)
        print(f"\nsymbols now loadable: {D.available_symbols(args.dest)}")


if __name__ == "__main__":
    main()
