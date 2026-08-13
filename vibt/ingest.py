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


def ingest(src: Path, dest: Path | None = None, dry_run: bool = False) -> pd.DataFrame:
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
        target = dest / f"{symbol}_{tf}.csv"
        if not dry_run:
            shutil.copyfile(path, target)
        rows.append({"symbol": symbol, "tf": tf, "source": path.name, **check(df, tf)})
    return pd.DataFrame(rows)


def _load_direct(path: Path, tf: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig").rename(columns=D._COLUMN_MAP)
    df["ts"] = pd.to_datetime(df["ts"]) - pd.Timedelta(hours=8)
    df = df.set_index("ts").sort_index()
    df = df[~df.index.duplicated(keep="last")][["open", "high", "low", "close"]].astype(float)
    df.index.name = "open_time"
    df["close_time"] = df.index + pd.Timedelta(minutes=D.TF_MINUTES[tf])
    return df


def cross_check(dest: Path | None = None) -> None:
    """Rebuild 4h/1d from 1h per symbol and confirm the files agree."""
    dest = dest or D.DATA_DIR
    for symbol in D.available_symbols(dest):
        f = D.load_all(dest, symbol)
        msgs = []
        for tf, rule in (("4h", "4h"), ("1d", "1D")):
            built = D.resample_from(f["1h"], rule)
            given = f[tf][["open", "high", "low", "close"]]
            common = built.index.intersection(given.index)
            if not len(common):
                msgs.append(f"{tf}: no overlap")
                continue
            rel = ((built.loc[common] - given.loc[common]).abs() / given.loc[common]).max().max()
            msgs.append(f"{tf}: {len(common)} shared bars, max rel diff {rel:.1e}")
        print(f"  {symbol:<10} " + "   ".join(msgs))


def main() -> None:
    ap = argparse.ArgumentParser(description="normalise exported klines into data/")
    ap.add_argument("src", type=Path, help="folder holding the exported CSVs")
    ap.add_argument("--dest", type=Path, default=None, help="target data dir (default: data/)")
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = ap.parse_args()

    report = ingest(args.src, args.dest, args.dry_run)
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
