"""Data loading and multi-timeframe alignment.

The CSV files are exported with Beijing-time (UTC+8) timestamps labelling the
*open* of each bar.  Because 8 is a multiple of 4, the 4h and 1d grids are the
same grid Binance uses in UTC -- Beijing 08:00 daily bars are UTC 00:00 days.
Internally everything is kept in UTC.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_SYMBOL = "BTCUSDT"

_COLUMN_MAP = {
    "日期时间(北京)": "ts",
    "开盘": "open",
    "最高": "high",
    "最低": "low",
    "收盘": "close",
}

TF_MINUTES = {"1h": 60, "4h": 240, "1d": 1440}


def available_symbols(data_dir: Path | None = None,
                      require: tuple[str, ...] = ("1h", "4h", "1d")) -> list[str]:
    """Symbols holding every timeframe in `require`.

    Screening a wide universe only needs daily bars, and exporting 1h for
    thirty-odd symbols is most of the download for data the cross-sectional
    work never reads -- so pass require=("1d",) to see daily-only symbols too.
    """
    data_dir = data_dir or DATA_DIR
    need = set(require)
    by_symbol: dict[str, set[str]] = {}
    for p in list(data_dir.glob("*_*.csv")) + list(data_dir.glob("*_*.csv.gz")):
        stem = p.name[:-7] if p.name.endswith(".csv.gz") else p.stem
        if "_" not in stem:
            continue
        sym, _, tf = stem.rpartition("_")
        if tf in TF_MINUTES:
            by_symbol.setdefault(sym, set()).add(tf)
    return sorted(s for s, tfs in by_symbol.items() if tfs >= need)


def resolve(tf: str, data_dir: Path, symbol: str) -> Path:
    """Locate a symbol/timeframe file, preferring plain CSV over gzip.

    1h files for a few dozen symbols run to tens of megabytes; storing them
    gzipped cuts that by roughly 4x and pandas reads .csv.gz transparently, so
    both layouts work and can be mixed in one directory.
    """
    plain = data_dir / f"{symbol}_{tf}.csv"
    if plain.exists():
        return plain
    gz = data_dir / f"{symbol}_{tf}.csv.gz"
    if gz.exists():
        return gz
    raise FileNotFoundError(
        f"no data for {symbol} {tf} in {data_dir} "
        f"(looked for {plain.name} and {gz.name})"
    )


def load(tf: str, data_dir: Path | None = None,
         symbol: str = DEFAULT_SYMBOL) -> pd.DataFrame:
    """Load one timeframe as a UTC-indexed OHLC frame indexed by bar OPEN time."""
    data_dir = data_dir or DATA_DIR
    path = resolve(tf, data_dir, symbol)
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.rename(columns=_COLUMN_MAP)
    missing = {"ts", "open", "high", "low", "close"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    df["ts"] = pd.to_datetime(df["ts"]) - pd.Timedelta(hours=8)  # Beijing -> UTC
    df = df.set_index("ts").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df[["open", "high", "low", "close"]].astype(float)
    df.index.name = "open_time"

    # close_time is what an executor is actually allowed to know the bar by.
    df["close_time"] = df.index + pd.Timedelta(minutes=TF_MINUTES[tf])
    return df


def load_all(data_dir: Path | None = None, symbol: str = DEFAULT_SYMBOL,
             tfs: tuple[str, ...] = ("1h", "4h", "1d")) -> dict[str, pd.DataFrame]:
    return {tf: load(tf, data_dir, symbol) for tf in tfs}


def load_universe(symbols: list[str] | None = None,
                  data_dir: Path | None = None,
                  tfs: tuple[str, ...] = ("1h", "4h", "1d")
                  ) -> dict[str, dict[str, pd.DataFrame]]:
    """{symbol: {tf: frame}} for symbols holding every timeframe in `tfs`."""
    symbols = symbols or available_symbols(data_dir, require=tfs)
    return {s: load_all(data_dir, s, tfs) for s in symbols}


def gap_report(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Rows where the spacing between consecutive bars is not exactly one period."""
    step = pd.Timedelta(minutes=TF_MINUTES[tf])
    delta = df.index.to_series().diff()
    bad = delta[(delta.notna()) & (delta != step)]
    return pd.DataFrame({"gap": bad, "bars_missing": (bad / step) - 1})


def align_higher_tf(
    base_index: pd.DatetimeIndex,
    higher: pd.DataFrame,
    higher_tf: str,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Project higher-timeframe values onto a lower-timeframe index without lookahead.

    A higher-timeframe bar is only usable from its *close_time* onwards, so the
    value carried at base bar ``t`` is the last higher bar that had already
    closed at or before ``t``.  ``base_index`` must be bar-open timestamps: a
    strategy deciding at the open of bar ``t`` may use any HTF bar that closed
    at or before ``t``.
    """
    columns = columns or [c for c in higher.columns if c != "close_time"]
    src = higher[columns].copy()
    src.index = higher["close_time"]
    src = src.sort_index()
    out = src.reindex(src.index.union(base_index)).ffill().reindex(base_index)
    out.columns = [f"{c}_{higher_tf}" for c in columns]
    return out


def resample_from(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Rebuild a higher timeframe from a lower one (used only to cross-check feeds)."""
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    out = df[["open", "high", "low", "close"]].resample(rule, label="left", closed="left").agg(agg)
    return out.dropna()


def bars_per_year(tf: str) -> float:
    return 365.0 * 24 * 60 / TF_MINUTES[tf]


def synthetic_intrabar_path(df: pd.DataFrame) -> np.ndarray:
    """Conservative high/low visit order: assume the adverse extreme comes first.

    Returns +1 when the low is assumed to be touched before the high (bearish
    bar), -1 otherwise.  Used by the backtester when both a stop and a target
    sit inside the same bar.
    """
    return np.where(df["close"].to_numpy() >= df["open"].to_numpy(), 1, -1)
