"""Pull BTCUSDT klines from Binance's public REST API and keep the CSVs current.

No API key needed -- /api/v3/klines is public.  Written to the same schema as
the supplied files (Beijing-time bar-open labels) so everything downstream is
unchanged.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

from . import data as D

BASES = [
    "https://api.binance.com",
    "https://api1.binance.com",
    "https://data-api.binance.vision",
]
INTERVAL = {"1h": "1h", "4h": "4h", "1d": "1d"}


def _get(path: str, params: dict, retries: int = 4) -> list:
    query = "&".join(f"{k}={v}" for k, v in params.items())
    last = None
    for attempt in range(retries):
        for base in BASES:
            url = f"{base}{path}?{query}"
            try:
                with urllib.request.urlopen(url, timeout=20) as r:
                    return json.loads(r.read().decode())
            except Exception as exc:  # noqa: BLE001 - any transport error is retryable
                last = exc
        time.sleep(2**attempt)
    raise RuntimeError(f"binance request failed after {retries} rounds: {last}")


def fetch_klines(symbol: str, tf: str, start_ms: int, end_ms: int | None = None) -> pd.DataFrame:
    rows: list[list] = []
    cursor = start_ms
    while True:
        params = {"symbol": symbol, "interval": INTERVAL[tf], "startTime": cursor, "limit": 1000}
        if end_ms:
            params["endTime"] = end_ms
        batch = _get("/api/v3/klines", params)
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < 1000:
            break
        cursor = batch[-1][0] + 1
        time.sleep(0.25)

    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close"])
    df = pd.DataFrame(rows).iloc[:, :5]
    df.columns = ["open_ms", "open", "high", "low", "close"]
    df["ts"] = pd.to_datetime(df["open_ms"], unit="ms", utc=True).dt.tz_localize(None)
    df = df.set_index("ts")[["open", "high", "low", "close"]].astype(float)
    df.index.name = "open_time"
    return df[~df.index.duplicated(keep="last")].sort_index()


def drop_unclosed(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Discard the bar currently forming -- acting on a partial bar is lookahead."""
    step = pd.Timedelta(minutes=D.TF_MINUTES[tf])
    now = pd.Timestamp.utcnow().tz_localize(None)
    return df[df.index + step <= now]


def update_csv(tf: str, symbol: str = "BTCUSDT", data_dir: Path | None = None) -> Path:
    data_dir = data_dir or D.DATA_DIR
    path = data_dir / f"{symbol}_{tf}.csv"
    if path.exists():
        existing = D.load(tf, data_dir)
        start = int((existing.index[-1]).timestamp() * 1000) + 1
    else:
        existing = None
        start = int(pd.Timestamp("2024-01-01").timestamp() * 1000)

    fresh = drop_unclosed(fetch_klines(symbol, tf, start), tf)
    if existing is not None:
        combined = pd.concat([existing[["open", "high", "low", "close"]], fresh])
    else:
        combined = fresh
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()

    out = combined.copy()
    out.index = out.index + pd.Timedelta(hours=8)          # back to Beijing labels
    out.index.name = "日期时间(北京)"
    out.columns = ["开盘", "最高", "最低", "收盘"]
    out.to_csv(path, encoding="utf-8-sig", float_format="%.2f")
    print(f"{tf}: {len(out)} bars, through {out.index[-1]} (Beijing), +{len(fresh)} new")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="refresh BTCUSDT OHLC csvs from Binance")
    ap.add_argument("--tf", nargs="*", default=["1h", "4h", "1d"], choices=["1h", "4h", "1d"])
    ap.add_argument("--symbol", default="BTCUSDT")
    args = ap.parse_args()
    for tf in args.tf:
        update_csv(tf, args.symbol)


if __name__ == "__main__":
    main()
