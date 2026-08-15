#!/usr/bin/env python3
"""Standalone Binance USDT-M fetcher: klines with volume, plus funding rates.

Run this ON THE VPS.  It talks to fapi.binance.com directly and depends on
nothing but the Python standard library -- no pip install, no VI-Dashboard
checkout, no pandas.

    python3 fetch_binance.py                    # everything, into ./binance_pull
    python3 fetch_binance.py --new-max 40       # cap the out-of-sample set
    python3 fetch_binance.py --skip-funding     # klines only
    python3 fetch_binance.py --dry-run          # list what it would fetch

It gathers three things in one pass:

  1. Daily klines WITH VOLUME for every eligible symbol never used in
     development, as a clean out-of-sample cross-section.
  2. Daily klines WITH VOLUME for the 26 symbols already in the repository.
     The existing files are OHLC only, so the cost assumption those backtests
     rest on (6.5bp per side, applied uniformly) has never been checked against
     how much any of these names actually trades.
  3. Funding-rate history for every symbol, which is a carry signal that exists
     independently of price momentum.

How the out-of-sample symbols are chosen matters, so the rule is fixed here in
advance and uses no return information whatsoever: every USDT-margined
perpetual that is TRADING and was onboarded on or before --onboard-before,
excluding the ones already in the repository.  Ranking on anything derived from
prices -- volume, volatility, past performance -- would quietly re-introduce the
selection effect this test exists to measure.

By default it takes ALL of them rather than a capped subset, because any cap
needs a tiebreaker and every tiebreaker is a choice.  An earlier version took
the 40 oldest, which is price-blind but not harmless: it returned the entire
2019-2021 cohort, structurally unlike the 2022-2024 names the strategy was built
on, so a failure could not be read as the strategy failing rather than the
sample being different in kind.  Taking everything eligible removes the question.

Only the daily panel is fetched.  Every downstream test -- the out-of-sample
cross-section, the cost check, funding carry -- reads daily bars, and
scripts/20_frequency.py already established that daily beats 4h beats 1h even at
zero cost.  Fetching 4h as well would multiply the download by about six for
data nothing would read.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

BASE = "https://fapi.binance.com"
UA = "Mozilla/5.0 (compatible; kline-fetch/1.0)"

# The 26 already in data/.  Listed explicitly so the script needs no repo access.
EXISTING = [
    "1000BONKUSDT", "1000FLOKIUSDT", "1000PEPEUSDT", "1000SHIBUSDT", "APTUSDT",
    "ARBUSDT", "BNBUSDT", "BTCUSDT", "DOGEUSDT", "DYDXUSDT", "ENAUSDT", "ETHUSDT",
    "FETUSDT", "HYPEUSDT", "LDOUSDT", "OPUSDT", "PENDLEUSDT", "RENDERUSDT",
    "SEIUSDT", "SOLUSDT", "SUIUSDT", "TAOUSDT", "TIAUSDT", "WIFUSDT", "WLDUSDT",
    "XRPUSDT",
]

# Named a priori in UNIVERSE.md but never exported.  Tagged in the manifest so the
# out-of-sample result can be split by whether I had picked the name in advance.
NAMED_NEVER_PULLED = [
    "ARKMUSDT", "AKTUSDT", "GRTUSDT", "STRKUSDT", "JUPUSDT", "GMXUSDT", "AAVEUSDT",
    "ADAUSDT", "LTCUSDT", "LINKUSDT", "AVAXUSDT", "DOTUSDT", "ATOMUSDT", "NEARUSDT",
]

KLINE_COLS = ["日期时间(北京)", "开盘", "最高", "最低", "收盘",
              "volume", "quote_volume", "trades", "taker_buy_base", "taker_buy_quote"]


class Limiter:
    """Keep well under the 2400 weight/minute budget without guessing at it.

    Binance reports the weight actually consumed in a response header, so the
    limiter reads that rather than assuming a cost per endpoint, and sleeps when
    the running total approaches the cap.  A 429 means we already misjudged it,
    so that is handled separately by honouring Retry-After.
    """

    def __init__(self, budget: int = 1800):
        self.budget = budget
        self.used = 0
        self.window_start = time.time()

    def note(self, weight: int) -> None:
        now = time.time()
        if now - self.window_start >= 60:
            self.window_start, self.used = now, 0
        self.used = max(self.used, weight)
        if self.used >= self.budget:
            nap = 60 - (now - self.window_start) + 1
            if nap > 0:
                print(f"    [rate limit] used {self.used}/min, sleeping {nap:.0f}s", flush=True)
                time.sleep(nap)
            self.window_start, self.used = time.time(), 0


LIMIT = Limiter()


def get(path: str, params: dict, tries: int = 6):
    url = f"{BASE}{path}?{urllib.parse.urlencode(params)}"
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30) as r:
                weight = r.headers.get("X-MBX-USED-WEIGHT-1M")
                if weight:
                    LIMIT.note(int(weight))
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code in (429, 418):
                wait = int(e.headers.get("Retry-After", 0) or 2 ** (attempt + 3))
                print(f"    [{e.code}] backing off {wait}s", flush=True)
                time.sleep(wait)
                continue
            if 500 <= e.code < 600:
                time.sleep(2 ** attempt)
                continue
            body = e.read().decode()[:200]
            raise RuntimeError(f"HTTP {e.code} on {path}: {body}") from e
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            if attempt == tries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError(f"gave up on {path}")


def ms(dt: str) -> int:
    return int(datetime.strptime(dt, "%Y-%m-%d")
               .replace(tzinfo=timezone.utc).timestamp() * 1000)


BEIJING = timezone(timedelta(hours=8))


def beijing(ts_ms: int) -> str:
    """Bar open time in UTC+8, matching every file already in data/.

    The loader subtracts 8 hours on the way back in, so emitting anything else
    here would shift every bar by a fixed offset without tripping a single
    integrity check.
    """
    return datetime.fromtimestamp(ts_ms / 1000, BEIJING).strftime("%Y-%m-%d %H:%M:%S")


def universe(onboard_before: str) -> tuple[list[dict], set[str]]:
    """(eligible symbols, every USDT-M perpetual currently listed).

    Both are needed.  Checking the repository's symbols against the *eligible*
    slice reports every recently-listed name as missing, which reads as
    "delisted" when it only means "listed after the cutoff".
    """
    info = get("/fapi/v1/exchangeInfo", {})
    cutoff = ms(onboard_before)
    listed = {s["symbol"] for s in info["symbols"]
              if s.get("status") == "TRADING" and s.get("contractType") == "PERPETUAL"}
    out = []
    for s in info["symbols"]:
        if (s.get("status") == "TRADING"
                and s.get("quoteAsset") == "USDT"
                and s.get("marginAsset") == "USDT"
                and s.get("contractType") == "PERPETUAL"
                and int(s.get("onboardDate", 0)) <= cutoff):
            out.append({"symbol": s["symbol"], "onboard": int(s["onboardDate"])})
    out.sort(key=lambda x: (x["onboard"], x["symbol"]))
    return out, listed


def klines(symbol: str, interval: str, start: int) -> list[list]:
    rows, cursor, seen = [], start, set()
    while True:
        batch = get("/fapi/v1/klines", {"symbol": symbol, "interval": interval,
                                        "startTime": cursor, "limit": 1500})
        if not batch:
            break
        fresh = [b for b in batch if b[0] not in seen]
        if not fresh:
            break
        for b in fresh:
            seen.add(b[0])
        rows.extend(fresh)
        if len(batch) < 1500:
            break
        cursor = batch[-1][0] + 1
    rows.sort(key=lambda b: b[0])
    # The final bar is still forming.  Keeping it would write a partial candle to
    # disk, and a later refresh would then look like the exchange restated it.
    if rows and rows[-1][6] > time.time() * 1000:
        rows.pop()
    return rows


def funding(symbol: str, start: int) -> list[dict]:
    rows, cursor, seen = [], start, set()
    while True:
        batch = get("/fapi/v1/fundingRate", {"symbol": symbol, "startTime": cursor,
                                             "limit": 1000})
        if not batch:
            break
        fresh = [b for b in batch if b["fundingTime"] not in seen]
        if not fresh:
            break
        for b in fresh:
            seen.add(b["fundingTime"])
        rows.extend(fresh)
        if len(batch) < 1000:
            break
        cursor = batch[-1]["fundingTime"] + 1
    rows.sort(key=lambda b: b["fundingTime"])
    return rows


def write_klines(path: Path, rows: list[list]) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(KLINE_COLS)
        for b in rows:
            # 0 openTime 1 o 2 h 3 l 4 c 5 vol 6 closeTime 7 quoteVol 8 trades
            # 9 takerBuyBase 10 takerBuyQuote
            w.writerow([beijing(b[0]), b[1], b[2], b[3], b[4],
                        b[5], b[7], b[8], b[9], b[10]])


def write_funding(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["日期时间(北京)", "funding_rate", "mark_price"])
        for b in rows:
            w.writerow([beijing(b["fundingTime"]), b["fundingRate"], b.get("markPrice", "")])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="./binance_pull", type=Path)
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--onboard-before", default="2024-07-01",
                    help="out-of-sample names must have listed by this date")
    ap.add_argument("--new-max", type=int, default=0,
                    help="cap the never-used set (0 = take all eligible, which is "
                         "the only genuinely selection-free option)")
    ap.add_argument("--intervals", default="1d",
                    help="the daily panel is what every downstream test uses; "
                         "4h costs ~6x the download and nothing reads it")
    ap.add_argument("--skip-funding", action="store_true")
    ap.add_argument("--skip-existing", action="store_true",
                    help="do not re-fetch volume for the 26 already in the repo")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    start = ms(args.start)
    intervals = [s.strip() for s in args.intervals.split(",") if s.strip()]

    print("reading exchangeInfo ...", flush=True)
    elig, listed = universe(args.onboard_before)
    have = set(EXISTING)
    fresh = [u for u in elig if u["symbol"] not in have]
    if args.new_max:
        fresh = fresh[:args.new_max]
    new_syms = [u["symbol"] for u in fresh]

    gone = [s for s in EXISTING if s not in listed]
    too_new = [s for s in EXISTING
               if s in listed and s not in {u["symbol"] for u in elig}]
    targets = new_syms + ([] if args.skip_existing else EXISTING)

    print(f"\n  eligible USDT-M perpetuals onboarded <= {args.onboard_before}: {len(elig)}")
    print(f"  already in the repo                    : {len(EXISTING)}")
    print(f"  never used, taking                     : {len(new_syms)}")
    if gone:
        print(f"  WARNING: no longer listed (renamed or delisted): {gone}")
    if too_new:
        print(f"  (listed after {args.onboard_before}, so not eligible as OOS "
              f"but still fetched: {too_new})")
    named = [s for s in new_syms if s in NAMED_NEVER_PULLED]
    print(f"    of which named a priori in UNIVERSE.md: {len(named)}  {named}")
    print(f"    never mentioned anywhere              : {len(new_syms) - len(named)}")
    print(f"\n  klines to fetch : {len(targets)} symbols x {len(intervals)} intervals")
    print(f"  funding to fetch: {0 if args.skip_funding else len(targets)} symbols")
    print(f"\n  out-of-sample symbols:\n    {' '.join(new_syms)}")

    if args.dry_run:
        print("\ndry run, nothing written")
        return

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = []
    t0 = time.time()

    for i, sym in enumerate(targets, 1):
        tag = "new" if sym in set(new_syms) else "existing"
        print(f"\n[{i}/{len(targets)}] {sym}  ({tag})", flush=True)
        row = {"symbol": sym, "group": tag,
               "named_a_priori": sym in NAMED_NEVER_PULLED}
        for iv in intervals:
            path = args.out / f"{sym}_{iv}_{args.start}_to_now.csv"
            if path.exists() and path.stat().st_size > 200:
                print(f"    {iv:<3} already on disk, skipping", flush=True)
                row[f"{iv}_bars"] = "skipped"
                continue
            try:
                rows = klines(sym, iv, start)
            except Exception as exc:                     # noqa: BLE001
                print(f"    {iv:<3} FAILED: {exc}", flush=True)
                row[f"{iv}_bars"] = "FAILED"
                continue
            if not rows:
                print(f"    {iv:<3} no data", flush=True)
                row[f"{iv}_bars"] = 0
                continue
            write_klines(path, rows)
            row[f"{iv}_bars"] = len(rows)
            row[f"{iv}_first"] = beijing(rows[0][0])
            print(f"    {iv:<3} {len(rows):>6} bars  {beijing(rows[0][0])} -> "
                  f"{beijing(rows[-1][0])}", flush=True)

        if not args.skip_funding:
            fpath = args.out / f"{sym}_funding.csv"
            if fpath.exists() and fpath.stat().st_size > 200:
                print("    fnd already on disk, skipping", flush=True)
                row["funding_rows"] = "skipped"
            else:
                try:
                    fr = funding(sym, start)
                    write_funding(fpath, fr)
                    row["funding_rows"] = len(fr)
                    if fr:
                        rates = [float(x["fundingRate"]) for x in fr]
                        row["funding_mean_bp"] = round(sum(rates) / len(rates) * 1e4, 4)
                        print(f"    fnd {len(fr):>6} rows  mean "
                              f"{row['funding_mean_bp']:+.4f}bp/period", flush=True)
                except Exception as exc:                 # noqa: BLE001
                    print(f"    fnd FAILED: {exc}", flush=True)
                    row["funding_rows"] = "FAILED"
        manifest.append(row)

    mpath = args.out / "MANIFEST.csv"
    keys = sorted({k for r in manifest for k in r})
    with open(mpath, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(manifest)

    total = sum(f.stat().st_size for f in args.out.glob("*.csv"))
    print(f"\n{'='*70}")
    print(f"done in {(time.time()-t0)/60:.1f} min")
    print(f"  {len(list(args.out.glob('*.csv')))} files, {total/1e6:.1f} MB in {args.out}")
    print(f"  manifest: {mpath}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    print(f"""
next, pack it (whichever exists on this box):

  cd {args.out} && zip -qr ~/binance_pull_{stamp}.zip . && ls -lh ~/binance_pull_{stamp}.zip
  # or, if zip is not installed:
  tar -czf ~/binance_pull_{stamp}.tar.gz -C {args.out} . && ls -lh ~/binance_pull_{stamp}.tar.gz

then from your local machine (PowerShell):

  scp vpn-sg:~/binance_pull_{stamp}.* "$env:USERPROFILE\\Desktop\\"
""")


if __name__ == "__main__":
    main()
