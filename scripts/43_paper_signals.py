"""每天跑一次：打印今天的目标持仓和要下的单，并记进日志.

    python scripts/43_paper_signals.py --equity 10000
    python scripts/43_paper_signals.py --equity 10000 --write     # 追加到日志
    python scripts/43_paper_signals.py --asof 2026-09-01          # 重放某一天
    python scripts/43_paper_signals.py --reconcile                # 看已实现滑点

It prints orders; it does not send them.  The reason to run it at all is the
last command: the strategy's breakeven cost out of sample was 6.38bp against
6.5bp charged, so whether it works is decided by the fills, and the only way to
learn those is to write down what was intended and compare later.

Run it just after 00:00 UTC.  The signal is formed on the daily bar that has
just closed and is meant to trade at that same instant; this repository measured
that a day's delay costs 0.3 to 0.6 of Sharpe, so the log records the intended
execution time and you record the real one.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vibt import paper as PP  # noqa: E402

pd.set_option("display.width", 200)


def show_book(book: PP.Book, orders: pd.DataFrame, equity: float) -> None:
    print("=" * 108)
    print(f"目标持仓   信号日 {book.asof.date()}（该日线已收盘）  "
          f"计划成交 {book.execute_at.date()} 00:00 UTC")
    print("=" * 108 + "\n")
    o = orders.copy()
    o.insert(0, "signal", book.signal)
    for c in ("target_w", "current_w", "delta_w"):
        o[c] = o[c].map(lambda v: f"{v:+.3f}")
    for c in ("target_notional", "trade_notional"):
        o[c] = o[c].map(lambda v: f"{v:+,.0f}")
    o["ref_price"] = o["ref_price"].map(lambda v: f"{v:,.4f}" if pd.notna(v) else "")
    o["trade_qty"] = o["trade_qty"].map(lambda v: f"{v:+.6f}" if pd.notna(v) else "")
    o["signal"] = o["signal"].map(lambda v: f"{v:+.2f}" if pd.notna(v) else "")
    print(o.to_string())

    gross = float(book.weights.abs().sum())
    net = float(book.weights.sum())
    turn = float(orders["delta_w"].abs().sum())
    print(f"""
  总敞口 {gross:.2f}   净敞口 {net:+.3f}   本次换手 {turn:.2f}
  账户权益 {equity:,.0f}   本次成交名义 {orders['trade_notional'].abs().sum():,.0f}
  按 {book.params.assumed_cost_bps:.1f}bp 估算的本次成本 ≈ """
          f"{orders['trade_notional'].abs().sum() * book.params.assumed_cost_bps / 1e4:,.2f}")
    if not book.is_rebalance:
        print(f"""
  **今天不是再平衡日**（每 {book.params.rebalance_days} 天一次）。
  上表是「如果今天调仓会怎样」，实际应当**不动**。加 --force 可覆盖。""")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--equity", type=float, default=10000.0)
    ap.add_argument("--asof", default=None, help="replay a past date, e.g. 2026-09-01")
    ap.add_argument("--write", action="store_true", help="append to paper/signals.csv")
    ap.add_argument("--force", action="store_true",
                    help="treat today as a rebalance day even if it is not")
    ap.add_argument("--reconcile", action="store_true",
                    help="report realised slippage from fills written into the log")
    args = ap.parse_args()

    p = PP.Params()

    if args.reconcile:
        r = PP.reconcile()
        if not len(r):
            print("""
  日志里还没有成交记录。对账的做法：打开 paper/signals.csv，
  给已经成交的行填上 fill_price / fill_qty / filled_at / fee_paid，再跑一次。
""")
            return
        print("=" * 96)
        print("已实现滑点")
        print("=" * 96 + "\n")
        print(r.to_string(index=False))
        s = r["slippage_bps"].dropna()
        print(f"""
  中位 {s.median():+.2f}bp   均值 {s.mean():+.2f}bp
  75% 分位 {s.quantile(0.75):+.2f}bp   最差 {s.max():+.2f}bp   笔数 {len(s)}

  回测假设的是 {p.assumed_cost_bps:.1f}bp/边（含手续费与滑点）。
  上面这个中位数只是**滑点**，还要再加手续费才能和它比。""")
        return

    log = PP.load_log()
    prev_w, prev_reb = PP.last_state(log, p.coins)
    book = PP.generate(args.asof, p, prev_reb)
    if args.force:
        book.is_rebalance = True

    orders = book.orders(prev_w, args.equity)
    show_book(book, orders, args.equity)

    # A stale data directory produces a perfectly well-formed book for a day that
    # has already gone.  Nothing downstream can tell the difference, so it has to
    # be said here.
    if args.asof is None:
        lag_h = (pd.Timestamp.utcnow().tz_localize(None) - book.execute_at) \
            .total_seconds() / 3600
        if lag_h > 6:
            print(f"""
  ⚠️  **数据过期 {lag_h/24:.1f} 天**：计划成交时间 {book.execute_at.date()} 已经过去。
      这张表是那一天的持仓，不是今天的。先刷新数据再跑：
        在能访问币安的机器上重抓 -> python -m vibt.ingest <目录> --merge --gzip""")

    if not args.write:
        print("\n  （未写入日志；加 --write 记录这次信号）")
        return
    if not book.is_rebalance:
        print("\n  非再平衡日，不写入。要强制记录请加 --force。")
        return
    path = PP.append(book, orders, args.equity)
    print(f"\n  已追加到 {path}")
    print("""
  成交之后，把这几列填回同一行：fill_price、fill_qty、filled_at、fee_paid。
  攒够几十笔再跑 --reconcile —— 那时候才知道 6.5bp 这个假设是松还是紧。""")


if __name__ == "__main__":
    main()
