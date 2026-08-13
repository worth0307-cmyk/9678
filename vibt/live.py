"""Daily runner: compute today's target exposure and emit an order plan.

Deliberately stops at the order plan.  Sending orders needs your keys, your
exchange, your risk limits -- wire `Broker.submit` to whatever you use.  The
default broker writes a paper fill to state.json so you can run the whole loop
for weeks and compare paper against the backtest before risking anything.

Run it once per day, a minute or two after 00:00 UTC (08:00 Beijing), which is
when the daily bar the system reads has closed:

    python -m vibt.live --equity 10000            # dry run, prints the plan
    python -m vibt.live --equity 10000 --paper    # dry run + record the fill
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import data as D, system as SYS

STATE_PATH = Path(__file__).resolve().parent.parent / "state.json"
MIN_TRADE_FRACTION = 0.05      # ignore rebalances smaller than 5% of equity


@dataclass
class OrderPlan:
    as_of: str
    bar_close_utc: str
    price: float
    current_exposure: float
    target_exposure: float
    delta_exposure: float
    side: str
    notional: float
    qty: float
    reason: str
    act: bool


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {"exposure": 0.0, "history": []}


def save_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2))


def build_plan(df: pd.DataFrame, equity: float, current: float,
               params: SYS.Params | None = None) -> tuple[OrderPlan, pd.DataFrame]:
    params = params or SYS.Params()
    target = float(SYS.target_exposure(df, params).iloc[-1])
    last_close_time = df["close_time"].iloc[-1]
    price = float(df["close"].iloc[-1])

    delta = target - current
    act = abs(delta) >= MIN_TRADE_FRACTION
    notional = abs(delta) * equity
    state = SYS.explain(df, params, 1)
    votes = int(state["bull_votes"].iloc[0])
    n = int(state["n_votes"].iloc[0])
    scalar = float(state["vol_scalar"].iloc[0])
    vol = float(state["realised_vol"].iloc[0])

    plan = OrderPlan(
        as_of=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        bar_close_utc=str(last_close_time),
        price=price,
        current_exposure=round(current, 4),
        target_exposure=round(target, 4),
        delta_exposure=round(delta, 4),
        side="BUY" if delta > 0 else ("SELL" if delta < 0 else "NONE"),
        notional=round(notional, 2),
        qty=round(notional / price, 6),
        reason=(f"{votes}/{n} bullish votes, realised vol {vol:.0%}, "
                f"vol scalar {scalar:.2f} -> target {target:+.0%}"),
        act=act,
    )
    return plan, state


def main() -> None:
    ap = argparse.ArgumentParser(description="daily BTCUSDT exposure runner")
    ap.add_argument("--equity", type=float, required=True, help="account equity in USDT")
    ap.add_argument("--paper", action="store_true", help="record the fill in state.json")
    ap.add_argument("--refresh", action="store_true", help="pull fresh candles first")
    ap.add_argument("--target-vol", type=float, default=SYS.Params().target_vol)
    ap.add_argument("--short-size", type=float, default=SYS.Params().short_size)
    args = ap.parse_args()

    if args.refresh:
        from . import fetch
        fetch.update_csv("1d")

    df = D.load("1d")
    params = SYS.Params(target_vol=args.target_vol, short_size=args.short_size)
    state = load_state()
    plan, snapshot = build_plan(df, args.equity, float(state.get("exposure", 0.0)), params)

    print(f"\nlast closed daily bar : {plan.bar_close_utc} UTC   close {plan.price:,.2f}")
    print(f"reason                : {plan.reason}")
    print(f"current exposure      : {plan.current_exposure:+.0%}")
    print(f"target  exposure      : {plan.target_exposure:+.0%}")
    if not plan.act:
        print(f"\n  NO ACTION -- change of {plan.delta_exposure:+.0%} is below the "
              f"{MIN_TRADE_FRACTION:.0%} rebalance threshold\n")
    else:
        print(f"\n  {plan.side} {plan.qty:.6f} BTC  (~{plan.notional:,.2f} USDT, "
              f"{plan.delta_exposure:+.0%} of equity)\n")

    if args.paper and plan.act:
        state["exposure"] = plan.target_exposure
        state.setdefault("history", []).append(asdict(plan))
        save_state(state)
        print(f"  paper fill recorded in {STATE_PATH}\n")


if __name__ == "__main__":
    main()
