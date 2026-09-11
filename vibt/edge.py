"""能不能做得成：把胜率、赔率、成本、波动率放进同一个算式.

The question "is a 51% win rate enough" has no answer as posed, and the reason
is worth having in code rather than in a conversation.  Three quantities decide
whether an idea can be traded, and a win rate is only the first:

  期望 e     p*b - (1-p), in units of the stop distance.  A win rate without a
             payoff ratio is not an edge -- 51% at 1:0.8 loses money, 45% at
             1:3 makes it.
  样本 n     how many trades before the edge can be distinguished from zero.
             At e = 0.02 that is about 15,000, which no daily strategy will ever
             accumulate: the edge is unconfirmable, not merely small.
  成本 c     a round trip costs 2c of notional, which is 2c/s in units of R.
             Narrow stops buy trade count and sell it straight back to the
             exchange, and the two effects fight each other with a closed-form
             optimum in between.

Two shapes of strategy need two forms of the cost test, and this repository has
run both:

  括号型     entry with a stop and a target (32/33 号脚本).  Cost enters as
             2c/s, trade count comes from how long price takes to travel s, and
             there is an optimal stop width.
  再平衡型   a weight panel rebalanced on a schedule (38/40 号脚本).  Cost enters
             as turnover x c, and the test is simply whether gross return per
             unit of turnover exceeds c -- the "打平成本" used there.

The first form's arithmetic is the substance of this module; the second is one
line, provided here so both live in the same place.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

TRADING_DAYS = 365.0        # crypto never closes


@dataclass(frozen=True)
class Edge:
    """A bracket-style idea, described by what actually determines its fate."""
    win: float              # probability of a win
    payoff: float = 1.0     # win size as a multiple of the stop distance
    cost: float = 0.00065   # per side, as a fraction of notional (6.5bp default)
    sigma_daily: float = 0.0357   # the instrument's daily return std

    # ---------------------------------------------------------------- 每笔
    @property
    def expectancy(self) -> float:
        """Mean outcome per trade, in units of the stop distance (R)."""
        return self.win * self.payoff - (1.0 - self.win)

    @property
    def sd_r(self) -> float:
        """Std of the per-trade outcome in R, which the Sharpe needs."""
        second = self.win * self.payoff ** 2 + (1.0 - self.win)
        return math.sqrt(max(second - self.expectancy ** 2, 1e-12))

    def cost_r(self, stop: float) -> float:
        """A round trip in R units: the narrower the stop, the dearer it gets."""
        return 2.0 * self.cost / stop if stop > 0 else float("inf")

    def net_expectancy(self, stop: float) -> float:
        return self.expectancy - self.cost_r(stop)

    # ---------------------------------------------------------------- 频率
    def trades_per_year(self, stop: float) -> float:
        """How often a driftless walk of this volatility travels to a barrier.

        For barriers at -s and +b*s the expected first-passage time is b*s^2/sigma^2,
        so trade count falls with the SQUARE of the stop.  That is the whole
        tension: widening the stop to afford the cost costs quadratically many
        trades, and the Sharpe only grows with the square root of them.
        """
        if stop <= 0 or self.sigma_daily <= 0:
            return 0.0
        days = self.payoff * (stop / self.sigma_daily) ** 2
        return TRADING_DAYS / days if days > 0 else 0.0

    def sharpe(self, stop: float) -> float:
        n = self.trades_per_year(stop)
        if n <= 0:
            return float("nan")
        return self.net_expectancy(stop) / self.sd_r * math.sqrt(n)

    # ---------------------------------------------------------------- 最优
    @property
    def optimal_stop(self) -> float:
        """s* = 4c/e.

        Setting d/ds of (e - 2c/s)/s to zero.  At this width the net expectancy
        is exactly e/2: **half the edge goes to the exchange, always**, whatever
        the edge happens to be.  There is no configuration that pays less.
        """
        e = self.expectancy
        return 4.0 * self.cost / e if e > 0 else float("nan")

    @property
    def max_sharpe(self) -> float:
        """Best annualised Sharpe available at any stop width.

        Closed form: e^2 * sigma * sqrt(365/b) / (8 * c * sd_R).  Note the SQUARE
        on the edge -- doubling the edge quadruples what can be achieved, which
        is why the gap between 51% and 55% is a gap between nothing and a
        business rather than a matter of degree.
        """
        e = self.expectancy
        if e <= 0:
            return float("nan")
        return (e ** 2 * self.sigma_daily * math.sqrt(TRADING_DAYS / self.payoff)
                / (8.0 * self.cost * self.sd_r))

    # ---------------------------------------------------------------- 样本
    def trades_to_confirm(self, alpha: float = 0.05, power: float = 0.80) -> float:
        """Trades needed before this edge is distinguishable from zero.

        A one-sided t-test on the mean R.  This is the number that makes small
        edges unusable in practice rather than merely unprofitable: you cannot
        run a strategy you will never be able to tell is working.
        """
        e = self.expectancy
        if e <= 0:
            return float("inf")
        z_a, z_b = _z(1 - alpha), _z(power)
        return ((z_a + z_b) * self.sd_r / e) ** 2

    def verdict(self) -> dict:
        e = self.expectancy
        s = self.optimal_stop
        return {
            "胜率": self.win, "赔率": self.payoff,
            "成本bp": self.cost * 1e4, "日波动": self.sigma_daily,
            "每笔期望R": e,
            "最优止损": s,
            "最优处每年笔数": self.trades_per_year(s) if e > 0 else float("nan"),
            "最优处净期望R": self.net_expectancy(s) if e > 0 else float("nan"),
            "可达最大Sharpe": self.max_sharpe,
            "确认所需笔数": self.trades_to_confirm(),
        }


def _z(q: float) -> float:
    """Normal quantile without dragging scipy into a module this small."""
    # Acklam's rational approximation; accurate to ~1e-9 over (0,1)
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if q < plow:
        r = math.sqrt(-2 * math.log(q))
        return (((((c[0]*r+c[1])*r+c[2])*r+c[3])*r+c[4])*r+c[5]) / \
               ((((d[0]*r+d[1])*r+d[2])*r+d[3])*r+1)
    if q > phigh:
        r = math.sqrt(-2 * math.log(1 - q))
        return -(((((c[0]*r+c[1])*r+c[2])*r+c[3])*r+c[4])*r+c[5]) / \
                ((((d[0]*r+d[1])*r+d[2])*r+d[3])*r+1)
    r = q - 0.5
    t = r * r
    return (((((a[0]*t+a[1])*t+a[2])*t+a[3])*t+a[4])*t+a[5]) * r / \
           (((((b[0]*t+b[1])*t+b[2])*t+b[3])*t+b[4])*t+1)


# ------------------------------------------------------------------ 再平衡型
def breakeven_cost(gross_annual: float, turnover_annual: float) -> float:
    """The other shape: cost per side a rebalancing strategy can absorb.

    Used in 38/40 号脚本, where the residual reversion earned +9.6% gross at 441x
    turnover -- 2.2bp of room against 6.5bp charged.  Unlike the bracket form
    there is no optimal anything to solve for; the number either clears the
    exchange's fee or it does not.
    """
    return gross_annual / turnover_annual if turnover_annual > 0 else float("nan")
