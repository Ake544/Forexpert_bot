"""SL/TP from the NNFX ATR, and lot sizing from a risk percentage.

Alert-only for now: `position_size` is computed and journalled so a future
executor never has to re-derive it, but no order is ever placed.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..settings import RiskParams
from ..strategy.state import Direction


@dataclass(frozen=True)
class TradeLevels:
    entry: float
    stop_loss: float
    take_profit: float
    atr: float
    sl_distance: float
    tp_distance: float
    risk_reward: float


def build_levels(
    entry: float,
    atr: float,
    direction: Direction,
    sl_multiplier: float,
    tp_multiplier: float,
) -> TradeLevels:
    """Apply the ATR distances around ``entry``.

    Raises ValueError on a non-positive ATR: an ATR of zero means the indicator
    is not ready, and a zero-distance stop is worse than no signal at all.
    """
    if atr is None or atr <= 0:
        raise ValueError("ATR must be a positive number to build levels")
    sl_distance = atr * sl_multiplier
    tp_distance = atr * tp_multiplier
    if direction == "buy":
        stop_loss, take_profit = entry - sl_distance, entry + tp_distance
    elif direction == "sell":
        stop_loss, take_profit = entry + sl_distance, entry - tp_distance
    else:
        raise ValueError("direction must be 'buy' or 'sell'")
    risk_reward = tp_distance / sl_distance if sl_distance else 0.0
    return TradeLevels(
        entry=entry,
        stop_loss=stop_loss,
        take_profit=take_profit,
        atr=atr,
        sl_distance=sl_distance,
        tp_distance=tp_distance,
        risk_reward=risk_reward,
    )


def position_size(levels: TradeLevels, risk: RiskParams) -> float:
    """Lot size risking ``risk_percent_per_trade`` of the account to the stop.

    Convention: ``sl_distance`` is in PRICE terms (e.g. 0.0030), and one lot
    moves ``assumed_contract_size`` currency units per 1.0 of price.

    Rounded DOWN to ``lot_step`` so rounding can never increase risk. If the
    result is smaller than ``min_lot`` the size is reported as 0.0 (skip the
    trade) rather than rounded UP to min_lot, which would silently breach the
    configured risk percentage.
    """
    loss_per_lot = levels.sl_distance * risk.assumed_contract_size
    if loss_per_lot <= 0:
        return 0.0
    risk_amount = risk.account_balance * risk.risk_percent_per_trade / 100.0
    raw_lots = risk_amount / loss_per_lot
    step = risk.lot_step if risk.lot_step > 0 else 0.01
    stepped = int(raw_lots / step) * step
    rounded = round(stepped, 8)
    if rounded < risk.min_lot:
        return 0.0
    return float(min(rounded, risk.max_lot))