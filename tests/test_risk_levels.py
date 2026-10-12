"""ATR level maths and risk-based position sizing."""
from __future__ import annotations

import pytest

from forexpert.risk.levels import build_levels, position_size
from forexpert.settings import RiskParams


def test_buy_levels_use_the_pine_direction() -> None:
    levels = build_levels(entry=100.0, atr=2.0, direction="buy",
                          sl_multiplier=1.5, tp_multiplier=2.25)
    assert levels.stop_loss == pytest.approx(97.0)
    assert levels.take_profit == pytest.approx(104.5)
    assert levels.sl_distance == pytest.approx(3.0)
    assert levels.tp_distance == pytest.approx(4.5)
    assert levels.risk_reward == pytest.approx(1.5)


def test_sell_levels_are_the_exact_mirror() -> None:
    levels = build_levels(entry=100.0, atr=2.0, direction="sell",
                          sl_multiplier=1.5, tp_multiplier=2.25)
    assert levels.stop_loss == pytest.approx(103.0)
    assert levels.take_profit == pytest.approx(95.5)
    assert levels.risk_reward == pytest.approx(1.5)


@pytest.mark.parametrize("bad_atr", [0.0, -1.0, None])
def test_non_positive_atr_raises_instead_of_producing_a_zero_stop(bad_atr) -> None:
    with pytest.raises(ValueError):
        build_levels(entry=100.0, atr=bad_atr, direction="buy",
                     sl_multiplier=1.5, tp_multiplier=2.25)


def test_unknown_direction_is_rejected() -> None:
    with pytest.raises(ValueError):
        build_levels(entry=100.0, atr=1.0, direction="long",
                     sl_multiplier=1.5, tp_multiplier=2.25)


def test_position_size_risks_the_configured_percentage() -> None:
    levels = build_levels(entry=1.1000, atr=0.0020, direction="buy",
                          sl_multiplier=1.5, tp_multiplier=2.25)
    risk = RiskParams(risk_percent_per_trade=1.0, account_balance=10_000.0,
                      lot_step=0.01, min_lot=0.01, max_lot=100.0,
                      assumed_contract_size=100_000.0)
    lots = position_size(levels, risk)
    # 1% of 10k is $100; a $0.0030 stop on 100k units costs $300 per lot.
    assert lots == pytest.approx(0.33)
    assert lots * levels.sl_distance * 100_000.0 == pytest.approx(99.0, abs=1.0)


def test_position_size_never_exceeds_max_lot() -> None:
    levels = build_levels(entry=1.1000, atr=0.0001, direction="buy",
                          sl_multiplier=1.5, tp_multiplier=2.25)
    risk = RiskParams(risk_percent_per_trade=5.0, account_balance=100_000.0,
                      max_lot=2.0, assumed_contract_size=100_000.0)
    assert position_size(levels, risk) == pytest.approx(2.0)


def test_tiny_risk_budget_below_min_lot_reports_zero_not_min_lot() -> None:
    """Rounding UP to min_lot would breach the risk percentage, so we skip."""
    levels = build_levels(entry=1.1000, atr=0.0100, direction="buy",
                          sl_multiplier=1.5, tp_multiplier=2.25)
    risk = RiskParams(risk_percent_per_trade=0.01, account_balance=1_000.0,
                      min_lot=0.01, lot_step=0.01, assumed_contract_size=100_000.0)
    assert position_size(levels, risk) == 0.0


def test_zero_stop_distance_is_never_divided_by() -> None:
    levels = build_levels(entry=100.0, atr=2.0, direction="buy",
                          sl_multiplier=0.0, tp_multiplier=2.25)
    assert levels.risk_reward == 0.0
    assert position_size(levels, RiskParams()) == 0.0