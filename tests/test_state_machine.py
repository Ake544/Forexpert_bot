"""State machine tests: the three-candle sequence, expiry, filters, dedupe.

Candles are built around a fixed reference level of 100.0 (see conftest.make_bar)
so every assertion about "above"/"below" is unambiguous.
"""
from __future__ import annotations

import pytest

from conftest import LEVEL, make_bar
from forexpert.settings import AtrParams, RiskParams, StrategyParams
from forexpert.strategy.machine import BarOrderError, StrategyMachine
from forexpert.strategy.state import Phase, SetupState


def machine(direction: str = "buy", **overrides) -> StrategyMachine:
    params = StrategyParams(**overrides)
    return StrategyMachine(
        symbol="EURUSD",
        timeframe="4h",
        direction=direction,
        params=params,
        risk=RiskParams(risk_percent_per_trade=1.0, account_balance=10_000.0),
        atr_params=AtrParams(sl_multiplier=1.5, tp_multiplier=2.25),
    )


def touch_bar(index: int, **overrides):
    """A bar that dips to the level and closes back on the setup side."""
    defaults = {"open": 101.0, "high": 101.2, "low": 99.8, "close": 100.5}
    defaults.update(overrides)
    return make_bar(index, **defaults)


# --------------------------------------------------------------------- buy side
def test_valid_buy_sequence_signals_on_the_third_candle() -> None:
    engine = machine("buy")
    assert engine.on_bar(make_bar(0, open=100.5, high=101.5, low=100.2, close=101.0)) is None
    assert engine.state.phase is Phase.CANDLE1_OK
    assert engine.on_bar(make_bar(1, open=101.0, high=102.5, low=100.9, close=102.0)) is None
    assert engine.state.phase is Phase.CANDLE2_OK

    signal = engine.on_bar(touch_bar(2))

    assert signal is not None
    assert signal.direction == "buy"
    assert signal.entry == pytest.approx(100.5)
    assert signal.stop_loss == pytest.approx(99.0)
    assert signal.take_profit == pytest.approx(102.75)
    assert signal.risk_reward == pytest.approx(1.5)
    assert signal.confirmations == ("VSLRT",)
    assert signal.bars_from_candle1 == 2
    assert signal.bars_from_candle2 == 1
    assert engine.state.phase is Phase.COOLDOWN


def test_entry_may_come_later_than_the_third_candle() -> None:
    engine = machine("buy")
    engine.on_bar(make_bar(0, close=101.0))
    engine.on_bar(make_bar(1, open=101.0, close=102.0))
    for index in range(2, 5):
        assert engine.on_bar(make_bar(index, open=101.0, high=101.5, low=100.6, close=101.2)) is None
    signal = engine.on_bar(touch_bar(5))
    assert signal is not None
    assert signal.bars_from_candle2 == 4


# -------------------------------------------------------------------- sell side
def test_valid_sell_sequence_is_the_exact_mirror() -> None:
    engine = machine("sell")
    assert engine.on_bar(make_bar(0, open=99.5, high=99.8, low=98.5, close=99.0)) is None
    assert engine.state.phase is Phase.CANDLE1_OK
    assert engine.on_bar(make_bar(1, open=99.0, high=99.1, low=97.5, close=98.0)) is None
    assert engine.state.phase is Phase.CANDLE2_OK

    signal = engine.on_bar(make_bar(2, open=99.5, high=100.2, low=98.5, close=99.5))

    assert signal is not None
    assert signal.direction == "sell"
    assert signal.entry == pytest.approx(99.5)
    assert signal.stop_loss == pytest.approx(101.0)
    assert signal.take_profit == pytest.approx(97.25)


# ------------------------------------------------------------- invalid sequences
def test_candle_two_that_closes_back_below_the_level_fails_the_setup() -> None:
    engine = machine("buy")
    engine.on_bar(make_bar(0, close=101.0))
    assert engine.on_bar(make_bar(1, open=100.5, high=101.0, low=99.0, close=99.5)) is None
    assert engine.state.phase is Phase.IDLE
    assert engine.state.reject_reason == "candle2_failed"
    assert engine.on_bar(touch_bar(2)) is None


def test_the_two_candles_must_be_consecutive() -> None:
    engine = machine("buy")
    engine.on_bar(make_bar(0, close=101.0))
    engine.on_bar(make_bar(1, open=101.0, close=99.0))       # fails candle 2
    assert engine.state.phase is Phase.IDLE
    engine.on_bar(make_bar(2, close=101.0))                  # new candle 1
    assert engine.state.phase is Phase.CANDLE1_OK
    engine.on_bar(make_bar(3, open=101.0, close=102.0))      # candle 2
    assert engine.state.phase is Phase.CANDLE2_OK


def test_reseed_option_lets_the_failed_candle_become_candle_one() -> None:
    engine = machine("buy", reseed_candle1_on_failure=True)
    engine.on_bar(make_bar(0, close=101.0))
    # Opens below the level (so candle 2 fails) but closes beyond it (so it can
    # serve as a fresh candle 1).
    engine.on_bar(make_bar(1, open=99.5, high=101.2, low=99.0, close=101.0))
    assert engine.state.phase is Phase.CANDLE1_OK
    assert engine.state.reject_reason == "candle2_failed_reseeded"


def test_candle_one_needs_a_close_beyond_the_level_not_just_a_wick() -> None:
    engine = machine("buy")
    engine.on_bar(make_bar(0, open=101.0, high=101.5, low=99.0, close=99.5))
    assert engine.state.phase is Phase.IDLE


# ------------------------------------------------------------------------ expiry
def test_setup_expires_after_the_configured_number_of_bars() -> None:
    engine = machine("buy", expiry_bars=2)
    engine.on_bar(make_bar(0, close=101.0))
    engine.on_bar(make_bar(1, open=101.0, close=102.0))
    assert engine.on_bar(make_bar(2, low=100.6, close=101.2)) is None
    assert engine.state.phase is Phase.WAITING_TOUCH
    assert engine.on_bar(make_bar(3, low=100.6, close=101.2)) is None
    assert engine.state.phase is Phase.WAITING_TOUCH
    assert engine.on_bar(make_bar(4, low=100.6, close=101.2)) is None
    assert engine.state.phase is Phase.IDLE
    assert engine.state.reject_reason == "expired"


def test_closing_on_the_wrong_side_kills_the_setup_immediately() -> None:
    engine = machine("buy", expire_on_wrong_side_close=True)
    engine.on_bar(make_bar(0, close=101.0))
    engine.on_bar(make_bar(1, open=101.0, close=102.0))
    assert engine.on_bar(make_bar(2, open=100.5, high=100.6, low=99.0, close=99.5)) is None
    assert engine.state.phase is Phase.IDLE
    assert engine.state.reject_reason == "closed_wrong_side"


def test_expiry_is_not_enforced_when_disabled() -> None:
    engine = machine("buy", expiry_bars=2, expire_on_wrong_side_close=False)
    engine.on_bar(make_bar(0, close=101.0))
    engine.on_bar(make_bar(1, open=101.0, close=102.0))
    for index in range(2, 8):
        engine.on_bar(make_bar(index, low=100.6, close=101.2))
    assert engine.state.phase is Phase.WAITING_TOUCH
    assert engine.on_bar(touch_bar(8)) is not None


# ----------------------------------------------------------------------- filters
@pytest.mark.parametrize("broken", ["vwap_bull", "trend_bull"])
def test_trend_filter_must_hold_on_the_entry_bar_by_default(broken: str) -> None:
    engine = machine("buy", filters_on_entry_bar=True)
    engine.on_bar(make_bar(0, close=101.0))
    engine.on_bar(make_bar(1, open=101.0, close=102.0))
    assert engine.on_bar(touch_bar(2, **{broken: False})) is None
    assert engine.state.reject_reason == "trend_filter_not_met"


def test_unknown_indicator_state_never_counts_as_a_pass() -> None:
    engine = machine("buy")
    engine.on_bar(make_bar(0, close=101.0))
    engine.on_bar(make_bar(1, open=101.0, close=102.0))
    assert engine.on_bar(touch_bar(2, vwap_bull=None, trend_bull=None)) is None
    assert engine.state.reject_reason == "trend_filter_not_met"


def test_sticky_filters_allow_a_setup_that_warmed_up_before_the_touch() -> None:
    engine = machine("buy", filters_on_entry_bar=False)
    engine.on_bar(make_bar(0, close=101.0, trend_bull=True))
    engine.on_bar(make_bar(1, open=101.0, close=102.0, trend_bull=False))
    signal = engine.on_bar(touch_bar(2, trend_bull=False, vwap_bull=False))
    assert signal is not None
    assert signal.bars_from_candle2 == 1


# ----------------------------------------------------------------- confirmations
def test_no_confirmation_means_no_signal() -> None:
    engine = machine("buy")
    engine.on_bar(make_bar(0, close=101.0))
    engine.on_bar(make_bar(1, open=101.0, close=102.0))
    assert engine.on_bar(touch_bar(2, vslrt_bull=False, kro_bottom_touch=False)) is None
    assert engine.state.reject_reason == "no_confirmation"


def test_kro_can_confirm_on_its_own_when_enabled() -> None:
    engine = machine("buy")
    engine.on_bar(make_bar(0, close=101.0))
    engine.on_bar(make_bar(1, open=101.0, close=102.0))
    signal = engine.on_bar(touch_bar(2, vslrt_bull=False, kro_bottom_touch=True))
    assert signal is not None
    assert signal.confirmations == ("KRO",)


def test_kro_top_touch_confirms_a_sell_not_a_buy() -> None:
    engine = machine("sell")
    engine.on_bar(make_bar(0, open=99.5, high=99.8, low=98.5, close=99.0))
    engine.on_bar(make_bar(1, open=99.0, high=99.1, low=97.5, close=98.0))
    signal = engine.on_bar(
        make_bar(2, open=99.5, high=100.2, low=98.5, close=99.5,
                 vslrt_bull=False, kro_top_touch=True)
    )
    assert signal is not None
    assert signal.confirmations == ("KRO",)


# --------------------------------------------------------------------- integrity
def test_level_and_atr_unavailable_block_the_signal() -> None:
    engine = machine("buy")
    engine.on_bar(make_bar(0, close=101.0))
    engine.on_bar(make_bar(1, open=101.0, close=102.0))
    assert engine.on_bar(touch_bar(2, level=None)) is None
    assert engine.state.reject_reason == "level_unavailable"
    assert engine.on_bar(touch_bar(3, atr=None)) is None
    assert engine.state.reject_reason == "atr_unavailable"


def test_repeating_a_bar_is_rejected_rather_than_signalling_twice() -> None:
    engine = machine("buy")
    engine.on_bar(make_bar(0, close=101.0))
    engine.on_bar(make_bar(1, open=101.0, close=102.0))
    engine.on_bar(touch_bar(2))
    with pytest.raises(BarOrderError):
        engine.on_bar(touch_bar(2))


def test_cooldown_blocks_an_immediate_second_signal() -> None:
    engine = machine("buy", cooldown_bars=3)
    engine.on_bar(make_bar(0, close=101.0))
    engine.on_bar(make_bar(1, open=101.0, close=102.0))
    assert engine.on_bar(touch_bar(2)) is not None
    assert engine.state.bars_until_ready == 3
    for index in (3, 4):
        assert engine.on_bar(touch_bar(index)) is None
        assert engine.state.phase is Phase.COOLDOWN
    assert engine.on_bar(touch_bar(5)) is None
    assert engine.state.phase is Phase.IDLE
    assert engine.state.reject_reason == "cooldown_elapsed"


def test_signal_ids_are_deterministic_across_processes() -> None:
    first = machine("buy")
    second = machine("buy")
    bars = [make_bar(0, close=101.0), make_bar(1, open=101.0, close=102.0), touch_bar(2)]
    ids = []
    for engine in (first, second):
        signal = None
        for bar in bars:
            signal = engine.on_bar(bar) or signal
        assert signal is not None
        ids.append(signal.signal_id)
    assert ids[0] == ids[1]


def test_entry_price_can_be_the_level_instead_of_the_close() -> None:
    engine = machine("buy", entry_price="level")
    engine.on_bar(make_bar(0, close=101.0))
    engine.on_bar(make_bar(1, open=101.0, close=102.0))
    signal = engine.on_bar(touch_bar(2))
    assert signal is not None
    assert signal.entry == pytest.approx(LEVEL)


def test_close_touch_definition_needs_the_wrong_side_expiry_disabled() -> None:
    engine = machine("buy", touch_definition="close", expire_on_wrong_side_close=False)
    engine.on_bar(make_bar(0, close=101.0))
    engine.on_bar(make_bar(1, open=101.0, close=102.0))
    signal = engine.on_bar(make_bar(2, open=100.5, high=100.6, low=99.0, close=99.8))
    assert signal is not None
    assert signal.entry == pytest.approx(99.8)


def test_unknown_direction_is_rejected() -> None:
    with pytest.raises(ValueError):
        machine("sideways")


def test_state_survives_a_serialization_round_trip() -> None:
    engine = machine("buy", cooldown_bars=4)
    engine.on_bar(make_bar(0, close=101.0))
    engine.on_bar(make_bar(1, open=101.0, close=102.0))
    assert engine.on_bar(touch_bar(2)) is not None

    restored = SetupState.from_dict(engine.state.to_dict())
    assert restored.phase is Phase.COOLDOWN
    assert restored.bars_until_ready == 4
    assert restored.candle1_time == engine.state.candle1_time
    assert restored.last_signal_id == engine.state.last_signal_id

    resumed = StrategyMachine(
        symbol="EURUSD", timeframe="4h", direction="buy",
        params=engine.params, risk=engine.risk, state=restored,
        atr_params=AtrParams(),
    )
    assert resumed.on_bar(touch_bar(3)) is None
    assert resumed.state.phase is Phase.COOLDOWN