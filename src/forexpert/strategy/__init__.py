"""Strategy state machine: VWAP + Trend Meter + VSLRT/ATR setup detection.

Stages are built so a future executor never has to touch the signal logic:
`StrategyMachine.on_bar` takes one CLOSED candle and returns a `TradeSignal` or
None. Nothing here fetches data, sends messages or reads the clock.
"""