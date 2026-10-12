"""KRO (Triple Confirmation Kernel Regression Oscillator) - DELIBERATELY NOT PORTED.

Licence: the original indicator is CC BY-SA 4.0 (QuantraSystems). Any derivative
must keep the CC BY-SA 4.0 licence and attribution.

Why this module raises instead of returning numbers
---------------------------------------------------
The Pine file in Indicators.txt contains only the CALL SITE. The maths lives in
two imported libraries that were not supplied:

    import QuantraAI/QuantraMAs/2 as DynamicMA            # Indicators.txt:930
    import QuantraSystems/DynamicFunctions/1 as DynamicFunc # Indicators.txt:931

    Wave = DynamicFunc.waveCalculation(source, bandwidth, width)      # :1007
    Ep   = DynamicFunc.kernelRegression(source, bandwidth, 'Epanechnikov')  # :1010
    Lo   = DynamicFunc.kernelRegression(source, bandwidth, 'Logistic')      # :1011
    Wa   = DynamicFunc.kernelRegression(source, bandwidth, 'Wave')          # :1012
    Mid  = DynamicMA.SMA(AV, sdLook)                                      # :1016
    [_, u1, l1] = DynamicFunc.stdv_bands(AV, sdLook, sdMult / 2)          # :1021
    [_, u2, l2] = DynamicFunc.stdv_bands(AV, sdLook, sdMult)              # :1022

Only ``DynamicMA.SMA`` is trivially reproducible (``ta.sma``). The other four
functions define the indicator, and reimplementing them from the published
LuxAlgo variant would silently change the signal. So KRO stays disabled until
the source of `QuantraSystems/DynamicFunctions` is supplied.

To enable it:
  1. Open the QuantraSystems library script on TradingView and paste
     `waveCalculation`, `kernelRegression` and `stdv_bands`.
  2. Implement them here, mirroring the Pine maths bar for bar.
  3. Set ``kro.enabled: true`` in config/config.yaml.
  4. Add parity cases in tests/parity/reference_values.yaml.
"""
from __future__ import annotations

MISSING_LIBRARY_FUNCTIONS: tuple[str, ...] = (
    "DynamicFunc.waveCalculation(source, bandwidth, width)",
    "DynamicFunc.kernelRegression(source, bandwidth, 'Epanechnikov')",
    "DynamicFunc.kernelRegression(source, bandwidth, 'Logistic')",
    "DynamicFunc.kernelRegression(source, bandwidth, 'Wave')",
    "DynamicFunc.stdv_bands(series, lookback, multiplier)",
)


class KroNotPortedError(RuntimeError):
    """Raised whenever a code path needs the KRO confirmation while it is unported."""


def kro_confirmation(*_args: object, **_kwargs: object) -> None:
    """Always raises: the KRO maths was never supplied, so it is never guessed."""
    listed = "\n  - ".join(MISSING_LIBRARY_FUNCTIONS)
    raise KroNotPortedError(
        "KRO confirmation is unavailable. The Pine imports QuantraSystems/DynamicFunctions, "
        "whose source is not in Indicators.txt. Missing:\n  - " + listed
    )


def assert_kro_disabled(enabled: bool) -> None:
    """Guard used by the strategy layer so a config flip cannot silently do nothing."""
    if enabled:
        kro_confirmation()