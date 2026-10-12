"""Configuration loading: YAML + environment, no hardcoded secrets.

Design rules:
  * Every tunable lives in config/config.yaml.
  * Secrets are `${ENV_VAR}` references resolved from the environment (a local
    config/.env is loaded automatically, no python-dotenv dependency).
  * Indicator profiles support one-level `extends:` inheritance so a symbol can
    reuse the base profile and change a single value.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"
_ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)\}")


class ConfigError(RuntimeError):
    """Raised when the configuration file is missing, malformed or incomplete."""


def load_dotenv(path: Path) -> dict[str, str]:
    """Minimal .env reader: KEY=value lines, # comments, optional quotes.

    Values are written into ``os.environ`` without overriding real environment
    variables, so a genuine env var always wins over the file.
    """
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if not key:
            continue
        values[key] = value
        os.environ.setdefault(key, value)
    return values


def _expand(value: Any) -> Any:
    """Recursively expand ${ENV_VAR} references; missing vars become empty string."""
    if isinstance(value, str):
        return _ENV_PATTERN.sub(lambda m: os.environ.get(m.group(1), ""), value)
    if isinstance(value, Mapping):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


@dataclass(frozen=True)
class MlVwapParams:
    """Inputs of 'VWAP Bands with ML [CryptoSea]'.

    ``session_tz`` / ``session_start_hour`` set the VWAP's daily anchor to match
    the chart's trading day. Defaults (UTC / 0) match a UTC-anchored chart;
    a broker whose daily bar opens off midnight UTC (most FX brokers) sets these
    to its server time zone so the ported VWAP resets on the same bar as Pine's
    ``timeframe.change("1D")``.
    """

    window: int = 300
    forecast: int = 1
    sigma: float = 0.1
    band1_multiplier: float = 2.0
    band2_multiplier: float = 3.0
    session_tz: str = "UTC"
    session_start_hour: float = 0.0


@dataclass(frozen=True)
class TrendMeterParams:
    """Which element of 'Trend meter by Lij_MC' is read as green.

    mode:
      crossover       -> fast_ma > slow_ma            (Trend Bar 'MA Crossover')
      slow_direction  -> slow_ma[t] > slow_ma[t-1]    (Trend Bar 'MA Direction - Slow MA')
      three_meters    -> RSI5>50 AND RSI13>50 AND fast-MACD histogram>0
                          ('All 3 Trend Meters align'; ignores slow_ma entirely)
    """

    mode: str = "crossover"
    fast_type: str = "EMA"
    fast_length: int = 5
    slow_type: str = "SMA"
    slow_length: int = 150


@dataclass(frozen=True)
class VslrtParams:
    """Inputs of 'Volume-Supported Linear Regression Trend' (LonesomeTheBlue)."""

    short_length: int = 20
    long_length: int = 64


@dataclass(frozen=True)
class KroParams:
    """Inputs of the QuantraSystems KRO overlay.

    enabled is False and MUST stay False until the imported library functions
    (waveCalculation / kernelRegression / stdv_bands) are ported. See
    indicators/kro.py for the exact list of missing functions.
    """

    enabled: bool = False
    bandwidth: int = 300
    width: float = 2.0
    sd_lookback: int = 150
    sd_extreme: float = 3.0
    mintick: float = 0.01


@dataclass(frozen=True)
class AtrParams:
    """Inputs of 'NNFX ATR' (sueun123)."""

    length: int = 14
    smoothing: str = "RMA"
    tp_multiplier: float = 2.25
    sl_multiplier: float = 1.5
    decimals: int = 5


@dataclass(frozen=True)
class IndicatorParams:
    ml_vwap: MlVwapParams = field(default_factory=MlVwapParams)
    trend_meter: TrendMeterParams = field(default_factory=TrendMeterParams)
    vslrt: VslrtParams = field(default_factory=VslrtParams)
    kro: KroParams = field(default_factory=KroParams)
    atr: AtrParams = field(default_factory=AtrParams)


@dataclass(frozen=True)
class StrategyParams:
    touch_line: str = "vwap"
    touch_definition: str = "wick"
    expiry_bars: int = 10
    expire_on_wrong_side_close: bool = True
    filters_on_entry_bar: bool = True
    entry_price: str = "close"
    cooldown_bars: int = 3
    reseed_candle1_on_failure: bool = False


@dataclass(frozen=True)
class RiskParams:
    risk_percent_per_trade: float = 1.0
    account_balance: float = 1000.0
    min_lot: float = 0.01
    lot_step: float = 0.01
    max_lot: float = 100.0
    assumed_contract_size: float = 100_000.0


@dataclass(frozen=True)
class NotifierParams:
    telegram: dict[str, Any] = field(default_factory=dict)
    discord: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SymbolConfig:
    symbol: str
    timeframe: str
    provider: str
    profile: str
    params: IndicatorParams
    enabled: bool = True

    @property
    def key(self) -> str:
        return f"{self.symbol}:{self.timeframe}"


@dataclass(frozen=True)
class AppConfig:
    runtime: dict[str, Any]
    data: dict[str, Any]
    strategy: StrategyParams
    risk: RiskParams
    notifier: NotifierParams
    journal: dict[str, Any]
    alerts: dict[str, Any]
    symbols: tuple[SymbolConfig, ...]
    source_path: Path

    def enabled_symbols(self) -> tuple[SymbolConfig, ...]:
        return tuple(s for s in self.symbols if s.enabled)


@dataclass(frozen=True)
class CandleColumns:
    """Canonical OHLCV column names used everywhere in the codebase."""

    time: str = "time"
    open: str = "open"
    high: str = "high"
    low: str = "low"
    close: str = "close"
    volume: str = "volume"

    @classmethod
    def names(cls) -> tuple[str, ...]:
        return (cls.time, cls.open, cls.high, cls.low, cls.close, cls.volume)


def _merge_profile(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """One-level deep merge of indicator sections (override wins per key)."""
    merged = dict(base)
    for section, values in override.items():
        if section in merged and isinstance(values, Mapping):
            merged[section] = {**merged[section], **values}
        else:
            merged[section] = values
    return merged


def _build_params(raw: Mapping[str, Any]) -> IndicatorParams:
    return IndicatorParams(
        ml_vwap=MlVwapParams(**_with_keys(raw.get("ml_vwap", {}), MlVwapParams)),
        trend_meter=TrendMeterParams(
            **_with_keys(raw.get("trend_meter", {}), TrendMeterParams)
        ),
        vslrt=VslrtParams(**_with_keys(raw.get("vslrt", {}), VslrtParams)),
        kro=KroParams(**_with_keys(raw.get("kro", {}), KroParams)),
        atr=AtrParams(**_with_keys(raw.get("atr", {}), AtrParams)),
    )


def _with_keys(raw: Mapping[str, Any], cls: type) -> dict[str, Any]:
    known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
    unknown = set(raw) - known
    if unknown:
        raise ConfigError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")
    return {k: v for k, v in raw.items() if k in known}


def load_config(path: Path | str | None = None) -> AppConfig:
    """Load, validate and return the application configuration."""
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        raise ConfigError(f"Config file not found: {config_path}")

    os.environ.setdefault("PYTHONHASHSEED", "0")
    load_dotenv(config_path.parent / ".env")

    with config_path.open("r", encoding="utf-8") as handle:
        raw = _expand(yaml.safe_load(handle) or {})
    if not isinstance(raw, dict):
        raise ConfigError("Top level of the config file must be a mapping")

    profiles: dict[str, dict[str, Any]] = {}
    for name, body in (raw.get("profiles") or {"default": {}}).items():
        parent = body.get("extends")
        if parent:
            if parent not in (raw.get("profiles") or {}):
                raise ConfigError(f"Profile '{name}' extends unknown profile '{parent}'")
            profiles[name] = _merge_profile(profiles[parent], body)
        else:
            profiles[name] = dict(body)

    symbols: list[SymbolConfig] = []
    for entry in raw.get("symbols") or []:
        profile_name = entry.get("profile", "default")
        if profile_name not in profiles:
            raise ConfigError(f"Symbol {entry.get('symbol')} uses unknown profile '{profile_name}'")
        symbols.append(
            SymbolConfig(
                symbol=str(entry["symbol"]),
                timeframe=str(entry["timeframe"]),
                provider=str(entry.get("provider", raw.get("data", {}).get("provider", "twelve_data"))),
                profile=profile_name,
                params=_build_params(profiles[profile_name]),
                enabled=bool(entry.get("enabled", True)),
            )
        )

    if not symbols:
        raise ConfigError("No symbols configured")

    strategy_raw = raw.get("strategy") or {}
    strategy = StrategyParams(**_with_keys(strategy_raw, StrategyParams))
    risk = RiskParams(**_with_keys(raw.get("risk") or {}, RiskParams))

    return AppConfig(
        runtime=raw.get("runtime") or {},
        data=raw.get("data") or {},
        strategy=strategy,
        risk=risk,
        notifier=NotifierParams(
            telegram=(raw.get("notifier") or {}).get("telegram") or {},
            discord=(raw.get("notifier") or {}).get("discord") or {},
        ),
        journal=raw.get("journal") or {},
        alerts=raw.get("alerts") or {},
        symbols=tuple(symbols),
        source_path=config_path,
    )


def with_param_overrides(params: IndicatorParams, **overrides: Any) -> IndicatorParams:
    """Return a copy of `params` with individual indicator sections replaced."""
    return replace(params, **overrides)