"""Config loading must fail loudly rather than silently running on defaults."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from forexpert.settings import ConfigError, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"


def test_default_config_loads() -> None:
    config = load_config(CONFIG_PATH)
    assert config.strategy.touch_line == "vwap"
    assert config.strategy.entry_price == "close"
    assert config.strategy.expiry_bars == 10
    assert config.enabled_symbols(), "no enabled symbols in the default config"


def test_every_symbol_resolves_its_profile() -> None:
    config = load_config(CONFIG_PATH)
    for symbol in config.symbols:
        assert symbol.params.ml_vwap.window in {300, 350}
        assert symbol.params.atr.tp_multiplier == 2.25
        assert symbol.params.atr.sl_multiplier == 1.5
        assert symbol.params.kro.enabled is False, "KRO must stay disabled until it is ported"


def test_profile_extends_overrides_only_the_named_key() -> None:
    config = load_config(CONFIG_PATH)
    profile_200 = next(s for s in config.symbols if s.profile == "tm_sma200")
    profile_default = next(s for s in config.symbols if s.profile == "default")
    assert profile_200.params.trend_meter.slow_length == 200
    assert profile_200.params.trend_meter.slow_type == "SMA"
    assert profile_200.params.ml_vwap.window == profile_default.params.ml_vwap.window


def test_unknown_indicator_key_is_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "config.yaml"
    bad.write_text(
        "profiles:\n"
        "  default:\n"
        "    ml_vwap:\n"
        "      window: 300\n"
        "      not_a_real_key: 1\n"
        "symbols:\n"
        "  - {symbol: EURUSD, timeframe: 4h, profile: default}\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError):
        load_config(bad)


def test_vwap_session_anchor_defaults_to_utc() -> None:
    config = load_config(CONFIG_PATH)
    for symbol in config.symbols:
        assert symbol.params.ml_vwap.session_tz == "UTC"
        assert symbol.params.ml_vwap.session_start_hour == 0.0


def test_vwap_session_anchor_is_configurable_per_profile(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "profiles:\n"
        "  default:\n"
        "    ml_vwap:\n"
        "      session_tz: Europe/Athens\n"
        "      session_start_hour: 22\n"
        "symbols:\n"
        "  - {symbol: EURUSD, timeframe: 4h, profile: default}\n",
        encoding="utf-8",
    )
    config = load_config(config_file)
    assert config.symbols[0].params.ml_vwap.session_tz == "Europe/Athens"
    assert config.symbols[0].params.ml_vwap.session_start_hour == 22.0


def test_missing_config_file_is_rejected() -> None:
    with pytest.raises(ConfigError):
        load_config(REPO_ROOT / "config" / "does_not_exist.yaml")


def test_symbol_keys_are_unique() -> None:
    config = load_config(CONFIG_PATH)
    keys = [s.key for s in config.enabled_symbols()]
    assert len(keys) == len(set(keys)), f"duplicate symbol/timeframe pairs: {keys}"


def test_dotenv_expansion_leaves_no_unresolved_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `load_config` on the real config exports config/.env into os.environ, and
    # .env values never override an existing variable, so clear it first.
    monkeypatch.delenv("TWELVE_DATA_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("TWELVE_DATA_API_KEY=abc123\n", encoding="utf-8")
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "data:\n"
        "  twelve_data:\n"
        "    api_key: ${TWELVE_DATA_API_KEY}\n"
        "symbols:\n"
        "  - {symbol: EURUSD, timeframe: 4h, profile: default}\n",
        encoding="utf-8",
    )
    config = load_config(config_file)
    assert config.data["twelve_data"]["api_key"] == "abc123"
    assert replace(config.strategy, expiry_bars=5).expiry_bars == 5