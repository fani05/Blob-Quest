from pathlib import Path

import pytest
import yaml

from blobquest.config import Config, ConfigError, load_config, save_config

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


@pytest.mark.parametrize("name", ["default.yaml", "sarsa.yaml", "handmade.yaml"])
def test_shipped_configs_load(name: str) -> None:
    config = load_config(CONFIGS / name)
    assert isinstance(config, Config)


def test_default_yaml_matches_dataclass_defaults() -> None:
    assert load_config(CONFIGS / "default.yaml") == Config()


def test_roundtrip(tmp_path: Path) -> None:
    config = Config().with_overrides(training={"seed": 3}, world={"start": (1, 2)})
    save_config(config, tmp_path / "c.yaml")
    assert load_config(tmp_path / "c.yaml") == config


def test_unknown_section_rejected() -> None:
    with pytest.raises(ConfigError, match="unknown config sections"):
        Config.from_dict({"bogus": {}})


def test_unknown_key_rejected() -> None:
    with pytest.raises(ConfigError, match="unknown keys in 'energy'"):
        Config.from_dict({"energy": {"initial": 10, "nope": 1}})


@pytest.mark.parametrize(
    "data",
    [
        {"energy": {"initial": 0}},
        {"energy": {"initial": 100, "maximum": 50}},
        {"agent": {"algorithm": "dqn"}},
        {"agent": {"epsilon": {"kind": "cosine"}}},
        {"world": {"wall_density": 1.5}},
        {"world": {"num_energy_boosters": 10, "max_boosters": 3}},
        {"training": {"episodes": 0}},
    ],
)
def test_invalid_values_rejected(data: dict) -> None:
    with pytest.raises(ConfigError):
        Config.from_dict(data)


def test_nested_schedule_from_dict() -> None:
    config = Config.from_dict({"agent": {"epsilon": {"kind": "exponential", "decay_rate": 0.99}}})
    assert config.agent.epsilon.kind == "exponential"
    assert config.agent.epsilon.decay_rate == 0.99


def test_to_dict_is_yaml_safe() -> None:
    yaml.safe_dump(Config().to_dict())


def test_missing_file() -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config("does/not/exist.yaml")
