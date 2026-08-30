from argparse import Namespace
from pathlib import Path
from typing import Any, Mapping, Union

import yaml


def dict_to_namespace(value: Any) -> Any:
    if isinstance(value, Mapping):
        return Namespace(
            **{key: dict_to_namespace(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return [dict_to_namespace(item) for item in value]
    return value


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a YAML mapping: {path}")
    return config


def load_model_config(path: Union[str, Path]) -> Namespace:
    config_path = Path(path).expanduser().resolve()
    config = _load_yaml_mapping(config_path)
    if "model" not in config:
        raise ValueError(f"Model config must define 'model': {config_path}")
    return dict_to_namespace(config)


def load_training_config(path: Union[str, Path]) -> Namespace:
    config_path = Path(path).expanduser().resolve()
    training_config = _load_yaml_mapping(config_path)
    model_config_value = training_config.pop("model_config", None)
    if not model_config_value:
        raise ValueError(
            f"Training config must define 'model_config': {config_path}"
        )

    model_config_path = Path(model_config_value).expanduser()
    if not model_config_path.is_absolute():
        model_config_path = config_path.parent / model_config_path
    model_config = _load_yaml_mapping(model_config_path.resolve())
    if "model" not in model_config:
        raise ValueError(
            f"Referenced model config must define 'model': {model_config_path}"
        )

    overlapping_keys = set(training_config).intersection(model_config)
    if overlapping_keys:
        names = ", ".join(sorted(overlapping_keys))
        raise ValueError(
            f"Training and model configs define the same top-level keys: {names}"
        )

    return dict_to_namespace({**model_config, **training_config})
