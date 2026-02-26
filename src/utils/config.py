import json
import os
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected object in {path}, got {type(data).__name__}")
    return data


def _set_nested_value(root: dict[str, Any], path_keys: list[str], value: str) -> None:
    cursor = root
    for key in path_keys[:-1]:
        next_value = cursor.get(key)
        if not isinstance(next_value, dict):
            next_value = {}
            cursor[key] = next_value
        cursor = next_value

    key = path_keys[-1]
    if value.isdigit():
        cursor[key] = int(value)
        return
    if value.lower() in {"true", "false"}:
        cursor[key] = value.lower() == "true"
        return
    try:
        cursor[key] = float(value)
        return
    except ValueError:
        cursor[key] = value


def apply_env_overrides(config: dict[str, Any], prefix: str = "ADAPTEACH__") -> dict[str, Any]:
    output = dict(config)
    for env_key, env_value in os.environ.items():
        if not env_key.startswith(prefix):
            continue
        nested_keys = env_key[len(prefix) :].lower().split("__")
        _set_nested_value(output, nested_keys, env_value)
    return output


def validate_config(config: dict[str, Any], schema_path: Path) -> None:
    with schema_path.open("r", encoding="utf-8") as handle:
        schema = json.load(handle)
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(config), key=lambda e: list(e.path))
    if errors:
        details = "; ".join(error.message for error in errors)
        raise ValueError(f"Invalid config: {details}")


def load_app_config(
    config_path: Path = Path("config.yaml"),
    schema_path: Path = Path("config.schema.json"),
) -> dict[str, Any]:
    config = _load_yaml(config_path)
    merged = apply_env_overrides(config)
    validate_config(merged, schema_path)
    return merged


def load_pipeline_config(config_name: str, config_dir: Path = Path("configs")) -> dict[str, Any]:
    cfg_path = config_dir / f"{config_name}.yaml"
    if not cfg_path.exists():
        raise ValueError(f"Unknown pipeline config: {config_name}")
    return _load_yaml(cfg_path)
