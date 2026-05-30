import yaml
from pathlib import Path


def load_config(path: str) -> dict:
    """Load YAML config file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def merge_config(base: dict, override: dict) -> dict:
    """Deep merge override into base config."""
    result = base.copy()
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = merge_config(result[key], val)
        else:
            result[key] = val
    return result
