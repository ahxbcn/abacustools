"""Manage default settings for abacustools."""

from copy import deepcopy
from typing import Any
from pathlib import Path

def read_config_file(config_file: Path) -> dict[str, Any]:
    """
    Read the config file in YAML format and return a dictionary of settings.
    """
    import yaml
    
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config


def _merge_config(default: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Merge user settings over the packaged defaults recursively."""
    config = deepcopy(default)
    for key, value in overrides.items():
        if isinstance(config.get(key), dict) and isinstance(value, dict):
            config[key] = _merge_config(config[key], value)
        else:
            config[key] = deepcopy(value)
    return config

def generate_default_config(override: bool=False):
    """
    Write default config file to ~/.abacustools/config.yaml.
    """
    import yaml
    original_config_file = Path(__file__).parent / "default_config.yaml"
    default_config = read_config_file(original_config_file)
    config_dir = Path.home() / ".abacustools"
    config_file = config_dir / "config.yaml"
    if config_file.exists() and not override:
        print("Found existing config file. Skipping generation.")
    else:
        print(f"Generating default config file at {config_file}")
        config_dir.mkdir(parents=True, exist_ok=True)
        with open(config_file, "w") as f:
            yaml.dump(default_config, f)


_DEFAULT_CONFIG_FILE = Path(__file__).parent / "default_config.yaml"
config_file = Path.home() / ".abacustools" / "config.yaml"
if not config_file.exists():
    generate_default_config()
CONFIG = _merge_config(
    read_config_file(_DEFAULT_CONFIG_FILE),
    read_config_file(config_file),
)

if __name__ == "__main__":
    generate_default_config()
