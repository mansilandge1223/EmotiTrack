"""Load config.yaml and provide a single access point."""

import yaml
from pathlib import Path

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"
_config = None


def get_config() -> dict:
    """Load and cache the project configuration."""
    global _config
    if _config is None:
        with open(_CONFIG_PATH) as f:
            _config = yaml.safe_load(f)
    return _config


def get(section: str, key: str | None = None):
    """Get a config section or a specific key within a section.

    Usage:
        get("audio", "sample_rate")  -> 16000
        get("paths")                 -> {"data_raw": "data/raw", ...}
    """
    cfg = get_config()
    if key is None:
        return cfg[section]
    return cfg[section][key]
