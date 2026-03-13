"""Configuration management for FedNS-Traffic."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


_DEFAULT_CONFIG_PATH = Path(__file__).parents[2] / "configs" / "default.yaml"


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load YAML configuration.

    Parameters
    ----------
    path : str | None
        Path to a YAML config file.  If *None*, the default config at
        ``configs/default.yaml`` is loaded.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    config_path = Path(path) if path else _DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f)


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge *override* into *base*.  Override values take precedence.
    """
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result


def get_device_str(device: str = "auto") -> str:
    """Resolve ``"auto"`` to the best available device string."""
    if device == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return device
