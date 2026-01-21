"""Configuration loading utilities"""
import yaml
from pathlib import Path


def load_config(config_path):
    """
    Load YAML configuration file

    Args:
        config_path: Path to YAML config file

    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config, save_path):
    """
    Save configuration to YAML file

    Args:
        config: Configuration dictionary
        save_path: Where to save the config
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def merge_configs(base_config, override_config):
    """
    Merge two configs (override takes precedence)

    Args:
        base_config: Base configuration
        override_config: Override values

    Returns:
        Merged configuration
    """
    merged = base_config.copy()
    merged.update(override_config)
    return merged

