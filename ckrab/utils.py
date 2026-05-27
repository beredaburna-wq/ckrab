"""
Utility functions for ckrab.
"""

import logging
import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    config_file = Path(config_path)

    if not config_file.exists():
        logging.warning(f"Config file not found: {config_path}. Using defaults.")
        return {}

    try:
        with open(config_file, "r") as f:
            config = yaml.safe_load(f) or {}
        logging.info(f"Loaded config from {config_path}")
        return config
    except Exception as e:
        logging.error(f"Failed to load config: {e}")
        raise


def setup_logging(level: str = "INFO", log_file: str = None):
    """
    Setup logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
    """
    level = getattr(logging, level.upper(), logging.INFO)

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(level)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


def ensure_directories(config: Dict[str, Any]):
    """
    Create necessary directories from config.

    Args:
        config: Configuration dictionary
    """
    paths = config.get("paths", {})
    for path_key, path_value in paths.items():
        if path_value:
            Path(path_value).mkdir(parents=True, exist_ok=True)
            logging.debug(f"Ensured directory: {path_value}")