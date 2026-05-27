"""
ckrab - Uncensored Image to Video Generator

A lightweight image-to-video generation tool optimized for mobile devices.
"""

__version__ = "0.1.0"
__author__ = "beredaburna-wq"

from .generator import ImageToVideoGenerator
from .utils import load_config, setup_logging

__all__ = [
    "ImageToVideoGenerator",
    "load_config",
    "setup_logging",
]