"""
xckrab - Text-to-Image-to-Video Generator

A lightweight generator for creating images from prompts and animating them into videos.
Optimized for mobile devices.
"""

__version__ = "0.2.0"
__author__ = "beredaburna-wq"

from .text_to_image import TextToImageGenerator
from .image_to_video import ImageToVideoGenerator
from .utils import load_config, setup_logging

__all__ = [
    "TextToImageGenerator",
    "ImageToVideoGenerator",
    "load_config",
    "setup_logging",
]