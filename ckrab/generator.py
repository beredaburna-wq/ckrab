"""
Main image-to-video generator module.
"""

import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
from PIL import Image
import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image
import cv2
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ImageToVideoGenerator:
    """Image to video generation using Stable Video Diffusion."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, device: Optional[str] = None):
        """
        Initialize the generator.

        Args:
            config: Configuration dictionary
            device: Device to use ('cpu', 'cuda', 'mps', or 'auto')
        """
        self.config = config or {}
        self.device = device or self._detect_device()
        self.pipe = None
        self.model_loaded = False

        logger.info(f"Initializing ImageToVideoGenerator on device: {self.device}")

    def _detect_device(self) -> str:
        """Auto-detect the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def load_model(self, model_name: str = "stable-video-diffusion-img2vid"):
        """
        Load the video generation model.

        Args:
            model_name: Name of the model to load
        """
        if self.model_loaded:
            logger.info("Model already loaded")
            return

        logger.info(f"Loading model: {model_name}")
        try:
            self.pipe = StableVideoDiffusionPipeline.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                variant="fp16" if self.device != "cpu" else None,
            )
            self.pipe.to(self.device)

            # Enable memory optimizations for mobile devices
            if self.config.get("performance", {}).get("enable_attention_slicing", True):
                self.pipe.enable_attention_slicing()

            if self.config.get("performance", {}).get("cpu_offload", True):
                self.pipe.enable_sequential_cpu_offload()

            self.model_loaded = True
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def generate(
        self,
        image_path: str,
        output_path: str = "output.mp4",
        duration: float = 3.0,
        fps: int = 30,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
    ) -> str:
        """
        Generate a video from an image.

        Args:
            image_path: Path to input image
            output_path: Path to save output video
            duration: Duration of video in seconds
            fps: Frames per second
            num_inference_steps: Number of diffusion steps (more = higher quality)
            guidance_scale: Guidance scale for generation
            seed: Random seed for reproducibility

        Returns:
            Path to generated video
        """
        if not self.model_loaded:
            self.load_model()

        # Load and prepare image
        logger.info(f"Loading image from: {image_path}")
        try:
            image = load_image(image_path)
            image = image.resize((1024, 576))  # Optimal size for model
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            raise

        # Set seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Generate video frames
        logger.info("Generating video frames...")
        try:
            with torch.no_grad():
                frames = self.pipe(
                    image=image,
                    height=576,
                    width=1024,
                    num_frames=int(duration * fps),
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                ).frames[0]
        except Exception as e:
            logger.error(f"Failed to generate frames: {e}")
            raise

        # Convert frames to video
        logger.info(f"Saving video to: {output_path}")
        try:
            self._save_video(frames, output_path, fps)
        except Exception as e:
            logger.error(f"Failed to save video: {e}")
            raise

        logger.info("Video generation complete!")
        return output_path

    def _save_video(self, frames: list, output_path: str, fps: int = 30):
        """
        Save frames as video file.

        Args:
            frames: List of PIL images
            output_path: Output video path
            fps: Frames per second
        """
        # Create output directory
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Convert PIL images to numpy arrays
        frames_array = []
        for frame in tqdm(frames, desc="Processing frames"):
            if isinstance(frame, Image.Image):
                frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
            frames_array.append(frame)

        # Get frame dimensions
        height, width = frames_array[0].shape[:2]

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Write frames
        for frame in tqdm(frames_array, desc="Writing video"):
            out.write(frame)

        out.release()
        logger.info(f"Video saved: {output_path}")

    def batch_generate(self, image_dir: str, output_dir: str = "outputs", **kwargs) -> list:
        """
        Generate videos from multiple images.

        Args:
            image_dir: Directory containing input images
            output_dir: Directory to save output videos
            **kwargs: Additional arguments for generate()

        Returns:
            List of output video paths
        """
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
        image_files = [
            f for f in os.listdir(image_dir)
            if Path(f).suffix.lower() in image_extensions
        ]

        logger.info(f"Found {len(image_files)} images to process")

        output_paths = []
        for img_file in tqdm(image_files, desc="Batch processing"):
            input_path = os.path.join(image_dir, img_file)
            output_name = Path(img_file).stem + ".mp4"
            output_path = os.path.join(output_dir, output_name)

            try:
                output = self.generate(input_path, output_path, **kwargs)
                output_paths.append(output)
            except Exception as e:
                logger.error(f"Failed to process {img_file}: {e}")

        return output_paths