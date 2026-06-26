"""
Main image-to-video generator module.
"""

import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
from PIL import Image
import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image
import cv2
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Constants
DEFAULT_DURATION = 3.0
DEFAULT_FPS = 30
DEFAULT_INFERENCE_STEPS = 25
DEFAULT_GUIDANCE_SCALE = 7.5
MIN_GUIDANCE_SCALE = 1.0
MAX_GUIDANCE_SCALE = 20.0
OPTIMAL_HEIGHT = 576
OPTIMAL_WIDTH = 1024


class ImageToVideoGenerator:
    """Image to video generation using Stable Video Diffusion."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, device: Optional[str] = None):
        """
        Initialize the generator.

        Args:
            config: Configuration dictionary
            device: Device to use ('cpu', 'cuda', 'mps', or None for auto-detect)
        """
        self.config = config or {}
        self.device = device or self._detect_device()
        self.pipe = None
        self.model_loaded = False

        logger.info(f"Initializing ImageToVideoGenerator on device: {self.device}")

    def _detect_device(self) -> str:
        """
        Auto-detect the best available device.

        Returns:
            Device string ('cuda', 'mps', or 'cpu')
        """
        if torch.cuda.is_available():
            device = "cuda"
            logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
            return device
        elif torch.backends.mps.is_available():
            device = "mps"
            logger.info("MPS (Metal Performance Shaders) available")
            return device
        else:
            device = "cpu"
            logger.warning("Using CPU - generation may be slow")
            return device

    def load_model(self, model_name: str = "stable-video-diffusion-img2vid"):
        """
        Load the video generation model.

        Args:
            model_name: Name of the model to load

        Raises:
            Exception: If model loading fails
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
                logger.debug("Attention slicing enabled")

            if self.config.get("performance", {}).get("cpu_offload", True):
                self.pipe.enable_sequential_cpu_offload()
                logger.debug("Sequential CPU offload enabled")

            self.model_loaded = True
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _validate_generate_params(
        self,
        duration: float,
        fps: int,
        num_inference_steps: int,
        guidance_scale: float,
    ) -> None:
        """
        Validate generation parameters.

        Args:
            duration: Video duration in seconds
            fps: Frames per second
            num_inference_steps: Number of inference steps
            guidance_scale: Guidance scale value

        Raises:
            ValueError: If parameters are invalid
        """
        if duration <= 0:
            raise ValueError("duration must be greater than 0")
        if fps <= 0:
            raise ValueError("fps must be greater than 0")
        if num_inference_steps < 1:
            raise ValueError("num_inference_steps must be at least 1")
        if not MIN_GUIDANCE_SCALE <= guidance_scale <= MAX_GUIDANCE_SCALE:
            raise ValueError(
                f"guidance_scale must be between {MIN_GUIDANCE_SCALE} and {MAX_GUIDANCE_SCALE}"
            )

    def _set_seed(self, seed: Optional[int]) -> None:
        """
        Set random seed for reproducibility.

        Args:
            seed: Seed value (None to skip)
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            logger.debug(f"Random seed set to: {seed}")

    def _preprocess_image(self, image_path: str) -> Image.Image:
        """
        Load and validate image.

        Args:
            image_path: Path to input image

        Returns:
            Preprocessed PIL Image resized to optimal dimensions

        Raises:
            FileNotFoundError: If image doesn't exist
            ValueError: If image is invalid
        """
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        try:
            image = load_image(image_path)
            original_size = image.size
            image = image.resize((OPTIMAL_WIDTH, OPTIMAL_HEIGHT))
            logger.debug(f"Resized image from {original_size} to ({OPTIMAL_WIDTH}, {OPTIMAL_HEIGHT})")
            return image
        except Exception as e:
            raise ValueError(f"Failed to load or process image: {e}")

    def _cleanup(self) -> None:
        """Clear GPU memory after generation."""
        if self.device == "cuda":
            torch.cuda.empty_cache()
            logger.debug("GPU cache cleared")

    def generate(
        self,
        image_path: str,
        output_path: str = "output.mp4",
        duration: float = DEFAULT_DURATION,
        fps: int = DEFAULT_FPS,
        num_inference_steps: int = DEFAULT_INFERENCE_STEPS,
        guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
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

        Raises:
            FileNotFoundError: If image file not found
            ValueError: If parameters are invalid
            Exception: If generation fails
        """
        if not self.model_loaded:
            self.load_model()

        # Validate parameters
        self._validate_generate_params(duration, fps, num_inference_steps, guidance_scale)

        try:
            # Load and prepare image
            logger.info(f"Loading image from: {image_path}")
            image = self._preprocess_image(image_path)

            # Set seed for reproducibility
            self._set_seed(seed)

            # Generate video frames
            num_frames = int(duration * fps)
            logger.info(f"Generating {num_frames} video frames...")
            with torch.no_grad():
                frames = self.pipe(
                    image=image,
                    height=OPTIMAL_HEIGHT,
                    width=OPTIMAL_WIDTH,
                    num_frames=num_frames,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                ).frames[0]

            # Convert frames to video
            logger.info(f"Saving video to: {output_path}")
            self._save_video(frames, output_path, fps)

            logger.info("Video generation complete!")
            return output_path

        except Exception as e:
            logger.error(f"Video generation failed: {e}")
            raise
        finally:
            self._cleanup()

    def _save_video(self, frames: List[Image.Image], output_path: str, fps: int = DEFAULT_FPS) -> None:
        """
        Save frames as video file.

        Args:
            frames: List of PIL images
            output_path: Output video path
            fps: Frames per second

        Raises:
            Exception: If video writing fails
        """
        # Create output directory
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        try:
            # Convert PIL images to numpy arrays
            frames_array = []
            for frame in tqdm(frames, desc="Processing frames"):
                if isinstance(frame, Image.Image):
                    frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
                frames_array.append(frame)

            # Get frame dimensions
            if not frames_array:
                raise ValueError("No frames to write")
            height, width = frames_array[0].shape[:2]

            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            if not out.isOpened():
                raise RuntimeError(f"Failed to create video writer for {output_path}")

            # Write frames
            for frame in tqdm(frames_array, desc="Writing video"):
                out.write(frame)

            out.release()
            logger.info(f"Video saved: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save video: {e}")
            raise

    def batch_generate(
        self,
        image_dir: str,
        output_dir: str = "outputs",
        skip_errors: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate videos from multiple images.

        Args:
            image_dir: Directory containing input images
            output_dir: Directory to save output videos
            skip_errors: If True, skip failed images; if False, raise on first error
            **kwargs: Additional arguments for generate()

        Returns:
            Dictionary with 'videos' (list of paths) and 'errors' (list of error messages)

        Raises:
            FileNotFoundError: If image directory not found
            Exception: If skip_errors is False and generation fails
        """
        if not Path(image_dir).exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
        image_files = [
            f for f in os.listdir(image_dir)
            if Path(f).suffix.lower() in image_extensions
        ]

        if not image_files:
            logger.warning(f"No image files found in {image_dir}")

        logger.info(f"Found {len(image_files)} images to process")

        output_paths = []
        errors = []

        for i, img_file in enumerate(tqdm(image_files, desc="Batch processing"), 1):
            input_path = os.path.join(image_dir, img_file)
            output_name = Path(img_file).stem + ".mp4"
            output_path = os.path.join(output_dir, output_name)

            try:
                logger.info(f"Processing {i}/{len(image_files)}: {img_file}")
                output = self.generate(input_path, output_path, **kwargs)
                output_paths.append(output)
            except Exception as e:
                error_msg = f"Failed to process {img_file}: {e}"
                if not skip_errors:
                    logger.error(error_msg)
                    raise
                errors.append(error_msg)
                logger.warning(error_msg)

        logger.info(f"Generated {len(output_paths)}/{len(image_files)} videos")
        if errors:
            logger.warning(f"{len(errors)} image(s) failed to generate")

        return {"videos": output_paths, "errors": errors}
