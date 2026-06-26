"""
Text-to-Image generation module.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

logger = logging.getLogger(__name__)

# Constants
DEFAULT_MODEL = "stabilityai/stable-diffusion-2.1"
DEFAULT_INFERENCE_STEPS = 30
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_HEIGHT = 512
DEFAULT_WIDTH = 768
MAX_PROMPT_LENGTH = 77
MIN_GUIDANCE_SCALE = 1.0
MAX_GUIDANCE_SCALE = 20.0


class TextToImageGenerator:
    """Text-to-Image generation using Stable Diffusion."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, device: Optional[str] = None):
        """
        Initialize the text-to-image generator.

        Args:
            config: Configuration dictionary
            device: Device to use ('cpu', 'cuda', 'mps', or 'auto')
        """
        self.config = config or {}
        self.device = device or self._detect_device()
        self.pipe = None
        self.model_loaded = False

        logger.info(f"Initializing TextToImageGenerator on device: {self.device}")

    def _detect_device(self) -> str:
        """Auto-detect the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _validate_generate_params(
        self, height: int, width: int, guidance_scale: float, num_inference_steps: int
    ) -> None:
        """
        Validate generation parameters.

        Args:
            height: Image height in pixels
            width: Image width in pixels
            guidance_scale: Guidance scale value
            num_inference_steps: Number of inference steps

        Raises:
            ValueError: If parameters are invalid
        """
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError("Height and width must be multiples of 8")
        if not MIN_GUIDANCE_SCALE <= guidance_scale <= MAX_GUIDANCE_SCALE:
            raise ValueError(f"guidance_scale must be between {MIN_GUIDANCE_SCALE} and {MAX_GUIDANCE_SCALE}")
        if num_inference_steps < 1:
            raise ValueError("num_inference_steps must be at least 1")

    def _preprocess_prompt(self, prompt: str) -> str:
        """
        Clean and validate prompt text.

        Args:
            prompt: Raw prompt text

        Returns:
            Cleaned prompt text

        Raises:
            ValueError: If prompt is invalid
        """
        if not prompt or not isinstance(prompt, str):
            raise ValueError("Prompt must be a non-empty string")
        return prompt.strip()[:MAX_PROMPT_LENGTH]

    def load_model(self, model_name: str = DEFAULT_MODEL) -> None:
        """
        Load the text-to-image model.

        Args:
            model_name: Name or path of the model to load

        Raises:
            Exception: If model loading fails
        """
        if self.model_loaded:
            logger.info("Model already loaded")
            return

        logger.info(f"Loading model: {model_name}")
        try:
            # Determine data type based on device
            if self.device == "cpu":
                torch_dtype = torch.float32
            else:
                torch_dtype = torch.float16

            # Load pipeline
            self.pipe = StableDiffusionPipeline.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                safety_checker=None,  # Disable safety checker for uncensored generation
            )

            # Use faster scheduler
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipe.scheduler.config
            )

            self.pipe.to(self.device)

            # Enable device-specific memory optimizations
            self._configure_memory_optimizations()

            self.model_loaded = True
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _configure_memory_optimizations(self) -> None:
        """Configure memory optimizations based on device type."""
        if self.device == "cpu":
            # CPU: Use sequential offloading
            if self.config.get("performance", {}).get("cpu_offload", True):
                self.pipe.enable_sequential_cpu_offload()
        elif self.device == "mps":
            # Metal Performance Shaders: Use attention slicing
            if self.config.get("performance", {}).get("enable_attention_slicing", True):
                self.pipe.enable_attention_slicing()
        else:  # cuda
            # CUDA: Only enable optimizations if VRAM is limited
            if self.config.get("performance", {}).get("enable_attention_slicing", False):
                self.pipe.enable_attention_slicing()
            if self.config.get("performance", {}).get("cpu_offload", False):
                self.pipe.enable_sequential_cpu_offload()

    def generate(
        self,
        prompt: str,
        output_path: str = "generated_image.png",
        negative_prompt: str = "blurry, low quality, distorted",
        num_inference_steps: int = DEFAULT_INFERENCE_STEPS,
        guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
        height: int = DEFAULT_HEIGHT,
        width: int = DEFAULT_WIDTH,
        seed: Optional[int] = None,
    ) -> str:
        """
        Generate an image from a text prompt.

        Args:
            prompt: Text description of the image to generate
            output_path: Path to save generated image
            negative_prompt: What NOT to generate
            num_inference_steps: Number of diffusion steps (more = higher quality)
            guidance_scale: How strongly to follow the prompt (1-20)
            height: Image height in pixels (must be multiple of 8)
            width: Image width in pixels (must be multiple of 8)
            seed: Random seed for reproducibility

        Returns:
            Path to generated image

        Raises:
            ValueError: If parameters are invalid
            Exception: If generation fails
        """
        if not self.model_loaded:
            self.load_model()

        # Validate parameters
        self._validate_generate_params(height, width, guidance_scale, num_inference_steps)
        prompt = self._preprocess_prompt(prompt)

        logger.info(f"Generating image from prompt: {prompt[:50]}...")

        try:
            # Set seed for reproducibility
            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(seed)
            else:
                generator = torch.Generator(device=self.device).manual_seed(0)

            # Generate image
            with torch.no_grad():
                image = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                ).images[0]

            # Save image
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            image.save(output_path)
            logger.info(f"Image saved to: {output_path}")

            return output_path

        except Exception as e:
            logger.error(f"Failed to generate image: {e}")
            raise

    def batch_generate(
        self,
        prompts: List[str],
        output_dir: str = "generated_images",
        skip_errors: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate multiple images from prompts.

        Args:
            prompts: List of text prompts
            output_dir: Directory to save images
            skip_errors: If True, skip failed prompts; if False, raise on first error
            **kwargs: Additional arguments for generate()

        Returns:
            Dictionary with 'images' (list of paths) and 'errors' (list of error messages)

        Raises:
            Exception: If skip_errors is False and generation fails
        """
        output_paths = []
        errors = []

        for i, prompt in enumerate(prompts, 1):
            try:
                output_name = f"image_{i:03d}.png"
                output_path = Path(output_dir) / output_name

                logger.info(f"Generating image {i}/{len(prompts)}: {prompt[:50]}...")
                path = self.generate(
                    prompt=prompt,
                    output_path=str(output_path),
                    **kwargs
                )
                output_paths.append(path)
            except Exception as e:
                error_msg = f"Failed to generate image for prompt '{prompt}': {e}"
                if not skip_errors:
                    logger.error(error_msg)
                    raise
                errors.append(error_msg)
                logger.warning(error_msg)

        logger.info(f"Generated {len(output_paths)}/{len(prompts)} images")
        if errors:
            logger.warning(f"{len(errors)} image(s) failed to generate")

        return {
            "images": output_paths,
            "errors": errors,
            "success_count": len(output_paths),
            "error_count": len(errors),
        }

    def unload_model(self) -> None:
        """Unload model and free GPU/memory resources."""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            self.model_loaded = False
            if self.device == "cuda":
                torch.cuda.empty_cache()
            logger.info("Model unloaded and memory freed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        self.unload_model()
