"""
Text-to-Image generation module.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image

logger = logging.getLogger(__name__)


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

    def load_model(self, model_name: str = "stabilityai/stable-diffusion-2.1"):
        """
        Load the text-to-image model.

        Args:
            model_name: Name or path of the model to load
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

            # Use faster scheduler for mobile devices
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipe.scheduler.config
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
        prompt: str,
        output_path: str = "generated_image.png",
        negative_prompt: str = "blurry, low quality, distorted",
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        height: int = 512,
        width: int = 768,
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
            height: Image height in pixels
            width: Image width in pixels
            seed: Random seed for reproducibility

        Returns:
            Path to generated image
        """
        if not self.model_loaded:
            self.load_model()

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
        prompts: list,
        output_dir: str = "generated_images",
        **kwargs
    ) -> list:
        """
        Generate multiple images from prompts.

        Args:
            prompts: List of text prompts
            output_dir: Directory to save images
            **kwargs: Additional arguments for generate()

        Returns:
            List of generated image paths
        """
        output_paths = []

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
                logger.error(f"Failed to generate image for prompt '{prompt}': {e}")

        logger.info(f"Generated {len(output_paths)} images")
        return output_paths