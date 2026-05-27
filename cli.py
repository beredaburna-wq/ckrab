"""
Command-line interface for xckrab.
"""

import click
import logging
from pathlib import Path
from xckrab import TextToImageGenerator, ImageToVideoGenerator
from xckrab.utils import load_config, setup_logging, ensure_directories


@click.group()
def cli():
    """xckrab - Text-to-Image-to-Video Generator"""
    pass


# ============================================================================
# TEXT-TO-IMAGE COMMANDS
# ============================================================================

@cli.command()
@click.option("--prompt", "-p", required=True, help="Text prompt to generate image from")
@click.option("--output", "-o", default="generated_image.png", help="Output image path")
@click.option("--negative", default="blurry, low quality, distorted", help="Negative prompt")
@click.option("--steps", default=30, type=int, help="Number of inference steps")
@click.option("--guidance", default=7.5, type=float, help="Guidance scale")
@click.option("--width", default=768, type=int, help="Image width")
@click.option("--height", default=512, type=int, help="Image height")
@click.option("--seed", default=None, type=int, help="Random seed")
@click.option("--device", default="auto", help="Device (cpu, cuda, mps, auto)")
@click.option("--config", default="config.yaml", help="Config file path")
@click.option("--verbose", is_flag=True, help="Verbose output")
def imagine(prompt, output, negative, steps, guidance, width, height, seed, device, config, verbose):
    """Generate an image from a text prompt (like Grok Imagine)."""

    # Setup logging
    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(log_level)
    logger = logging.getLogger(__name__)

    try:
        # Load configuration
        cfg = load_config(config)
        ensure_directories(cfg)

        # Initialize generator
        logger.info("Initializing text-to-image generator...")
        generator = TextToImageGenerator(config=cfg, device=device if device != "auto" else None)

        # Generate image
        logger.info(f"Generating image from prompt: {prompt}")
        output_path = generator.generate(
            prompt=prompt,
            output_path=output,
            negative_prompt=negative,
            num_inference_steps=steps,
            guidance_scale=guidance,
            width=width,
            height=height,
            seed=seed,
        )

        click.echo(f"\n✓ Image saved to: {output_path}")

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        click.echo(f"✗ Error: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.option("--prompts", "-p", required=True, help="Text file with prompts (one per line)")
@click.option("--output-dir", "-o", default="generated_images", help="Output directory")
@click.option("--steps", default=30, type=int, help="Number of inference steps")
@click.option("--width", default=768, type=int, help="Image width")
@click.option("--height", default=512, type=int, help="Image height")
@click.option("--config", default="config.yaml", help="Config file path")
@click.option("--verbose", is_flag=True, help="Verbose output")
def imagine_batch(prompts, output_dir, steps, width, height, config, verbose):
    """Generate multiple images from a file of prompts."""

    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(log_level)
    logger = logging.getLogger(__name__)

    try:
        # Read prompts from file
        with open(prompts, 'r') as f:
            prompt_list = [line.strip() for line in f if line.strip()]

        logger.info(f"Loaded {len(prompt_list)} prompts from {prompts}")

        cfg = load_config(config)
        ensure_directories(cfg)

        generator = TextToImageGenerator(config=cfg)

        output_paths = generator.batch_generate(
            prompts=prompt_list,
            output_dir=output_dir,
            num_inference_steps=steps,
            width=width,
            height=height,
        )

        click.echo(f"\n✓ Generated {len(output_paths)} images in: {output_dir}")

    except Exception as e:
        logger.error(f"Batch generation failed: {e}")
        click.echo(f"✗ Error: {e}", err=True)
        raise click.Abort()


# ============================================================================
# IMAGE-TO-VIDEO COMMANDS
# ============================================================================

@cli.command()
@click.option("--input", "-i", required=True, help="Input image path")
@click.option("--output", "-o", default="output.mp4", help="Output video path")
@click.option("--duration", "-d", default=5, type=float, help="Video duration in seconds")
@click.option("--fps", default=30, type=int, help="Frames per second")
@click.option("--steps", default=25, type=int, help="Number of inference steps")
@click.option("--guidance", default=7.5, type=float, help="Guidance scale")
@click.option("--seed", default=None, type=int, help="Random seed")
@click.option("--device", default="auto", help="Device (cpu, cuda, mps, auto)")
@click.option("--config", default="config.yaml", help="Config file path")
@click.option("--verbose", is_flag=True, help="Verbose output")
def animate(input, output, duration, fps, steps, guidance, seed, device, config, verbose):
    """Animate an image into a video."""

    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(log_level)
    logger = logging.getLogger(__name__)

    try:
        cfg = load_config(config)
        ensure_directories(cfg)

        logger.info("Initializing image-to-video generator...")
        generator = ImageToVideoGenerator(config=cfg, device=device if device != "auto" else None)

        logger.info(f"Animating image: {input} -> {output}")
        output_path = generator.generate(
            image_path=input,
            output_path=output,
            duration=duration,
            fps=fps,
            num_inference_steps=steps,
            guidance_scale=guidance,
            seed=seed,
        )

        click.echo(f"\n✓ Video saved to: {output_path}")

    except Exception as e:
        logger.error(f"Animation failed: {e}")
        click.echo(f"✗ Error: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.option("--input-dir", "-i", required=True, help="Input directory with images")
@click.option("--output-dir", "-o", default="generated_videos", help="Output directory")
@click.option("--duration", "-d", default=5, type=float, help="Video duration")
@click.option("--fps", default=30, type=int, help="Frames per second")
@click.option("--config", default="config.yaml", help="Config file path")
@click.option("--verbose", is_flag=True, help="Verbose output")
def animate_batch(input_dir, output_dir, duration, fps, config, verbose):
    """Animate multiple images into videos."""

    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(log_level)
    logger = logging.getLogger(__name__)

    try:
        cfg = load_config(config)
        ensure_directories(cfg)

        logger.info("Initializing generator...")
        generator = ImageToVideoGenerator(config=cfg)

        logger.info(f"Processing images from: {input_dir}")
        output_paths = generator.batch_generate(
            image_dir=input_dir,
            output_dir=output_dir,
            duration=duration,
            fps=fps,
        )

        click.echo(f"\n✓ Generated {len(output_paths)} videos in: {output_dir}")

    except Exception as e:
        logger.error(f"Batch animation failed: {e}")
        click.echo(f"✗ Error: {e}", err=True)
        raise click.Abort()


# ============================================================================
# COMBINED PIPELINE
# ============================================================================

@cli.command()
@click.option("--prompt", "-p", required=True, help="Text prompt")
@click.option("--output", "-o", default="final_video.mp4", help="Output video path")
@click.option("--image-steps", default=30, type=int, help="Image generation steps")
@click.option("--video-steps", default=25, type=int, help="Video generation steps")
@click.option("--duration", "-d", default=5, type=float, help="Video duration")
@click.option("--config", default="config.yaml", help="Config file path")
@click.option("--verbose", is_flag=True, help="Verbose output")
def create(prompt, output, image_steps, video_steps, duration, config, verbose):
    """Create video directly from text prompt (prompt → image → video)."""

    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(log_level)
    logger = logging.getLogger(__name__)

    try:
        cfg = load_config(config)
        ensure_directories(cfg)

        # Step 1: Generate image from prompt
        logger.info("Step 1: Generating image from prompt...")
        text_gen = TextToImageGenerator(config=cfg)
        temp_image = "/tmp/temp_generated.png"
        text_gen.generate(
            prompt=prompt,
            output_path=temp_image,
            num_inference_steps=image_steps,
        )

        # Step 2: Generate video from image
        logger.info("Step 2: Animating image to video...")
        video_gen = ImageToVideoGenerator(config=cfg)
        video_gen.generate(
            image_path=temp_image,
            output_path=output,
            duration=duration,
            num_inference_steps=video_steps,
        )

        click.echo(f"\n✓ Video created and saved to: {output}")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        click.echo(f"✗ Error: {e}", err=True)
        raise click.Abort()


# ============================================================================
# UTILITY COMMANDS
# ============================================================================

@cli.command()
def info():
    """Display system information and model status."""
    import torch
    from xckrab import __version__

    click.echo(f"\n{'='*50}")
    click.echo(f"xckrab v{__version__}")
    click.echo(f"{'='*50}")
    click.echo(f"\nSystem Information:")
    click.echo(f"  PyTorch version: {torch.__version__}")
    click.echo(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        click.echo(f"  GPU: {torch.cuda.get_device_name(0)}")
        click.echo(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    click.echo(f"  MPS available: {torch.backends.mps.is_available()}")
    
    click.echo(f"\nRecommendations:")
    if not torch.cuda.is_available():
        click.echo("  • No GPU detected - generation will be slower on CPU")
        click.echo("  • Consider using Google Colab with GPU for faster processing")
    click.echo(f"\n{'='*50}\n")


if __name__ == "__main__":
    cli()