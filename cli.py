"""
Command-line interface for ckrab.
"""

import click
import logging
from ckrab import ImageToVideoGenerator
from ckrab.utils import load_config, setup_logging, ensure_directories


@click.group()
def cli():
    """ckrab - Image to Video Generator"""
    pass


@cli.command()
@click.option("--input", "-i", required=True, help="Input image path")
@click.option("--output", "-o", default="output.mp4", help="Output video path")
@click.option("--duration", "-d", default=3, type=float, help="Video duration in seconds")
@click.option("--fps", default=30, type=int, help="Frames per second")
@click.option("--steps", default=25, type=int, help="Number of inference steps")
@click.option("--guidance", default=7.5, type=float, help="Guidance scale")
@click.option("--seed", default=None, type=int, help="Random seed")
@click.option("--device", default="auto", help="Device (cpu, cuda, mps, auto)")
@click.option("--config", default="config.yaml", help="Config file path")
@click.option("--verbose", is_flag=True, help="Verbose output")
def generate(input, output, duration, fps, steps, guidance, seed, device, config, verbose):
    """Generate a video from an image."""

    # Setup logging
    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(log_level)
    logger = logging.getLogger(__name__)

    try:
        # Load configuration
        cfg = load_config(config)
        ensure_directories(cfg)

        # Initialize generator
        logger.info("Initializing generator...")
        generator = ImageToVideoGenerator(config=cfg, device=device if device != "auto" else None)

        # Generate video
        logger.info(f"Generating video: {input} -> {output}")
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
        logger.error(f"Generation failed: {e}")
        click.echo(f"✗ Error: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.option("--input-dir", "-i", required=True, help="Input directory with images")
@click.option("--output-dir", "-o", default="outputs", help="Output directory for videos")
@click.option("--duration", "-d", default=3, type=float, help="Video duration in seconds")
@click.option("--fps", default=30, type=int, help="Frames per second")
@click.option("--config", default="config.yaml", help="Config file path")
@click.option("--verbose", is_flag=True, help="Verbose output")
def batch(input_dir, output_dir, duration, fps, config, verbose):
    """Generate videos from multiple images."""

    # Setup logging
    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(log_level)
    logger = logging.getLogger(__name__)

    try:
        # Load configuration
        cfg = load_config(config)
        ensure_directories(cfg)

        # Initialize generator
        logger.info("Initializing generator...")
        generator = ImageToVideoGenerator(config=cfg)

        # Generate videos
        logger.info(f"Processing images from: {input_dir}")
        output_paths = generator.batch_generate(
            image_dir=input_dir,
            output_dir=output_dir,
            duration=duration,
            fps=fps,
        )

        click.echo(f"\n✓ Generated {len(output_paths)} videos in: {output_dir}")

    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        click.echo(f"✗ Error: {e}", err=True)
        raise click.Abort()


@cli.command()
def info():
    """Display system information and model status."""
    import torch
    from ckrab import __version__

    click.echo(f"\n{'='*50}")
    click.echo(f"ckrab v{__version__}")
    click.echo(f"{'='*50}")
    click.echo(f"\nPython Information:")
    click.echo(f"  PyTorch version: {torch.__version__}")
    click.echo(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        click.echo(f"  GPU: {torch.cuda.get_device_name(0)}")
        click.echo(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    click.echo(f"  MPS available: {torch.backends.mps.is_available()}")
    click.echo(f"\nRecommendations:")
    if not torch.cuda.is_available():
        click.echo("  • No CUDA GPU detected - generation will be slower on CPU")
        click.echo("  • Consider using Google Colab with GPU for faster processing")
    click.echo(f"\n{'='*50}\n")


if __name__ == "__main__":
    cli()