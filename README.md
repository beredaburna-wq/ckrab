# xckrab - Text-to-Image-to-Video Generator

An uncensored AI generator that creates images from text prompts and animates them into videos. Perfect for running on mobile devices via Termux or GitHub Codespaces.

## Features

- **Text-to-Image**: Generate unlimited unique images from text descriptions (no content filters)
- **Image-to-Video**: Convert static images into smooth animated videos
- **Two-Step Pipeline**: Generate → Animate workflow
- Lightweight and mobile-friendly
- Works with Termux on Android
- CLI and Python API support

## Requirements

- Python 3.8+
- pip or pip3
- 3GB+ RAM (recommended for image generation)
- 2GB+ storage for models

## Installation

### On Termux (Android)

```bash
# Install Python and dependencies
pkg install python pip

# Clone the repository
git clone https://github.com/beredaburna-wq/ckrab.git
cd ckrab

# Install Python requirements
pip install -r requirements.txt
```

### On Linux/macOS

```bash
git clone https://github.com/beredaburna-wq/ckrab.git
cd ckrab
pip install -r requirements.txt
```

## Quick Start

### Step 1: Generate Image from Text Prompt

```python
from xckrab import TextToImageGenerator

generator = TextToImageGenerator()
image_path = generator.generate(
    prompt="a futuristic city at sunset, neon lights, cyberpunk style",
    output_path="generated_image.png",
    num_inference_steps=30
)
```

### Step 2: Animate Image to Video

```python
from xckrab import ImageToVideoGenerator

video_gen = ImageToVideoGenerator()
video_path = video_gen.generate(
    image_path="generated_image.png",
    output_path="video.mp4",
    duration=5,
    fps=30
)
```

## CLI Usage

### Generate Images

```bash
# Single prompt
python cli.py imagine --prompt "a beautiful landscape" --output image.png

# Batch generate from file
python cli.py imagine-batch --prompts prompts.txt --output-dir ./images
```

### Animate Videos

```bash
# Single image
python cli.py animate --input image.png --output video.mp4 --duration 5

# Batch animate multiple images
python cli.py animate-batch --input-dir ./images --output-dir ./videos
```

### Full Pipeline (Generate + Animate)

```bash
# Create image and video in one command
python cli.py create --prompt "a dragon flying over mountains" --output video.mp4 --duration 5
```

## Configuration

Edit `config.yaml` to customize:

```yaml
# Image Generation
text_to_image:
  model: "stable-diffusion-3-medium"
  resolution: "768x512"
  num_inference_steps: 30
  guidance_scale: 7.5

# Video Animation
image_to_video:
  duration: 5
  fps: 30
  num_inference_steps: 25

# Performance (for mobile)
performance:
  batch_size: 1
  enable_attention_slicing: true
  cpu_offload: true
```

## Workflow

### Separate Workflow (Recommended for Mobile)

```
Text Prompt
    ↓
[Text-to-Image Generator]
    ↓
Generated Image (PNG)
    ↓
[Image-to-Video Generator]
    ↓
Animated Video (MP4)
```

This allows you to:
- Generate multiple images and choose your favorites
- Reuse images to create different videos
- Run each step separately for better memory management

### Combined Workflow (Faster)

```
Text Prompt → Image → Video (all in one)
```

## Mobile-Specific Notes

For best performance on Termux/Android:
1. Close background apps before generation
2. Use CPU mode (default)
3. Start with lower resolution images (512x384)
4. Use fewer inference steps for faster generation

See `TERMUX_SETUP.md` for detailed setup instructions.

## Model Options

**Text-to-Image Models:**
- `stable-diffusion-3` - High quality, slow
- `stable-diffusion-2.1` - Good balance
- `flux-1-dev` - Grok-like quality (experimental)

**Image-to-Video Models:**
- `stable-video-diffusion-img2vid` - Smooth motion
- `damo-vidar` - Alternative (if available)

## License

MIT License - Free for personal and commercial use

## Disclaimer

This tool is for educational purposes. Users are responsible for the content they generate and must comply with local laws and platform terms of service.