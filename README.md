# ckrab - Image to Video Generator

An uncensored image-to-video generator built with Python. Perfect for running on mobile devices via Termux or GitHub Codespaces.

## Features

- Convert static images to dynamic videos
- No content filters or restrictions
- Lightweight and mobile-friendly
- Works with Termux on Android
- CLI and Python API support

## Requirements

- Python 3.8+
- pip or pip3
- 2GB+ RAM (recommended)
- 1GB+ storage for models

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

### Using Python API

```python
from ckrab import ImageToVideoGenerator

# Initialize generator
generator = ImageToVideoGenerator()

# Generate video from image
generator.generate(
    image_path="input.jpg",
    output_path="output.mp4",
    duration=5,  # seconds
    fps=30
)
```

### Using CLI

```bash
# Single image
python cli.py generate --input image.jpg --output video.mp4 --duration 5

# Batch process
python cli.py batch --input-dir ./images --output-dir ./videos

# View system info
python cli.py info
```

## Configuration

Edit `config.yaml` to customize:
- Model selection
- Video resolution
- Frame rate
- Processing parameters
- Performance settings for mobile devices

## Mobile-Specific Notes

For best performance on Termux/Android:
1. Close background apps before generation
2. Use CPU mode (default)
3. Reduce resolution in config
4. Lower inference steps for faster generation

See `TERMUX_SETUP.md` for detailed setup instructions.

## License

MIT License - Free for personal and commercial use

## Disclaimer

This tool is for educational purposes. Users are responsible for the content they generate and must comply with local laws and platform terms of service.