# ckrab - Image to Video Generator

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![License](https://img.shields.io/badge/License-MIT-green) ![Status](https://img.shields.io/badge/Status-Active-brightgreen)

An image-to-video generator built with Python. Perfect for running on mobile devices via Termux or GitHub Codespaces. Create dynamic videos from static images with full control and no artificial restrictions.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Performance & Benchmarks](#performance--benchmarks)
- [Mobile-Specific Setup](#mobile-specific-setup)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)
- [Contributing](#contributing)
- [License](#license)
- [Disclaimer](#disclaimer)

## Features

- ✨ Convert static images to dynamic videos
- 🎯 Full control over generation parameters
- 📱 Lightweight and mobile-friendly (runs on Termux/Android)
- ⚡ Works with GitHub Codespaces
- 🐍 CLI and Python API support
- 🔧 Highly configurable for different hardware
- 🖥️ CPU-based processing (no GPU required)

## Requirements

- **Python**: 3.8 or higher
- **RAM**: 2GB+ (recommended 4GB+ for faster generation)
- **Storage**: 1GB+ for models and outputs
- **Package Manager**: pip or pip3
- **Internet**: Required for initial model download (~500MB-1GB depending on model)

### Supported Platforms

- Termux on Android
- Linux (Ubuntu, Debian, etc.)
- macOS (Intel & Apple Silicon)
- GitHub Codespaces
- Windows (via WSL2)

## Installation

### On Termux (Android)

```bash
# Install Python and dependencies
pkg install python pip git

# Clone the repository
git clone https://github.com/beredaburna-wq/ckrab.git
cd ckrab

# Install Python requirements
pip install -r requirements.txt
```

For detailed Termux setup, see [TERMUX_SETUP.md](TERMUX_SETUP.md).

### On Linux/macOS

```bash
# Clone the repository
git clone https://github.com/beredaburna-wq/ckrab.git
cd ckrab

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python requirements
pip install -r requirements.txt
```

### On GitHub Codespaces

```bash
git clone https://github.com/beredaburna-wq/ckrab.git
cd ckrab
pip install -r requirements.txt
python cli.py generate --input sample.jpg --output video.mp4
```

## Quick Start

### Using Python API

```python
from ckrab import ImageToVideoGenerator

# Initialize generator with default settings
generator = ImageToVideoGenerator()

# Generate video from image
generator.generate(
    image_path="input.jpg",
    output_path="output.mp4",
    duration=5,          # seconds
    fps=30,              # frames per second
    inference_steps=50   # higher = better quality, slower
)

# Advanced usage with custom configuration
generator.generate(
    image_path="input.jpg",
    output_path="output.mp4",
    duration=8,
    fps=24,
    inference_steps=75,
    motion_intensity=0.7,  # 0.0-1.0
    device="cpu"           # or "cuda" if GPU available
)
```

### Using CLI

```bash
# Single image to video
python cli.py generate --input image.jpg --output video.mp4 --duration 5

# Batch process multiple images
python cli.py batch --input-dir ./images --output-dir ./videos --duration 5

# View system information and model status
python cli.py info

# Show help and all available options
python cli.py generate --help
```

## Configuration

Edit `config.yaml` to customize default behavior:

```yaml
# Model Configuration
model:
  name: "stable-video-diffusion"  # Model to use
  device: "cpu"                    # cpu or cuda
  dtype: "float32"                 # float32 or float16

# Video Settings
video:
  resolution: [576, 960]           # [height, width] - reduce for mobile
  fps: 30                          # frames per second
  duration: 5                       # default seconds
  
# Performance
inference:
  steps: 50                        # 25-100: higher = better quality
  guidance_scale: 7.5              # 1.0-20.0
  num_frames: 25                   # frames to generate

# Mobile Optimization (Termux)
mobile:
  cpu_only: true
  max_memory_percent: 80           # Don't use more than 80% RAM
  enable_memory_optimization: true
```

### Configuration Precedence

1. Command-line arguments (highest priority)
2. Environment variables
3. `config.yaml`
4. Built-in defaults (lowest priority)

## Performance & Benchmarks

### Expected Generation Times

| Device | Duration | FPS | Inference Steps | Time |
|--------|----------|-----|-----------------|------|
| Desktop CPU (8 cores) | 5s | 30 | 50 | ~2-3 min |
| Laptop CPU (4 cores) | 5s | 30 | 50 | ~4-6 min |
| Termux (Android) | 5s | 30 | 50 | ~8-15 min |
| Desktop GPU (RTX3080) | 5s | 30 | 50 | ~20-30 sec |

**Tip**: For faster results on mobile, reduce inference steps to 25-30 or lower resolution.

## Mobile-Specific Setup

### Termux Performance Optimization

1. **Close background apps** - Free up RAM before generation
2. **Use CPU mode** - Default and safest for Termux (no CUDA support)
3. **Reduce resolution** - Set to `[360, 640]` in config for faster processing
4. **Lower inference steps** - Start with 25-30 for testing, increase for quality
5. **Monitor storage** - Models are large; ensure 2GB+ free space
6. **Use battery saver** - Run long generations on charger

### Increasing Termux RAM Limit

Edit `~/.termux/termux.properties`:

```bash
# Allow Termux to use more RAM (be careful)
allow_external_apps=false
```

See [TERMUX_SETUP.md](TERMUX_SETUP.md) for detailed instructions.

## Troubleshooting

### Generation is Very Slow

**Possible causes:**
- Too many inference steps (start with 25-30)
- Running on Termux with limited RAM
- High resolution setting
- Background processes consuming resources

**Solutions:**
- Reduce `inference_steps` in config or CLI
- Close other apps
- Lower video resolution
- Use `--fast-mode` flag for quick testing

### Out of Memory Error

**Error message**: `MemoryError` or `CUDA out of memory`

**Solutions:**
```bash
# Use CPU instead of GPU
python cli.py generate --input image.jpg --device cpu

# Enable memory optimization
python cli.py generate --input image.jpg --optimize-memory

# Reduce resolution
# Edit config.yaml and lower resolution setting
```

### Model Download Fails

**Error message**: `Connection error` or `Failed to download model`

**Solutions:**
- Check internet connection
- Try again (sometimes servers are busy)
- Use manual model download:
  ```bash
  python cli.py download-models --model stable-video-diffusion
  ```

### Black or Corrupted Output Video

**Possible causes:**
- Corrupted input image
- Insufficient inference steps
- Model not fully downloaded

**Solutions:**
- Verify input image is valid: `file input.jpg`
- Increase inference steps to 50+
- Re-download models: `python cli.py download-models --force`
- Check available disk space

### Termux-Specific Issues

See [TERMUX_SETUP.md](TERMUX_SETUP.md#troubleshooting) for Termux-specific troubleshooting.

## FAQ

**Q: What image formats are supported?**
A: JPG, PNG, WebP, and BMP. Ensure images are RGB (not grayscale). Recommended: 512x768 to 1024x1536 pixels.

**Q: How long does generation take?**
A: 5-30 seconds on GPU, 2-15 minutes on CPU depending on hardware and settings. See [Performance & Benchmarks](#performance--benchmarks).

**Q: Can I use GPU acceleration?**
A: Yes, if you have NVIDIA GPU with CUDA support. Install `torch[cuda]` variant. Termux/Android doesn't support GPU.

**Q: Why is my video jerky or low quality?**
A: Increase `inference_steps` (50+) and `fps` in config. Higher values = better quality, longer generation.

**Q: Can I batch process videos?**
A: Yes, use `python cli.py batch --input-dir ./images --output-dir ./videos`

**Q: Is this legal?**
A: Yes. The tool itself is legal (MIT license). You're responsible for respecting copyright, privacy, and local laws when generating and sharing content.

**Q: How much storage do I need?**
A: Models: ~500MB-1GB. Cache: ~100-500MB per generation. Output videos: varies by duration/resolution. Have at least 2GB free.

**Q: Can I use this commercially?**
A: Yes, MIT license allows commercial use. Respect copyright of input images and comply with model licenses.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

MIT License - Free for personal and commercial use. See [LICENSE](LICENSE) file for details.

## Disclaimer

**Use Responsibly**: This tool generates videos from images for legitimate purposes including education, research, and creative projects. Users are responsible for ensuring their use complies with:

- Local laws and regulations in your jurisdiction
- Copyright and intellectual property rights of input images
- Platform terms of service where content is shared or published
- Privacy rights of any individuals depicted in images

The creators are not responsible for misuse of this tool. Always verify you have rights to use input images and respect copyright laws.

---

**Need help?** Open an [issue](https://github.com/beredaburna-wq/ckrab/issues) on GitHub or check [TERMUX_SETUP.md](TERMUX_SETUP.md) for detailed setup guidance.
