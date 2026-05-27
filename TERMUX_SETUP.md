# ckrab on Termux (Android)

Guide to running ckrab on Android devices using Termux.

## Prerequisites

- Android device with at least 2GB RAM
- Termux app installed from F-Droid (recommended) or Play Store
- ~5GB free storage space for models

## Installation

### 1. Update Termux

```bash
apt update && apt upgrade
```

### 2. Install Python and System Dependencies

```bash
apt install python pip git clang make libjpeg-turbo-dev zlib-dev
```

### 3. Clone the Repository

```bash
git clone https://github.com/beredaburna-wq/ckrab.git
cd ckrab
```

### 4. Install Python Dependencies

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

Note: On Termux with limited resources, this may take a while. Consider installing packages one by one if you encounter memory issues.

## Memory Management

For best results on mobile devices:

1. **Close background apps** before running generation
2. **Use CPU mode** (default) or reduce `batch_size` in config
3. **Reduce video resolution** in config to 640x360 or lower
4. **Use fewer inference steps** (e.g., 15 instead of 25)

Edit `config.yaml`:

```yaml
video:
  resolution: "640x360"
  quality: "medium"

generation:
  num_frames: 12
  num_inference_steps: 15

performance:
  batch_size: 1
  enable_attention_slicing: true
```

## Usage

### Basic Usage

```bash
python cli.py generate --input image.jpg --output video.mp4 --duration 3
```

### Using CLI

```bash
# Install as command
pip install -e .

# Then use anywhere
ckrab generate --input image.jpg --output video.mp4
```

### Using Python API

```python
from ckrab import ImageToVideoGenerator

generator = ImageToVideoGenerator()
generator.generate(
    image_path="image.jpg",
    output_path="video.mp4",
    duration=3,
    fps=24,
    num_inference_steps=15  # Lower on mobile
)
```

## File Transfer

### Using Termux Shared Storage

```bash
# Copy images to shared storage
cp /sdcard/DCIM/image.jpg ~/storage/pictures/

# Generate video
python cli.py generate --input ~/storage/pictures/image.jpg --output ~/storage/movies/video.mp4

# Share the video
# Navigate to /sdcard/Movies/ on your phone
```

## Troubleshooting

### "Not enough memory" Error

1. Close all background apps
2. Reduce batch size: `batch_size: 1` in config
3. Use fewer inference steps
4. Clear Termux cache: `rm -rf ~/.cache/huggingface/`

### Model Download Issues

If model download is interrupted:

```bash
# Clear cache and retry
rm -rf ~/.cache/huggingface/
python cli.py generate --input image.jpg --output video.mp4
```

### Permission Denied

Grant Termux storage permission:
1. Open Termux settings
2. Go to Permissions
3. Enable Storage access

Then run:
```bash
termux-setup-storage
```

### Slow Performance

CPU inference on mobile is intentionally slow. To speed up:

1. **Use Google Colab** with GPU for faster processing
2. **Reduce complexity** in config (lower steps, resolution, etc.)
3. **Pre-process images** to smaller sizes before generation

## Recommended Settings for Mobile

```yaml
video:
  resolution: "512x288"
  fps: 24
  quality: "low"

generation:
  num_frames: 8
  num_inference_steps: 10
  guidance_scale: 5.0

performance:
  batch_size: 1
  enable_attention_slicing: true
  cpu_offload: true
```

## Performance Tips

- First generation may take 5-15 minutes as models download and cache
- Subsequent generations will be faster
- JPEG compression of output video reduces file size
- Use lower resolution inputs (512x288) for faster generation

## Resources

- **Termux Wiki**: https://wiki.termux.com/
- **Hugging Face Models**: https://huggingface.co/models
- **PyTorch on Mobile**: https://pytorch.org/mobile/

## Support

For issues specific to Termux setup, check:
- Termux troubleshooting guide
- GitHub issues on this repository