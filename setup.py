"""Setup script for ckrab package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="ckrab",
    version="0.1.0",
    author="beredaburna-wq",
    description="Uncensored image-to-video generator optimized for mobile devices",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/beredaburna-wq/ckrab",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Multimedia :: Video",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "opencv-python>=4.5.0",
        "Pillow>=8.0.0",
        "requests>=2.26.0",
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "diffusers>=0.21.0",
        "transformers>=4.30.0",
        "moviepy>=1.0.0",
        "ffmpeg-python>=0.2.1",
        "tqdm>=4.62.0",
        "pyyaml>=5.4.0",
        "click>=8.0.0",
    ],
    entry_points={
        "console_scripts": [
            "ckrab=cli:cli",
        ],
    },
)