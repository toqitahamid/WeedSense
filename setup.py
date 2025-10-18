"""
WeedSense: Multi-Task Learning for Weed Analysis
Setup script for installation
"""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="weedsense",
    version="1.0.0",
    author="Toqi Tahamid Sarker, Khaled R Ahmed, Taminul Islam, Cristiana Bernardi Rankrape, Karla Gage",
    author_email="toqitahamid.sarker@siu.edu",
    description="Multi-Task Learning for Weed Segmentation, Height Estimation, and Growth Stage Classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/toqitahamid/weedsense",
    project_urls={
        "Bug Tracker": "https://github.com/toqitahamid/weedsense/issues",
        "Documentation": "https://weedsense.github.io",
        "Paper": "https://arxiv.org/abs/2508.14486",
        "Base Library": "https://github.com/sithu31296/semantic-segmentation",
    },
    packages=find_packages(exclude=["tools", "configs", "docs", "*.egg-info"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.7.0",
            "isort>=5.12.0",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
        ],
        "train": [
            "wandb>=0.15.0",
            "tensorboard>=2.13.0",
        ],
        "export": [
            "onnx>=1.14.0",
            "onnxruntime>=1.15.0",
            "thop>=0.1.1",
        ],
    },
    include_package_data=True,
    package_data={
        "weedsense": ["*.yaml", "*.yml"],
    },
    zip_safe=False,
    keywords=[
        "deep learning",
        "computer vision",
        "semantic segmentation",
        "multi-task learning",
        "agriculture",
        "weed detection",
        "plant phenotyping",
        "growth stage classification",
        "height estimation",
        "pytorch",
    ],
)

