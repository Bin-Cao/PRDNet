#!/usr/bin/env python3
"""
Setup script for PRDNET: Pseudo-particle Ray Diffraction Network

Author: Bin Cao (bcao686@connect.hkust-gz.edu.cn)
Affiliations:
- Hong Kong University of Science and Technology (Guangzhou)
- City University of Hong Kong
"""

import os
import sys
from pathlib import Path
from setuptools import setup, find_packages

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements from requirements.txt
def read_requirements():
    """Read requirements from requirements.txt file."""
    requirements_path = this_directory / "requirements.txt"
    if requirements_path.exists():
        with open(requirements_path, 'r', encoding='utf-8') as f:
            requirements = []
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if line and not line.startswith('#'):
                    # Remove inline comments
                    if '#' in line:
                        line = line.split('#')[0].strip()
                    requirements.append(line)
            return requirements
    return []

# Version information
def get_version():
    """Get version from __init__.py or git."""
    try:
        # Try to get version from git
        import subprocess
        version = subprocess.check_output(
            ["git", "describe", "--tags", "--always"], 
            stderr=subprocess.DEVNULL
        ).decode().strip()
        if version.startswith('v'):
            version = version[1:]
        return version
    except:
        # Fallback version
        return "0.1.0"

# Package metadata
setup(
    name="prdnet",
    version=get_version(),
    author="Bin Cao",
    author_email="bcao686@connect.hkust-gz.edu.cn",
    description="A physics-informed neural network that combines graph neural networks with pseudo-particle-ray diffraction for crystal property prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/PRDNET",  # Update with actual repository URL
    project_urls={
        "Bug Reports": "https://github.com/your-username/PRDNET/issues",
        "Source": "https://github.com/your-username/PRDNET",
        "Documentation": "https://github.com/your-username/PRDNET#readme",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
            "pre-commit>=2.20.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.18.0",
        ],
        "viz": [
            "seaborn>=0.11.0",
            "plotly>=5.0.0",
        ],
        "distributed": [
            "accelerate>=0.20.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "prdnet-train=trainer:main",
        ],
    },
    include_package_data=True,
    package_data={
        "prdnet": [
            "*.json",
            "*.yaml",
            "*.yml",
        ],
    },
    keywords=[
        "machine learning",
        "deep learning",
        "graph neural networks",
        "materials science",
        "crystal structure",
        "diffraction",
        "physics-informed neural networks",
        "property prediction",
        "pytorch",
        "pytorch geometric",
    ],
    zip_safe=False,
)
