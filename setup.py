#!/usr/bin/env python3
"""
Setup script for nanoNAS: Neural Architecture Search Made Simple
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="nanonas",
    version="1.0.0",
    author="AutoML Research Team",
    author_email="research@nanonas.ai",
    description="Neural Architecture Search Made Simple: Educational and Production-Ready NAS",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/Ratz-innovator/AutoMl-CodeGen",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
            "pre-commit>=2.20.0",
        ],
        "docs": [
            "mkdocs>=1.4.0",
            "mkdocs-material>=8.5.0",
            "mkdocstrings[python]>=0.19.0",
        ],
        "viz": [
            "tensorboard>=2.10.0",
            "graphviz>=0.20.0",
            "networkx>=2.8.0",
            "plotly>=5.11.0",
        ],
        "benchmarks": [
            "torchvision>=0.15.0",
            "timm>=0.6.0",
            "thop>=0.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "nanonas=nanonas.cli:main",
            "nanonas-benchmark=nanonas.benchmarks.cli:main",
            "nanonas-viz=nanonas.visualization.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "nanonas": [
            "configs/*.yaml",
            "configs/search_spaces/*.yaml",
            "configs/benchmarks/*.yaml",
        ],
    },
) 