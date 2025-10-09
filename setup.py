#!/usr/bin/env python3
"""
Setup script for JAX NoProp implementation.
"""

from setuptools import setup, find_packages

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="jax-noprop",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="JAX/Flax implementation of NoProp: Training Neural Networks without Back-propagation or Forward-propagation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/jax-noprop",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/jax-noprop/issues",
        "Documentation": "https://github.com/yourusername/jax-noprop#readme",
        "Source Code": "https://github.com/yourusername/jax-noprop",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
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
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
        ],
    },
    keywords="jax, flax, noprop, neural-networks, machine-learning, deep-learning, pytorch-alternative",
    include_package_data=True,
    zip_safe=False,
)
