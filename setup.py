#!/usr/bin/env python3
"""
Setup script for HSI_PSI - Hyperspectral Image Processing Library
"""

from setuptools import setup, find_packages
import os

# Read the README file for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements from requirements.txt
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        lines = fh.readlines()
    
    # Parse requirements, skip comments and empty lines
    requirements = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith("#"):
            requirements.append(line)
    
    return requirements

# Get version from hsi_psi/__init__.py
def get_version():
    version_file = os.path.join(os.path.dirname(__file__), "hsi_psi", "__init__.py")
    with open(version_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip('"').strip("'")
    return "2.0.0"

def print_package_info():
    """Print HSI_PSI package information"""
    print("HSI_PSI - Advanced Hyperspectral Image Analysis Library")
    print("Optimized for close-range vegetation monitoring")

setup(
    name="hsi-psi",
    version=get_version(),
    author="Ivan Kashkan",
    author_email="kashkan@psi.cz",
    description="Advanced Python library for hyperspectral image analysis with intelligent wavelength mapping, spectral cropping, and optimized processing pipelines for close-range vegetation monitoring",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/lamewarden/HSI_PSI",
    project_urls={
        "Bug Reports": "https://github.com/lamewarden/HSI_PSI/issues",
        "Source": "https://github.com/lamewarden/HSI_PSI",
        "Documentation": "https://github.com/lamewarden/HSI_PSI#readme",
        "Examples": "https://github.com/lamewarden/HSI_PSI/tree/main/examples",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Agriculture",
        "Natural Language :: English",
    ],
    python_requires=">=3.7",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=21.0.0",
            "flake8>=3.8.0",
            "twine>=3.4.0",
            "wheel>=0.36.0",
            "jupyter>=1.0.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
            "myst-parser>=0.17.0",
        ],
        "analysis": [
            "seaborn>=0.11.0",
            "plotly>=5.0.0",
            "ipywidgets>=7.6.0",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="hyperspectral imaging, remote sensing, spectral analysis, image processing, VNIR, SWIR, vegetation monitoring, agriculture, spectral cropping, wavelength mapping, close-range sensing",
    entry_points={
        "console_scripts": [
            "hsi-psi-info=hsi_psi.utils:print_package_info",
        ],
    },
)
