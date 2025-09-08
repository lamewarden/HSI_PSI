"""
HSI_PSI - Hyperspectral Image Analysis Library

HSI_PSI is designed with simplicity and reproducibility in mind. It uses advanced 
hyperspectral data manipulation methods, packed together into self-explanatory 
functions. HSI_PSI tools are based on everyday real-life experience working with 
HSI data as individual files or extensive image batches.

The package consists of three core modules:
- core: Foundation classes for data handling and visualization
- preprocessing: Advanced pipelines for data preparation and correction  
- utils: Helper functions and utilities for file operations

Developed for PSI VNIR/SWIR/MSC cameras and compatible hyperspectral formats.
"""

# Import main classes and functions for easy access
from .core import (
    HS_image, 
    MS_image, 
    get_hdr_images, 
    get_polygon_masks_from_json, 
    standardize_image, 
    convert_header_to_envi
)
from .preprocessing import HS_preprocessor


# Version info
__version__ = "1.0.0"
__author__ = "Ivan Kashkan"
__email__ = "kashkan@psi.cz"

# Define what gets imported with "from HSI_PSI import *"
__all__ = [
    # Core classes
    "HS_image",
    "MS_image", 
    "HS_preprocessor",
    
    # Utility functions
    "get_hdr_images",
    "get_polygon_masks_from_json", 
    "standardize_image",
    "convert_header_to_envi"


]
