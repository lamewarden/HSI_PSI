"""
HSI_PSI - Hyperspectral Image Processing for PSI VNIR/SWIR/MSC cameras

A comprehensive library for hyperspectral image loading, preprocessing, analysis, and visualization.

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
