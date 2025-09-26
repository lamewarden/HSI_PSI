"""
HSI_PSI - Advanced Hyperspectral Image Analysis Library v0.3.0

HSI_PSI is designed with simplicity, flexibility, and reproducibility in mind. It uses 
advanced hyperspectral data manipulation methods with intelligent wavelength mapping, 
spectral cropping capabilities, and optimized processing pipelines. HSI_PSI tools are 
based on real-world experience working with HSI data in close-range vegetation monitoring.

New Features in v0.3.0:
- Generic transformer class supporting both PCA and MNF (Minimum Noise Fraction)
- Integration with pysptools for professional-grade MNF implementation
- Enhanced dimensionality reduction with robust noise estimation
- Advanced normalization methods: SNV, RNV (Robust Normal Variate), and L2
- Improved preprocessing pipeline with consolidated normalization checks
- Extended configuration templates with new normalization options

Features from v0.2.0:
- Spectral range cropping with automatic metadata updates
- Intelligent wavelength mapping between different sensor configurations  
- Advanced noise analysis and spectral quality assessment
- Optimized processing pipeline order for maximum data quality
- Enhanced reference teflon library creation with automatic adaptation
- Wavelength-based calibration mapping for mixed sensor datasets
- Multi-spectrum plotting functionality (plot_spectra)
- Advanced visualization tools (vis_clust_2D, plot_confusion_matrix)
- Package information display (print_package_info)

The package consists of four enhanced core modules:
- core: Foundation classes with spectral cropping and wavelength mapping
- preprocessing: Optimized pipelines with intelligent wavelength handling
- dim_red: Dimensionality reduction with PCA and MNF (Minimum Noise Fraction) transformations
- utils: Enhanced utilities with noise analysis, visualization, and data extraction functions

Optimized for PSI VNIR/SWIR/MSC cameras and close-range vegetation applications.
"""

# Import main classes and functions for easy access
from .core import (
    HS_image, 
    MS_image, 
    get_polygon_masks_from_json, 
    convert_header_to_envi
)
from .preprocessing import HS_preprocessor
from .dim_red import (
    transformer,
    HS_PCA_transformer  # Backward compatibility alias
)
from .utils import (
    get_hdr_images,
    get_rgb_sample,
    extract_masks_from_hs_image,
    extract_masked_spectra_to_df,
    stretch_image,
    standardize_image,
    create_config_template,
    rank_noisy_bands,
    summarize_noisiest_bands,
    print_package_info,
    plot_spectra,
    vis_clust_2D,
    plot_confusion_matrix
)


# Version info
__version__ = "0.3.0"
__author__ = "Ivan Kashkan, HSI_PSI Development Team"
__email__ = "kashkan@psi.cz"

# Define what gets imported with "from HSI_PSI import *"
__all__ = [
    # Core classes
    "HS_image",
    "MS_image", 
    "HS_preprocessor",
    
    # Dimensionality reduction classes
    "transformer",
    "HS_PCA_transformer",  # Backward compatibility
    
    # Utility functions
    "get_hdr_images",
    "get_polygon_masks_from_json", 
    "convert_header_to_envi",
    
    # Additional utility functions from utils module
    "get_rgb_sample",
    "extract_masks_from_hs_image",
    "extract_masked_spectra_to_df",
    "stretch_image",
    "standardize_image",
    "create_config_template",
    "rank_noisy_bands",
    "summarize_noisiest_bands",
    "print_package_info",
    "plot_spectra",
    "vis_clust_2D",
    "plot_confusion_matrix"
]
