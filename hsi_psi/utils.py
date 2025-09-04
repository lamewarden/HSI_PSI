"""
HS_tools.utils - Utility functions for hyperspectral image analysis

Contains helper functions for loading, processing, and visualizing hyperspectral data.
"""

import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, Optional, Union, List, Tuple
from .core import HS_image, MS_image


def get_hdr_images(folder: str, min_rows: int = 1, format: str = 'hdr') -> Dict[str, HS_image]:
    """
    Load all HDR images from a folder as HS_image or MS_image objects.
    
    Args:
        folder: Path to folder containing image files
        min_rows: Minimum number of rows required
        format: File format to look for ('hdr' by default)
        
    Returns:
        Dictionary mapping filenames to HS_image/MS_image objects
        
    Example:
        >>> images = get_hdr_images("/path/to/images/")
        >>> print(f"Loaded {len(images)} images")
    """
    all_images = {}
    
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(f'.{format}'):
                try:
                    file_path = os.path.join(root, file)
                    
                    # Try to load as HS_image first
                    try:
                        img = HS_image(file_path)
                    except Exception:
                        # Fallback to MS_image for multispectral data
                        img = MS_image(file_path)
                    
                    # Check minimum rows requirement
                    if img.rows >= min_rows:
                        all_images[file] = img
                        print(f"✓ Loaded: {file} ({img.rows}x{img.cols}x{img.bands})")
                    else:
                        print(f"⚠ Skipped {file}: insufficient rows ({img.rows} < {min_rows})")
                        
                except Exception as e:
                    print(f"❌ Failed to load {file}: {str(e)}")
    
    return all_images


def get_rgb_sample(image: HS_image, normalize: bool = True, correct: bool = True,
                  show: bool = True, title: str = "RGB Sample", repeat: int = 1,
                  gamma: float = 0.8, axes: bool = False) -> np.ndarray:
    """
    Extract RGB representation from hyperspectral image.
    
    Args:
        image: HS_image object
        normalize: Whether to normalize RGB values
        correct: Whether to apply gamma correction
        show: Whether to display the image
        title: Title for the plot
        repeat: Repeat factor for stretching
        gamma: Gamma correction factor
        axes: Whether to show axes
        
    Returns:
        RGB array (height, width, 3)
        
    Example:
        >>> rgb = get_rgb_sample(hs_image, show=True)
        >>> print(f"RGB shape: {rgb.shape}")
    """
    # Get RGB band indices (closest to 670nm, 560nm, 470nm)
    r_idx, g_idx, b_idx = image.get_rgb_bands()
    
    # Extract RGB bands
    red_band = image.img[:, :, r_idx]
    green_band = image.img[:, :, g_idx]
    blue_band = image.img[:, :, b_idx]
    
    # Stack RGB bands
    rgb_image = np.stack([red_band, green_band, blue_band], axis=2)
    
    # Handle SNV-normalized data (negative values)
    has_negative = np.any(rgb_image < 0)
    if has_negative:
        # Shift and scale for SNV data
        for i in range(3):
            band = rgb_image[:, :, i]
            band_min, band_max = np.percentile(band, [2, 98])
            rgb_image[:, :, i] = np.clip((band - band_min) / (band_max - band_min), 0, 1)
    else:
        # Standard normalization for regular data
        if normalize:
            # Use percentile-based normalization for robustness
            for i in range(3):
                band = rgb_image[:, :, i]
                band_max = np.percentile(band, 99)
                if band_max > 0:
                    rgb_image[:, :, i] = np.clip(band / band_max, 0, 1)
    
    # Apply gamma correction if requested
    if correct and gamma != 1.0:
        rgb_image = np.power(rgb_image, gamma)
    
    # Repeat for stretching if needed
    if repeat > 1:
        rgb_image = np.repeat(rgb_image, repeat, axis=0)
    
    # Display if requested
    if show:
        plt.figure(figsize=(10, 6))
        plt.imshow(rgb_image)
        plt.title(title, fontsize=14, fontweight='bold')
        
        if not axes:
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return rgb_image


def extract_masks_from_hs_image(hs_image: HS_image, pri_thr: float = -0.1,
                               ndvi_thr: float = 0.2, hbsi_thr: float = -0.6,
                               min_pix_size: int = 2) -> np.ndarray:
    """
    Extract vegetation masks from hyperspectral image using spectral indices.
    
    Args:
        hs_image: HS_image object
        pri_thr: PRI threshold for vegetation detection
        ndvi_thr: NDVI threshold for vegetation detection  
        hbsi_thr: HBSI threshold for vegetation detection
        min_pix_size: Minimum object size in pixels
        
    Returns:
        Binary mask array (height, width, 1)
        
    Example:
        >>> mask = extract_masks_from_hs_image(hs_image, pri_thr=-0.11)
        >>> masked_pixels = np.sum(mask)
    """
    from .spectral_indices import calculate_ndvi, calculate_pri, calculate_hbsi
    from skimage.morphology import remove_small_objects
    
    # Calculate vegetation indices
    ndvi_image = calculate_ndvi(hs_image)
    hbsi_image = calculate_hbsi(hs_image) 
    pri_image = calculate_pri(hs_image)
    
    # Create individual masks
    ndvi_mask = (ndvi_image > ndvi_thr)[:,:,np.newaxis]
    hbsi_mask = (hbsi_image > hbsi_thr)[:,:,np.newaxis]
    pri_mask = (pri_image < pri_thr)[:,:,np.newaxis]
    
    # Combine masks (vegetation = low PRI AND low NDVI AND low HBSI)
    combined_mask = pri_mask.astype(bool) & ~ndvi_mask.astype(bool) & ~hbsi_mask.astype(bool)
    
    # Remove small objects
    mask_2d = remove_small_objects(combined_mask[:,:,0], min_size=min_pix_size)
    
    # Convert back to 3D uint8 array
    final_mask = mask_2d.astype(np.uint8)[:,:,np.newaxis]
    
    return final_mask


def extract_masked_spectra_to_df(processed_images_dict: Dict[str, HS_image],
                                save_path: Optional[str] = None) -> pd.DataFrame:
    """
    Extract spectra from masked pixels and convert to pandas DataFrame.
    
    Args:
        processed_images_dict: Dictionary of {filename: HS_image} with masks
        save_path: Optional path to save the DataFrame as CSV
        
    Returns:
        DataFrame with columns for each wavelength plus 'label' column
        
    Example:
        >>> df = extract_masked_spectra_to_df(processed_images)
        >>> print(f"Extracted {len(df)} spectra")
    """
    extracted_spectra = []
    
    for filename, hs_image in processed_images_dict.items():
        if hs_image is not None and hasattr(hs_image, 'mask') and hs_image.mask is not None:
            # Apply mask to the image
            masked_image = hs_image.img * hs_image.mask
            
            # Reshape to 2D: (pixels, bands)
            masked_img_flat = masked_image.reshape(-1, masked_image.shape[2])
            
            # Filter out zero pixels (non-masked pixels)
            masked_img_filtered = masked_img_flat[~np.all(masked_img_flat == 0, axis=1)]
            
            if len(masked_img_filtered) > 0:
                # Create DataFrame with wavelengths as column names
                masked_img_df = pd.DataFrame(masked_img_filtered, columns=hs_image.ind)
                
                # Extract label from filename (format: "X-X-LABEL-X-...")
                try:
                    label = hs_image.name.split("-")[2] if hasattr(hs_image, 'name') else filename.split("-")[2]
                except IndexError:
                    label = "unknown"
                
                masked_img_df['label'] = label
                extracted_spectra.append(masked_img_df)
                
                print(f"Extracted {len(masked_img_filtered)} masked pixels from {filename}")
            else:
                print(f"No masked pixels found in {filename}")
        else:
            print(f"Skipping {filename}: no image or mask data")
    
    if not extracted_spectra:
        print("No spectral data extracted!")
        return pd.DataFrame()
    
    # Concatenate all DataFrames
    concatenated_df = pd.concat(extracted_spectra, axis=0, ignore_index=True)
    
    print(f"Total extracted spectra: {len(concatenated_df)} pixels from {len(extracted_spectra)} images")
    print(f"DataFrame shape: {concatenated_df.shape}")
    print(f"Labels found: {concatenated_df['label'].unique()}")
    
    # Save to file if path provided
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        concatenated_df.to_csv(save_path, index=False)
        print(f"DataFrame saved to: {save_path}")
    
    return concatenated_df


def stretch_image(image: np.ndarray, to_size: int = 400, axis: int = 0) -> np.ndarray:
    """
    Stretch image along specified axis to target size.
    
    Args:
        image: Input image array
        to_size: Target size for the specified axis
        axis: Axis to stretch (0 for rows, 1 for columns)
        
    Returns:
        Stretched image array
    """
    import cv2
    
    if axis == 0:
        # Stretch rows
        current_size = image.shape[0]
        if current_size != to_size:
            scale_factor = to_size / current_size
            new_shape = (image.shape[1], to_size) if len(image.shape) == 2 else (image.shape[1], to_size)
            return cv2.resize(image, new_shape)
    elif axis == 1:
        # Stretch columns  
        current_size = image.shape[1]
        if current_size != to_size:
            scale_factor = to_size / current_size
            new_shape = (to_size, image.shape[0]) if len(image.shape) == 2 else (to_size, image.shape[0])
            return cv2.resize(image, new_shape)
    
    return image


def standardize_image(image: np.ndarray) -> np.ndarray:
    """
    Standardize image to zero mean and unit variance.
    
    Args:
        image: Input image array
        
    Returns:
        Standardized image array
    """
    mean = np.mean(image)
    std = np.std(image)
    
    if std > 0:
        return (image - mean) / std
    else:
        return image - mean


def create_config_template() -> Dict[str, Any]:
    """
    Create a template configuration dictionary for preprocessing pipeline.
    
    Returns:
        Dictionary with default configuration parameters
        
    Example:
        >>> config = create_config_template()
        >>> config['sensor_calibration']['clip_to'] = 15
    """
    template = {
        'sensor_calibration': {
            'clip_to': 10,
            'dark_calibration': False,
            'white_ref_path': None
        },
        'solar_correction': {
            'teflon_edge_coord': [-10, -3],
            'smooth_window': 35
        },
        'spectral_smoothing': {
            'sigma': 11,
            'mode': 'reflect'
        },
        'normalization': {
            'method': 'to_wl',
            'to_wl': 751,
            'clip_to': 10
        },
        'mask_extraction': {
            'pri_thr': -0.1,
            'ndvi_thr': 0.2, 
            'hbsi_thr': -0.6,
            'min_pix_size': 2
        }
    }
    
    return template
