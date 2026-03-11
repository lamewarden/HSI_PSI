"""
HS_tools.utils - Utility functions for hyperspectral image analysis

Contains helper functions for loading, processing, and visualizing hyperspectral data.
"""

import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, Optional, Union, List, Tuple, Any
from .core import HS_image, MS_image
from sklearn.metrics import classification_report, confusion_matrix


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


def get_rgb_sample(image: Union[HS_image, MS_image], normalize: bool = True, correct: bool = True,
                   show: bool = True, title: str = 'RGB Sample', axes: bool = False,
                   repeat: int = 1, verbose: bool = False):
    """
    Generate and optionally display an RGB representation for a single `HS_image` or `MS_image`.

    Parameters
    ----------
    image : HS_image or MS_image
        The image object to produce RGB from.
    normalize : bool
        Whether to perform traditional normalization for non-SNV data.
    correct : bool
        Remove obvious outliers and fix NaNs/Infs.
    show : bool
        Display the RGB image with matplotlib if True.
    title : str
        Plot title.
    axes : bool
        Whether to show axes in the matplotlib figure.
    repeat : int
        If >1, repeat the RGB sample along rows to make visualization larger.
    verbose : bool
        Print debug info when True.

    Returns
    -------
    np.ndarray
        RGB image array with shape (H, W, 3) or repeated along rows if `repeat` > 1.
    """
    if image is None:
        raise ValueError("No image provided.")
    
    # Check if image contains negative values (likely SNV-normalized)
    has_negative_values = np.any(image.img < 0)
    is_snv_normalized = hasattr(image, 'normalized') and image.normalized and has_negative_values
        
    # Extract RGB bands based on wavelength ranges
    if len(image.ind) <= 6:
        R = (image[670] / 4095)
        G = (image[595] / 4095)
        B = (image[495] / 4095)
    elif np.mean(image.ind) < 900 and len(image.ind) > 6 and image.bits == 12 and image.calibrated == False:
        R = np.mean([image[value] for value in image.ind if value >= 570 and value <= 650], axis=0)/4095
        G = np.mean([image[value] for value in image.ind if value >= 520 and value <= 570], axis=0)/4095
        B = np.mean([image[value] for value in image.ind if value >= 450 and value <= 520], axis=0)/4095
    elif np.mean(image.ind) < 900 and len(image.ind) > 6 and image.bits == 12 and image.calibrated == True and not is_snv_normalized:
        # Scaling to 95% of total reflectance (only for non-SNV data)
        global_95 = np.percentile(image.img[1:-1, 20:-20, :], 95)
        R = np.clip(np.mean([image[value] for value in image.ind if value >= 570 and value <= 650], axis=0)/global_95, 0, 1)
        G = np.clip(np.mean([image[value] for value in image.ind if value >= 520 and value <= 570], axis=0)/global_95, 0, 1)
        B = np.clip(np.mean([image[value] for value in image.ind if value >= 450 and value <= 520], axis=0)/global_95, 0, 1)
    elif np.mean(image.ind) < 900 and len(image.ind) > 6 and image.bits == 12 and is_snv_normalized:
        # Special handling for SNV-normalized data
        R = np.mean([image[value] for value in image.ind if value >= 570 and value <= 650], axis=0)
        G = np.mean([image[value] for value in image.ind if value >= 520 and value <= 570], axis=0)
        B = np.mean([image[value] for value in image.ind if value >= 450 and value <= 520], axis=0)
    elif np.mean(image.ind) > 900 and image.bits == 12:
        R = np.mean([image[value] for value in image.ind if value >= 1000 and value <= 1100], axis=0)/4095
        G = np.mean([image[value] for value in image.ind if value >= 1200 and value <= 1300], axis=0)/4095
        B = np.mean([image[value] for value in image.ind if value >= 1400 and value <= 1500], axis=0)/4095
    else:
        # Fallback for other cases
        R = np.mean([image[value] for value in image.ind if value >= 570 and value <= 650], axis=0)
        G = np.mean([image[value] for value in image.ind if value >= 520 and value <= 570], axis=0)
        B = np.mean([image[value] for value in image.ind if value >= 450 and value <= 520], axis=0)
    
    if correct:
        # Remove outliers only if the value is an outlier in all 3 channels
        R_mean, R_std = np.mean(R), np.std(R)
        G_mean, G_std = np.mean(G), np.std(G)
        B_mean, B_std = np.mean(B), np.std(B)
        outlier_mask = (
            (np.abs(R - R_mean) > 4 * R_std) &
            (np.abs(G - G_mean) > 4 * G_std) &
            (np.abs(B - B_mean) > 4 * B_std)
        )
        R = np.where(outlier_mask, R_mean, R)
        G = np.where(outlier_mask, G_mean, G)
        B = np.where(outlier_mask, B_mean, B)
        
        # Replace NaNs and Infs
        R = np.nan_to_num(R, nan=np.nanmin(R))
        G = np.nan_to_num(G, nan=np.nanmin(G))
        B = np.nan_to_num(B, nan=np.nanmin(B))
        
        R = np.where(np.isinf(R), np.nanmax(R), R)
        G = np.where(np.isinf(G), np.nanmax(G), G)
        B = np.where(np.isinf(B), np.nanmax(B), B)
    
    # Handle normalization differently for SNV vs regular data
    if is_snv_normalized:
        # For SNV data: use percentile-based normalization to preserve contrast
        if verbose:
            print(f" RGB ranges before processing: R[{np.min(R):.3f}, {np.max(R):.3f}], G[{np.min(G):.3f}, {np.max(G):.3f}], B[{np.min(B):.3f}, {np.max(B):.3f}]")
        
        # Use robust percentile-based scaling to preserve contrast
        # This prevents outliers from compressing the main data range
        def robust_normalize(channel, low_percentile=2, high_percentile=98):
            """Robust normalization using percentiles to preserve contrast."""
            # Calculate percentiles to avoid outlier compression
            low_val = np.percentile(channel, low_percentile)
            high_val = np.percentile(channel, high_percentile)
            
            if high_val > low_val:
                # Scale to [0, 1] based on percentile range
                normalized = (channel - low_val) / (high_val - low_val)
                # Clip to [0, 1] but preserve relative intensities
                normalized = np.clip(normalized, 0, 1)
            else:
                # If no variation, set to middle gray
                normalized = np.full_like(channel, 0.5)
            
            return normalized
        
        # Apply robust normalization to each channel independently
        R = robust_normalize(R)
        G = robust_normalize(G)
        B = robust_normalize(B)
        
        # Enhance contrast further by stretching the histogram
        def enhance_contrast(channel, gamma=0.8):
            """Apply gamma correction to enhance contrast."""
            return np.power(channel, gamma)
        
        R = enhance_contrast(R)
        G = enhance_contrast(G)
        B = enhance_contrast(B)
            
        if verbose:
            print(f"   Applied robust normalization: R[{np.min(R):.3f}, {np.max(R):.3f}], G[{np.min(G):.3f}, {np.max(G):.3f}], B[{np.min(B):.3f}, {np.max(B):.3f}]")
    
    elif normalize:
        # Traditional normalization for non-SNV data
        # Only use interior region for normalization to avoid edge effects
        try:
            R_norm_val = np.max(R.squeeze()[5:-5, 20:-20])
            G_norm_val = np.max(G.squeeze()[5:-5, 20:-20])
            B_norm_val = np.max(B.squeeze()[5:-5, 20:-20])
            
            if R_norm_val > 0:
                R = R / R_norm_val
            if G_norm_val > 0:
                G = G / G_norm_val
            if B_norm_val > 0:
                B = B / B_norm_val
        except (IndexError, ValueError):
            # Fallback if slicing fails
            R_max, G_max, B_max = np.max(R), np.max(G), np.max(B)
            if R_max > 0: R = R / R_max
            if G_max > 0: G = G / G_max
            if B_max > 0: B = B / B_max
    
    # Final clipping to ensure valid RGB range [0, 1]
    R = np.clip(R, 0, 1)
    G = np.clip(G, 0, 1)
    B = np.clip(B, 0, 1)
    
    # Create RGB sample
    rgb_sample = np.dstack((R, G, B))

    if repeat > 1:
        # Repeat the RGB channels to match the original image shape
        # Ensure rgb_sample is at least 3D
        if rgb_sample.ndim == 2:
            rgb_sample = rgb_sample[:, :, np.newaxis]
        # Repeat along axis=0 (rows)
        rgb_sample = np.repeat(rgb_sample, repeat, axis=0)
    
    if show:
        plt.figure(figsize=(8, 6))
        plt.imshow(rgb_sample)
        
        # Enhanced title with processing info
        if is_snv_normalized:
            enhanced_title = f"{title} (SNV-normalized)"
        else:
            enhanced_title = title
        
        plt.title(enhanced_title)
        if not axes:
            plt.axis('off')
        plt.show()
    
    return rgb_sample


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
            'white_ref_path': None,
            'dark_calibration': False,
            'clip_to': 10
        },
        'spike_removal': {
            'win': 7,
            'k': 6.0,
            'replace': 'median'
        },
        'spectral_cropping': {
            'wl_start': None,
            'wl_end': None,
            'band_start': None,
            'band_end': None
        },
        'solar_correction': {
            'teflon_edge_coord': (-10, -3),
            'reference_teflon': None,
            'smooth_window': 35
        },
        'spectral_smoothing': {
            'sigma': 11,
            'mode': 'reflect'
        },
        'normalization': {
            'to_wl': 751,
            'clip_to': 10,
            'method': 'to_wl'
        },
        'mask_extraction': {
            'pri_thr': -0.1,
            'ndvi_thr': 0.2, 
            'hbsi_thr': -0.6,
            'min_pix_size': 2,
            'repeat': 10,
            'show_visualization': True
        }
    }
    
    return template


# utils.py
import numpy as np

try:
    from scipy.signal import savgol_filter
    _HAS_SG = True
except Exception:
    _HAS_SG = False


def _extract_cube_and_meta(hs_image, mask=None):
    """
    Best-effort extractor for HSI_PSI.HS_image-like objects.
    Expects a spectral cube shaped (H, W, B).
    """
    # cube - check for .img first (HSI_PSI convention)
    cube = None
    for attr in ("img", "cube", "data", "array", "X"):
        if hasattr(hs_image, attr):
            arr = getattr(hs_image, attr)
            if isinstance(arr, np.ndarray) and arr.ndim == 3:
                cube = arr
                break
    if cube is None:
        raise ValueError("Cannot locate 3D spectral cube on hs_image (tried .img/.cube/.data/.array/.X).")

    # wavelengths (optional) - check for .ind first (HSI_PSI convention)
    wl = None
    for attr in ("ind", "wavelengths", "bands", "wl", "lambda_"):
        if hasattr(hs_image, attr):
            w = getattr(hs_image, attr)
            w = np.asarray(w).ravel()
            if w.ndim == 1 and w.size == cube.shape[2]:
                wl = w
                break

    # mask
    if mask is None:
        for attr in ("mask", "roi_mask", "valid_mask"):
            if hasattr(hs_image, attr):
                m = getattr(hs_image, attr)
                if isinstance(m, np.ndarray) and m.shape[:2] == cube.shape[:2]:
                    mask = m.astype(bool)
                    break
    if mask is None:
        mask = np.ones(cube.shape[:2], dtype=bool)

    return cube, wl, mask


def _odd_at_most(val, max_allowed):
    """Nearest odd integer ≤ min(val, max_allowed). Minimum returned is 3 if possible."""
    k = int(min(val, max_allowed))
    if k % 2 == 0:
        k -= 1
    if k < 3:
        k = 3 if max_allowed >= 3 else max_allowed | 1  # last resort odd
    return max(1, k)


def rank_noisy_bands(
    hs_image,
    mask=None,
    window_frac=0.02,
    window=None,
    poly=3,
    n_top=10,
    use_relative=True,
    robust=True,
    eps=1e-12,
):
    """
    Rank spectral bands by noise using residuals to a smoothed spectrum.

    Parameters
    ----------
    hs_image : HS_image or compatible
        Object with a (H,W,B) cube in attribute .cube/.data/.array/.X.
        Optional attributes: .wavelengths, .mask.
    mask : ndarray[H,W], optional
        Valid pixel mask. If None, tries hs_image.mask, else uses all pixels.
    window_frac : float
        Savitzky–Golay window as fraction of band count (used if `window` is None).
    window : int, optional
        Explicit Savitzky–Golay window length (odd). Overrides window_frac.
    poly : int
        Savitzky–Golay polynomial order.
    n_top : int
        Number of top noisy bands to return.
    use_relative : bool
        If True, rank by NSR = noise/signal_median. If False, rank by absolute noise.
    robust : bool
        If True, noise = 1.4826 * MAD; else standard deviation.
    eps : float
        Small constant to stabilize division.

    Returns
    -------
    result : dict
        {
          "order": np.ndarray[B]               # indices sorted descending by score
          "score": np.ndarray[B]               # NSR or abs noise per band
          "noise": np.ndarray[B]               # robust noise estimate per band
          "signal_med": np.ndarray[B]          # median signal per band
          "top_idx": np.ndarray[min(n_top,B)]  # top band indices
          "top_score": np.ndarray[min(n_top,B)]
          "wavelengths": np.ndarray[B] or None
          "B": int                              # number of bands
        }
    """
    cube, wl, m = _extract_cube_and_meta(hs_image, mask=mask)
    H, W, B = cube.shape
    if B < 5:
        raise ValueError(f"Too few bands for smoothing: B={B}")

    # choose window
    if window is None:
        window = max(5, int(round(B * float(window_frac))))
    win = _odd_at_most(window, B - 1)
    if poly >= win:
        poly = max(1, min(poly, win - 1))

    # collect spectra over ROI
    # Ensure mask is 2D boolean array matching spatial dimensions
    if m.ndim == 3:
        # If mask is 3D, take first channel or flatten to 2D
        if m.shape[2] == 1:
            m = m[:, :, 0]
        else:
            m = m.any(axis=2)  # Any band marked as True
    m = m.astype(bool)
    
    # Reshape cube to (Npix, B) and apply mask
    cube_reshaped = cube.reshape(-1, B)  # (H*W, B)
    mask_flat = m.flatten()  # (H*W,)
    S = cube_reshaped[mask_flat].astype(np.float64)  # (Npix, B)
    if S.size == 0:
        raise ValueError("Mask selects zero pixels.")

    # smooth per spectrum
    if _HAS_SG:
        S_smooth = savgol_filter(S, window_length=win, polyorder=poly, axis=1, mode="interp")
    else:
        # fallback: moving median filter as a crude smoother
        k = win
        pad = k // 2
        S_pad = np.pad(S, ((0, 0), (pad, pad)), mode="edge")
        S_smooth = np.empty_like(S)
        for b in range(B):
            S_smooth[:, b] = np.median(S_pad[:, b:b + k], axis=1)

    # residuals
    R = S - S_smooth  # (Npix, B)

    # noise per band
    if robust:
        med_R = np.median(R, axis=0)                # (B,)
        mad = np.median(np.abs(R - med_R), axis=0)  # (B,)
        noise = 1.4826 * mad
    else:
        noise = np.std(R, axis=0, ddof=1)

    # signal level per band
    signal_med = np.median(S, axis=0)  # (B,)

    # score
    if use_relative:
        score = noise / (np.abs(signal_med) + eps)
    else:
        score = noise.copy()

    # rank
    order = np.argsort(score)[::-1]
    k = int(min(n_top if n_top is not None else 10, B))
    top_idx = order[:k]
    top_score = score[top_idx]

    return {
        "order": order,
        "score": score,
        "noise": noise,
        "signal_med": signal_med,
        "top_idx": top_idx,
        "top_score": top_score,
        "wavelengths": wl,
        "B": B,
    }


def summarize_noisiest_bands(result, n=None):
    """
    Convenience printer. Returns a small dict with indices, wavelengths, and scores.
    """
    order = result["order"]
    wl = result.get("wavelengths", None)
    score = result["score"]
    noise = result["noise"]
    signal = result["signal_med"]
    k = int(min(n if n is not None else result["top_idx"].size, result["B"]))
    idx = order[:k]
    out = {
        "band_index": idx,
        "wavelength": (wl[idx] if wl is not None else None),
        "score": score[idx],
        "noise": noise[idx],
        "signal_med": signal[idx],
    }
    return out


def print_package_info() -> None:
    """
    Print comprehensive HSI_PSI package information including version,
    features, and system details.
    """
    import platform
    import sys
    from datetime import datetime
    
    print("=" * 60)
    print("HSI_PSI - Hyperspectral Image Processing Package")
    print("=" * 60)
    print(f"Version: 0.2.0")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    print("Key Features:")
    print("  • Advanced spectral cropping with wavelength mapping")
    print("  • Intelligent sensor calibration and solar correction")
    print("  • Optimized 7-step processing pipeline")
    print("  • Noise analysis and band ranking")
    print("  • Automated mask extraction and vegetation indices")
    print("  • Flexible configuration system")
    print()
    
    print("Core Components:")
    print("  • HS_image/MS_image classes for data handling")
    print("  • HS_preprocessor for automated processing")
    print("  • Utility functions for analysis and visualization")
    print("  • Reference teflon library management")
    print()
    
    print("Processing Pipeline:")
    print("  1. Sensor calibration (dark current & flat field)")
    print("  2. Spike removal (cosmic ray correction)")
    print("  3. Spectral cropping (wavelength selection)")
    print("  4. Solar irradiance correction")
    print("  5. Spectral smoothing (noise reduction)")
    print("  6. Reflectance normalization")
    print("  7. Mask extraction (vegetation/non-vegetation)")
    print()
    
    print("Use Cases:")
    print("  • Close-range vegetation monitoring")
    print("  • Plant stress detection and analysis")
    print("  • Spectral signature characterization")
    print("  • Agricultural phenotyping")
    print("=" * 60)


def plot_spectra(spectra_dicts_list, dict_names=None, scale=False, 
                 title="Multiple Spectra Comparison", figure_size=(12, 8)):
    """
    Plot multiple spectra from a list of dictionaries (format from spectrum_probe).
    
    Args:
        spectra_dicts_list: List of dictionaries, each containing spectra data 
                          (format from spectrum_probe output)
        dict_names: List of names for each dictionary. If None, uses "Dataset 1", "Dataset 2", etc.
        scale: If True, scale all spectra to have the same maximum value as the first spectrum
        title: Title for the plot
        figure_size: Tuple of (width, height) for the figure size
        
    Example:
        # Get spectra from different images
        spectra1 = preprocessor1.spectrum_probe(rois=rois, show=False)
        spectra2 = preprocessor2.spectrum_probe(rois=rois, show=False) 
        
        # Plot them together
        from hsi_psi.utils import plot_spectra
        plot_spectra([spectra1, spectra2], 
                    dict_names=["Image 1", "Image 2"],
                    scale=True)
    """
    if not spectra_dicts_list:
        raise ValueError("spectra_dicts_list cannot be empty")
    
    # Generate default names if not provided
    if dict_names is None:
        dict_names = [f"Dataset {i+1}" for i in range(len(spectra_dicts_list))]
    elif len(dict_names) != len(spectra_dicts_list):
        raise ValueError("Length of dict_names must match length of spectra_dicts_list")
    
    # Create figure
    plt.figure(figsize=figure_size)
    
    # Get a colormap with enough colors
    colors = plt.cm.tab10(np.linspace(0, 1, 10))  # 10 distinct colors
    color_idx = 0
    
    # Find reference spectrum for scaling if needed
    reference_spectrum = None
    reference_max = None
    if scale:
        # Use first spectrum from first dataset as reference
        first_dict = spectra_dicts_list[0]
        if first_dict:
            first_key = next(iter(first_dict.keys()))
            reference_spectrum = first_dict[first_key]["spectrum"]
            reference_max = np.max(reference_spectrum)
    
    # Plot each dataset
    for dict_idx, (spectra_dict, dataset_name) in enumerate(zip(spectra_dicts_list, dict_names)):
        if not spectra_dict:
            continue
            
        # Plot each spectrum in the dictionary
        for roi_name, data in spectra_dict.items():
            spectrum = data["spectrum"]
            wavelengths = data["wavelengths"]
            
            # Apply scaling if requested
            if scale and reference_max is not None:
                current_max = np.max(spectrum)
                if current_max > 0:  # Avoid division by zero
                    scale_factor = reference_max / current_max
                    spectrum = spectrum * scale_factor
            
            # Create label combining dataset name and ROI name
            label = f"{dataset_name} - {roi_name}"
            
            # Plot with unique color
            color = colors[color_idx % len(colors)]
            plt.plot(wavelengths, spectrum, color=color, label=label, linewidth=2)
            color_idx += 1
    
    # Customize plot
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Reflectance' + (' (scaled)' if scale else ''))
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def vis_clust_2D(X, pc_to_visualize):
    fig, ax = plt.subplots(figsize=(10, 8))

    # Map unique labels to integer values to color them
    unique_labels = X['label'].unique()
    label_mapping = {label: i for i, label in enumerate(unique_labels)}
    colors = X['label'].map(label_mapping)

    # Create a scatter plot, color by 'label' column values
    scatter = ax.scatter(x=X.iloc[:, pc_to_visualize[0]], y=X.iloc[:, pc_to_visualize[1]], c=colors, cmap='rainbow')

    # Create a legend with the actual 'label' values
    handles, _ = scatter.legend_elements()
    labels = [str(label) for label in unique_labels]
    ax.legend(handles=handles, labels=labels, title="Label")
    plt.xlabel(f'PC{pc_to_visualize[0]}')
    plt.ylabel(f'PC{pc_to_visualize[1]}')

    # Show the plot
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names=None, fig_size=(8,6)):
    """
    Plot a confusion matrix with absolute counts and row-wise percentages.
    The percentage values are displayed in a smaller font.
    
    Args:
        y_true (array-like): Ground truth labels.
        y_pred (array-like): Predicted labels.
        class_names (list, optional): List of class names (strings).
            If None, indices [0..n_classes-1] will be used.
        fig_size (tuple, optional): Figure size in inches. Default is (8, 6).
    """
    cm = confusion_matrix(y_true, y_pred)
    n_classes = cm.shape[0]

    # If no class names are provided, just use numeric class indices
    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]
    
    # Compute row-wise percentages
    cm_percent = np.zeros_like(cm, dtype=float)
    for i in range(n_classes):
        row_sum = cm[i].sum()
        if row_sum != 0:
            cm_percent[i] = cm[i] / row_sum * 100
        else:
            cm_percent[i] = 0

    fig, ax = plt.subplots(figsize=fig_size)
    cax = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar(cax)

    # Set tick marks and labels
    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)

    # Determine threshold for text color
    threshold = cm.max() / 2.

    # Loop over data dimensions and create text annotations.
    for i in range(n_classes):
        for j in range(n_classes):
            abs_val = cm[i, j]
            perc_val = cm_percent[i, j]
            color = 'white' if abs_val > threshold else 'black'
            # Place absolute count above center
            ax.text(j, i - 0.2, f"{abs_val}", ha='center', va='center', color=color, fontsize=12)
            # Place percentage value below center in a smaller font
            ax.text(j, i + 0.2, f"{perc_val:.1f}%", ha='center', va='center', color=color, fontsize=8)

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    plt.tight_layout()
    plt.show()