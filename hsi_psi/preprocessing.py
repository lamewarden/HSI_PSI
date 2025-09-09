import os
import copy
import json
import warnings
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from skimage.morphology import remove_small_objects
import glob
import pandas as pd

# Import from local modules using relative imports
from .core import HS_image

warnings.filterwarnings("ignore")



class HS_preprocessor:
    """
    Enhanced hyperspectral image preprocessing pipeline with integrated mask extraction.
    
    Features:
    - All original HS_preprocessor functionality preserved
    - Integrated mask extraction using vegetation indices (NDVI, HBSI, PRI)
    - Configurable segmentation parameters stored in config file
    - Mask storage as hs_image.mask attribute
    - Updated run_full_pipeline with mask extraction as final step
    - Updated process_folder with mask integration
    
    Processing Pipeline:
    1. Sensor calibration (white/dark reference correction)
    2. Solar spectrum correction (using reference teflon)  
    3. Spectral smoothing (Gaussian filtering)
    4. Normalization (wavelength-based or SNV)
    5. Mask extraction (vegetation index segmentation) [NEW]
    
    Features:
    - Step-by-step processing with configuration storage
    - Visualization tools for each step
    - Step-by-step testing and visualization
    - Save/load configurations 
    - Folder processing
    - Verbose control for all print outputs
    - Reference teflon spectrum storage and persistence
    - Integrated mask extraction with configurable parameters
    """
    
    def __init__(self, image_path=None, verbose=True):
        """Initialize with optional image path and verbose control."""
        self.image_path = image_path
        self.image = None
        self.original_image = None
        self.white_calibration = None
        self.verbose = verbose
        self.reference_teflon = None  # Store reference teflon data as dict: {wavelength: reflectance}
        
        # Store parameters for each step
        self.config = {}
        self.step_results = {}  # Store intermediate results for visualization
        
        if image_path:
            self.load_image(image_path)
    
    def load_image(self, image_path):
        """Step 1: Load hyperspectral image."""
        self.image_path = image_path
        self.image = HS_image(image_path)
        self.original_image = copy.deepcopy(self.image)
        if self.verbose:
            print(f"✓ Loaded image: {os.path.basename(image_path)}")
            print(f"  Shape: {self.image.img.shape}")
            print(f"  Wavelength range: {min(self.image.ind)}-{max(self.image.ind)} nm")
        return self
    
    def _upload_calibration(self, dark_calibration=None, white_ref_path=None):
        """Upload white and dark calibration matrices."""
        if white_ref_path:
            # Use external white reference
            white_calibration = HS_image(white_ref_path)
            white_matrix = np.mean(white_calibration.img, axis=0)

        
        if dark_calibration:
            # Use image edges as dark reference
            dark_calib = HS_image(dark_calibration)  
            dark_matrix = np.mean(dark_calib.img, axis=0)
        else:
            # Use image edges as dark reference
            dark_matrix = np.zeros(white_matrix.shape)

                
        return white_matrix, dark_matrix
    

  
    
    def extract_masks(self, pri_thr=None, ndvi_thr=None, hbsi_thr=None, min_pix_size=None, 
                     repeat=10, show_visualization=True):
        """Step 6: Extract vegetation masks using vegetation indices."""
        
        # Use parameters from loaded config if available, otherwise use provided parameters or hardcoded defaults
        if pri_thr is None:
            pri_thr = self.config.get('mask_extraction', {}).get('pri_thr', -0.1)
        if ndvi_thr is None:
            ndvi_thr = self.config.get('mask_extraction', {}).get('ndvi_thr', 0.2)
        if hbsi_thr is None:
            hbsi_thr = self.config.get('mask_extraction', {}).get('hbsi_thr', -0.6)
        if min_pix_size is None:
            min_pix_size = self.config.get('mask_extraction', {}).get('min_pix_size', 2)
        
        # Store parameters for this step (update config with actual values used)
        self.config['mask_extraction'] = {
            'pri_thr': pri_thr,
            'ndvi_thr': ndvi_thr,
            'hbsi_thr': hbsi_thr,
            'min_pix_size': min_pix_size
        }
        
        # Calculate vegetation indices
        ndvi_image = (self.image[751] - self.image[670]) / (self.image[751] + self.image[670] + 1e-7)
        ndvi_mask = (ndvi_image > ndvi_thr)[:,:,np.newaxis]
        
        hbsi_image = (self.image[470] - self.image[751]) / (self.image[470] + self.image[751] + 1e-7)
        hbsi_mask = (hbsi_image > hbsi_thr)[:,:,np.newaxis]

        pri_image = (self.image[531] - self.image[570]) / (self.image[531] + self.image[570] + 1e-7)
        pri_mask = (pri_image < pri_thr)[:,:,np.newaxis]

        # Display all vegetation indices for calibration
        if show_visualization:
            orig_rgb = self.get_rgb_sample(normalize=True, correct=False)
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 6))
            fig.suptitle('Vegetation Indices for Threshold Calibration', fontsize=16, fontweight='bold')
            
            # NDVI
            im1 = axes[0,0].imshow(np.repeat(ndvi_image, repeat, axis=0), cmap='RdYlGn')
            axes[0,0].set_title(f"NDVI Image\n(Threshold: {ndvi_thr})")
            axes[0,0].axis('off')
            plt.colorbar(im1, ax=axes[0,0], shrink=0.8)
            
            # HBSI
            im2 = axes[0,1].imshow(np.repeat(hbsi_image, repeat, axis=0), cmap='viridis')
            axes[0,1].set_title(f"HBSI Image\n(Threshold: {hbsi_thr})")
            axes[0,1].axis('off')
            plt.colorbar(im2, ax=axes[0,1], shrink=0.8)
            
            # PRI
            im3 = axes[0,2].imshow(np.repeat(pri_image, repeat, axis=0), cmap='plasma')
            axes[0,2].set_title(f"PRI Image\n(Threshold: {pri_thr})")
            axes[0,2].axis('off')
            plt.colorbar(im3, ax=axes[0,2], shrink=0.8)

            # Masks
            axes[1,0].imshow(np.repeat(ndvi_mask, repeat, axis=0))
            axes[1,0].set_title(f"NDVI mask\n(Threshold: {ndvi_thr})")
            axes[1,0].axis('off')
            
            axes[1,1].imshow(np.repeat(hbsi_mask, repeat, axis=0))
            axes[1,1].set_title(f"HBSI mask\n(Threshold: {hbsi_thr})")
            axes[1,1].axis('off')
            
            axes[1,2].imshow(np.repeat(pri_mask, repeat, axis=0))
            axes[1,2].set_title(f"PRI mask\n(Threshold: {pri_thr})")
            axes[1,2].axis('off')
            
            plt.show()

        # Combine masks
        mask = pri_mask.astype(bool) & ~ndvi_mask.astype(bool) & ~hbsi_mask.astype(bool)
        
        # Remove small objects - note that remove_small_objects expects and returns 2D boolean array
        mask_2d = remove_small_objects(mask[:,:,0], min_size=min_pix_size)

        # Convert back to 3D uint8 array
        mask = mask_2d.astype(np.uint8)[:,:,np.newaxis]
        
        # Store mask as image attribute
        self.image.mask = mask
        
        # Store result for visualization
        self.step_results['mask_extraction'] = copy.deepcopy(self.image)
        
        if show_visualization:
            # Create visualization with proper transparency
            plt.figure(figsize=(8, 4))
            plt.imshow(np.repeat(orig_rgb[:,:,:], repeat, axis=0))
            
            # Create turquoise overlay only where mask == 1
            mask_repeated = np.repeat(mask[:,:,0], repeat, axis=0)
            turquoise_overlay = np.full((mask_repeated.shape[0], mask_repeated.shape[1], 3), np.nan)
            
            # Only set colors where mask is 1 (true transparency for mask == 0)
            turquoise_overlay[mask_repeated == 1, 0] = 64/255    # Red
            turquoise_overlay[mask_repeated == 1, 1] = 224/255   # Green  
            turquoise_overlay[mask_repeated == 1, 2] = 208/255   # Blue
            
            plt.imshow(turquoise_overlay, alpha=0.6)
        
            plt.title("Final Segmentation Result\n(Turquoise = Plant pixels)", fontsize=14, fontweight='bold')    
            plt.axis('off')
            plt.show()
        
        if self.verbose:
            print(f"✓ Mask extraction completed")
            print(f"  PRI threshold: {pri_thr}")
            print(f"  NDVI threshold: {ndvi_thr}")
            print(f"  HBSI threshold: {hbsi_thr}")
            print(f"  Min pixel size: {min_pix_size}")
            print(f"  Mask shape: {mask.shape}")
            mask_pixels = np.sum(mask)
            total_pixels = mask.shape[0] * mask.shape[1]
            print(f"  Masked pixels: {mask_pixels}/{total_pixels} ({100*mask_pixels/total_pixels:.1f}%)")
        
        return self

    
    def get_spectrum(self, roi=(slice(None), slice(None), slice(None)), 
                    ref_spectrum=None, title="Spectrum", scale=False, show=True):
        """Extract and visualize spectrum from ROI."""
        # Extract spectrum from ROI
        roi_data = self.image.img[roi]
        mean_spectrum = np.mean(roi_data, axis=(0, 1))
        wavelengths = np.array(self.image.ind)
        
        if show:
            plt.figure(figsize=(10, 6))
            plt.plot(wavelengths, mean_spectrum, 'b-', linewidth=2, label='Current')
            
            if ref_spectrum is not None:
                if scale:
                    # Scale reference to match current spectrum magnitude
                    scale_factor = np.max(mean_spectrum) / np.max(ref_spectrum)
                    scaled_ref = ref_spectrum * scale_factor
                    plt.plot(wavelengths, scaled_ref, 'r--', linewidth=2, 
                           label=f'Reference (scaled)')
                else:
                    plt.plot(wavelengths, ref_spectrum, 'r--', linewidth=2, label='Reference')
            
            plt.xlabel('Wavelength (nm)')
            plt.ylabel('Reflectance')
            plt.title(title)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()
        
        return mean_spectrum, wavelengths
    
    # === CONVENIENCE METHODS (updated with mask extraction) ===
    
    def run_full_pipeline(self, config_override=None, extract_masks_flag=True):
        """
        Run the complete preprocessing pipeline using loaded configuration.
        
        Args:
            config_override (dict, optional): Override specific parameters for any step.
                                            Format: {'step_name': {'param': value}}
            extract_masks_flag (bool): Whether to run mask extraction as final step
                                            
        Returns:
            self: For method chaining
            
        Example:
            # Use default config with mask extraction
            processor.run_full_pipeline()
            
            # Override specific parameters
            processor.run_full_pipeline({
                'sensor_calibration': {'clip_to': 15},
                'mask_extraction': {'pri_thr': -0.15}
            })
        """
        
        # Check if config is loaded
        if not hasattr(self, 'config') or not self.config:
            raise ValueError("No configuration loaded. Please run load_config() first.")
        
        # Deep copy config to avoid modifying original
        import copy
        pipeline_config = copy.deepcopy(self.config)
        
        # Apply overrides if provided
        if config_override:
            for step, step_params in config_override.items():
                if step in pipeline_config:
                    pipeline_config[step].update(step_params)
                else:
                    pipeline_config[step] = step_params
        
        if self.verbose:
            print("🔄 Running full preprocessing pipeline...")
            steps_to_run = ['sensor_calibration', 'solar_correction', 'spectral_smoothing', 'normalization']
            if extract_masks_flag:
                steps_to_run.append('mask_extraction')
            print(f"Pipeline steps: {steps_to_run}")
        
        # Run preprocessing pipeline steps in order
        pipeline_steps = ['sensor_calibration', 'solar_correction', 'spectral_smoothing', 'normalization']
        
        for step in pipeline_steps:
            if step in pipeline_config:
                if self.verbose:
                    print(f"Running {step}...")
                
                step_params = pipeline_config[step].copy()
                
                # Handle sensor calibration
                if step == 'sensor_calibration':
                    # Get white reference path from config
                    white_ref_path = step_params.pop('white_ref_path', None)
                    self.sensor_calibration(white_ref_path=white_ref_path, **step_params)
                
                # Handle solar correction
                elif step == 'solar_correction':
                    # Filter out metadata parameters that shouldn't be passed to the method
                    # These are computed internally and saved for reference but not method parameters
                    method_params = {k: v for k, v in step_params.items() 
                                   if k not in ['reference_source', 'has_reference']}
                    
                    # Use loaded reference teflon if available
                    if hasattr(self, 'reference_teflon') and self.reference_teflon is not None:
                        if isinstance(self.reference_teflon, dict):
                            # Check if it's the new wavelength-as-keys format
                            if all(isinstance(k, (int, float)) for k in self.reference_teflon.keys()):
                                # Extract spectrum from wavelength-as-keys format
                                current_wavelengths = np.array(self.image.ind)
                                ref_wavelengths = np.array(sorted(self.reference_teflon.keys()))
                                ref_spectrum = np.array([self.reference_teflon[wl] for wl in ref_wavelengths])
                                
                                if np.array_equal(ref_wavelengths, current_wavelengths):
                                    method_params['reference_teflon'] = ref_spectrum
                                else:
                                    # Interpolate to match current wavelengths
                                    from scipy.interpolate import interp1d
                                    interp_func = interp1d(ref_wavelengths, ref_spectrum, 
                                                         kind='linear', bounds_error=False, fill_value='extrapolate')
                                    method_params['reference_teflon'] = interp_func(current_wavelengths)
                            elif 'spectrum' in self.reference_teflon:
                                # Legacy dictionary format
                                method_params['reference_teflon'] = self.reference_teflon['spectrum']
                            else:
                                # Unknown dict format
                                if self.verbose:
                                    print("  ⚠ Warning: Unknown reference teflon dictionary format")
                        else:
                            # Legacy array format
                            method_params['reference_teflon'] = self.reference_teflon
                    self.solar_correction(**method_params)
                
                # Handle spectral smoothing
                elif step == 'spectral_smoothing':
                    self.spectral_smoothing(**step_params)
                
                # Handle normalization
                elif step == 'normalization':
                    self.normalization(**step_params)
        
        # Run mask extraction as final step if requested
        if extract_masks_flag:
            if self.verbose:
                print(f"Running mask_extraction...")
            
            # Use mask extraction parameters from config, or defaults
            mask_params = pipeline_config.get('mask_extraction', {
                'pri_thr': -0.1,
                'ndvi_thr': 0.2,
                'hbsi_thr': -0.6,
                'min_pix_size': 2
            })
            
            # Don't show visualization in batch processing unless verbose
            mask_params['show_visualization'] = self.verbose
            
            self.extract_masks(**mask_params)
        
        if self.verbose:
            print("Full pipeline completed successfully!")
            if extract_masks_flag and hasattr(self.image, 'mask'):
                print(f"Mask stored in image.mask attribute")
            
        return self
    
    def get_step_result(self, step_name):
        """Get the HS_image result from a specific processing step."""
        if step_name not in self.step_results:
            available = list(self.step_results.keys())
            raise ValueError(f"Step '{step_name}' not found. Available: {available}")
        return self.step_results[step_name]
    
    def get_current_image(self):
        """Get the current processed HS_image."""
        return self.image
    
    def get_HScube(self):
        """Get the current processed hyperspectral data cube."""
        if self.image is None:
            raise ValueError("No image loaded.")
        return self.image.img
    
    def get_config(self):
        """Get the current configuration dictionary."""
        return copy.deepcopy(self.config)
    
    def get_reference_teflon_data(self):
        """
        Get reference teflon data in a readable format.
        
        Returns:
            dict or None: Dictionary with wavelength information, or None if no reference
        """
        if self.reference_teflon is None:
            return None
        
        if isinstance(self.reference_teflon, dict):
            # Check if it's the new wavelength-as-keys format
            if all(isinstance(k, (int, float)) for k in self.reference_teflon.keys()):
                # New format: {wavelength: reflectance}
                wavelengths = sorted(self.reference_teflon.keys())
                spectrum = [self.reference_teflon[wl] for wl in wavelengths]
                return {
                    'format': 'wavelength_keys',
                    'wavelengths': wavelengths,
                    'spectrum': spectrum,
                    'num_bands': len(wavelengths),
                    'wavelength_range': f"{wavelengths[0]:.1f}-{wavelengths[-1]:.1f} nm"
                }
            elif 'spectrum' in self.reference_teflon and 'wavelengths' in self.reference_teflon:
                # Legacy dictionary format
                wavelengths = self.reference_teflon['wavelengths']
                spectrum = self.reference_teflon['spectrum']
                if wavelengths is not None:
                    return {
                        'format': 'legacy_dict_with_wavelengths',
                        'wavelengths': wavelengths,
                        'spectrum': spectrum,
                        'num_bands': len(spectrum),
                        'wavelength_range': f"{wavelengths[0]:.1f}-{wavelengths[-1]:.1f} nm"
                    }
                else:
                    return {
                        'format': 'legacy_dict_no_wavelengths',
                        'wavelengths': None,
                        'spectrum': spectrum,
                        'num_bands': len(spectrum),
                        'wavelength_range': 'unknown'
                    }
            else:
                return {
                    'format': 'unknown_dict',
                    'wavelengths': None,
                    'spectrum': None,
                    'num_bands': 0,
                    'wavelength_range': 'unknown'
                }
        else:
            # Legacy array format
            return {
                'format': 'legacy_array',
                'wavelengths': None,
                'spectrum': self.reference_teflon,
                'num_bands': len(self.reference_teflon),
                'wavelength_range': 'unknown'
            }
    
    def save_config(self, filepath):
        """Save current configuration to JSON file, including reference teflon spectrum with wavelengths."""
        # Prepare reference teflon data
        reference_teflon_data = None
        
        if self.reference_teflon is not None:
            if isinstance(self.reference_teflon, dict):
                # Check if it's the new wavelength-as-keys format
                if all(isinstance(k, (int, float)) for k in self.reference_teflon.keys()):
                    # New format: {wavelength: reflectance}
                    reference_teflon_data = {
                        'format': 'wavelength_keys',
                        'data': self.reference_teflon
                    }
                elif 'spectrum' in self.reference_teflon and 'wavelengths' in self.reference_teflon:
                    # Legacy dictionary format - convert to new format for saving
                    wavelengths = self.reference_teflon['wavelengths']
                    spectrum = self.reference_teflon['spectrum']
                    if wavelengths is not None:
                        reference_teflon_data = {
                            'format': 'wavelength_keys',
                            'data': {float(wl): float(refl) for wl, refl in zip(wavelengths, spectrum)}
                        }
                    else:
                        # Can't convert without wavelengths - save as legacy
                        reference_teflon_data = {
                            'format': 'legacy_dict',
                            'data': {
                                'spectrum': spectrum.tolist() if hasattr(spectrum, 'tolist') else spectrum,
                                'wavelengths': None
                            }
                        }
                else:
                    # Unknown dict format
                    reference_teflon_data = {
                        'format': 'unknown_dict',
                        'data': self.reference_teflon
                    }
            else:
                # Legacy array format
                reference_teflon_data = {
                    'format': 'legacy_array',
                    'data': self.reference_teflon.tolist() if hasattr(self.reference_teflon, 'tolist') else self.reference_teflon
                }
        
        config_to_save = {
            'image_path': self.image_path,
            'processing_config': self.config,
            'reference_teflon_data': reference_teflon_data
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_to_save, f, indent=2, default=str)
        
        if self.verbose:
            print(f"✓ Configuration saved to: {filepath}")
            if self.reference_teflon is not None:
                ref_info = self.get_reference_teflon_data()
                if ref_info:
                    print(f"  ✓ Included reference teflon spectrum ({ref_info['num_bands']} bands, {ref_info['format']} format)")
                    if ref_info['wavelength_range'] != 'unknown':
                        print(f"    Wavelength range: {ref_info['wavelength_range']}")
    
    def load_config(self, filepath):
        """Load configuration from JSON file, including reference teflon spectrum with wavelengths."""
        with open(filepath, 'r') as f:
            saved_config = json.load(f)
        
        self.config = saved_config.get('processing_config', {})
        
        # Load reference teflon spectrum - handle multiple formats
        if 'reference_teflon_data' in saved_config and saved_config['reference_teflon_data'] is not None:
            ref_data = saved_config['reference_teflon_data']
            format_type = ref_data.get('format', 'unknown')
            
            if format_type == 'wavelength_keys':
                # New format: {wavelength: reflectance}
                self.reference_teflon = {float(k): float(v) for k, v in ref_data['data'].items()}
                if self.verbose:
                    print(f"✓ Configuration loaded from: {filepath}")
                    wavelengths = sorted(self.reference_teflon.keys())
                    print(f"  ✓ Restored reference teflon spectrum ({len(wavelengths)} bands)")
                    print(f"    Wavelength range: {wavelengths[0]:.1f}-{wavelengths[-1]:.1f} nm")
                    
            elif format_type == 'legacy_dict':
                # Legacy dictionary format
                spectrum = np.array(ref_data['data']['spectrum'])
                wavelengths = ref_data['data'].get('wavelengths')
                if wavelengths is not None:
                    wavelengths = np.array(wavelengths)
                    self.reference_teflon = {
                        'spectrum': spectrum,
                        'wavelengths': wavelengths,
                        'format_version': '2.0'
                    }
                    if self.verbose:
                        print(f"✓ Configuration loaded from: {filepath}")
                        print(f"  ✓ Restored reference teflon spectrum ({len(spectrum)} bands, legacy dict format)")
                        print(f"    Wavelength range: {wavelengths[0]:.1f}-{wavelengths[-1]:.1f} nm")
                else:
                    self.reference_teflon = spectrum
                    if self.verbose:
                        print(f"✓ Configuration loaded from: {filepath}")
                        print(f"  ✓ Restored reference teflon spectrum ({len(spectrum)} bands, legacy dict format, no wavelengths)")
                        
            elif format_type == 'legacy_array':
                # Legacy array format
                self.reference_teflon = np.array(ref_data['data'])
                if self.verbose:
                    print(f"✓ Configuration loaded from: {filepath}")
                    print(f"  ✓ Restored reference teflon spectrum ({len(self.reference_teflon)} bands, legacy array format)")
                    print(f"  ⚠ Warning: No wavelength information found. Consider regenerating for multi-camera compatibility")
            else:
                # Unknown format
                self.reference_teflon = ref_data['data']
                if self.verbose:
                    print(f"✓ Configuration loaded from: {filepath}")
                    print(f"  ⚠ Warning: Unknown reference teflon format: {format_type}")
                    
        elif 'reference_teflon' in saved_config and saved_config['reference_teflon'] is not None:
            # Old config file format - spectrum only (backward compatibility)
            self.reference_teflon = np.array(saved_config['reference_teflon'])
            if self.verbose:
                print(f"✓ Configuration loaded from: {filepath}")
                print(f"  ✓ Restored reference teflon spectrum ({len(self.reference_teflon)} bands) - legacy format")
                print(f"  ⚠ Warning: No wavelength information found. Consider regenerating for multi-camera compatibility")
                
        else:
            # No reference teflon
            self.reference_teflon = None
            if self.verbose:
                print(f"✓ Configuration loaded from: {filepath}")
                print(f"  ℹ No reference teflon spectrum found in config")
        
        return self
    
    def create_reference_teflon_library(self, hs_images, teflon_edge_coord=(-10,-3), 
                                       white_ref_path=None, dark_calibration=False, clip_to=10):
        """
        Create a reference teflon spectrum from multiple images taken under optimal conditions.
        
        Args:
            hs_images: List of paths to hyperspectral images OR directory path containing images
            teflon_edge_coord: Tuple of (start, end) coordinates for teflon panel location
            white_ref_path: Path to white reference file (None for auto-detection)
            dark_calibration: Use dark calibration (True/False)
            clip_to: Maximum reflectance value for sensor calibration
            
        Returns:
            numpy.ndarray: Reference teflon spectrum (median of all input spectra)
        """
        # Check if input is a directory or list of files
        if isinstance(hs_images, str) and os.path.isdir(hs_images):
            # It's a directory - find all files ending with "Data.hdr"
            hs_images_list = glob.glob(os.path.join(hs_images, "*Data.hdr"))
            if not hs_images_list:
                raise ValueError(f"No files ending with 'Data.hdr' found in directory: {hs_images}")
            if self.verbose:
                print(f"Found {len(hs_images_list)} Data.hdr files in directory")
        else:
            # It's a list of file paths
            hs_images_list = hs_images
        
        if not hs_images_list:
            raise ValueError("hs_images_list cannot be empty")
        
        teflon_spectra = []
        
        for hs_image_path in hs_images_list:
            try:
                # Create temporary processor for this image (inherit verbose setting)
                temp_processor = HS_preprocessor(hs_image_path, verbose=False)
                
                # Apply sensor calibration with specified parameters
                temp_processor.sensor_calibration(
                    white_ref_path=white_ref_path,
                    dark_calibration=dark_calibration,
                    clip_to=clip_to
                )
                
                # Extract teflon spectrum from the edge
                teflon_spectrum = temp_processor.image.img[:, teflon_edge_coord[0]:teflon_edge_coord[1], :].mean(axis=(0,1))
                teflon_spectra.append(teflon_spectrum)
                
            except Exception as e:
                if self.verbose:
                    print(f"Failed to process {os.path.basename(hs_image_path)}: {str(e)}")
                continue
        
        if not teflon_spectra:
            raise ValueError("No valid teflon spectra could be extracted")
        
        # Use median to get robust reference (less affected by outliers)
        reference_teflon = np.median(teflon_spectra, axis=0)
        
        # Store the created reference teflon spectrum and wavelengths as class attribute
        reference_wavelengths = None
        # Get wavelengths from the last processed image (they should all be the same)
        if hasattr(temp_processor, 'image') and temp_processor.image is not None:
            reference_wavelengths = np.array(temp_processor.image.ind)
        else:
            # Fallback: use current image wavelengths if available
            if hasattr(self, 'image') and self.image is not None:
                reference_wavelengths = np.array(self.image.ind)
            else:
                reference_wavelengths = None
                if self.verbose:
                    print("  ⚠ Warning: Could not determine wavelengths for reference teflon")
        
        # Store in new wavelength-as-keys format
        if reference_wavelengths is not None:
            self.reference_teflon = {float(wl): float(refl) for wl, refl in zip(reference_wavelengths, reference_teflon)}
        else:
            # Fallback to old array format if no wavelengths available
            self.reference_teflon = reference_teflon.copy()
        
        if self.verbose:
            print(f"✓ Created reference teflon from {len(teflon_spectra)} images")
            if reference_wavelengths is not None:
                wl_range = f"{reference_wavelengths[0]:.1f}-{reference_wavelengths[-1]:.1f} nm"
                print(f"  ✓ Stored reference teflon spectrum ({len(reference_teflon)} bands) with wavelength keys")
                print(f"    Wavelength range: {wl_range}")
                print(f"    Access example: reference_teflon[{reference_wavelengths[len(reference_wavelengths)//2]:.1f}] = {reference_teflon[len(reference_teflon)//2]:.4f}")
            else:
                print(f"  ✓ Stored reference teflon spectrum ({len(reference_teflon)} bands) - array format")
        
        return reference_teflon

    @staticmethod
    def create_config_template():
        """Create a template configuration dictionary for user to fill."""
        template = {
            'sensor_calibration': {
                'clip_to': 10,  # Maximum reflectance value
                'white_ref_path': None,  # Path to white reference, or None for auto-detection
                'dark_calibration': False  # Use dark calibration (True/False)
            },
            'solar_correction': {
                'teflon_edge_coord': (-10, -3),  # Teflon panel location in image
                'smooth_window': 35,  # Smoothing window for teflon spectrum
                'reference_teflon': None  # Reference teflon spectrum, or None for auto
            },
            'spectral_smoothing': {
                'sigma': 11,  # Gaussian smoothing parameter
                'mode': 'reflect'  # Boundary handling mode
            },
            'normalization': {
                'method': 'to_wl',  # 'to_wl' for wavelength normalization, 'snv' for SNV
                'to_wl': 751,  # Wavelength for normalization (nm) - only used if method='to_wl'
                'clip_to': 10  # Maximum normalized value - only used if method='to_wl'
            },
            'mask_extraction': {
                'pri_thr': -0.1,  # PRI threshold for vegetation detection
                'ndvi_thr': 0.2,  # NDVI threshold for vegetation detection
                'hbsi_thr': -0.6,  # HBSI threshold for vegetation detection
                'min_pix_size': 2  # Minimum object size in pixels
            }
        }
        
        print("Enhanced configuration template created with mask extraction parameters!")
        return template
    
    @staticmethod
    def process_folder(folder_path, white_ref_path=None, reference_teflon=None, 
                      config=None, config_path=None, pattern="*Data.hdr", verbose=True,
                      extract_masks=True):
        """
        Process all hyperspectral images in a folder with integrated mask extraction.
        
        Args:
            folder_path (str): Path to folder containing images
            white_ref_path (str, optional): Path to white reference file
            reference_teflon (dict/array, optional): Reference teflon spectrum
            config (dict, optional): Configuration dictionary
            config_path (str, optional): Path to config file to load (takes precedence over config)
            pattern (str): File pattern to match (default: "*Data.hdr")
            verbose (bool): Enable verbose output
            extract_masks (bool): Whether to extract masks as final step
            
        Returns:
            dict: Dictionary of processed images with masks {filename: HS_image}
        """
        # Load config from file if config_path is provided
        loaded_config = None
        loaded_reference_teflon = None
        
        if config_path is not None:
            if verbose:
                print(f"Loading configuration from: {config_path}")
            
            # Create temporary processor to load config
            temp_processor = HS_preprocessor(verbose=False)
            temp_processor.load_config(config_path)
            
            loaded_config = temp_processor.get_config()
            loaded_reference_teflon = temp_processor.reference_teflon
            
            if verbose:
                print(f"  ✓ Loaded configuration with {len(loaded_config)} pipeline steps")
                if loaded_reference_teflon is not None:
                    if isinstance(loaded_reference_teflon, dict) and all(isinstance(k, (int, float)) for k in loaded_reference_teflon.keys()):
                        wl_keys = sorted(loaded_reference_teflon.keys())
                        print(f"Loaded reference teflon ({len(wl_keys)} bands, wavelength-keys format)")
                        print(f"Range: {wl_keys[0]:.1f}-{wl_keys[-1]:.1f} nm")
                    else:
                        print(f"Loaded reference teflon (legacy format)")
                else:
                    print(f" No reference teflon in config file")
        
        # Use loaded config/reference if available, otherwise use provided parameters
        final_config = loaded_config if loaded_config is not None else config
        final_reference_teflon = loaded_reference_teflon if loaded_reference_teflon is not None else reference_teflon
        
        # Find all images
        image_paths = glob.glob(os.path.join(folder_path, pattern))
        
        if not image_paths:
            if verbose:
                print(f"No images found matching pattern '{pattern}' in {folder_path}")
            return {}
        
        if verbose:
            print(f"Found {len(image_paths)} images to process")
        
        results = {}
        
        for img_path in image_paths:
            try:
                filename = os.path.basename(img_path)
                if verbose:
                    print(f"\nProcessing: {filename}")
                
                # Create processor and run pipeline (inherit verbose setting)
                processor = HS_preprocessor(img_path, verbose=verbose)
                
                # Set the processor's config from the loaded/provided configuration
                if final_config is not None:
                    processor.config = final_config.copy()
                
                # Set reference teflon if provided
                if final_reference_teflon is not None:
                    if isinstance(final_reference_teflon, dict):
                        # Check if it's the new wavelength-as-keys format
                        if all(isinstance(k, (int, float)) for k in final_reference_teflon.keys()):
                            # Already in new format
                            processor.reference_teflon = final_reference_teflon.copy()
                        elif 'spectrum' in final_reference_teflon:
                            # Legacy dictionary format
                            processor.reference_teflon = final_reference_teflon.copy()
                        else:
                            # Unknown dict format
                            processor.reference_teflon = final_reference_teflon
                            if verbose:
                                print(f"Warning: Unknown reference teflon dictionary format for {filename}")
                    else:
                        # Legacy array format
                        processor.reference_teflon = final_reference_teflon
                        if verbose:
                            print(f"Warning: Using legacy reference teflon array format for {filename}")
                
                # Set up config override for external parameters
                config_override = {}
                
                # Add white reference path to sensor calibration config if provided
                if white_ref_path:
                    config_override['sensor_calibration'] = {'white_ref_path': white_ref_path}
                
                processor.run_full_pipeline(config_override=config_override if config_override else None,
                                          extract_masks_flag=extract_masks)
                
                # Store result
                results[filename] = processor.get_current_image()
                
                if verbose:
                    print(f"Completed: {filename}")
                    if extract_masks and hasattr(processor.image, 'mask'):
                        mask_pixels = np.sum(processor.image.mask)
                        total_pixels = processor.image.mask.shape[0] * processor.image.mask.shape[1]
                        print(f"Mask: {mask_pixels}/{total_pixels} pixels ({100*mask_pixels/total_pixels:.1f}%)")
                
            except Exception as e:
                if verbose:
                    print(f"Failed {filename}: {str(e)}")
                results[filename] = None
        
        if verbose:
            successful_results = [r for r in results.values() if r is not None]
            print(f"\nProcessed {len(successful_results)}/{len(image_paths)} images successfully")
            if extract_masks:
                mask_count = sum(1 for r in successful_results if hasattr(r, 'mask') and r.mask is not None)
                print(f"Masks extracted for {mask_count}/{len(successful_results)} processed images")
        return results
    
    
    
    def sensor_calibration(self, white_ref_path=None, dark_calibration=False, clip_to=10):
        """Step 2: Apply white reference calibration (transferred from readHS.py)."""
        if self.image is None:
            raise ValueError("No image loaded. Call load_image() first.")
        
        # Get calibration matrices
        white_matrix, dark_matrix = self._upload_calibration(dark_calibration=dark_calibration, white_ref_path=white_ref_path)
        
        # Apply calibration: (raw - dark) / (white - dark)
        self.image.img = np.clip((self.image.img - dark_matrix) / (white_matrix - dark_matrix), 0, clip_to)
        
        # Clean up NaN and Inf values
        self.image.img[np.isnan(self.image.img)] = 0
        self.image.img[np.isinf(self.image.img)] = clip_to
        self.image.calibrated = True
        
        # Store config and results
        self.config['sensor_calibration'] = {
            'clip_to': clip_to, 
            'white_ref_path': white_ref_path,
            'dark_calibration': dark_calibration
        }
        self.step_results['sensor_calibrated'] = copy.deepcopy(self.image)
        
        if self.verbose:
            print(f"✓ Applied sensor calibration (clip_to={clip_to}, dark_cal={dark_calibration})")
            if white_ref_path:
                print(f"  Used external white reference: {os.path.basename(white_ref_path)}")
            else:
                print(f"  Used auto-detected white calibration")
        return self
    
    def solar_correction(self, teflon_edge_coord=(-10, -3), reference_teflon=None, smooth_window=35):
        """Step 3: Solar spectrum correction using side teflon panel."""
        if self.image is None:
            raise ValueError("No image loaded.")
        if not hasattr(self.image, 'calibrated') or not self.image.calibrated:
            if self.verbose:
                print("Warning: Image not sensor calibrated. Consider running sensor_calibration() first.")
        
        # Extract current teflon spectrum from image edge
        current_teflon = self.image.img[:, teflon_edge_coord[0]:teflon_edge_coord[1], :].mean(axis=(0,1))
        
        # Smooth the teflon spectrum
        current_teflon_smooth = savgol_filter(current_teflon, window_length=smooth_window, polyorder=3)
        
        # Determine which reference teflon to use (priority order):
        # 1. User provided reference_teflon parameter
        # 2. Stored reference_teflon attribute from previous use
        # 3. Create idealized synthetic spectrum
        if reference_teflon is not None:
            # Use provided reference teflon (highest priority)
            used_reference = reference_teflon
            reference_source = "user_provided"
            if self.verbose:
                print("  Using user-provided reference teflon spectrum")
        elif self.reference_teflon is not None:
            # Use stored reference teflon (medium priority)
            if isinstance(self.reference_teflon, dict):
                current_wavelengths = np.array(self.image.ind)
                
                # Check if it's the new wavelength-as-keys format
                if all(isinstance(k, (int, float)) for k in self.reference_teflon.keys()):
                    # New format: {wavelength: reflectance}
                    ref_wavelengths = np.array(sorted(self.reference_teflon.keys()))
                    ref_spectrum = np.array([self.reference_teflon[wl] for wl in ref_wavelengths])
                    
                    if np.array_equal(ref_wavelengths, current_wavelengths):
                        # Wavelengths match - use directly
                        used_reference = ref_spectrum
                        reference_source = "stored_wavelength_keys_direct"
                        if self.verbose:
                            print("  Using stored reference teflon spectrum (wavelength keys, direct match)")
                    else:
                        # Wavelengths don't match - interpolate
                        from scipy.interpolate import interp1d
                        interp_func = interp1d(ref_wavelengths, ref_spectrum, 
                                             kind='linear', bounds_error=False, fill_value='extrapolate')
                        used_reference = interp_func(current_wavelengths)
                        reference_source = "stored_wavelength_keys_interpolated"
                        if self.verbose:
                            print(f"  Using stored reference teflon spectrum (wavelength keys, interpolated)")
                            print(f"    Original range: {ref_wavelengths[0]:.1f}-{ref_wavelengths[-1]:.1f} nm ({len(ref_wavelengths)} bands)")
                            print(f"    Target range: {current_wavelengths[0]:.1f}-{current_wavelengths[-1]:.1f} nm ({len(current_wavelengths)} bands)")
                
                elif 'spectrum' in self.reference_teflon and 'wavelengths' in self.reference_teflon:
                    # Legacy dictionary format with 'spectrum' and 'wavelengths' keys
                    ref_spectrum = self.reference_teflon['spectrum']
                    ref_wavelengths = self.reference_teflon['wavelengths']
                    
                    if ref_wavelengths is not None:
                        if np.array_equal(ref_wavelengths, current_wavelengths):
                            used_reference = ref_spectrum
                            reference_source = "stored_legacy_dict_direct"
                            if self.verbose:
                                print("  Using stored reference teflon spectrum (legacy dict format, wavelengths match)")
                        else:
                            from scipy.interpolate import interp1d
                            interp_func = interp1d(ref_wavelengths, ref_spectrum, 
                                                 kind='linear', bounds_error=False, fill_value='extrapolate')
                            used_reference = interp_func(current_wavelengths)
                            reference_source = "stored_legacy_dict_interpolated"
                            if self.verbose:
                                print(f"  Using stored reference teflon spectrum (legacy dict format, interpolated)")
                    else:
                        used_reference = ref_spectrum
                        reference_source = "stored_legacy_dict_no_wavelengths"
                        if self.verbose:
                            print("  Using stored reference teflon spectrum (legacy dict format, no wavelength info)")
                else:
                    # Unknown dictionary format
                    raise ValueError("Unknown reference teflon dictionary format")
            else:
                # Old format - assume it's just the spectrum array (backward compatibility)
                used_reference = self.reference_teflon
                reference_source = "stored_legacy_array"
                if self.verbose:
                    print("  Using stored reference teflon spectrum (legacy array format - assuming wavelengths match)")
                    print("  ⚠ Warning: Consider regenerating reference with wavelength information for multi-camera compatibility")
        else:
            # Create idealized teflon spectrum (lowest priority)
            wavelengths = np.array(self.image.ind)
            used_reference = 0.95 + 0.05 * (wavelengths - wavelengths.min()) / (wavelengths.max() - wavelengths.min())
            used_reference = used_reference * np.mean(current_teflon_smooth)
            reference_source = "synthetic"
            if self.verbose:
                print("  Using synthetic idealized teflon spectrum")
        
        # Store the reference teflon spectrum as class attribute if it came from user
        if reference_teflon is not None:
            # Store in new wavelength-as-keys format
            current_wavelengths = np.array(self.image.ind)
            self.reference_teflon = {float(wl): float(refl) for wl, refl in zip(current_wavelengths, reference_teflon)}
            if self.verbose:
                print(f"  ✓ Stored reference teflon spectrum with {len(self.reference_teflon)} wavelength keys")
        
        # Apply smoothing to the reference if needed
        if reference_source != "synthetic":
            reference_teflon_smooth = savgol_filter(used_reference, window_length=smooth_window, polyorder=3)
        else:
            reference_teflon_smooth = used_reference  # Already smooth synthetic spectrum
        
        # Calculate solar correction
        solar_correction = reference_teflon_smooth / (current_teflon_smooth + 1e-7)
        
        # Apply correction with reasonable limits
        solar_correction = np.clip(solar_correction, 0.1, 5.0)
        self.image.img = self.image.img * solar_correction
        
        # Clean up
        self.image.img[np.isnan(self.image.img)] = 0
        self.image.img[np.isinf(self.image.img)] = 0
        
        # Store config and results
        self.config['solar_correction'] = {
            'teflon_edge_coord': teflon_edge_coord,
            'smooth_window': smooth_window,
            'reference_source': reference_source,
            'has_reference': reference_teflon is not None,
        }
        self.step_results['solar_corrected'] = copy.deepcopy(self.image)
        
        if self.verbose:
            print(f"✓ Applied solar spectrum correction (window={smooth_window}, source={reference_source})")
        return self
    
    def spectral_smoothing(self, sigma=11, mode='reflect'):
        """Step 4: Apply Gaussian smoothing to reduce spectral noise."""
        if self.image is None:
            raise ValueError("No image loaded.")
        
        # Apply Gaussian smoothing along spectral dimension (axis=2)
        self.image.img = gaussian_filter1d(
            self.image.img,
            sigma=sigma,
            axis=2,  # Spectral dimension
            mode=mode
        )
        
        # Store config and results
        self.config['spectral_smoothing'] = {'sigma': sigma, 'mode': mode}
        self.step_results['smoothed'] = copy.deepcopy(self.image)
        
        if self.verbose:
            print(f"✓ Applied spectral smoothing (sigma={sigma})")
        return self
    
    def normalization(self, to_wl=751, clip_to=10, method="to_wl"):
        """Step 5: Final normalization to specific wavelength or SNV normalization."""
        if self.image is None:
            raise ValueError("No image loaded.")
        
        if method == "snv":
            # Apply SNV normalization along spectral axis
            if hasattr(self.image, 'normalized') and self.image.normalized:
                if self.verbose:
                    print("Image already normalized. Skipping.")
            else:
                # Use HS_image's SNV method if available, otherwise implement here
                if hasattr(self.image, 'apply_snv'):
                    self.image.apply_snv()
                else:
                    # Implement SNV directly
                    original_shape = self.image.img.shape
                    # Reshape to 2D: (pixels, bands)
                    reshaped_img = self.image.img.reshape(-1, original_shape[2])
                    
                    # Calculate mean and std along spectral axis (axis=1)
                    mean_spectrum = np.mean(reshaped_img, axis=1, keepdims=True)
                    std_spectrum = np.std(reshaped_img, axis=1, keepdims=True)
                    
                    # Apply SNV transformation with small epsilon to prevent division by zero
                    epsilon = 1e-7
                    snv_data = (reshaped_img - mean_spectrum) / (std_spectrum + epsilon)
                    
                    # Reshape back to original shape
                    self.image.img = snv_data.reshape(original_shape)
                    
                    # Clean up any remaining NaN or Inf values
                    self.image.img[np.isnan(self.image.img)] = 0
                    self.image.img[np.isinf(self.image.img)] = 0
                
                self.image.normalized = True
                if self.verbose:
                    print(f"✓ Applied SNV normalization along spectral axis")
        
        elif method == "to_wl":
            # Original wavelength-specific normalization
            if to_wl is None:
                raise ValueError("to_wl cannot be None when method='to_wl'")
                
            if hasattr(self.image, 'normalized') and self.image.normalized:
                if self.verbose:
                    print("Image already normalized. Skipping.")
            else:
                # Use HS_image's normalize method
                self.image.normalize(to_wl=to_wl, clip_to=clip_to)
            
            if self.verbose:
                print(f"✓ Applied final normalization to {to_wl}nm (clip_to={clip_to})")
        
        else:
            raise ValueError(f"Unknown normalization method: {method}. Use 'to_wl' or 'snv'")
        
        # Store config and results
        self.config['normalization'] = {
            'method': method,
            'to_wl': to_wl if method == "to_wl" else None,
            'clip_to': clip_to if method == "to_wl" else None
        }
        self.step_results['normalized'] = copy.deepcopy(self.image)
        
        return self
    
    # === VISUALIZATION METHODS ===
    
    def get_rgb_sample(self, normalize=True, correct=True, show=True, title='RGB Sample', axes=False, repeat=1):
        """
        Generate and optionally display RGB representation of the current processed image.
        Adapted from get_rgb_sample function in readHS.py with enhanced SNV support.
        """
        if self.image is None:
            raise ValueError("No image loaded.")
        
        image = self.image
        
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
            if self.verbose:
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
                
            if self.verbose:
                print(f"  ✓ Applied robust normalization: R[{np.min(R):.3f}, {np.max(R):.3f}], G[{np.min(G):.3f}, {np.max(G):.3f}], B[{np.min(B):.3f}, {np.max(B):.3f}]")
        
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
    

    def spectrum_probe(
            self, 
            normalize=True, 
            correct=False, 
            show=True, 
            title='RGB Sample', 
            axes=False, 
            repeat=1, 
            rois=None,
            figure_size=(8, 6)):
        
        rgb_img = self.get_rgb_sample(normalize=normalize, correct=correct, repeat=repeat, show=False)
        if rois is not None:
            probed_spectra = {}
            for roi_name, slices in rois.items():
                mean_spectrum, wavelengths = self.get_spectrum(slices, show=False)
                probed_spectra[roi_name] = {
                    "spectrum": mean_spectrum,
                    "wavelengths": wavelengths,
                    "roi": slices
                }
        if show:
            # Create subplots: RGB image on top, spectrum plot below
            if rois is not None:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(figure_size[0], figure_size[1] * 1.5))
            else:
                fig, ax1 = plt.subplots(1, 1, figsize=figure_size)
            
            # Display RGB image
            ax1.imshow(rgb_img)
            
            if rois is not None:
                # Get matplotlib default colors
                colors = plt.cm.tab10(np.linspace(0, 1, len(probed_spectra)))
                
                # Add colored ROI bounding boxes and labels on RGB image
                for i, (roi_name, data) in enumerate(probed_spectra.items()):
                    slices = data["roi"]
                    y_slice, x_slice = slices
                    
                    # Apply repeat amplification to y-coordinates to match stretched image
                    y_start_stretched = y_slice.start * repeat
                    y_stop_stretched = y_slice.stop * repeat
                    x_start = x_slice.start
                    x_stop = x_slice.stop
                    
                    # Calculate center for text label
                    y_center = ((y_slice.start + y_slice.stop) // 2) * repeat
                    x_center = (x_slice.start + x_slice.stop) // 2
                    
                    # Get the color for this ROI
                    color = colors[i]
                    
                    # Add colored bounding box rectangle
                    from matplotlib.patches import Rectangle
                    rect = Rectangle((x_start, y_start_stretched), 
                                   x_stop - x_start, 
                                   y_stop_stretched - y_start_stretched,
                                   linewidth=3, edgecolor=color, facecolor='none', alpha=0.8)
                    ax1.add_patch(rect)
                    
                    # Add colored text box
                    ax1.text(x_center, y_start_stretched-2, roi_name, color='white', fontsize=12,
                            ha='center', va='center', 
                            bbox=dict(facecolor=color, alpha=0.3, pad=2))
                
                # Plot spectra with matching colors
                for i, (roi_name, data) in enumerate(probed_spectra.items()):
                    spectrum = data["spectrum"]
                    wavelengths = data["wavelengths"]
                    color = colors[i]
                    
                    ax2.plot(wavelengths, spectrum, color=color, label=roi_name, linewidth=2)
                
                ax2.set_xlabel('Wavelength (nm)')
                ax2.set_ylabel('Reflectance')
                ax2.set_title('Spectra from ROIs')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            
            ax1.set_title(title)
            if not axes:
                ax1.axis('off')
            
            plt.tight_layout()
            plt.show()
        
        if rois is not None:
            return probed_spectra
        else:
            return rgb_img



    
    
    @staticmethod
    def extract_masked_spectra_to_df(processed_images_dict, save_path=None, segmentation=False):
        """
        Extract pixel spectra from hyperspectral images into a DataFrame.

        Parameters
        ----------
        processed_images_dict : dict
            {filename: HS_image} where HS_image has .img (H, W, B), .mask, .ind (wavelengths), .name
        save_path : str, optional
            If given, save the resulting DataFrame to CSV at this path.
        segmentation : bool, optional (default False)
            If True, also extract spectra of non-masked pixels and label them as 'BG'.

        Returns
        -------
        pd.DataFrame
            Columns are wavelengths from hs_image.ind plus a 'label' column.
        """
        extracted_spectra = []

        for filename, hs_image in processed_images_dict.items():
            if hs_image is None or not hasattr(hs_image, 'img') or hs_image.img is None:
                print(f"Skipping {filename}: no image data")
                continue
            if not hasattr(hs_image, 'mask') or hs_image.mask is None:
                print(f"Skipping {filename}: no mask data")
                continue

            img = hs_image.img  # (H, W, B)
            if img.ndim != 3:
                print(f"Skipping {filename}: unexpected image shape {img.shape}")
                continue

            H, W, B = img.shape
            flat_img = img.reshape(-1, B)

            # Normalize mask to 2D boolean (H, W)
            m = hs_image.mask
            if m.ndim == 3:
                if m.shape[2] == 1:
                    m2d = m[:, :, 0]
                elif m.shape[2] == B:
                    # Per-band mask -> any band marked as True counts as FG
                    m2d = m.any(axis=2)
                else:
                    # Fallback: any nonzero across channels
                    m2d = (m != 0).any(axis=2)
            elif m.ndim == 2:
                m2d = m
            else:
                print(f"Skipping {filename}: unexpected mask shape {m.shape}")
                continue

            mask_flat = m2d.reshape(-1).astype(bool)

            # Wavelengths -> ensure correct length B
            if not hasattr(hs_image, 'ind'):
                print(f"Skipping {filename}: hs_image.ind (wavelengths) not found")
                continue
            cols = list(hs_image.ind)
            if len(cols) != B:
                print(f"Skipping {filename}: wavelengths length {len(cols)} != bands {B}")
                continue

            # Label from name or filename
            try:
                base = hs_image.name if hasattr(hs_image, 'name') and hs_image.name else filename
                parts = base.split("-")
                label_fg = parts[2] if len(parts) > 2 else "FG"
            except Exception:
                label_fg = "FG"

            # Foreground spectra
            fg_rows = flat_img[mask_flat]
            if fg_rows.size > 0:
                fg_df = pd.DataFrame(fg_rows, columns=cols)
                fg_df["label"] = label_fg
                extracted_spectra.append(fg_df)

            # Background spectra (optional)
            if segmentation:
                bg_rows = flat_img[~mask_flat]
                if bg_rows.size > 0:
                    bg_df = pd.DataFrame(bg_rows, columns=cols)
                    bg_df["label"] = "BG"
                    extracted_spectra.append(bg_df)
                print(f"Extracted {len(fg_rows)} FG and {len(bg_rows)} BG pixels from {filename}")
            else:
                if fg_rows.size > 0:
                    print(f"Extracted {len(fg_rows)} masked (FG) pixels from {filename}")
                else:
                    print(f"No masked pixels found in {filename}")

        if not extracted_spectra:
            print("No spectral data extracted!")
            return pd.DataFrame()

        out_df = pd.concat(extracted_spectra, ignore_index=True)

        if save_path is not None:
            out_df.to_csv(save_path, index=False)

        return out_df
