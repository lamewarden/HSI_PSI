"""
Enhanced usage example for HSI_PSI package v0.3.0

This example demonstrates the new features including:
1. Spectral range cropping capabilities
2. Enhanced preprocessing pipeline with optimized order
3. Noise analysis functions
4. Wavelength mapping for different sensor configurations
5. Reference teflon library creation
6. NEW: Dimensionality reduction with PCA and MNF transformers
"""

import hsi_psi
from hsi_psi import HS_image, HS_preprocessor, get_rgb_sample
from hsi_psi.dim_red import transformer
from hsi_psi.utils import rank_noisy_bands, summarize_noisiest_bands, create_config_template

def basic_example():
    """Enhanced basic usage example with new features"""
    print(f"HSI_PSI version: {hsi_psi.__version__}")
    print("Enhanced Hyperspectral Image Processing Library")
    print("=" * 50)
    
    # Example image path (replace with your actual path)
    image_path = "data/sample_image.hdr"
    
    try:
        # Load hyperspectral image
        print(f"\n1. Loading image: {image_path}")
        image = HS_image(image_path)
        
        # Display basic info
        print(f"   Original shape: {image.img.shape}")
        print(f"   Wavelength range: {min(image.ind):.1f}-{max(image.ind):.1f} nm")
        print(f"   Number of bands: {len(image.ind)}")
        
        # NEW: Demonstrate spectral cropping
        print(f"\n2. Spectral Range Cropping (NEW FEATURE)")
        original_bands = len(image.ind)
        image.crop_spectral_range(wl_start=450, wl_end=800)
        print(f"   Cropped to vegetation range: {image.ind[0]}-{image.ind[-1]} nm")
        print(f"   Bands reduced: {original_bands} → {len(image.ind)}")
        
        # Extract RGB representation
        print(f"\n3. RGB Extraction")
        rgb = get_rgb_sample(image, show=False, title="Cropped HS Image")
        print(f"   RGB image shape: {rgb.shape}")
        
        # NEW: Enhanced preprocessor with optimized pipeline
        print(f"\n4. Enhanced Preprocessing Pipeline")
        processor = HS_preprocessor(image_path, verbose=True)
        
        # NEW: Create enhanced configuration
        config = create_config_template()
        config['spectral_cropping']['wl_start'] = 450
        config['spectral_cropping']['wl_end'] = 800
        
        print("   Configuration created with spectral cropping")
        print(f"   Pipeline order: sensor_cal → spike_removal → cropping → solar → smoothing → normalization")
        
        # Show available processing steps
        pipeline_steps = ['sensor_calibration', 'spike_removal', 'spectral_cropping', 
                         'solar_correction', 'spectral_smoothing', 'normalization', 'mask_extraction']
        print(f"   Available steps: {', '.join(pipeline_steps)}")
        
    except FileNotFoundError:
        print(f"\nImage file not found: {image_path}")
        print("Please update the image_path variable with a valid .hdr file")
        print("\nDemonstrating configuration template creation instead:")
        demonstrate_config_template()
    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure you have a valid hyperspectral image file")
        print("\nDemonstrating configuration template creation instead:")
        demonstrate_config_template()

def demonstrate_config_template():
    """Demonstrate new configuration template features"""
    print("\n" + "=" * 50)
    print("CONFIGURATION TEMPLATE DEMONSTRATION")
    print("=" * 50)
    
    # Create configuration template
    config = create_config_template()
    
    print("\nNew enhanced configuration template includes:")
    for section, params in config.items():
        print(f"   {section}:")
        if isinstance(params, dict):
            for key, value in params.items():
                print(f"      {key}: {value}")
        print()

def demonstrate_noise_analysis():
    """Demonstrate noise analysis capabilities (requires actual image)"""
    print("\n" + "=" * 50)
    print("NOISE ANALYSIS CAPABILITIES")
    print("=" * 50)
    
    print("New noise analysis functions available:")
    print("   - rank_noisy_bands(): Identify problematic spectral bands")
    print("   - summarize_noisiest_bands(): Generate noise reports")
    print("   - Robust statistical methods using Savitzky-Golay filtering")
    print("\nExample usage:")
    print("   noise_ranking = rank_noisy_bands(hs_image, method='savgol_residuals')")
    print("   summary = summarize_noisiest_bands(noise_ranking, top_n=10)")

def demonstrate_wavelength_mapping():
    """Demonstrate wavelength mapping capabilities"""
    print("\n" + "=" * 50)
    print("WAVELENGTH MAPPING CAPABILITIES")
    print("=" * 50)
    
    print("Enhanced wavelength mapping features:")
    print("   ✓ Automatic interpolation between different sensor configurations")
    print("   ✓ Calibration files can have any spectral range/resolution")
    print("   ✓ Reference teflon spectra automatically adapt to current image")
    print("   ✓ No manual spectral alignment required")
    print("\nExample scenarios handled automatically:")
    print("   - Image: 450-800nm (cropped) + Calibration: 350-1000nm (full range)")
    print("   - Image: 200 bands + Reference teflon: 651 bands")
    print("   - Different sensors with different wavelength sampling")

if __name__ == "__main__":
    basic_example()
    demonstrate_noise_analysis()
    demonstrate_wavelength_mapping()
    
    print("\n" + "=" * 50)
    print("HSI_PSI v0.2.0 - Ready for advanced hyperspectral analysis!")
    print("Optimized for close-range vegetation monitoring")
    print("=" * 50)
