"""
Basic usage example for HSI_PSI package

This example demonstrates how to:
1. Import the package
2. Load a hyperspectral image
3. Extract RGB representation
4. Run basic preprocessing
"""

import hsi_psi
from hsi_psi import HS_image, HS_preprocessor, get_rgb_sample

def basic_example():
    """Basic usage example"""
    print(f"HSI_PSI version: {hsi_psi.__version__}")
    
    # Example image path (replace with your actual path)
    image_path = "data/sample_image.hdr"
    
    try:
        # Load hyperspectral image
        print(f"Loading image: {image_path}")
        image = HS_image(image_path)
        
        # Display basic info
        print(f"Image shape: {image.img.shape}")
        print(f"Wavelength range: {min(image.ind):.1f}-{max(image.ind):.1f} nm")
        print(f"Number of bands: {len(image.ind)}")
        
        # Extract RGB representation
        rgb = get_rgb_sample(image, show=False, title="Sample HS Image")
        print(f"RGB image shape: {rgb.shape}")
        
        # Initialize preprocessor
        processor = HS_preprocessor(image_path, verbose=True)
        print("Preprocessor initialized successfully")
        
    except FileNotFoundError:
        print(f"Image file not found: {image_path}")
        print("Please update the image_path variable with a valid .hdr file")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have a valid hyperspectral image file")

if __name__ == "__main__":
    basic_example()
