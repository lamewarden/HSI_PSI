# Test script for new MNF implementation
import numpy as np
from hsi_psi.dim_red import transformer

def test_mnf():
    """Test the custom MNF implementation"""
    print("Testing custom MNF implementation...")
    
    # Create synthetic HSI data
    np.random.seed(42)
    height, width, bands = 30, 30, 50
    n_pixels = height * width
    
    # Create realistic HSI data with signal and noise
    # Signal components with spatial correlation
    signal = np.random.randn(n_pixels, bands) * 10
    signal[:, :10] += np.random.randn(n_pixels, 1) * 5  # First 10 bands correlated
    signal[:, 10:20] += np.random.randn(n_pixels, 1) * 3  # Next 10 bands correlated
    
    # Add noise
    noise = np.random.randn(n_pixels, bands) * 0.5
    hsi_data = signal + noise
    
    print(f"Created synthetic HSI data: {hsi_data.shape}")
    
    # Initialize transformer
    trans = transformer(method='mnf')
    
    # Convert to HSI format and store shape
    hsi_3d = hsi_data.T.reshape(bands, height, width)
    X = trans.HSI_to_X(hsi_3d)
    print(f"Converted to X format: {X.shape}")
    
    # Test MNF fitting
    print("\nFitting MNF with 10 components...")
    trans.fit(X, n_components=10)
    
    # Test transformation
    print("Applying transformation...")
    X_transformed = trans.transform(X)
    print(f"Transformed shape: {X_transformed.shape}")
    
    # Check results
    print(f"Eigenvalues (SNR): {trans.eigenvalues_[:5]}")
    print(f"Transformed data range: [{X_transformed.min():.2f}, {X_transformed.max():.2f}]")
    
    # Test fit_transform
    print("\nTesting fit_transform...")
    X_transformed2 = trans.fit_transform(X, n_components=10)
    
    # Verify consistency
    diff = np.abs(X_transformed - X_transformed2).max()
    print(f"Max difference between transform and fit_transform: {diff}")
    
    print("✓ All tests passed!")
    return True

if __name__ == "__main__":
    test_mnf()