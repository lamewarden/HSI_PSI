"""
Example comparing PCA and MNF transformations on hyperspectral data.

This example demonstrates the differences between PCA and MNF (Minimum Noise Fraction)
transformations on synthetic hyperspectral data.
"""

import numpy as np
import matplotlib.pyplot as plt
from hsi_psi.dim_red import transformer

def create_noisy_hsi_data():
    """Create synthetic HSI data with signal and noise components."""
    np.random.seed(42)
    
    # Dimensions
    height, width, bands = 40, 40, 80
    n_pixels = height * width
    
    # Create signal with spatial patterns
    x, y = np.meshgrid(np.linspace(0, 4*np.pi, width), np.linspace(0, 4*np.pi, height))
    
    # Different spectral signatures for different spatial regions
    sig1 = np.sin(x) * np.cos(y)  # Pattern 1
    sig2 = np.cos(x + np.pi/4) * np.sin(y + np.pi/4)  # Pattern 2
    sig3 = np.sin(2*x) * np.cos(2*y)  # Pattern 3
    
    # Create spectral profiles for each pattern
    wavelengths = np.linspace(400, 2500, bands)  # Typical HSI range
    
    # Spectral signatures (simplified vegetation, soil, water)
    spec1 = np.exp(-(wavelengths - 800)**2 / (2 * 100**2))  # Vegetation-like
    spec2 = np.exp(-(wavelengths - 1200)**2 / (2 * 200**2))  # Soil-like  
    spec3 = np.exp(-(wavelengths - 1500)**2 / (2 * 150**2))  # Mineral-like
    
    # Combine spatial and spectral components
    hsi_data = np.zeros((n_pixels, bands))
    
    for i in range(bands):
        spatial_component = (sig1.flatten() * spec1[i] + 
                           sig2.flatten() * spec2[i] + 
                           sig3.flatten() * spec3[i])
        hsi_data[:, i] = spatial_component
    
    # Add correlated noise (simulating sensor noise)
    noise_correlation = np.random.randn(n_pixels, 5)  # Few noise sources
    noise_profile = np.random.randn(5, bands)  # Spectral noise patterns
    correlated_noise = noise_correlation @ noise_profile * 0.1
    
    # Add independent noise
    independent_noise = np.random.randn(n_pixels, bands) * 0.05
    
    # Combine signal and noise
    hsi_data += correlated_noise + independent_noise
    
    return hsi_data, (height, width, bands)

def compare_pca_mnf():
    """Compare PCA and MNF transformations."""
    print("Comparing PCA and MNF transformations...")
    
    # Create synthetic data
    hsi_data, (height, width, bands) = create_noisy_hsi_data()
    print(f"Created HSI data: {height}x{width}x{bands}")
    
    # Convert to 3D format
    hsi_3d = hsi_data.T.reshape(bands, height, width)
    
    # Number of components to compare
    n_components = 5
    
    # Test PCA
    print("\n--- PCA Transformation ---")
    pca_trans = transformer(method='pca')
    X_pca = pca_trans.HSI_to_X(hsi_3d)
    pca_trans.fit(X_pca, n_components)
    X_pca_transformed = pca_trans.transform(X_pca)
    
    print(f"PCA - Explained variance ratios: {pca_trans.pca_obj.explained_variance_ratio_[:n_components]}")
    print(f"PCA - Total variance explained: {pca_trans.pca_obj.explained_variance_ratio_[:n_components].sum():.3f}")
    
    # Test MNF
    print("\n--- MNF Transformation ---")
    mnf_trans = transformer(method='mnf')
    X_mnf = mnf_trans.HSI_to_X(hsi_3d)
    mnf_trans.fit(X_mnf, n_components)
    X_mnf_transformed = mnf_trans.transform(X_mnf)
    
    print(f"MNF - Signal-to-Noise ratios: {mnf_trans.eigenvalues_[:n_components]}")
    print(f"MNF - Mean SNR for first {n_components} components: {mnf_trans.eigenvalues_[:n_components].mean():.3f}")
    
    # Compare transformed data statistics
    print("\n--- Comparison ---")
    print(f"PCA transformed data range: [{X_pca_transformed.min():.2f}, {X_pca_transformed.max():.2f}]")
    print(f"MNF transformed data range: [{X_mnf_transformed.min():.2f}, {X_mnf_transformed.max():.2f}]")
    
    # Component variances
    pca_vars = np.var(X_pca_transformed, axis=0)
    mnf_vars = np.var(X_mnf_transformed, axis=0)
    
    print("\nComponent variances:")
    for i in range(n_components):
        print(f"Component {i+1}: PCA={pca_vars[i]:.2f}, MNF={mnf_vars[i]:.2f}")
    
    return X_pca_transformed, X_mnf_transformed, pca_trans, mnf_trans

if __name__ == "__main__":
    X_pca, X_mnf, pca_trans, mnf_trans = compare_pca_mnf()
    print("\n✓ Comparison completed successfully!")
    
    # Additional info about the methods
    print("\nMethod Information:")
    print("PCA: Maximizes variance - finds directions of maximum data spread")
    print("MNF: Maximizes SNR - finds directions that separate signal from noise")
    print("MNF is particularly useful for hyperspectral data with correlated noise")