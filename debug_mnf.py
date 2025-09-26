# Debug script to analyze MNF components and identify noise issues
import numpy as np
import matplotlib.pyplot as plt
from hsi_psi.dim_red import transformer

def analyze_mnf_components():
    """Analyze MNF components to understand why second component might be noisy."""
    print("MNF Component Analysis")
    print("=" * 50)
    
    # Create synthetic HSI data with known signal structure
    np.random.seed(42)
    height, width, bands = 50, 50, 100
    n_pixels = height * width
    
    print(f"Creating synthetic HSI data: {height}x{width}x{bands}")
    
    # Create structured signal
    x, y = np.meshgrid(np.linspace(0, 2*np.pi, width), np.linspace(0, 2*np.pi, height))
    
    # Multiple signal patterns with different spectral signatures
    pattern1 = np.sin(x) * np.cos(y)  # High frequency spatial pattern
    pattern2 = np.sin(x/2) * np.cos(y/2)  # Lower frequency pattern
    pattern3 = np.cos(x + y)  # Diagonal pattern
    
    # Spectral signatures
    wavelengths = np.linspace(0, 1, bands)
    spec1 = np.exp(-(wavelengths - 0.3)**2 / (2 * 0.05**2))  # Peak at 30%
    spec2 = np.exp(-(wavelengths - 0.6)**2 / (2 * 0.08**2))  # Peak at 60%
    spec3 = np.exp(-(wavelengths - 0.8)**2 / (2 * 0.06**2))  # Peak at 80%
    
    # Combine spatial and spectral
    hsi_data = np.zeros((n_pixels, bands))
    for i, spec in enumerate([spec1, spec2, spec3]):
        if i == 0:
            spatial = pattern1.flatten() * 10  # Strong signal
        elif i == 1:
            spatial = pattern2.flatten() * 7   # Medium signal
        else:
            spatial = pattern3.flatten() * 5   # Weaker signal
            
        for b in range(bands):
            hsi_data[:, b] += spatial * spec[b]
    
    # Add realistic noise
    # Correlated noise across bands
    noise_bands = 5
    noise_correlation = np.random.randn(n_pixels, noise_bands) * 0.5
    noise_profile = np.random.randn(noise_bands, bands)
    correlated_noise = noise_correlation @ noise_profile
    
    # Independent noise
    independent_noise = np.random.randn(n_pixels, bands) * 0.2
    
    # Combine
    noisy_hsi = hsi_data + correlated_noise + independent_noise
    
    print(f"Signal power: {np.var(hsi_data):.4f}")
    print(f"Noise power: {np.var(correlated_noise + independent_noise):.4f}")
    print(f"SNR: {np.var(hsi_data) / np.var(correlated_noise + independent_noise):.2f}")
    
    # Test MNF
    mnf_trans = transformer(method='mnf')
    hsi_3d = noisy_hsi.T.reshape(bands, height, width)
    X = mnf_trans.HSI_to_X(hsi_3d)
    
    print(f"\nInput data shape: {X.shape}")
    print(f"Input data range: [{X.min():.2f}, {X.max():.2f}]")
    
    # Fit MNF
    mnf_trans.fit(X, n_components=10)
    X_transformed = mnf_trans.transform(X)
    
    print(f"\nTransformed data shape: {X_transformed.shape}")
    
    # Analyze each component
    print("\nComponent Analysis:")
    print("-" * 30)
    
    for i in range(min(5, X_transformed.shape[1])):
        component = X_transformed[:, i]
        
        # Reshape to spatial for analysis
        comp_spatial = component.reshape(height, width)
        
        # Calculate spatial statistics
        mean_val = np.mean(component)
        std_val = np.std(component)
        
        # Calculate spatial correlation (neighboring pixel similarity)
        spatial_corr_h = np.corrcoef(comp_spatial[:-1, :].flatten(), comp_spatial[1:, :].flatten())[0, 1]
        spatial_corr_v = np.corrcoef(comp_spatial[:, :-1].flatten(), comp_spatial[:, 1:].flatten())[0, 1]
        avg_spatial_corr = (spatial_corr_h + spatial_corr_v) / 2
        
        # High frequency content (measure of noisiness)
        grad_x = np.diff(comp_spatial, axis=1)
        grad_y = np.diff(comp_spatial, axis=0)
        gradient_magnitude = np.sqrt(grad_x[:-1, :]**2 + grad_y[:, :-1]**2)
        high_freq_content = np.mean(gradient_magnitude)
        
        print(f"Component {i+1}:")
        print(f"  Eigenvalue (SNR): {mnf_trans.eigenvalues_[i]:.4f}")
        print(f"  Mean: {mean_val:.4f}, Std: {std_val:.4f}")
        print(f"  Spatial correlation: {avg_spatial_corr:.4f}")
        print(f"  High-freq content: {high_freq_content:.4f}")
        
        # Classify as signal or noise based on characteristics
        if avg_spatial_corr > 0.1 and mnf_trans.eigenvalues_[i] > 1.0:
            classification = "SIGNAL"
        elif avg_spatial_corr < 0.05 or high_freq_content > std_val * 0.5:
            classification = "NOISE"
        else:
            classification = "MIXED"
        
        print(f"  Classification: {classification}")
        print()
    
    # Additional diagnostics
    print("Covariance Matrix Diagnostics:")
    print("-" * 30)
    
    # Check if noise estimation is reasonable
    noise_cov = mnf_trans._estimate_noise_covariance(X)
    signal_cov = np.cov(X.T)
    
    noise_eigenvals = linalg.eigvals(noise_cov)
    signal_eigenvals = linalg.eigvals(signal_cov)
    
    print(f"Noise cov eigenvalue range: [{noise_eigenvals.min():.6f}, {noise_eigenvals.max():.6f}]")
    print(f"Signal cov eigenvalue range: [{signal_eigenvals.min():.6f}, {signal_eigenvals.max():.6f}]")
    print(f"Noise cov condition number: {np.linalg.cond(noise_cov):.2e}")
    print(f"Signal cov condition number: {np.linalg.cond(signal_cov):.2e}")
    
    # Check ratio of signal to noise variance
    noise_power = np.trace(noise_cov) / bands
    signal_power = np.trace(signal_cov) / bands
    print(f"Average noise power: {noise_power:.6f}")
    print(f"Average signal power: {signal_power:.6f}")
    print(f"Signal-to-noise ratio: {signal_power / noise_power:.2f}")

if __name__ == "__main__":
    # Import scipy.linalg for the analysis
    from scipy import linalg
    analyze_mnf_components()