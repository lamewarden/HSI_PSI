# Quick test of improved MNF implementation
import numpy as np
from hsi_psi.dim_red import transformer

def quick_mnf_test():
    """Quick test to check if second component noise issue is resolved."""
    print("Quick MNF Test - Checking Component Quality")
    print("=" * 50)
    
    # Create test data with known structure
    np.random.seed(123)  # Different seed for variety
    height, width, bands = 40, 40, 60
    
    # Create structured signal patterns
    x, y = np.meshgrid(np.linspace(0, 4*np.pi, width), np.linspace(0, 4*np.pi, height))
    
    # Three distinct spatial patterns
    pattern1 = np.sin(x) * np.cos(y) * 10      # Strong signal
    pattern2 = np.cos(x/2) * np.sin(y/2) * 6  # Medium signal  
    pattern3 = np.sin(x+y) * 3                # Weaker signal
    
    # Create hyperspectral data
    hsi_data = np.zeros((height * width, bands))
    
    # Add spectral signatures to spatial patterns
    for b in range(bands):
        wavelength = b / bands
        
        # Spectral response for each pattern
        spec1 = np.exp(-((wavelength - 0.3)**2) / 0.02)  # Peak at 30%
        spec2 = np.exp(-((wavelength - 0.6)**2) / 0.03)  # Peak at 60% 
        spec3 = np.exp(-((wavelength - 0.8)**2) / 0.025) # Peak at 80%
        
        spatial_band = (pattern1.flatten() * spec1 + 
                       pattern2.flatten() * spec2 + 
                       pattern3.flatten() * spec3)
        
        hsi_data[:, b] = spatial_band
    
    # Add realistic noise
    noise_power = np.std(hsi_data) * 0.15  # 15% noise
    noise = np.random.randn(height * width, bands) * noise_power
    
    # Add some correlated noise
    corr_noise = np.random.randn(height * width, 3) @ np.random.randn(3, bands) * noise_power * 0.5
    
    noisy_hsi = hsi_data + noise + corr_noise
    
    print(f"Created test data: {height}x{width}x{bands}")
    print(f"Signal power: {np.var(hsi_data):.4f}")
    print(f"Noise power: {np.var(noise + corr_noise):.4f}")
    print(f"SNR: {np.var(hsi_data) / np.var(noise + corr_noise):.2f}")
    
    # Test MNF
    print("\nTesting MNF...")
    mnf_trans = transformer(method='mnf')
    
    # Convert to proper format
    hsi_3d = noisy_hsi.T.reshape(bands, height, width)
    X = mnf_trans.HSI_to_X(hsi_3d)
    
    # Fit and transform
    mnf_trans.fit(X, n_components=8)
    X_transformed = mnf_trans.transform(X)
    
    print(f"\nMNF Results:")
    print(f"Eigenvalues (SNR): {mnf_trans.eigenvalues_}")
    
    # Validate components
    print("\nValidating component quality...")
    validation = mnf_trans.validate_components(X_transformed, n_check=5)
    
    for comp_info in validation['component_quality']:
        print(f"\nComponent {comp_info['component']}:")
        print(f"  SNR: {comp_info['snr']:.4f}")
        print(f"  Spatial correlation: {comp_info['spatial_correlation']:.4f}")
        print(f"  Quality: {comp_info['overall_quality']}")
        print(f"  Flags: {', '.join(comp_info['flags'])}")
    
    if validation['recommendations']:
        print("\nRecommendations:")
        for rec in validation['recommendations']:
            print(f"  • {rec}")
    else:
        print("\n✓ No quality issues detected!")
    
    # Check specifically if second component is good
    if len(validation['component_quality']) >= 2:
        second_comp = validation['component_quality'][1]  # Index 1 = component 2
        if second_comp['overall_quality'] != "POOR (likely noise)":
            print(f"\n✓ Second component quality: {second_comp['overall_quality']} (Fixed!)")
        else:
            print(f"\n⚠ Second component still has issues: {second_comp['overall_quality']}")
    
    return mnf_trans, validation

if __name__ == "__main__":
    mnf_trans, validation = quick_mnf_test()