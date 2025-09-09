"""
Test script for spectral spike removal functionality
"""
import numpy as np
import matplotlib.pyplot as plt

def test_spike_removal():
    """Test the spike removal function with synthetic data"""
    
    # Create synthetic spectrum with known spikes
    n_bands = 100
    wavelengths = np.linspace(400, 1000, n_bands)
    
    # Base smooth spectrum (vegetation-like)
    spectrum = 0.05 + 0.3 * np.exp(-(wavelengths - 750)**2 / (2 * 50**2))
    
    # Add some spikes at known positions
    spike_positions = [20, 35, 60, 80]
    spike_values = [0.8, 0.9, 0.7, 0.85]
    
    corrupted_spectrum = spectrum.copy()
    for pos, val in zip(spike_positions, spike_values):
        corrupted_spectrum[pos] = val
    
    # Test the spike removal function directly
    from hsi_psi.preprocessing import HS_preprocessor
    
    # Create a dummy preprocessor to access the internal method
    preprocessor = HS_preprocessor()
    
    # Apply spike removal
    cleaned_spectrum, spike_mask = preprocessor._remove_spikes_1d(
        corrupted_spectrum, win=7, k=6.0, replace="median"
    )
    
    # Visualize results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(wavelengths, spectrum, 'g-', label='Original clean', linewidth=2)
    plt.plot(wavelengths, corrupted_spectrum, 'r-', label='With spikes', alpha=0.7)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Reflectance')
    plt.title('Before Spike Removal')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.plot(wavelengths, spectrum, 'g-', label='Original clean', linewidth=2)
    plt.plot(wavelengths, cleaned_spectrum, 'b-', label='After cleaning', alpha=0.7)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Reflectance')
    plt.title('After Spike Removal')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    spike_positions_detected = wavelengths[spike_mask]
    plt.stem(spike_positions_detected, np.ones(len(spike_positions_detected)), 
             linefmt='r-', markerfmt='ro', basefmt=' ')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Detected')
    plt.title(f'Detected Spikes ({len(spike_positions_detected)} found)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    residuals = corrupted_spectrum - cleaned_spectrum
    plt.plot(wavelengths, residuals, 'k-', alpha=0.7)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Removed values')
    plt.title('Removed Spike Values')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print results
    print(f"✅ Test completed!")
    print(f"📊 Original spikes at positions: {spike_positions}")
    print(f"🔍 Detected spikes at positions: {np.where(spike_mask)[0].tolist()}")
    print(f"📈 Detection accuracy: {len(spike_positions_detected)}/{len(spike_positions)} spikes found")
    
    return cleaned_spectrum, spike_mask

if __name__ == "__main__":
    test_spike_removal()
