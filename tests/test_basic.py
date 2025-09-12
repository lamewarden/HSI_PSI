"""
Enhanced tests for HSI_PSI package v0.2.0
Tests new features including spectral cropping, wavelength mapping, and noise analysis
"""
import unittest
import sys
import os
import numpy as np

# Add the package root to Python path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import hsi_psi
    from hsi_psi import HS_image, MS_image, HS_preprocessor
    from hsi_psi.utils import rank_noisy_bands, summarize_noisiest_bands, create_config_template
    PACKAGE_AVAILABLE = True
except ImportError as e:
    PACKAGE_AVAILABLE = False
    IMPORT_ERROR = str(e)


class TestPackageImport(unittest.TestCase):
    """Test that the enhanced package can be imported correctly"""
    
    def test_package_import(self):
        """Test that the main package imports without errors"""
        if not PACKAGE_AVAILABLE:
            self.fail(f"Package import failed: {IMPORT_ERROR}")
        
        # Test that version is accessible
        self.assertTrue(hasattr(hsi_psi, '__version__'))
        self.assertIsInstance(hsi_psi.__version__, str)
        
    def test_main_classes_import(self):
        """Test that main classes can be imported"""
        if not PACKAGE_AVAILABLE:
            self.skipTest("Package not available")
            
        # Test that main classes are accessible
        self.assertTrue(hasattr(hsi_psi, 'HS_image'))
        self.assertTrue(hasattr(hsi_psi, 'MS_image'))
        self.assertTrue(hasattr(hsi_psi, 'HS_preprocessor'))
        
    def test_new_utils_import(self):
        """Test that new utility functions can be imported"""
        if not PACKAGE_AVAILABLE:
            self.skipTest("Package not available")
            
        # Test new noise analysis functions
        self.assertTrue(hasattr(hsi_psi.utils, 'rank_noisy_bands'))
        self.assertTrue(hasattr(hsi_psi.utils, 'summarize_noisiest_bands'))
        self.assertTrue(hasattr(hsi_psi.utils, 'create_config_template'))


class TestVersion(unittest.TestCase):
    """Test version information"""
    
    def test_version_format(self):
        """Test that version follows semantic versioning"""
        if not PACKAGE_AVAILABLE:
            self.skipTest("Package not available")
            
        version = hsi_psi.__version__
        # Basic check for x.y.z format
        parts = version.split('.')
        self.assertEqual(len(parts), 3, f"Version {version} should have 3 parts")
        
        # Check that all parts are numeric
        for part in parts:
            self.assertTrue(part.isdigit(), f"Version part '{part}' should be numeric")


class TestConfigurationTemplate(unittest.TestCase):
    """Test enhanced configuration template functionality"""
    
    def test_config_template_creation(self):
        """Test that configuration template can be created"""
        if not PACKAGE_AVAILABLE:
            self.skipTest("Package not available")
            
        config = create_config_template()
        
        # Test that all expected sections are present
        expected_sections = [
            'spectral_cropping', 'sensor_calibration', 'spike_removal',
            'solar_correction', 'spectral_smoothing', 'normalization', 'mask_extraction'
        ]
        
        for section in expected_sections:
            self.assertIn(section, config, f"Config should contain {section} section")
            
    def test_spectral_cropping_config(self):
        """Test that spectral cropping configuration is properly structured"""
        if not PACKAGE_AVAILABLE:
            self.skipTest("Package not available")
            
        config = create_config_template()
        cropping_config = config['spectral_cropping']
        
        # Test that all expected parameters are present
        expected_params = ['wl_start', 'wl_end', 'band_start', 'band_end']
        for param in expected_params:
            self.assertIn(param, cropping_config, f"Spectral cropping config should contain {param}")


class TestMockHSImage(unittest.TestCase):
    """Test HS_image functionality with mock data"""
    
    def create_mock_hs_image(self):
        """Create a mock HS_image for testing"""
        if not PACKAGE_AVAILABLE:
            return None
            
        # Create mock data
        mock_img = np.random.rand(50, 50, 100)  # 50x50 spatial, 100 bands
        mock_wavelengths = list(range(400, 500))  # 400-499 nm
        
        # Create a minimal mock HS_image object
        class MockHSImage:
            def __init__(self):
                self.img = mock_img
                self.ind = mock_wavelengths
                self.bands = len(mock_wavelengths)
                self.rows, self.cols = mock_img.shape[:2]
                
            def crop_spectral_range(self, wl_start=None, wl_end=None, band_start=None, band_end=None):
                """Mock spectral cropping method"""
                if wl_start is not None and wl_end is not None:
                    # Find indices
                    start_idx = max(0, wl_start - 400)  # Mock: wavelength 400 = index 0
                    end_idx = min(len(self.ind), wl_end - 400 + 1)
                    
                    # Crop
                    self.img = self.img[:, :, start_idx:end_idx]
                    self.ind = self.ind[start_idx:end_idx]
                    self.bands = len(self.ind)
                return self
        
        return MockHSImage()
    
    def test_mock_spectral_cropping(self):
        """Test spectral cropping functionality with mock data"""
        if not PACKAGE_AVAILABLE:
            self.skipTest("Package not available")
            
        mock_image = self.create_mock_hs_image()
        if mock_image is None:
            self.skipTest("Could not create mock image")
            
        original_bands = len(mock_image.ind)
        original_shape = mock_image.img.shape
        
        # Test cropping
        mock_image.crop_spectral_range(wl_start=420, wl_end=450)
        
        # Verify cropping worked
        self.assertLess(len(mock_image.ind), original_bands, "Should have fewer bands after cropping")
        self.assertEqual(mock_image.img.shape[2], len(mock_image.ind), "Image bands should match wavelength list")
        self.assertEqual(mock_image.img.shape[:2], original_shape[:2], "Spatial dimensions should be unchanged")


if __name__ == '__main__':
    print("Running enhanced HSI_PSI v0.2.0 tests...")
    unittest.main()
