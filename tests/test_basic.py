"""
Basic tests for HSI_PSI package
"""
import unittest
import sys
import os

# Add the package root to Python path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import hsi_psi
    from hsi_psi import HS_image, MS_image, HS_preprocessor
    PACKAGE_AVAILABLE = True
except ImportError as e:
    PACKAGE_AVAILABLE = False
    IMPORT_ERROR = str(e)


class TestPackageImport(unittest.TestCase):
    """Test that the package can be imported correctly"""
    
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


if __name__ == '__main__':
    unittest.main()
