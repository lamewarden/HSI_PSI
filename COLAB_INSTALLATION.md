# Installing HSI_PSI in Google Colab

## 🚀 **Quick Installation (Recommended)**

### **Method 1: Direct pip install from GitHub**
```python
# In a Colab cell, run:
!pip install git+https://github.com/lamewarden/HSI_PSI.git
```

### **Method 2: Install with specific branch/tag**
```python
# Install specific version/tag
!pip install git+https://github.com/lamewarden/HSI_PSI.git@v1.0.0

# Or install from a specific branch
!pip install git+https://github.com/lamewarden/HSI_PSI.git@main
```

### **Method 3: Development install (if you plan to modify code)**
```python
# Clone and install in editable mode
!git clone https://github.com/lamewarden/HSI_PSI.git
%cd HSI_PSI
!pip install -e .
```

## 📋 **Complete Colab Setup Example**

Here's a complete cell you can copy-paste into Colab:

```python
# Install HSI_PSI package
!pip install git+https://github.com/lamewarden/HSI_PSI.git

# Import and test the package
import hsi_psi
from hsi_psi import HS_image, MS_image, HS_preprocessor

print(f"✅ HSI_PSI successfully installed!")
print(f"📦 Version: {hsi_psi.__version__}")
print(f"👤 Author: {hsi_psi.__author__}")

# Show available functions
print(f"🔧 Available classes and functions:")
for item in hsi_psi.__all__:
    print(f"  - {item}")
```

## 🔄 **Updating the Package**

If you make updates to the GitHub repository and want to update in Colab:

```python
# Method 1: Reinstall
!pip install --upgrade --force-reinstall git+https://github.com/lamewarden/HSI_PSI.git

# Method 2: Uninstall and reinstall
!pip uninstall hsi-psi -y
!pip install git+https://github.com/lamewarden/HSI_PSI.git
```

## 🧪 **Testing Installation**

After installation, test with this simple example:

```python
# Test basic functionality
try:
    import hsi_psi
    print("✅ Package imported successfully")
    
    # Test version access
    print(f"📦 Version: {hsi_psi.__version__}")
    
    # Test class imports
    from hsi_psi import HS_image, HS_preprocessor
    print("✅ Main classes imported successfully")
    
    print("🎉 HSI_PSI is ready to use in Colab!")
    
except ImportError as e:
    print(f"❌ Import failed: {e}")
```

## 📁 **Working with Data in Colab**

### **Upload your hyperspectral data:**
```python
from google.colab import files
import os

# Upload .hdr and corresponding data files
uploaded = files.upload()

# List uploaded files
for filename in uploaded.keys():
    print(f"📄 Uploaded: {filename}")
```

### **Example usage with uploaded data:**
```python
# Assuming you uploaded 'sample.hdr' and 'sample.raw'
from hsi_psi import HS_image, get_rgb_sample

# Load the hyperspectral image
image = HS_image("sample.hdr")

# Display basic info
print(f"Image shape: {image.img.shape}")
print(f"Wavelength range: {min(image.ind):.1f}-{max(image.ind):.1f} nm")

# Extract RGB visualization
rgb = get_rgb_sample(image, show=True, title="My Hyperspectral Image")
```

## 🔧 **Dependencies**

The package will automatically install these dependencies:
- numpy
- pandas  
- matplotlib
- scikit-learn
- scikit-image
- spectral
- opencv-python
- scipy

## ⚠️ **Troubleshooting**

### **If installation fails:**
```python
# Update pip first
!pip install --upgrade pip

# Try installing dependencies separately
!pip install numpy pandas matplotlib scikit-learn scikit-image spectral opencv-python scipy

# Then install HSI_PSI
!pip install git+https://github.com/lamewarden/HSI_PSI.git
```

### **If import fails:**
```python
# Restart runtime (Runtime > Restart runtime)
# Then try importing again

# Or check if package is installed
!pip list | grep hsi
```

## 🎯 **Ready-to-Use Colab Template**

Copy this complete cell for a quick start:

```python
# ==================================================
# HSI_PSI Package Setup for Google Colab
# ==================================================

print("🔄 Installing HSI_PSI package...")
!pip install git+https://github.com/lamewarden/HSI_PSI.git

print("\n📦 Testing installation...")
import hsi_psi
from hsi_psi import HS_image, MS_image, HS_preprocessor, get_rgb_sample

print(f"✅ HSI_PSI v{hsi_psi.__version__} successfully installed!")
print("🎉 Ready to process hyperspectral images!")

# Uncomment to upload your data files:
# from google.colab import files
# uploaded = files.upload()
```

## 📚 **Next Steps**

After installation, check out the examples in the repository:
- `examples/basic_usage.py` - Basic usage examples
- Repository README: https://github.com/lamewarden/HSI_PSI

Happy hyperspectral image processing! 🔬📊
