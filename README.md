# HSI_PSI - Hyperspectral Image Analysis Library

**HSI_PSI** is a comprehensive Python library for hyperspectral image analysis, designed with simplicity and reproducibility in mind. It provides advanced hyperspectral data manipulation methods, packed together into self-explanatory functions that are based on everyday real-life experience working with HSI data as individual files or extensive image batches.

## 🔬 What HSI_PSI Can Do

HSI_PSI empowers researchers and practitioners to:

- **📁 Load & Handle Data**: Seamlessly work with hyperspectral images from PSI VNIR/SWIR/MSC cameras
- **⚡ Process Efficiently**: Apply sophisticated preprocessing pipelines with sensor calibration, solar correction, and spectral smoothing
- **🎭 Extract Information**: Generate vegetation masks and calculate spectral indices (NDVI, PRI, HBSI, EVI, and more)
- **📊 Visualize Results**: Create RGB representations and interactive spectral plots
- **🔧 Batch Process**: Handle entire folders of hyperspectral images with consistent preprocessing
- **💾 Save Configurations**: Store and reuse preprocessing settings for reproducible workflows

## 🏗️ Package Architecture

HSI_PSI consists of three core modules, each designed for specific aspects of hyperspectral analysis:

### **🔧 Core Module**
The foundation of HSI_PSI, providing:
- `HS_image` and `MS_image` classes for data handling
- Image loading and spectral band access
- RGB extraction and visualization tools
- Spectral indices calculation
- Data standardization and format conversion

### **⚙️ Preprocessing Module** 
Advanced data preparation capabilities:
- `HS_preprocessor` class for automated pipelines
- Sensor calibration and radiometric correction
- Solar irradiance normalization
- Spectral smoothing and noise reduction
- Configuration management and batch processing

### **🛠️ Utils Module**
Helper functions and utilities:
- File I/O operations
- Data validation and quality checks
- Polygon mask generation from JSON
- Image enhancement and stretching
- Format conversions and compatibility tools

## Features

- **Core Classes**: `HS_image` and `MS_image` for hyperspectral/multispectral data handling
- **Complete Preprocessing Pipeline**: Sensor calibration, solar correction, spectral smoothing, normalization
- **Mask Extraction**: Vegetation segmentation using spectral indices (NDVI, PRI, HBSI)
- **Configuration Management**: Save/load preprocessing configurations
- **Batch Processing**: Process entire folders of hyperspectral images
- **Visualization**: RGB extraction and spectral plotting

## Installation

### From PyPI (coming soon)
```bash
pip install hsi-psi
```

### From GitHub
```bash
pip install git+https://github.com/lamewarden/HSI_PSI.git
```

### Development Installation
```bash
git clone https://github.com/lamewarden/HSI_PSI.git
cd HSI_PSI
pip install -e .[dev]
```

## Dependencies

```
numpy
pandas
matplotlib
scikit-learn
scikit-image
spectral
opencv-python
scipy
```

## Quick Start

### Basic Usage

```python
import hsi_psi
from hsi_psi import HS_image, get_rgb_sample

# Load hyperspectral image
image = HS_image("data/sample_image.hdr")

# Display basic info
print(f"Image shape: {image.img.shape}")
print(f"Wavelength range: {min(image.ind)}-{max(image.ind)} nm")

# Extract and show RGB
rgb = get_rgb_sample(image, show=True, title="My HS Image")

# Access spectral band at specific wavelength
red_band = image[670]  # Band closest to 670nm
```

### Preprocessing Pipeline

```python
from hsi_psi import HS_preprocessor, create_config_template

# Create and configure processor
processor = HS_preprocessor("data/image.hdr", verbose=True)

# Option 1: Use default configuration
config = create_config_template()
processor.config = config

# Option 2: Load from file
processor.load_config("config/my_config.json")

# Run complete pipeline
processor.run_full_pipeline()

# Get results
processed_image = processor.get_current_image()
rgb = processor.get_rgb_sample(show=True)
spectrum = processor.get_spectrum(roi=(slice(10,20), slice(50,60), slice(None)))
```


### Batch Processing

```python
from HS_tools import HS_preprocessor

# Process entire folder with configuration
processed_images = HS_preprocessor.process_folder(
    folder_path="data/hyperspectral_images/",
    config_path="config/processing_config.json",
    verbose=True
)

print(f"Processed {len(processed_images)} images")
```

### Data Extraction

```python
from HS_tools.utils import extract_masked_spectra_to_df

# Extract spectra from masked regions to DataFrame
df = extract_masked_spectra_to_df(
    processed_images, 
    save_path="results/extracted_spectra.csv"
)

print(f"Extracted {len(df)} spectral samples")
print(f"Features: {df.shape[1]-1} wavelengths")  # -1 for label column
```

## Library Structure

```
HS_tools/
├── __init__.py          # Main imports and version info
├── core.py              # HS_image and MS_image classes
├── preprocessing.py     # HS_preprocessor pipeline class
├── utils.py             # Utility functions (loading, extraction, etc.)
├── spectral_indices.py  # Spectral index calculations
└── README.md           # This file
```

## Configuration System

The library uses JSON configuration files for reproducible processing:

```json
{
  "sensor_calibration": {
    "clip_to": 10,
    "dark_calibration": false
  },
  "solar_correction": {
    "teflon_edge_coord": [-10, -3],
    "smooth_window": 35
  },
  "spectral_smoothing": {
    "sigma": 11,
    "mode": "reflect"
  },
  "normalization": {
    "method": "to_wl",
    "to_wl": 751,
    "clip_to": 10
  },
  "mask_extraction": {
    "pri_thr": -0.1,
    "ndvi_thr": 0.2,
    "hbsi_thr": -0.6,
    "min_pix_size": 2
  }
}
```

## Available Spectral Indices

- **NDVI**: Normalized Difference Vegetation Index
- **PRI**: Photochemical Reflectance Index  
- **HBSI**: Hyperspectral Blue Spectral Index
- **EVI**: Enhanced Vegetation Index
- **SAVI**: Soil-Adjusted Vegetation Index
- **GNDVI**: Green NDVI
- **ARI**: Anthocyanin Reflectance Index
- **CRI**: Carotenoid Reflectance Index

## Method Chaining

Most methods support chaining for clean workflows:

```python
result = (HS_preprocessor("image.hdr")
          .load_config("config.json")
          .run_full_pipeline()
          .get_rgb_sample(show=True))
```

## Error Handling

The library includes comprehensive error handling and informative messages:

- File loading errors with fallback format conversion
- Missing calibration file warnings
- Configuration validation
- Verbose logging for debugging

## Tips for EPPS2025_workshop_draft.ipynb

1. **Import at the top of your notebook**:
   ```python
   from HS_tools import *  # Import all main functions
   ```

2. **Use configuration files** for reproducible results across different datasets

3. **Enable verbose mode** when developing to see detailed processing steps

4. **Save intermediate results** using the step_results attribute

5. **Batch process** similar images using the same configuration

## Support

For issues related to the EPPS2025 Workshop, please refer to the workshop materials or contact the workshop organizers.

---

**Version**: 1.0.0  
**Author**: EPPS2025 Workshop  
**License**: Educational Use
