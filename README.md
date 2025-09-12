# HSI_PSI - Advanced Hyperspectral Image Analysis Library

**HSI_PSI** is a comprehensive Python library for hyperspectral image analysis, designed with simplicity, flexibility, and reproducibility in mind. Built specifically for close-range vegetation monitoring and agricultural applications, it provides advanced hyperspectral data manipulation methods packed into self-explanatory functions based on real-world HSI processing experience.

## What HSI_PSI Can Do

HSI_PSI empowers researchers and practitioners to:

- **Load & Handle Data**: Seamlessly work with hyperspectral images from PSI VNIR/SWIR/MSC cameras
- **Process Efficiently**: Apply sophisticated preprocessing pipelines with optimized step ordering
- **Flexible Spectral Processing**: Intelligent wavelength mapping handles different sensor configurations automatically
- **Extract Information**: Generate vegetation masks and calculate spectral indices (NDVI, PRI, HBSI, EVI, and more)
- **Spectral Range Cropping**: Focus processing on relevant wavelength ranges
- **Noise Analysis**: Advanced spectral noise detection and ranking capabilities
- **Save Configurations**: Store and reuse preprocessing settings for reproducible workflows
- **Visualize Results**: Create RGB representations and interactive spectral plots including the new plot_spectra function
- **Batch Process**: Handle entire folders of hyperspectral images with consistent preprocessing

## Package Architecture

HSI_PSI consists of three core modules, each designed for specific aspects of hyperspectral analysis:

### Core Module (`core.py`)
The foundation of HSI_PSI, providing:
- `HS_image` and `MS_image` classes for advanced data handling
- **NEW**: `crop_spectral_range()` method for wavelength selection
- **Enhanced**: Intelligent wavelength mapping for calibration compatibility
- Image loading with automatic format conversion
- Spectral band access with wavelength indexing
- RGB extraction and visualization tools
- Data standardization and format conversion

### Preprocessing Module (`preprocessing.py`)

Advanced data preparation capabilities with optimized processing pipeline:
- `HS_preprocessor` class for automated, scientifically-ordered processing
- Spectral range cropping as integrated pipeline step
- Enhanced wavelength mapping for calibration files with different spectral ranges
- Optimized Pipeline Order:
  1. Sensor calibration (raw → reflectance)
  2. Spike removal (artifact correction)
  3. Spectral cropping (focus on ROI)
  4. Solar correction (illumination normalization)
  5. Spectral smoothing (noise reduction)
  6. Normalization (data standardization)
  7. Mask extraction (vegetation segmentation)
- Configuration management and batch processing
- Reference teflon library creation with automatic wavelength adaptation

### Utils Module (`utils.py`)

Helper functions and utilities:
- Advanced noise analysis and ranking functions (rank_noisy_bands, summarize_noisiest_bands)
- Spectral data extraction and DataFrame conversion
- Configuration template generation (create_config_template)
- Multi-spectrum plotting functionality (plot_spectra)
- Visualization tools (vis_clust_2D, plot_confusion_matrix)
- RGB sample extraction with enhancement
- Image stretching and contrast adjustment
- Polygon mask generation from JSON annotations
- File I/O operations and format conversions

## Key Features & Recent Enhancements

### Spectral Range Cropping

- Crop hyperspectral images to specific wavelength ranges
- Supports both wavelength (nm) and band index specifications
- Automatic metadata and attribute updates
- Integrated as first pipeline step for computational efficiency

### Intelligent Wavelength Mapping

- Automatic interpolation between different spectral configurations
- Calibration files can have any spectral range/resolution
- Reference teflon spectra adapt to current image wavelengths
- No more manual spectral alignment required

### Advanced Noise Analysis

- `rank_noisy_bands()`: Identify and rank problematic spectral bands
- `summarize_noisiest_bands()`: Generate comprehensive noise reports
- Robust statistical methods using Savitzky-Golay filtering
- Integrated with visualization tools

### Multi-Spectrum Visualization

- `plot_spectra()`: Plot multiple spectra with customizable scaling
- Support for dictionary-based spectrum input format
- Interactive visualization capabilities
- Consistent with reference teflon library format

### Optimized Pipeline Architecture

- Scientifically-informed step ordering for maximum data quality
- Each step designed to preserve information for subsequent processing
- Modular design allows individual step execution or full pipeline
- Comprehensive error handling and verbose logging

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

### Basic Usage with New Features

```python
import hsi_psi
from hsi_psi import HS_image, HS_preprocessor, get_rgb_sample

# Load hyperspectral image
image = HS_image("data/sample_image.hdr")

# Display basic info
print(f"Image shape: {image.img.shape}")
print(f"Wavelength range: {min(image.ind)}-{max(image.ind)} nm")

# NEW: Crop spectral range to focus on vegetation-relevant wavelengths
image.crop_spectral_range(wl_start=450, wl_end=800)
print(f"Cropped range: {min(image.ind)}-{max(image.ind)} nm")

# Extract and show RGB
rgb = get_rgb_sample(image, show=True, title="Cropped HS Image")

# Access spectral band at specific wavelength
red_band = image[670]  # Band closest to 670nm
```

### Advanced Preprocessing Pipeline

```python
from hsi_psi import HS_preprocessor, create_config_template

# Create and configure processor
processor = HS_preprocessor("data/image.hdr", verbose=True)

# Create configuration template with new spectral cropping options
config = create_config_template()

# Configure spectral cropping (NEW)
config['spectral_cropping'] = {
    'wl_start': 450,    # Start at 450 nm
    'wl_end': 800,      # End at 800 nm
    'band_start': None,  # Alternative: use band indices
    'band_end': None
}

# Configure other processing steps
config['sensor_calibration']['white_ref_path'] = "calibration/white_ref.hdr"
config['solar_correction']['teflon_edge_coord'] = [-10, -3]

# Run optimized pipeline (NEW ORDER: sensor_cal → spike_removal → cropping → solar → smoothing → normalization)
processor.run_full_pipeline(config, extract_masks=True)

# Get results
processed_image = processor.get_current_image()
rgb = processor.get_rgb_sample(show=True)
```

### Noise Analysis (NEW)

```python
from hsi_psi import rank_noisy_bands, summarize_noisiest_bands

# Load and preprocess image
processor = HS_preprocessor("data/noisy_image.hdr")
processor.sensor_calibration(white_ref_path="cal/white.hdr")

# Analyze spectral noise
noise_ranking = rank_noisy_bands(processor.image, 
                                method='savgol_residuals',
                                window_length=7, 
                                polyorder=2)

# Get summary of noisiest bands
summary = summarize_noisiest_bands(noise_ranking, top_n=10)
print("Top 10 noisiest bands:")
for band_info in summary['top_noisy_bands']:
    print(f"Band {band_info['band_idx']} ({band_info['wavelength']} nm): "
          f"noise score = {band_info['noise_score']:.4f}")
```

### Reference Teflon Library with Wavelength Mapping (ENHANCED)

```python
# Create reference teflon library from multiple images
processor = HS_preprocessor("data/target_image.hdr")  # Define target wavelength range
processor.crop_spectral_range(wl_start=450, wl_end=750)  # Focus on vegetation range

# Create reference from full-range library images (e.g., 350-1000nm)
# The library automatically maps to the target range (450-750nm)
reference_spectrum = processor.create_reference_teflon_library(
    hs_images="library/full_range_images/",  # Full spectral range images
    teflon_edge_coord=(-10, -3),
    white_ref_path="calibration/white_ref.hdr"
)

# The reference is now automatically mapped to 450-750nm range
print(f"Reference spectrum adapted to {len(processor.image.ind)} bands")
```


### Batch Processing with Enhanced Pipeline

```python
from hsi_psi import HS_preprocessor

# Process entire folder with enhanced configuration
processed_images = HS_preprocessor.process_folder(
    folder_path="data/hyperspectral_images/",
    config_path="config/enhanced_processing_config.json",
    verbose=True,
    extract_masks=True  # Include vegetation mask extraction
)

print(f"Processed {len(processed_images)} images")
print("Each image includes masks and optimized preprocessing")
```

### Multi-Spectrum Visualization (NEW)

```python
from hsi_psi import plot_spectra

# Create reference teflon library in plot_spectra compatible format
processor = HS_preprocessor("data/image.hdr")
reference_spectrum = processor.create_reference_teflon_library(
    hs_images="library/",
    teflon_edge_coord=(-10, -3),
    white_ref_path="calibration/white_ref.hdr"
)

# Plot multiple spectra with customizable options
plot_spectra(
    [reference_spectrum, another_spectrum],
    dict_names=["Reference Teflon", "Sample Spectrum"],
    scale=True,  # Auto-scale for better comparison
    title="Spectral Comparison",
    x_label="Wavelength (nm)",
    y_label="Reflectance"
)
```

### Advanced Visualization Tools

```python
from hsi_psi import vis_clust_2D, plot_confusion_matrix

# 2D clustering visualization
vis_clust_2D(spectral_data, pc_to_visualize=[0, 1])

# Classification results visualization
plot_confusion_matrix(y_true, y_pred, 
                     class_names=["Healthy", "Stressed", "Dead"],
                     fig_size=(8, 6))
```

### Data Extraction with New Utils

```python
from hsi_psi.utils import extract_masked_spectra_to_df, extract_masks_from_hs_image

# Extract vegetation masks
masks = extract_masks_from_hs_image(processed_image, 
                                   pri_thr=-0.1, 
                                   ndvi_thr=0.2, 
                                   hbsi_thr=-0.6)

# Extract spectra from masked regions to DataFrame
df = extract_masked_spectra_to_df(
    processed_images, 
    save_path="results/extracted_spectra.csv"
)

print(f"Extracted {len(df)} spectral samples")
print(f"Features: {df.shape[1]-1} wavelengths")  # -1 for label column
```

## Enhanced Configuration System

The library now uses enhanced JSON configuration files with spectral cropping support:

```json
{
  "spectral_cropping": {
    "wl_start": 450,
    "wl_end": 800,
    "band_start": null,
    "band_end": null
  },
  "sensor_calibration": {
    "clip_to": 10,
    "dark_calibration": false,
    "white_ref_path": "calibration/white_ref.hdr"
  },
  "spike_removal": {
    "win": 7,
    "k": 6.0,
    "replace": "median"
  },
  "solar_correction": {
    "teflon_edge_coord": [-10, -3],
    "smooth_window": 35,
    "reference_teflon": null
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

## Optimized Processing Pipeline

The pipeline has been scientifically optimized for close-range hyperspectral vegetation analysis:

### Pipeline Order & Rationale:

1. **Sensor Calibration** - Convert raw counts to physical reflectance units
   - *Why first*: All subsequent processing needs calibrated data
   - *Wavelength mapping*: Automatically handles different calibration file spectral ranges

2. **Spike Removal** - Fix instrumental artifacts and dead pixels
   - *Why early*: Prevents artifacts from propagating through pipeline
   - *Full context*: Uses complete spectral information for better detection

3. **Spectral Cropping** - Focus on wavelength range of interest
   - *Why after calibration*: Preserves accuracy of calibration step
   - *Efficiency*: Reduces computational load for remaining steps

4. **Solar Correction** - Normalize illumination variations using teflon reference
   - *Why after cropping*: More efficient processing on focused spectral range
   - *Wavelength mapping*: Reference spectra automatically adapt to current range

5. **Spectral Smoothing** - Reduce noise while preserving spectral features
   - *Why late*: Preserves all real spectral information until final noise reduction

6. **Normalization** - Standardize data for vegetation index calculations
   - *Why near end*: Works on fully processed spectral data

7. **Mask Extraction** - Extract vegetation regions using spectral indices
   - *Why last*: Uses final processed data for most accurate calculations

## Library Structure

```
hsi_psi/
├── __init__.py          # Main imports and version info
├── core.py              # HS_image and MS_image classes with cropping
├── preprocessing.py     # Enhanced HS_preprocessor with optimized pipeline
├── utils.py             # Utility functions with noise analysis
└── README.md           # This file
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

## New Utility Functions

### Noise Analysis Functions

```python
from hsi_psi import rank_noisy_bands, summarize_noisiest_bands

# Rank bands by noise level
noise_ranking = rank_noisy_bands(hs_image, method='savgol_residuals')

# Get detailed summary
summary = summarize_noisiest_bands(noise_ranking, top_n=10)
```

### Data Extraction Functions

```python
from hsi_psi import extract_masks_from_hs_image, extract_masked_spectra_to_df

# Extract masks using vegetation indices
masks = extract_masks_from_hs_image(hs_image, pri_thr=-0.1, ndvi_thr=0.2)

# Convert to DataFrame for analysis
df = extract_masked_spectra_to_df([hs_image], save_path="data.csv")
```

### Visualization Functions

```python
from hsi_psi import plot_spectra, vis_clust_2D, plot_confusion_matrix, get_rgb_sample

# Multi-spectrum plotting
plot_spectra([spectrum1, spectrum2], dict_names=["Sample A", "Sample B"])

# 2D clustering visualization
vis_clust_2D(spectral_data, pc_to_visualize=[0, 1])

# Classification results
plot_confusion_matrix(y_true, y_pred, class_names=["Class A", "Class B"])

# RGB extraction and enhancement
rgb = get_rgb_sample(hs_image, normalize=True, show=True)
```
```python
from hsi_psi.utils import get_rgb_sample, stretch_image

# Enhanced RGB extraction
rgb = get_rgb_sample(hs_image, show=True, title="Enhanced RGB")

# Image enhancement
enhanced = stretch_image(rgb, stretch_type='histogram')
```

## Method Chaining

Most methods support chaining for clean workflows:

```python
# Chain operations for efficient processing
result = (HS_preprocessor("image.hdr", verbose=True)
          .load_config("config.json")
          .run_full_pipeline(extract_masks=True)
          .get_rgb_sample(show=True))

# Chain core operations
cropped_image = (HS_image("full_range_image.hdr")
                .crop_spectral_range(wl_start=450, wl_end=800)
                .normalize(to_wl=751))
```

## Error Handling & Compatibility

The library includes comprehensive error handling and automatic compatibility:

- **Automatic Format Conversion**: Handles various hyperspectral formats
- **Wavelength Mapping**: Automatic interpolation between different spectral configurations
- **Missing File Handling**: Graceful fallbacks for missing calibration files
- **Configuration Validation**: Comprehensive parameter checking
- **Verbose Logging**: Detailed processing information for debugging
- **Backward Compatibility**: Maintains compatibility with older configurations

## Performance Optimizations

### **For Close-Range Applications (1-5m height):**
- ✅ **No atmospheric correction needed** - minimal atmospheric path
- ✅ **Optimized pipeline order** - maximum data quality with minimum computation
- ✅ **Spectral cropping early** - reduces computational load
- ✅ **Intelligent wavelength mapping** - handles mixed sensor configurations
- ✅ **Efficient batch processing** - parallel-friendly design

### **Memory Management:**
- **Lazy loading options** for large datasets
- **Step-wise result caching** with `step_results` attribute
- **Configurable verbosity** to reduce output overhead
- **Efficient spectral interpolation** using scipy optimizations

## Best Practices for Close-Range HSI

### **1. Spectral Range Selection**
```python
# Focus on vegetation-relevant wavelengths
processor.crop_spectral_range(wl_start=450, wl_end=800)  # Visible to NIR
```

### **2. Reference Teflon Management**
```python
# Create robust reference from multiple images
reference = processor.create_reference_teflon_library(
    "library_images/",
    teflon_edge_coord=(-10, -3)
)
```

### **3. Configuration Management**
```python
# Save configurations for reproducibility
processor.save_config("configs/vegetation_analysis.json")

# Reuse across similar datasets
processor.load_config("configs/vegetation_analysis.json")
```

## Support & Development

- **Issues**: Report bugs and feature requests via GitHub Issues
- **Documentation**: Comprehensive docstrings and examples
- **Testing**: Integrated test suite for core functionality
- **Contributions**: Welcome via pull requests

---

**Version**: 0.2.0  
**Author**: HSI_PSI Development Team  
**License**: MIT  
**Optimized for**: Close-range vegetation monitoring and agricultural applications
