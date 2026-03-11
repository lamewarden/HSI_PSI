# HSI_PSI — Hyperspectral Image Analysis Library

**HSI_PSI** is a Python library for hyperspectral image analysis, designed with simplicity, flexibility, and reproducibility in mind. Built for close-range vegetation monitoring and agricultural applications, it covers the full pipeline from raw data loading to supervised segmentation.

## What HSI_PSI Can Do

- **Load & Handle Data**: Work with hyperspectral and multispectral images from PSI VNIR/SWIR/MSC cameras
- **Preprocess**: Run full preprocessing pipelines (calibration, spike removal, spectral cropping, solar correction, smoothing, normalization)
- **Reduce Dimensionality**: Apply PCA or MNF (Minimum Noise Fraction) transformations
- **Annotate Interactively**: Label image regions using an integrated Napari-based annotation tool
- **Segment Automatically**: Train and deploy ML classifiers for automated pixel-level segmentation
- **Extract Information**: Generate vegetation masks and calculate spectral indices (NDVI, PRI, HBSI, EVI, and more)
- **Visualize**: Create RGB representations, interactive spectral plots, and segmentation overlays
- **Batch Process**: Handle entire folders of hyperspectral images with consistent preprocessing
- **Save Configurations**: Store and reuse preprocessing settings for reproducible workflows

## Package Architecture

```
hsi_psi/
├── __init__.py       # Main imports and version info
├── core.py           # HS_image and MS_image base classes
├── preprocessing.py  # HS_preprocessor with full pipeline and batch processing
├── dim_red.py        # transformer class (PCA and MNF)
├── annotation.py     # NapariHS_Annotator (requires napari)
├── segmentation.py   # SpectralSegmenter (requires scikit-learn, optuna)
└── utils.py          # Utility functions
```

### Core Module (`core.py`)
Provides the foundational data classes:
- `HS_image` — loads and handles hyperspectral images (HDR/ENVI format)
  - Wavelength-indexed band access (`image[670]`)
  - Sensor calibration (dark & white reference)
  - Spectral cropping (`crop_spectral_range`)
  - Normalization: band-ratio (`normalize`), SNV (`apply_snv`), RNV (`apply_rnv`), L2 (`apply_l2`)
  - Mask visualization and management
  - Spectral extraction to DataFrame
- `MS_image` — multispectral image support with channel mapping

### Preprocessing Module (`preprocessing.py`)
`HS_preprocessor` orchestrates the full data preparation pipeline:

**Pipeline order:**
1. Sensor calibration — raw counts → reflectance (with dark/white reference)
2. Spike removal — artifact and dead pixel correction
3. Spectral cropping — focus on wavelength range of interest
4. Solar correction — illumination normalization via teflon reference
5. Spectral smoothing — Gaussian noise reduction
6. Normalization — band-ratio, SNV, RNV, or L2
7. Mask extraction — vegetation segmentation via spectral indices

Additional capabilities:
- Configuration management (JSON save/load)
- Batch folder processing
- Reference teflon library creation with automatic wavelength mapping
- Segmentation model integration (`load_segmentation_model`, `apply_segmentation`)
- Spectra extraction to DataFrame

### Dimensionality Reduction Module (`dim_red.py`)
`transformer` — generic dimensionality reduction with two methods:
- **PCA** — standard principal component analysis via scikit-learn
- **MNF** — Minimum Noise Fraction (custom implementation, no external HSI dependencies)
  - Custom noise covariance estimation using shift-difference noise model
  - `validate_components` for quality assessment

`HS_PCA_transformer` is provided as a backward-compatible alias for `transformer(method='pca')`.

### Annotation Module (`annotation.py`)
`NapariHS_Annotator` — interactive annotation using Napari:
- Supports multiple `HS_image` / `MS_image` inputs simultaneously
- Up to 10 annotation classes with customizable colors
- Paint, erase, fill tools with undo/redo (Ctrl+Z)
- Automatic mask de-repeating for tiled image visualizations
- Export annotations as numpy arrays
- Save/load masks as pickle files
- Integrates with Jupyter notebooks (including VS Code)

### Segmentation Module (`segmentation.py`)
`SpectralSegmenter` — supervised pixel-level classification:
- Load annotations from `NapariHS_Annotator` outputs
- Extract and inspect training data with per-class spectral visualization
- Optuna-based automated model optimization (Random Forest, Gradient Boosting, SVM, PCA+RF pipeline)
- Configurable class balancing and downsampling
- Predict segmentation maps for new images
- Batch prediction across image folders
- Visualize results with class overlays
- Save/load trained models (`.pkl`)

### Utils Module (`utils.py`)
- `rank_noisy_bands`, `summarize_noisiest_bands` — spectral noise analysis
- `plot_spectra` — multi-spectrum plotting with optional scaling
- `get_rgb_sample` — RGB extraction and enhancement
- `extract_masks_from_hs_image`, `extract_masked_spectra_to_df` — spectral data extraction
- `vis_clust_2D`, `plot_confusion_matrix` — visualization tools
- `create_config_template` — JSON configuration scaffolding
- `print_package_info` — display package metadata

## Installation

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

**Core:**
```
numpy, pandas, matplotlib, scipy
scikit-learn, scikit-image, opencv-python
spectral
```

**Optional:**
```
napari[pyqt5]>=0.4.0    # annotation module
optuna                   # segmentation model optimization
seaborn, ipywidgets      # enhanced plots and notebooks
```

## Quick Start

### Load and Inspect an Image

```python
from hsi_psi import HS_image, get_rgb_sample

image = HS_image("data/sample_image.hdr")
print(image)  # shape and wavelength range

# Access a specific band
red_band = image[670]  # band closest to 670 nm

# Extract RGB for display
rgb = get_rgb_sample(image, show=True)
```

### Run the Preprocessing Pipeline

```python
from hsi_psi import HS_preprocessor, create_config_template

config = create_config_template()
config['sensor_calibration']['white_ref_path'] = "calibration/white_ref.hdr"
config['spectral_cropping'] = {'wl_start': 450, 'wl_end': 900}
config['solar_correction']['teflon_edge_coord'] = [-10, -3]
config['normalization']['method'] = 'snv'

processor = HS_preprocessor("data/image.hdr", verbose=True)
processor.run_full_pipeline(config)

processed = processor.get_current_image()
processor.get_rgb_sample(show=True)
```

### Dimensionality Reduction

```python
from hsi_psi import transformer

# PCA
pca = transformer(method='pca')
X = pca.HSI_to_X(image)
pca.fit(X, n_components=10)
img_pca = pca.X_to_img(pca.transform(X))

# MNF (no extra dependencies required)
mnf = transformer(method='mnf')
X = mnf.HSI_to_X(image)
mnf.fit(X, n_components=10)
img_mnf = mnf.X_to_img(mnf.transform(X))

print(mnf.get_explained_variance_ratio())
print(mnf.get_method_info())
```

### Interactive Annotation

```python
from hsi_psi import NapariHS_Annotator

annotator = NapariHS_Annotator(
    images=[image1, image2],
    classes=['Plant', 'Background', 'Stressed'],
)
annotator.annotate()  # opens Napari GUI; close window when done

masks = annotator.get_masks()
annotator.save_masks("annotations/run1.pkl")
```

### Train and Apply a Segmentation Model

```python
from hsi_psi import SpectralSegmenter

segmenter = SpectralSegmenter(verbose=True)
segmenter.load_annotations("annotations/run1.pkl", images=[image1, image2])
segmenter.extract_training_data()

segmenter.visualize_spectra(show_boxplots=True)  # inspect class spectra

segmenter.optimize_model(n_trials=50, n_jobs=-1)  # Optuna search

mask = segmenter.predict_image(new_image)
segmenter.visualize_results(new_image, mask)

segmenter.batch_predict(image_list, output_dir="results/masks/")
segmenter.save_model("models/segmenter_v1.pkl")
```

### Batch Processing

```python
from hsi_psi import HS_preprocessor

results = HS_preprocessor.process_folder(
    folder_path="data/raw_images/",
    config_path="configs/pipeline.json",
    verbose=True,
    extract_masks=True
)
```

### Noise Analysis

```python
from hsi_psi import rank_noisy_bands, summarize_noisiest_bands

noise = rank_noisy_bands(image)
summary = summarize_noisiest_bands(noise, top_n=10)
```

## Configuration File Format

```json
{
  "spectral_cropping": {
    "wl_start": 450,
    "wl_end": 900,
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

Available normalization methods: `"to_wl"`, `"snv"`, `"rnv"`, `"l2"`.

## Support & Development

- **Issues**: Report bugs and feature requests via [GitHub Issues](https://github.com/lamewarden/HSI_PSI/issues)
- **Contributions**: Welcome via pull requests

---

**Version**: 0.4.0  
**Author**: Ivan Kashkan  
**Contact**: kashkan@psi.cz  
**License**: MIT  
**Optimized for**: Close-range vegetation monitoring and agricultural applications
