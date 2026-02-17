"""
Interactive annotation tools for hyperspectral images using Napari.

This module provides professional-grade annotation interfaces that work seamlessly 
in Jupyter notebooks and standalone applications.

Classes:
    NapariHS_Annotator: Interactive annotation tool using the Napari viewer
    
Dependencies:
    - napari[pyqt5]: For the interactive annotation interface
    
Example:
    >>> from hsi_psi import HS_preprocessor, NapariHS_Annotator
    >>> preprocessor = HS_preprocessor(image)
    >>> annotator = NapariHS_Annotator(preprocessor, classes=['Plant', 'Background'])
    >>> annotator.annotate()  # Opens interactive viewer
    >>> annotator.save_masks('output/annotations/')
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

try:
    import napari
    NAPARI_AVAILABLE = True
except ImportError:
    NAPARI_AVAILABLE = False


class NapariHS_Annotator:
    """
    Interactive annotation tool for hyperspectral images using Napari.
    
    This tool provides a professional-grade annotation interface that works
    seamlessly in Jupyter notebooks (including VSCode) and standalone applications.
    
    Features:
        - Up to 10 annotation classes
        - Interactive painting with adjustable brush size (mouse scroll)
        - Eraser tool (hold Alt/Option while painting)
        - Fill tool for large regions
        - Undo/Redo support (Ctrl+Z / Ctrl+Shift+Z)
        - RGB visualization with repeat functionality
        - Export masks as numpy arrays or images
        - Automatic de-repeating of masks to match original image dimensions
    
    Attributes:
        preprocessor: The HS_preprocessor object containing the hyperspectral image
        image: The HS_image object from the preprocessor
        classes: List of annotation class names
        colors: List of RGBA colors for each class
        repeat: Repetition factor for Y-axis visualization
        rgb_image: The RGB visualization array
        label_layers: Dictionary of annotation masks for each class
        viewer: The napari viewer instance (when active)
        
    Example:
        >>> annotator = NapariHS_Annotator(
        ...     preprocessor, 
        ...     classes=['Plant', 'Background', 'Soil'],
        ...     repeat=10
        ... )
        >>> annotator.annotate()  # Opens interactive napari viewer
        >>> masks = annotator.get_masks()  # Get annotation masks
        >>> annotator.save_masks('output/annotations/')  # Save to disk
    """
    
    def __init__(self, 
                 preprocessor,
                 classes: List[str] = ['Plant', 'Background'],
                 colors: Optional[List[List[float]]] = None,
                 repeat: int = 1,
                 normalize: bool = True,
                 correct: bool = True):
        """
        Initialize the napari annotation tool.
        
        Parameters:
            preprocessor: HS_preprocessor
                The preprocessor object containing the hyperspectral image
            classes: List[str]
                List of class names (max 10), default: ['Plant', 'Background']
            colors: Optional[List[List[float]]]
                List of RGBA colors for each class (values 0-1)
                If None, uses default color scheme
            repeat: int
                Repetition factor for Y-axis visualization (like get_rgb_sample)
            normalize: bool
                Normalize RGB visualization
            correct: bool
                Correct outliers in RGB visualization
                
        Raises:
            ImportError: If napari is not installed
            ValueError: If more than 10 classes are specified
        """
        if not NAPARI_AVAILABLE:
            raise ImportError(
                "Napari is required. Install with: pip install napari[pyqt5]"
            )
        
        if len(classes) > 10:
            raise ValueError("Maximum 10 annotation classes supported")
        
        self.preprocessor = preprocessor
        self.image = preprocessor.image
        self.classes = classes
        self.repeat = repeat
        self.normalize = normalize
        self.correct = correct
        
        # Store original image dimensions (before repeat)
        self.original_height = self.image.img.shape[0]
        self.original_width = self.image.img.shape[1]
        
        # Default colors for up to 10 classes (napari format: RGBA)
        self.colors = colors if colors else [
            [0.0, 1.0, 0.0, 1.0],  # Green - Plant
            [0.545, 0.271, 0.075, 1.0],  # Brown - Background/Soil
            [1.0, 0.0, 0.0, 1.0],  # Red
            [0.0, 0.0, 1.0, 1.0],  # Blue
            [1.0, 1.0, 0.0, 1.0],  # Yellow
            [1.0, 0.0, 1.0, 1.0],  # Magenta
            [0.0, 1.0, 1.0, 1.0],  # Cyan
            [1.0, 0.647, 0.0, 1.0],  # Orange
            [0.5, 0.0, 0.5, 1.0],  # Purple
            [1.0, 0.753, 0.796, 1.0],  # Pink
        ][:len(classes)]
        
        # Generate RGB visualization similar to get_rgb_sample
        self.rgb_image = self._generate_rgb()
        
        # Initialize label layers (one per class)
        h, w = self.rgb_image.shape[:2]
        self.label_layers = {}
        for i, class_name in enumerate(self.classes):
            self.label_layers[class_name] = np.zeros((h, w), dtype=np.uint8)
        
        self.viewer = None
        
    def _generate_rgb(self) -> np.ndarray:
        """
        Generate RGB visualization similar to get_rgb_sample.
        
        Returns:
            np.ndarray: RGB image array with shape (H, W, 3)
        """
        image = self.image
        
        # Check if SNV-normalized
        has_negative_values = np.any(image.img < 0)
        is_snv_normalized = (
            hasattr(image, 'normalized') and 
            image.normalized and 
            has_negative_values
        )
        
        # Extract RGB bands based on image type
        if len(image.ind) <= 6:
            R = (image[670] / 4095)
            G = (image[595] / 4095)
            B = (image[495] / 4095)
        elif (np.mean(image.ind) < 900 and len(image.ind) > 6 and 
              image.bits == 12 and not image.calibrated):
            R = np.mean([image[value] for value in image.ind if 570 <= value <= 650], axis=0) / 4095
            G = np.mean([image[value] for value in image.ind if 520 <= value <= 570], axis=0) / 4095
            B = np.mean([image[value] for value in image.ind if 450 <= value <= 520], axis=0) / 4095
        elif (np.mean(image.ind) < 900 and len(image.ind) > 6 and 
              image.bits == 12 and image.calibrated and not is_snv_normalized):
            global_95 = np.percentile(image.img[1:-1, 20:-20, :], 95)
            R = np.clip(np.mean([image[value] for value in image.ind if 570 <= value <= 650], axis=0) / global_95, 0, 1)
            G = np.clip(np.mean([image[value] for value in image.ind if 520 <= value <= 570], axis=0) / global_95, 0, 1)
            B = np.clip(np.mean([image[value] for value in image.ind if 450 <= value <= 520], axis=0) / global_95, 0, 1)
        elif np.mean(image.ind) < 900 and len(image.ind) > 6 and is_snv_normalized:
            R = np.mean([image[value] for value in image.ind if 570 <= value <= 650], axis=0)
            G = np.mean([image[value] for value in image.ind if 520 <= value <= 570], axis=0)
            B = np.mean([image[value] for value in image.ind if 450 <= value <= 520], axis=0)
        else:
            R = np.mean([image[value] for value in image.ind if 570 <= value <= 650], axis=0)
            G = np.mean([image[value] for value in image.ind if 520 <= value <= 570], axis=0)
            B = np.mean([image[value] for value in image.ind if 450 <= value <= 520], axis=0)
        
        # Correct outliers if requested
        if self.correct:
            for channel in [R, G, B]:
                channel_mean, channel_std = np.mean(channel), np.std(channel)
                outlier_mask = np.abs(channel - channel_mean) > 4 * channel_std
                channel[outlier_mask] = channel_mean
                np.nan_to_num(channel, copy=False, nan=np.nanmin(channel))
        
        # Normalize
        if self.normalize and not is_snv_normalized:
            for channel in [R, G, B]:
                ch_min, ch_max = np.min(channel), np.max(channel)
                if ch_max > ch_min:
                    channel[:] = (channel - ch_min) / (ch_max - ch_min)
        elif is_snv_normalized:
            # Robust normalization for SNV data
            for channel in [R, G, B]:
                low_val = np.percentile(channel, 2)
                high_val = np.percentile(channel, 98)
                if high_val > low_val:
                    channel[:] = np.clip((channel - low_val) / (high_val - low_val), 0, 1)
        
        # Stack RGB and repeat along Y-axis if requested
        rgb = np.dstack([R, G, B])
        if self.repeat > 1:
            rgb = np.repeat(rgb, self.repeat, axis=0)
        
        return rgb
    
    def _derepeat_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        De-repeat a mask to original image dimensions.
        
        If the image was repeated N times along Y-axis for visualization,
        this method shrinks the mask back to original size by taking every Nth row.
        
        Parameters:
            mask: np.ndarray
                The mask from repeated image
                
        Returns:
            np.ndarray: Mask with original image dimensions
        """
        if self.repeat > 1:
            # Take every repeat-th row to reverse the repeat operation
            return mask[::self.repeat, :]
        else:
            return mask
    
    def annotate(self):
        """
        Launch the interactive napari annotation viewer.
        
        Opens a napari window with the RGB visualization and annotation layers.
        Use the napari interface to create annotations for each class.
        
        Controls:
            - **Paint Mode**: Click the "paint" button in the layer controls
            - **Erase Mode**: Click the "erase" button or hold Alt/Option while painting
            - **Fill Mode**: Click the "fill" button for large regions
            - **Brush Size**: Use mouse scroll wheel or slider in controls
            - **Undo/Redo**: Ctrl+Z / Ctrl+Shift+Z
            - **Switch Class**: Click on different label layers to annotate different classes
            - **Toggle Visibility**: Eye icon to show/hide annotation layers
        
        Tips:
            - Start with one class at a time - make one layer visible and annotate
            - Use fill tool for large homogeneous regions
            - Use paint with large brush for bulk annotation
            - Use paint with small brush for precise boundaries
            - Save your work periodically using annotator.save_masks()
        """
        # Create napari viewer
        self.viewer = napari.Viewer(title="Hyperspectral Image Annotation")
        
        # Add RGB image as base layer
        self.viewer.add_image(
            self.rgb_image, 
            name='RGB Image',
            rgb=True,
            opacity=1.0
        )
        
        # Add label layer for each class
        for i, (class_name, color) in enumerate(zip(self.classes, self.colors)):
            layer = self.viewer.add_labels(
                self.label_layers[class_name],
                name=f'{class_name}',
                opacity=0.5,
                visible=(i == 0)  # Only first class visible by default
            )
            # Set color for this label layer (label value 1 = annotated pixels)
            layer.color = {1: np.array(color)}
            # Set brush size
            layer.brush_size = 10
        
        # Print instructions
        print("=" * 70)
        print("🎨 NAPARI ANNOTATION TOOL - INTERACTIVE MODE")
        print("=" * 70)
        print("\n📋 INSTRUCTIONS:")
        print("  1. Select a label layer (class) from the layer list on the left")
        print("  2. Click the 'paint brush' icon to enter paint mode")
        print("  3. Adjust brush size with mouse scroll wheel")
        print("  4. Click and drag to paint annotations")
        print("  5. Use 'eraser' or hold Alt/Option to erase")
        print("  6. Use 'fill' tool for large regions")
        print("  7. Switch between classes by selecting different label layers")
        print("\n⌨️  KEYBOARD SHORTCUTS:")
        print("  • Ctrl+Z: Undo")
        print("  • Ctrl+Shift+Z: Redo")
        print("  • 1: Paint mode")
        print("  • 2: Pan/Zoom mode")
        print("  • 3: Fill mode")
        print("  • 4: Erase mode")
        print("\n💾 SAVING:")
        print("  • When done, close the napari window")
        print("  • Then run: annotator.save_masks('output/path/')")
        if self.repeat > 1:
            print(f"\n📏 NOTE: Image repeated {self.repeat}x for visualization")
            print(f"   Masks will be automatically de-repeated to match original image size")
            print(f"   Original: {self.original_height}x{self.original_width}")
            print(f"   Displayed: {self.rgb_image.shape[0]}x{self.rgb_image.shape[1]}")
        print("=" * 70)
        
        # Start the viewer
        napari.run()
        
    def get_masks(self, original_size: bool = True) -> Dict[str, np.ndarray]:
        """
        Get binary masks for each class.
        
        Parameters:
            original_size: bool
                If True (default), returns masks de-repeated to original image size.
                If False, returns masks at the repeated visualization size.
        
        Returns:
            Dict[str, np.ndarray]: Dictionary with class names as keys and binary masks as values
        """
        masks = {}
        for class_name, label_layer in self.label_layers.items():
            mask = (label_layer > 0).astype(np.uint8)
            if original_size:
                mask = self._derepeat_mask(mask)
            masks[class_name] = mask
        return masks
    
    def get_combined_mask(self, original_size: bool = True) -> np.ndarray:
        """
        Get a combined annotation mask with unique class indices.
        
        Parameters:
            original_size: bool
                If True (default), returns mask de-repeated to original image size.
                If False, returns mask at the repeated visualization size.
        
        Returns:
            np.ndarray: Mask array where each pixel value represents the class index (0 = unlabeled)
        """
        h, w = self.rgb_image.shape[:2]
        combined = np.zeros((h, w), dtype=np.uint8)
        
        for i, (class_name, label_layer) in enumerate(self.label_layers.items(), start=1):
            mask = label_layer > 0
            combined[mask] = i
        
        if original_size:
            combined = self._derepeat_mask(combined)
        
        return combined
    
    def save_masks(self, output_dir: str, prefix: str = 'annotation'):
        """
        Save annotation masks to disk.
        
        The masks are automatically de-repeated to match the original image dimensions.
        Saves:
            - Combined mask as .npy file
            - Individual class masks as .npy and .png files
            - Metadata as .json file
        
        Parameters:
            output_dir: str
                Directory to save masks
            prefix: str
                Prefix for saved files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save combined mask as numpy array (de-repeated to original size)
        combined = self.get_combined_mask(original_size=True)
        combined_path = output_path / f'{prefix}_combined_{timestamp}.npy'
        np.save(combined_path, combined)
        print(f"✅ Combined mask saved: {combined_path}")
        print(f"   Shape: {combined.shape} (original image dimensions)")
        
        # Save individual class masks (de-repeated to original size)
        masks = self.get_masks(original_size=True)
        for class_name, mask in masks.items():
            # Numpy array
            mask_path = output_path / f'{prefix}_{class_name}_{timestamp}.npy'
            np.save(mask_path, mask)
            
            # PNG for visualization
            import matplotlib.pyplot as plt
            png_path = output_path / f'{prefix}_{class_name}_{timestamp}.png'
            plt.imsave(png_path, mask, cmap='gray')
            
            pixel_count = mask.sum()
            percentage = (pixel_count / mask.size) * 100
            print(f"   • {class_name}: {pixel_count} pixels ({percentage:.2f}%) → {mask_path.name}")
        
        # Save metadata
        metadata = {
            'classes': self.classes,
            'colors': [color for color in self.colors],
            'shape': combined.shape,
            'original_shape': (self.original_height, self.original_width),
            'timestamp': timestamp,
            'repeat': self.repeat,
            'total_annotated_pixels': int(np.sum(combined > 0))
        }
        metadata_path = output_path / f'{prefix}_metadata_{timestamp}.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        print(f"✅ Metadata saved: {metadata_path}")
        print(f"\n📊 Total annotated pixels: {metadata['total_annotated_pixels']} / {combined.size}")
        print(f"    Coverage: {(metadata['total_annotated_pixels'] / combined.size * 100):.2f}%")
        if self.repeat > 1:
            print(f"\n📏 Masks de-repeated from {self.rgb_image.shape[:2]} to {combined.shape}")
    
    def load_masks(self, mask_path: str):
        """
        Load previously saved annotation masks.
        
        Parameters:
            mask_path: str
                Path to the saved combined .npy mask file
        """
        loaded_mask = np.load(mask_path)
        
        # The loaded mask is at original size, need to repeat if necessary
        expected_original_shape = (self.original_height, self.original_width)
        expected_display_shape = self.rgb_image.shape[:2]
        
        if loaded_mask.shape != expected_original_shape:
            print(f"⚠️ Warning: Loaded mask shape {loaded_mask.shape} doesn't match original image shape {expected_original_shape}")
            print("   Attempting to resize...")
            from scipy.ndimage import zoom
            zoom_factors = (
                expected_original_shape[0] / loaded_mask.shape[0], 
                expected_original_shape[1] / loaded_mask.shape[1]
            )
            loaded_mask = zoom(loaded_mask, zoom_factors, order=0).astype(np.uint8)
        
        # Repeat the loaded mask if necessary to match display size
        if self.repeat > 1:
            loaded_mask = np.repeat(loaded_mask, self.repeat, axis=0)
        
        # Split combined mask into individual class masks
        for i, class_name in enumerate(self.classes, start=1):
            self.label_layers[class_name] = (loaded_mask == i).astype(np.uint8)
        
        print(f"✅ Masks loaded from: {mask_path}")
        print(f"   Classes loaded: {', '.join(self.classes)}")
        if self.repeat > 1:
            print(f"   Masks repeated {self.repeat}x to match display size")


# Module-level check for napari availability
if not NAPARI_AVAILABLE:
    import warnings
    warnings.warn(
        "Napari is not installed. Install with 'pip install napari[pyqt5]' to use annotation tools.",
        ImportWarning
    )
