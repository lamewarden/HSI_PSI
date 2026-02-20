"""
Interactive annotation tools for hyperspectral images using Napari.

This module provides professional-grade annotation interfaces that work seamlessly 
in Jupyter notebooks and standalone applications.

Classes:
    NapariHS_Annotator: Interactive annotation tool using the Napari viewer
    
Dependencies:
    - napari[pyqt5]: For the interactive annotation interface
    
Example:
    >>> from hsi_psi import HS_image, NapariHS_Annotator
    >>> images = [HS_image('image1.hdr'), HS_image('image2.hdr')]
    >>> annotator = NapariHS_Annotator(images, classes=['Plant', 'Background'])
    >>> annotator.annotate()  # Opens interactive viewer
    >>> annotator.save_masks('output/annotations/')
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Union
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
    It accepts a list of HS_image or MS_image objects and displays them in napari
    with corresponding annotation layers.
    
    Features:
        - Support for multiple images (list of HS_image/MS_image)
        - Up to 10 annotation classes per image
        - Interactive painting with adjustable brush size (mouse scroll)
        - Eraser tool (hold Alt/Option while painting)
        - Fill tool for large regions
        - Undo/Redo support (Ctrl+Z / Ctrl+Shift+Z)
        - RGB visualization with repeat functionality
        - Export masks as numpy arrays or images
        - Automatic de-repeating of masks to match original image dimensions
    
    Attributes:
        images: List of HS_image or MS_image objects
        classes: List of annotation class names
        colors: List of RGBA colors for each class
        repeat: Repetition factor for Y-axis visualization
        rgb_images: List of RGB visualization arrays for each image
        label_layers: Dictionary of annotation masks {image_idx: {class_name: mask}}
        viewer: The napari viewer instance (when active)
        
    Example:
        >>> from hsi_psi import HS_image, NapariHS_Annotator
        >>> images = [HS_image('img1.hdr'), HS_image('img2.hdr')]
        >>> annotator = NapariHS_Annotator(
        ...     images, 
        ...     classes=['Plant', 'Background', 'Soil'],
        ...     repeat=10
        ... )
        >>> annotator.annotate()  # Opens interactive napari viewer
        >>> masks = annotator.get_masks()  # Get annotation masks
        >>> annotator.save_masks('output/annotations/')  # Save to disk
    """
    
    def __init__(self, 
                 images: Union[List, object],
                 classes: List[str] = ['Plant', 'Background'],
                 colors: Optional[List[List[float]]] = None,
                 repeat: int = 1,
                 normalize: bool = True,
                 correct: bool = True):
        """
        Initialize the napari annotation tool.
        
        Parameters:
            images: Union[List, HS_image, MS_image]
                List of HS_image or MS_image objects, or a single image object
                (will be converted to a list)
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
        
        # Convert single image to list
        if not isinstance(images, list):
            images = [images]
        
        self.images = images
        self.classes = classes
        self.repeat = repeat
        self.normalize = normalize
        self.correct = correct
        
        # Store original image dimensions (before repeat) for each image
        self.original_dims = [(img.img.shape[0], img.img.shape[1]) for img in self.images]
        
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
        
        # Generate RGB visualizations for all images
        self.rgb_images = [self._generate_rgb(img) for img in self.images]
        
        # Initialize label layers (one per class per image)
        # Structure: {image_idx: {class_name: np.ndarray}}
        self.label_layers = {}
        for img_idx, rgb_img in enumerate(self.rgb_images):
            h, w = rgb_img.shape[:2]
            self.label_layers[img_idx] = {}
            for class_name in self.classes:
                self.label_layers[img_idx][class_name] = np.zeros((h, w), dtype=np.uint8)
        
        self.viewer = None
        
    def _generate_rgb(self, image) -> np.ndarray:
        """
        Generate RGB visualization similar to get_rgb_sample.
        
        Parameters:
            image: HS_image or MS_image object
        
        Returns:
            np.ndarray: RGB image array with shape (H, W, 3)
        """
        
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
        
        Opens a napari window with RGB visualizations for all images and annotation layers.
        Use the napari interface to create annotations for each class on each image.
        
        Controls:
            - **Paint Mode**: Click the "paint" button in the layer controls
            - **Erase Mode**: Click the "erase" button or hold Alt/Option while painting
            - **Fill Mode**: Click the "fill" button for large regions
            - **Brush Size**: Use mouse scroll wheel or slider in controls
            - **Undo/Redo**: Ctrl+Z / Ctrl+Shift+Z
            - **Switch Class**: Click on different label layers to annotate different classes
            - **Switch Image**: Toggle visibility of RGB and label layers for different images
            - **Toggle Visibility**: Eye icon to show/hide layers
        
        Tips:
            - Work on one image at a time - show/hide RGB layers
            - Start with one class at a time - make one layer visible and annotate
            - Use fill tool for large homogeneous regions
            - Use paint with large brush for bulk annotation
            - Use paint with small brush for precise boundaries
            - Save your work periodically using annotator.save_masks()
        """
        # Create napari viewer
        self.viewer = napari.Viewer(title="Hyperspectral Image Annotation - Multi-Image Mode")
        
        # Add layers for each image
        for img_idx, (image, rgb_img) in enumerate(zip(self.images, self.rgb_images)):
            image_name = getattr(image, 'name', f'Image_{img_idx}')
            
            # Add RGB image layer
            self.viewer.add_image(
                rgb_img, 
                name=f'{image_name} (RGB)',
                rgb=True,
                opacity=1.0,
                visible=(img_idx == 0)  # Only first image visible by default
            )
            
            # Add label layer for each class for this image
            for class_idx, (class_name, color) in enumerate(zip(self.classes, self.colors)):
                layer = self.viewer.add_labels(
                    self.label_layers[img_idx][class_name],
                    name=f'{image_name} - {class_name}',
                    opacity=0.5,
                    visible=(img_idx == 0 and class_idx == 0)  # Only first class of first image visible
                )
                # Set color for this label layer (label value 1 = annotated pixels)
                layer.color = {1: np.array(color)}
                # Set brush size
                layer.brush_size = 10
        
        # Print instructions
        num_images = len(self.images)
        print("=" * 70)
        print("🎨 NAPARI ANNOTATION TOOL - MULTI-IMAGE MODE")
        print("=" * 70)
        print(f"\n📷 IMAGES LOADED: {num_images}")
        for idx, image in enumerate(self.images):
            image_name = getattr(image, 'name', f'Image_{idx}')
            h, w = self.original_dims[idx]
            print(f"   {idx+1}. {image_name} ({h}×{w})")
        
        print("\n📋 INSTRUCTIONS:")
        print("  1. Toggle RGB image layers on/off to work on specific images")
        print("  2. Select a label layer (Image - Class) from the layer list")
        print("  3. Click the 'paint brush' icon to enter paint mode")
        print("  4. Adjust brush size with mouse scroll wheel")
        print("  5. Click and drag to paint annotations")
        print("  6. Use 'eraser' or hold Alt/Option to erase")
        print("  7. Use 'fill' tool for large regions")
        print("  8. Switch between images by toggling RGB layer visibility")
        
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
            print(f"\n📏 NOTE: Images repeated {self.repeat}x for visualization")
            print(f"   Masks will be automatically de-repeated to match original image sizes")
        print("=" * 70)
        
        # Start the viewer
        napari.run()
        
    def get_masks(self, original_size: bool = True, image_idx: Optional[int] = None) -> Dict:
        """
        Get binary masks for each class.
        
        Parameters:
            original_size: bool
                If True (default), returns masks de-repeated to original image size.
                If False, returns masks at the repeated visualization size.
            image_idx: Optional[int]
                If specified, returns masks only for that image index.
                If None (default), returns masks for all images.
        
        Returns:
            Dict: Nested dictionary structure:
                - If image_idx specified: {class_name: np.ndarray}
                - If image_idx is None: {image_idx: {class_name: np.ndarray}}
        """
        if image_idx is not None:
            # Return masks for single image
            masks = {}
            for class_name, label_layer in self.label_layers[image_idx].items():
                mask = (label_layer > 0).astype(np.uint8)
                if original_size:
                    mask = self._derepeat_mask(mask)
                masks[class_name] = mask
            return masks
        else:
            # Return masks for all images
            all_masks = {}
            for idx in range(len(self.images)):
                all_masks[idx] = {}
                for class_name, label_layer in self.label_layers[idx].items():
                    mask = (label_layer > 0).astype(np.uint8)
                    if original_size:
                        mask = self._derepeat_mask(mask)
                    all_masks[idx][class_name] = mask
            return all_masks
    
    def get_combined_mask(self, original_size: bool = True, image_idx: Optional[int] = None) -> Union[np.ndarray, Dict[int, np.ndarray]]:
        """
        Get a combined annotation mask with unique class indices.
        
        Parameters:
            original_size: bool
                If True (default), returns mask de-repeated to original image size.
                If False, returns mask at the repeated visualization size.
            image_idx: Optional[int]
                If specified, returns mask only for that image index.
                If None (default), returns masks for all images.
        
        Returns:
            Union[np.ndarray, Dict[int, np.ndarray]]:
                - If image_idx specified: Single mask array where each pixel value represents the class index (0 = unlabeled)
                - If image_idx is None: Dictionary {image_idx: mask_array}
        """
        if image_idx is not None:
            # Return combined mask for single image
            h, w = self.rgb_images[image_idx].shape[:2]
            combined = np.zeros((h, w), dtype=np.uint8)
            
            for class_idx, (class_name, label_layer) in enumerate(self.label_layers[image_idx].items(), start=1):
                mask = label_layer > 0
                combined[mask] = class_idx
            
            if original_size:
                combined = self._derepeat_mask(combined)
            
            return combined
        else:
            # Return combined masks for all images
            all_combined = {}
            for idx in range(len(self.images)):
                h, w = self.rgb_images[idx].shape[:2]
                combined = np.zeros((h, w), dtype=np.uint8)
                
                for class_idx, (class_name, label_layer) in enumerate(self.label_layers[idx].items(), start=1):
                    mask = label_layer > 0
                    combined[mask] = class_idx
                
                if original_size:
                    combined = self._derepeat_mask(combined)
                
                all_combined[idx] = combined
            
            return all_combined
    
    def save_masks(self, output_dir: str, prefix: str = 'annotation'):
        """
        Save annotation masks to disk for all images.
        
        The masks are automatically de-repeated to match the original image dimensions.
        For each image, saves:
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
        
        # Save masks for each image
        for img_idx, image in enumerate(self.images):
            image_name = getattr(image, 'name', f'Image_{img_idx}').replace('.hdr', '')
            
            print(f"\n{'='*70}")
            print(f"💾 Saving masks for: {image_name}")
            print(f"{'='*70}")
            
            # Save combined mask as numpy array (de-repeated to original size)
            combined = self.get_combined_mask(original_size=True, image_idx=img_idx)
            combined_path = output_path / f'{prefix}_{image_name}_combined_{timestamp}.npy'
            np.save(combined_path, combined)
            print(f"✅ Combined mask saved: {combined_path.name}")
            print(f"   Shape: {combined.shape} (original image dimensions)")
            
            # Save individual class masks (de-repeated to original size)
            masks = self.get_masks(original_size=True, image_idx=img_idx)
            for class_name, mask in masks.items():
                # Numpy array
                mask_path = output_path / f'{prefix}_{image_name}_{class_name}_{timestamp}.npy'
                np.save(mask_path, mask)
                
                # PNG for visualization
                import matplotlib.pyplot as plt
                png_path = output_path / f'{prefix}_{image_name}_{class_name}_{timestamp}.png'
                plt.imsave(png_path, mask, cmap='gray')
                
                pixel_count = mask.sum()
                percentage = (pixel_count / mask.size) * 100
                print(f"   • {class_name}: {pixel_count} pixels ({percentage:.2f}%) → {mask_path.name}")
            
            # Save metadata for this image
            orig_h, orig_w = self.original_dims[img_idx]
            metadata = {
                'image_name': image_name,
                'image_index': img_idx,
                'classes': self.classes,
                'colors': [color for color in self.colors],
                'shape': combined.shape,
                'original_shape': (orig_h, orig_w),
                'timestamp': timestamp,
                'repeat': self.repeat,
                'total_annotated_pixels': int(np.sum(combined > 0))
            }
            metadata_path = output_path / f'{prefix}_{image_name}_metadata_{timestamp}.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            
            print(f"✅ Metadata saved: {metadata_path.name}")
            print(f"📊 Total annotated pixels: {metadata['total_annotated_pixels']} / {combined.size}")
            print(f"    Coverage: {(metadata['total_annotated_pixels'] / combined.size * 100):.2f}%")
            if self.repeat > 1:
                print(f"📏 Masks de-repeated from {self.rgb_images[img_idx].shape[:2]} to {combined.shape}")
        
        print(f"\n{'='*70}")
        print(f"✅ All masks saved to: {output_path}")
        print(f"{'='*70}")
    
    def load_masks(self, mask_paths: Union[str, List[str]]):
        """
        Load previously saved annotation masks.
        
        Parameters:
            mask_paths: Union[str, List[str]]
                Path to the saved combined .npy mask file(s).
                Can be a single path (for single image) or list of paths (for multiple images).
                Number of paths must match number of images.
        """
        # Convert single path to list for consistent handling
        if isinstance(mask_paths, (str, Path)):
            mask_paths = [mask_paths]
        
        if len(mask_paths) != len(self.images):
            raise ValueError(f"Number of mask paths ({len(mask_paths)}) must match number of images ({len(self.images)})")
        
        # Load masks for each image
        for img_idx, mask_path in enumerate(mask_paths):
            loaded_mask = np.load(mask_path)
            
            # The loaded mask is at original size, need to repeat if necessary
            expected_original_shape = self.original_dims[img_idx]
            expected_display_shape = self.rgb_images[img_idx].shape[:2]
            
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
            
            # Split combined mask into individual class masks for this image
            for i, class_name in enumerate(self.classes, start=1):
                self.label_layers[img_idx][class_name] = (loaded_mask == i).astype(np.uint8)
            
            print(f"✅ Masks loaded for image {img_idx} from: {Path(mask_path).name}")
        
        print(f"\n{'='*70}")
        print(f"✅ All masks loaded successfully")
        print(f"   Classes loaded: {', '.join(self.classes)}")
        if self.repeat > 1:
            print(f"   Masks repeated {self.repeat}x to match display size")
        print(f"{'='*70}")


# Module-level check for napari availability
if not NAPARI_AVAILABLE:
    import warnings
    warnings.warn(
        "Napari is not installed. Install with 'pip install napari[pyqt5]' to use annotation tools.",
        ImportWarning
    )
