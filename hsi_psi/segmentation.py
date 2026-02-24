"""
Spectral Segmentation Module for HSI_PSI

This module provides tools for automated hyperspectral image segmentation using
machine learning models trained on manually annotated data.

Main Features:
- Extract training data from annotated images
- Visualize spectral signatures by class
- Automated model selection with Optuna (Random Forest, Gradient Boosting, SVM, PCA+RF)
- Image segmentation with trained models
- Batch processing of image folders
- Model persistence (save/load)

Example:
    >>> from hsi_psi.segmentation import SpectralSegmenter
    >>> 
    >>> # Initialize segmenter
    >>> segmenter = SpectralSegmenter()
    >>> 
    >>> # Extract training data from annotated images
    >>> segmenter.extract_training_data(images, masks)
    >>> 
    >>> # Visualize spectra (optional)
    >>> segmenter.visualize_spectra(show_boxplots=True)
    >>> 
    >>> # Optimize model selection
    >>> segmenter.optimize_model(n_trials=50, n_jobs=-1)
    >>> 
    >>> # Predict on new image
    >>> mask = segmenter.predict_image(new_image)
    >>> 
    >>> # Visualize results
    >>> segmenter.visualize_results(new_image, mask)
    >>> 
    >>> # Save model for later use
    >>> segmenter.save_model('my_segmentation_model.pkl')
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import warnings
from pathlib import Path
from typing import List, Dict, Union, Tuple, Optional
import joblib
import json
from datetime import datetime

# Machine learning imports
try:
    import optuna
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn and/or optuna not available. Install with: pip install scikit-learn optuna")


class SpectralSegmenter:
    """
    Automated hyperspectral image segmentation using machine learning.
    
    This class handles the complete workflow from annotated training data
    to automated segmentation of new images.
    
    Attributes:
        X_train (np.ndarray): Training features (spectral data)
        X_test (np.ndarray): Test features
        y_train (np.ndarray): Training labels
        y_test (np.ndarray): Test labels
        wavelengths (np.ndarray): Wavelength values for spectral bands
        class_names (list): Names of segmentation classes
        best_model: Trained classification model
        study (optuna.Study): Optuna optimization study
        training_data (pd.DataFrame): Combined training data with labels
    """
    
    def __init__(self, verbose=False):
        """
        Initialize SpectralSegmenter.
        
        Args:
            verbose (bool): Print progress information
        """
        self.verbose = verbose
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.wavelengths = None
        self.class_names = []
        self.best_model = None
        self.study = None
        self.training_data = None
        self.label_mapping = {}  # Maps class names to numeric labels
        self.reverse_label_mapping = {}  # Maps numeric labels to class names
        
        if not SKLEARN_AVAILABLE:
            warnings.warn("Machine learning features require scikit-learn and optuna")
    
    def extract_training_data(
        self, 
        images: List, 
        masks: Union[List[Dict[str, np.ndarray]], Dict[int, Dict[str, np.ndarray]]],
        test_size: float = 0.3,
        random_state: int = 42
    ):
        """
        Extract training data from annotated images.
        
        Args:
            images (list): List of HS_image or MS_image objects
            masks: Mask data in one of two formats:
                   - List of dicts: [{class_name: mask_array}, ...] one dict per image
                   - Dict of dicts: {image_idx: {class_name: mask_array}} from NapariHS_Annotator
            test_size (float): Fraction of data for test set (0.0-1.0)
            random_state (int): Random seed for reproducibility
            
        Returns:
            self: Returns self for method chaining
            
        Example:
            >>> # From NapariHS_Annotator
            >>> masks = annotator.get_masks()
            >>> segmenter.extract_training_data(images, masks)
            >>> 
            >>> # Or with list format
            >>> masks = [{'Plant': mask1, 'Background': mask2}, {...}]
            >>> segmenter.extract_training_data(images, masks)
        """
        # Convert dict format from NapariHS_Annotator to list format
        if isinstance(masks, dict):
            # Convert {image_idx: {class_name: mask}} to list [{class_name: mask}, ...]
            masks_list = []
            for img_idx in sorted(masks.keys()):
                masks_list.append(masks[img_idx])
            masks = masks_list
        
        if len(images) != len(masks):
            raise ValueError(f"Number of images ({len(images)}) must match number of mask dicts ({len(masks)})")
        
        if self.verbose:
            print("="*60)
            print("EXTRACTING TRAINING DATA")
            print("="*60)
        
        all_spectra = []
        all_labels = []
        
        # Get wavelengths from first image
        self.wavelengths = np.array(images[0].ind)
        
        # Collect all unique class names
        all_class_names = set()
        for mask_dict in masks:
            all_class_names.update(mask_dict.keys())
        self.class_names = sorted(list(all_class_names))
        
        # Create label mapping
        self.label_mapping = {name: idx for idx, name in enumerate(self.class_names)}
        self.reverse_label_mapping = {idx: name for name, idx in self.label_mapping.items()}
        
        if self.verbose:
            print(f"Classes found: {self.class_names}")
            print(f"Label mapping: {self.label_mapping}\n")
        
        # Extract data from each image
        for img_idx, (image, mask_dict) in enumerate(zip(images, masks)):
            if self.verbose:
                print(f"Processing image {img_idx + 1}/{len(images)}: {getattr(image, 'name', f'Image_{img_idx}')}")
            
            img_data = image.img  # Shape: (rows, cols, bands)
            rows, cols, n_bands = img_data.shape
            
            # Flatten spatial dimensions
            img_flat = img_data.reshape(-1, n_bands)
            
            # Process each class
            for class_name, mask in mask_dict.items():
                if mask is None or not np.any(mask):
                    if self.verbose:
                        print(f"  ⚠ {class_name}: No pixels annotated, skipping")
                    continue
                
                # Flatten mask
                mask_flat = mask.flatten()
                
                # Extract masked pixels
                class_pixels = img_flat[mask_flat == 1]
                n_pixels = len(class_pixels)
                
                if n_pixels == 0:
                    if self.verbose:
                        print(f"  ⚠ {class_name}: No pixels after extraction, skipping")
                    continue
                
                # Create labels
                class_label = self.label_mapping[class_name]
                labels = np.full(n_pixels, class_label, dtype=int)
                
                # Append to collection
                all_spectra.append(class_pixels)
                all_labels.append(labels)
                
                if self.verbose:
                    print(f"  ✓ {class_name}: {n_pixels} pixels extracted")
        
        if not all_spectra:
            raise ValueError("No training data extracted. Check that masks contain annotated pixels.")
        
        # Combine all data
        X = np.vstack(all_spectra)
        y = np.concatenate(all_labels)
        
        # Create DataFrame for easy inspection
        columns = [f"{wl:.1f}" for wl in self.wavelengths]
        self.training_data = pd.DataFrame(X, columns=columns)
        self.training_data['label'] = y
        self.training_data['class_name'] = [self.reverse_label_mapping[label] for label in y]
        
        # Train/test split with stratification
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"EXTRACTION COMPLETE")
            print(f"{'='*60}")
            print(f"Total samples: {len(X)}")
            print(f"Features (wavelengths): {X.shape[1]}")
            print(f"\nClass distribution:")
            for class_name in self.class_names:
                class_label = self.label_mapping[class_name]
                count = np.sum(y == class_label)
                print(f"  {class_name}: {count} pixels ({count/len(y)*100:.1f}%)")
            print(f"\nTrain set: {len(self.X_train)} samples")
            print(f"Test set: {len(self.X_test)} samples")
            print(f"{'='*60}")
        
        return self
    
    def visualize_spectra(
        self,
        show_mean=True,
        show_individuals=False,
        show_boxplots=False,
        show_violins=False,
        n_individual_samples=100,
        n_boxplot_features=8,
        figsize=(16, 5)
    ):
        """
        Visualize spectral signatures by class.
        
        Args:
            show_mean (bool): Show mean spectra with std bands
            show_individuals (bool): Show sample individual spectra
            show_boxplots (bool): Show boxplots of most discriminative features
            show_violins (bool): Show violin plots across wavelengths
            n_individual_samples (int): Number of individual spectra to plot
            n_boxplot_features (int): Number of features for boxplots
            figsize (tuple): Figure size for plots
            
        Returns:
            dict: Dictionary with mean spectra and statistics
        """
        if self.training_data is None:
            raise ValueError("No training data. Call extract_training_data() first.")
        
        results = {}
        
        # Calculate statistics per class
        for class_name in self.class_names:
            class_label = self.label_mapping[class_name]
            mask = self.training_data['label'] == class_label
            class_data = self.training_data[mask].drop(['label', 'class_name'], axis=1)
            
            results[class_name] = {
                'mean': class_data.mean(axis=0).values,
                'std': class_data.std(axis=0).values,
                'median': class_data.median(axis=0).values,
                'min': class_data.min(axis=0).values,
                'max': class_data.max(axis=0).values,
                'n_samples': len(class_data)
            }
        
        # Plot mean spectra
        if show_mean:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            
            colors = plt.cm.tab10(np.linspace(0, 1, len(self.class_names)))
            
            for i, class_name in enumerate(self.class_names):
                stats = results[class_name]
                ax.plot(self.wavelengths, stats['mean'], 
                       color=colors[i], linewidth=2, label=f"{class_name} (n={stats['n_samples']})")
                ax.fill_between(self.wavelengths, 
                               stats['mean'] - stats['std'], 
                               stats['mean'] + stats['std'],
                               color=colors[i], alpha=0.2)
            
            ax.set_xlabel('Wavelength (nm)', fontsize=12)
            ax.set_ylabel('Reflectance', fontsize=12)
            ax.set_title('Mean Spectral Signatures ± 1 Std Dev', fontsize=14, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        
        # Plot individual spectra
        if show_individuals:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            
            colors = plt.cm.tab10(np.linspace(0, 1, len(self.class_names)))
            
            for i, class_name in enumerate(self.class_names):
                class_label = self.label_mapping[class_name]
                mask = self.training_data['label'] == class_label
                class_data = self.training_data[mask].drop(['label', 'class_name'], axis=1)
                
                n_samples = min(n_individual_samples, len(class_data))
                sample_indices = np.random.choice(len(class_data), n_samples, replace=False)
                
                for idx in sample_indices:
                    ax.plot(self.wavelengths, class_data.iloc[idx].values, 
                           color=colors[i], alpha=0.05, linewidth=0.5)
                
                # Overlay mean
                stats = results[class_name]
                ax.plot(self.wavelengths, stats['mean'], 
                       color=colors[i], linewidth=2.5, label=class_name)
            
            ax.set_xlabel('Wavelength (nm)', fontsize=12)
            ax.set_ylabel('Reflectance', fontsize=12)
            ax.set_title(f'Individual Spectra (showing {n_individual_samples} samples per class)', 
                        fontsize=14, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        
        # Plot boxplots for most discriminative wavelengths
        if show_boxplots and len(self.class_names) == 2:
            # Calculate differences between classes (only works for 2 classes)
            class1, class2 = self.class_names
            diff = np.abs(results[class1]['mean'] - results[class2]['mean'])
            top_indices = np.argsort(diff)[-n_boxplot_features:][::-1]
            
            n_cols = 4
            n_rows = int(np.ceil(n_boxplot_features / n_cols))
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 4))
            axes = axes.flatten() if n_boxplot_features > 1 else [axes]
            
            colors_dict = {name: plt.cm.tab10(i) for i, name in enumerate(self.class_names)}
            
            for plot_idx, wl_idx in enumerate(top_indices):
                wl = self.wavelengths[wl_idx]
                
                data_to_plot = []
                labels = []
                for class_name in self.class_names:
                    class_label = self.label_mapping[class_name]
                    mask = self.training_data['label'] == class_label
                    class_data = self.training_data[mask].iloc[:, wl_idx]
                    data_to_plot.append(class_data.values)
                    labels.append(class_name)
                
                bp = axes[plot_idx].boxplot(data_to_plot, labels=labels, patch_artist=True, widths=0.6)
                
                for patch, class_name in zip(bp['boxes'], self.class_names):
                    patch.set_facecolor(colors_dict[class_name])
                    patch.set_alpha(0.6)
                
                for median in bp['medians']:
                    median.set(color='black', linewidth=2)
                
                axes[plot_idx].set_ylabel('Reflectance', fontsize=10)
                axes[plot_idx].set_title(f'{wl:.1f} nm\nΔ = {diff[wl_idx]:.4f}', 
                                        fontsize=11, fontweight='bold')
                axes[plot_idx].grid(True, alpha=0.3, axis='y')
            
            # Hide unused subplots
            for idx in range(n_boxplot_features, len(axes)):
                axes[idx].axis('off')
            
            plt.suptitle('Reflectance Distribution at Most Discriminative Wavelengths', 
                        fontsize=14, fontweight='bold', y=1.00)
            plt.tight_layout()
            plt.show()
        
        # Plot violin plots
        if show_violins:
            n_show = 15
            step = max(1, len(self.wavelengths) // n_show)
            indices = list(range(0, len(self.wavelengths), step))[:n_show]
            indices_sorted = sorted(indices, key=lambda i: self.wavelengths[i])
            
            fig, ax = plt.subplots(1, 1, figsize=(16, 6))
            
            positions_offset = np.linspace(-0.3, 0.3, len(self.class_names))
            colors = plt.cm.tab10(np.linspace(0, 1, len(self.class_names)))
            
            for i, class_name in enumerate(self.class_names):
                class_label = self.label_mapping[class_name]
                mask = self.training_data['label'] == class_label
                class_data = self.training_data[mask].drop(['label', 'class_name'], axis=1)
                
                data_to_plot = [class_data.iloc[:, idx].values for idx in indices_sorted]
                positions = np.arange(len(indices_sorted)) + positions_offset[i]
                
                parts = ax.violinplot(data_to_plot, positions=positions, widths=0.6/len(self.class_names),
                                     showmeans=True, showmedians=False)
                
                for pc in parts['bodies']:
                    pc.set_facecolor(colors[i])
                    pc.set_alpha(0.6)
                    pc.set_edgecolor(colors[i])
                
                parts['cmeans'].set_edgecolor(colors[i])
                parts['cmeans'].set_linewidth(2)
                
                for key in ['cbars', 'cmins', 'cmaxes']:
                    parts[key].set_visible(False)
            
            ax.set_xticks(np.arange(len(indices_sorted)))
            ax.set_xticklabels([f'{self.wavelengths[idx]:.0f}' for idx in indices_sorted], rotation=45)
            ax.set_xlabel('Wavelength (nm)', fontsize=12)
            ax.set_ylabel('Reflectance', fontsize=12)
            ax.set_title('Reflectance Distribution Across Wavelengths (Violin Plots)', 
                        fontsize=14, fontweight='bold')
            
            legend_elements = [Patch(facecolor=colors[i], alpha=0.6, label=name) 
                             for i, name in enumerate(self.class_names)]
            ax.legend(handles=legend_elements, fontsize=11, loc='upper right')
            
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.show()
        
        return results
    
    def optimize_model(
        self,
        n_trials: int = 50,
        n_jobs: int = -1,
        cv_folds: int = 5,
        random_state: int = 42,
        timeout: Optional[int] = None,
        param_ranges: Optional[Dict] = None,
        downsample: Optional[Union[float, int]] = None,
        balance_classes: bool = False,
        upsample_minority: bool = False,
        upsample_to: Optional[int] = None
    ):
        """
        Optimize model selection using Optuna with optional data sampling.
        
        Tests Random Forest, Gradient Boosting, SVM, and PCA+RandomForest with different
        hyperparameters to find the best model for the data.
        
        Note: RandomForest is automatically disabled for datasets with >30 features 
        to avoid performance issues. Use PCA_RF instead for high-dimensional data.
        
        Args:
            n_trials (int): Number of optimization trials
            n_jobs (int): Number of parallel jobs (-1 for all cores)
            cv_folds (int): Number of cross-validation folds
            random_state (int): Random seed
            timeout (int, optional): Timeout in seconds for optimization
            param_ranges (dict, optional): Custom hyperparameter ranges. Example:
                {
                    'RandomForest': {
                        'n_estimators': (100, 500),
                        'max_depth': (5, 30),
                        'min_samples_split': (2, 30),
                        'min_samples_leaf': (1, 15)
                    },
                    'GradientBoosting': {
                        'n_estimators': (100, 400),
                        'learning_rate': (0.001, 0.5),  # will use log scale
                        'max_depth': (3, 15),
                        'subsample': (0.6, 1.0)
                    },
                    'SVM': {
                        'C': (0.01, 1000),  # will use log scale
                        'gamma': (1e-5, 10),  # will use log scale
                        'kernel': ['rbf', 'linear', 'poly']
                    },
                    'PCA_RF': {
                        'n_components': (10, 50),
                        'n_estimators': (100, 500),
                        'max_depth': (5, 30),
                        'min_samples_split': (2, 30),
                        'min_samples_leaf': (1, 15)
                    }
                }
            downsample (float or int, optional): Downsample training data.
                - If float (0-1): Keep this percentage of samples (e.g., 0.5 = 50%)
                - If int: Keep this exact number of samples (randomly selected)
                - Without balance_classes: maintains original class proportions
                - With balance_classes: distributes samples equally across classes
            balance_classes (bool): Balance class distribution during sampling.
                - When used alone: downsample majority classes to minority class size
                - When used with downsample: distribute downsampled data equally across classes
                - Cannot be used with upsample_minority or upsample_to
            upsample_minority (bool): If True, upsample minority classes to match
                the majority class size (automatic class balancing)
            upsample_to (int, optional): Upsample all classes to this exact number
                of samples (overrides upsample_minority)
            
        Returns:
            self: Returns self for method chaining
            
        Examples:
            # Downsample to 50% of data (maintains class proportions)
            segmenter.optimize_model(downsample=0.5)
            
            # Downsample to 10000 samples with balanced classes
            segmenter.optimize_model(downsample=10000, balance_classes=True)
            # → Each class gets 10000/n_classes samples
            
            # Balance classes without changing total size (downsample to minority)
            segmenter.optimize_model(balance_classes=True)
            # → All classes get minority_class_size samples
            
            # Balance classes by upsampling minority classes
            segmenter.optimize_model(upsample_minority=True)
            
            # Upsample all classes to 5000 samples each
            segmenter.optimize_model(upsample_to=5000)
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("optimize_model requires scikit-learn and optuna")
        
        if self.X_train is None:
            raise ValueError("No training data. Call extract_training_data() first.")
        
        # Validate parameter combinations
        if balance_classes and (upsample_minority or upsample_to is not None):
            raise ValueError("balance_classes cannot be used with upsample_minority or upsample_to. "
                           "Use either balance_classes (downsampling) or upsampling, not both.")
        
        # Apply sampling if requested
        X_train_sampled = self.X_train.copy()
        y_train_sampled = self.y_train.copy()
        
        if downsample is not None or balance_classes or upsample_minority or upsample_to is not None:
            if self.verbose:
                print("="*60)
                print("DATA SAMPLING")
                print("="*60)
                print(f"Original training data: {len(X_train_sampled)} samples")
                for class_name in self.class_names:
                    class_label = self.label_mapping[class_name]
                    count = np.sum(y_train_sampled == class_label)
                    print(f"  {class_name}: {count} samples")
        
        # Downsampling (with optional class balancing)
        if downsample is not None or balance_classes:
            if balance_classes:
                # Get class counts
                class_counts = {class_label: np.sum(y_train_sampled == class_label) 
                               for class_label in range(len(self.class_names))}
                
                if downsample is not None:
                    # Balance classes AND downsample: distribute target evenly
                    if isinstance(downsample, float) and 0 < downsample < 1:
                        total_target = int(len(X_train_sampled) * downsample)
                    elif isinstance(downsample, int) and downsample > 0:
                        total_target = min(downsample, len(X_train_sampled))
                    else:
                        raise ValueError("downsample must be float (0-1) or positive int")
                    
                    # Distribute equally across classes
                    per_class_target = total_target // len(self.class_names)
                    
                    if self.verbose:
                        print(f"\n✓ Balancing + Downsampling to {total_target} samples")
                        print(f"  Target per class: {per_class_target} samples")
                else:
                    # Balance only: downsample to minority class size
                    per_class_target = min(class_counts.values())
                    if self.verbose:
                        print(f"\n✓ Balancing classes (downsampling to minority)")
                        print(f"  Target per class: {per_class_target} samples")
                
                # Downsample each class to target
                balanced_X = []
                balanced_y = []
                
                np.random.seed(random_state)
                
                for class_label in range(len(self.class_names)):
                    class_mask = y_train_sampled == class_label
                    class_X = X_train_sampled[class_mask]
                    class_y = y_train_sampled[class_mask]
                    
                    current_count = len(class_X)
                    
                    if current_count > per_class_target:
                        # Downsample this class
                        indices = np.random.choice(current_count, size=per_class_target, replace=False)
                        balanced_X.append(class_X[indices])
                        balanced_y.append(class_y[indices])
                    else:
                        # Keep all samples (already at or below target)
                        balanced_X.append(class_X)
                        balanced_y.append(class_y)
                
                X_train_sampled = np.vstack(balanced_X)
                y_train_sampled = np.concatenate(balanced_y)
                
                if self.verbose:
                    for class_name in self.class_names:
                        class_label = self.label_mapping[class_name]
                        count = np.sum(y_train_sampled == class_label)
                        print(f"  {class_name}: {count} samples")
                        
            else:
                # Downsample WITHOUT balancing: stratified sampling
                if isinstance(downsample, float) and 0 < downsample < 1:
                    n_samples = int(len(X_train_sampled) * downsample)
                elif isinstance(downsample, int) and downsample > 0:
                    n_samples = min(downsample, len(X_train_sampled))
                else:
                    raise ValueError("downsample must be float (0-1) or positive int")
                
                # Stratified downsampling (maintains class proportions)
                indices = np.arange(len(X_train_sampled))
                from sklearn.model_selection import StratifiedShuffleSplit
                splitter = StratifiedShuffleSplit(n_splits=1, train_size=n_samples, random_state=random_state)
                sampled_idx, _ = next(splitter.split(X_train_sampled, y_train_sampled))
                
                X_train_sampled = X_train_sampled[sampled_idx]
                y_train_sampled = y_train_sampled[sampled_idx]
                
                if self.verbose:
                    print(f"\n✓ Downsampled to {n_samples} samples (stratified)")
                    for class_name in self.class_names:
                        class_label = self.label_mapping[class_name]
                        count = np.sum(y_train_sampled == class_label)
                        print(f"  {class_name}: {count} samples")
        
        # Upsampling
        if upsample_to is not None or upsample_minority:
            # Determine target count per class
            if upsample_to is not None:
                target_count = upsample_to
            elif upsample_minority:
                # Find majority class size
                class_counts = {class_label: np.sum(y_train_sampled == class_label) 
                               for class_label in range(len(self.class_names))}
                target_count = max(class_counts.values())
            
            # Upsample each class
            upsampled_X = []
            upsampled_y = []
            
            np.random.seed(random_state)
            
            for class_label in range(len(self.class_names)):
                class_mask = y_train_sampled == class_label
                class_X = X_train_sampled[class_mask]
                class_y = y_train_sampled[class_mask]
                
                current_count = len(class_X)
                
                if current_count >= target_count:
                    # Already at or above target, keep as is (or downsample if needed)
                    upsampled_X.append(class_X[:target_count])
                    upsampled_y.append(class_y[:target_count])
                else:
                    # Upsample by random resampling with replacement
                    n_to_add = target_count - current_count
                    random_indices = np.random.choice(current_count, size=n_to_add, replace=True)
                    
                    upsampled_X.append(np.vstack([class_X, class_X[random_indices]]))
                    upsampled_y.append(np.concatenate([class_y, class_y[random_indices]]))
            
            X_train_sampled = np.vstack(upsampled_X)
            y_train_sampled = np.concatenate(upsampled_y)
            
            if self.verbose:
                print(f"\n✓ Upsampled to {target_count} samples per class")
                print(f"Total samples after upsampling: {len(X_train_sampled)}")
                for class_name in self.class_names:
                    class_label = self.label_mapping[class_name]
                    count = np.sum(y_train_sampled == class_label)
                    print(f"  {class_name}: {count} samples")
        
        if downsample is not None or balance_classes or upsample_minority or upsample_to is not None:
            if self.verbose:
                print(f"{'='*60}\n")
        
        if self.verbose:
            print("="*60)
            print("STARTING MODEL OPTIMIZATION")
            print("="*60)
            print(f"Trials: {n_trials}")
            print(f"CV Folds: {cv_folds}")
            print(f"Parallel Jobs: {n_jobs if n_jobs > 0 else 'All available cores'}")
            if timeout:
                print(f"Timeout: {timeout} seconds")
            if param_ranges:
                print(f"Custom hyperparameter ranges provided")
            print(f"Training samples (after sampling): {len(X_train_sampled)}")
            print(f"{'='*60}\n")
        
        # Default hyperparameter ranges
        default_ranges = {
            'RandomForest': {
                'n_estimators': (50, 300),
                'max_depth': (3, 20),
                'min_samples_split': (2, 20),
                'min_samples_leaf': (1, 10)
            },
            'GradientBoosting': {
                'n_estimators': (50, 300),
                'learning_rate': (0.01, 0.3),
                'max_depth': (3, 10),
                'subsample': (0.5, 1.0)
            },
            'SVM': {
                'C': (0.1, 100),
                'gamma': (1e-4, 1),
                'kernel': ['rbf', 'linear']
            },
            'PCA_RF': {
                'n_components': (15, 50),  # Default centered around 30
                'n_estimators': (50, 300),
                'max_depth': (3, 10),  # Max depth up to 10
                'min_samples_split': (2, 20),
                'min_samples_leaf': (1, 10)
            }
        }
        
        # Merge custom ranges with defaults
        if param_ranges is None:
            param_ranges = default_ranges
        else:
            # Update defaults with custom ranges
            for model_type in default_ranges:
                if model_type not in param_ranges:
                    param_ranges[model_type] = default_ranges[model_type]
                else:
                    # Merge with defaults
                    param_ranges[model_type] = {**default_ranges[model_type], **param_ranges[model_type]}
        
        # Check number of features and automatically block RF for high-dimensional data
        n_features = X_train_sampled.shape[1]
        available_models = ['RandomForest', 'GradientBoosting', 'SVM', 'PCA_RF']
        
        if n_features > 30:
            available_models.remove('RandomForest')
            if self.verbose:
                print(f"⚠️  RandomForest disabled: Dataset has {n_features} features (>30)")
                print(f"   Available models: {available_models}")
                print(f"   Use PCA_RF for dimensionality reduction + RandomForest\n")
        
        def objective(trial):
            """Optuna objective function."""
            model_type = trial.suggest_categorical('model_type', available_models)
            
            if model_type == 'RandomForest':
                rf_ranges = param_ranges['RandomForest']
                n_estimators = trial.suggest_int('rf_n_estimators', *rf_ranges['n_estimators'])
                max_depth = trial.suggest_int('rf_max_depth', *rf_ranges['max_depth'])
                min_samples_split = trial.suggest_int('rf_min_samples_split', *rf_ranges['min_samples_split'])
                min_samples_leaf = trial.suggest_int('rf_min_samples_leaf', *rf_ranges['min_samples_leaf'])
                
                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    random_state=random_state,
                    n_jobs=1  # cv does parallelization
                )
                
            elif model_type == 'GradientBoosting':
                gb_ranges = param_ranges['GradientBoosting']
                n_estimators = trial.suggest_int('gb_n_estimators', *gb_ranges['n_estimators'])
                learning_rate = trial.suggest_float('gb_learning_rate', *gb_ranges['learning_rate'], log=True)
                max_depth = trial.suggest_int('gb_max_depth', *gb_ranges['max_depth'])
                subsample = trial.suggest_float('gb_subsample', *gb_ranges['subsample'])
                
                model = GradientBoostingClassifier(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    subsample=subsample,
                    random_state=random_state
                )
                
            elif model_type == 'SVM':
                svm_ranges = param_ranges['SVM']
                C = trial.suggest_float('svm_C', *svm_ranges['C'], log=True)
                gamma = trial.suggest_float('svm_gamma', *svm_ranges['gamma'], log=True)
                kernel = trial.suggest_categorical('svm_kernel', svm_ranges['kernel'])
                
                model = SVC(
                    C=C,
                    gamma=gamma,
                    kernel=kernel,
                    random_state=random_state
                )
            
            else:  # PCA_RF
                pca_rf_ranges = param_ranges['PCA_RF']
                n_components = trial.suggest_int('pca_rf_n_components', *pca_rf_ranges['n_components'])
                n_estimators = trial.suggest_int('pca_rf_n_estimators', *pca_rf_ranges['n_estimators'])
                max_depth = trial.suggest_int('pca_rf_max_depth', *pca_rf_ranges['max_depth'])
                min_samples_split = trial.suggest_int('pca_rf_min_samples_split', *pca_rf_ranges['min_samples_split'])
                min_samples_leaf = trial.suggest_int('pca_rf_min_samples_leaf', *pca_rf_ranges['min_samples_leaf'])
                
                model = Pipeline([
                    ('pca', PCA(n_components=n_components, random_state=random_state)),
                    ('rf', RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        random_state=random_state,
                        n_jobs=1
                    ))
                ])
            
            # Cross-validation on sampled data
            cv_scores = cross_val_score(model, X_train_sampled, y_train_sampled, 
                                       cv=cv_folds, scoring='accuracy', n_jobs=n_jobs)
            
            return cv_scores.mean()
        
        # Create and run study
        self.study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=random_state)
        )
        
        self.study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=self.verbose,
            n_jobs=1  # Optuna parallelization (conflicts with sklearn's n_jobs)
        )
        
        # Build best model
        best_params = self.study.best_params
        model_type = best_params['model_type']
        
        if self.verbose:
            print(f"\n{'='*60}")
            print("OPTIMIZATION COMPLETE")
            print(f"{'='*60}")
            print(f"Best trial: {self.study.best_trial.number}")
            print(f"Best CV Accuracy: {self.study.best_value:.4f}")
            print(f"\nBest model: {model_type}")
            print("Best hyperparameters:")
            for key, value in best_params.items():
                print(f"  {key}: {value}")
        
        # Train best model on full training set
        if model_type == 'RandomForest':
            self.best_model = RandomForestClassifier(
                n_estimators=best_params['rf_n_estimators'],
                max_depth=best_params['rf_max_depth'],
                min_samples_split=best_params['rf_min_samples_split'],
                min_samples_leaf=best_params['rf_min_samples_leaf'],
                random_state=random_state,
                n_jobs=n_jobs
            )
        elif model_type == 'GradientBoosting':
            self.best_model = GradientBoostingClassifier(
                n_estimators=best_params['gb_n_estimators'],
                learning_rate=best_params['gb_learning_rate'],
                max_depth=best_params['gb_max_depth'],
                subsample=best_params['gb_subsample'],
                random_state=random_state
            )
        elif model_type == 'SVM':
            self.best_model = SVC(
                C=best_params['svm_C'],
                gamma=best_params['svm_gamma'],
                kernel=best_params['svm_kernel'],
                random_state=random_state
            )
        else:  # PCA_RF
            self.best_model = Pipeline([
                ('pca', PCA(n_components=best_params['pca_rf_n_components'], random_state=random_state)),
                ('rf', RandomForestClassifier(
                    n_estimators=best_params['pca_rf_n_estimators'],
                    max_depth=best_params['pca_rf_max_depth'],
                    min_samples_split=best_params['pca_rf_min_samples_split'],
                    min_samples_leaf=best_params['pca_rf_min_samples_leaf'],
                    random_state=random_state,
                    n_jobs=n_jobs
                ))
            ])
        
        if self.verbose:
            print(f"\nTraining {model_type} on full training set...")
        
        # Train on sampled data (use original data if no sampling was done)
        self.best_model.fit(X_train_sampled, y_train_sampled)
        
        # Store info about whether sampling was used
        self.used_sampling = (downsample is not None or balance_classes or upsample_minority or upsample_to is not None)
        if self.used_sampling:
            self.sampled_train_size = len(X_train_sampled)
        
        # Evaluate on test set
        y_pred = self.best_model.predict(self.X_test)
        test_accuracy = accuracy_score(self.y_test, y_pred)
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Test Set Accuracy: {test_accuracy:.4f}")
            if self.used_sampling:
                print(f"(Model trained on {self.sampled_train_size} sampled training examples)")
            print(f"{'='*60}\n")
            
            print("Classification Report:")
            print(classification_report(self.y_test, y_pred, 
                                       target_names=self.class_names))
            
            print("Confusion Matrix:")
            cm = confusion_matrix(self.y_test, y_pred)
            print(cm)
            print(f"{'='*60}")
        
        return self
    
    def predict_image(self, image) -> np.ndarray:
        """
        Segment an image using the trained model.
        
        Args:
            image: HS_image or MS_image object to segment
            
        Returns:
            np.ndarray: Segmentation mask with shape (rows, cols)
                       Values are class indices (use reverse_label_mapping to get names)
        """
        if self.best_model is None:
            raise ValueError("No trained model. Call optimize_model() first.")
        
        img_data = image.img
        rows, cols, n_bands = img_data.shape
        
        # Flatten spatial dimensions
        img_flat = img_data.reshape(-1, n_bands)
        
        # Predict
        predictions = self.best_model.predict(img_flat)
        
        # Reshape to image dimensions
        mask = predictions.reshape(rows, cols)
        
        if self.verbose:
            print(f"Segmented image: {getattr(image, 'name', 'Unknown')}")
            print(f"  Shape: {mask.shape}")
            for class_name in self.class_names:
                class_label = self.label_mapping[class_name]
                n_pixels = np.sum(mask == class_label)
                print(f"  {class_name}: {n_pixels} pixels ({n_pixels/mask.size*100:.1f}%)")
        
        return mask
    
    def visualize_results(
        self,
        image,
        predicted_mask: np.ndarray,
        ground_truth_mask: Optional[Dict[str, np.ndarray]] = None,
        show_rgb: bool = True,
        show_overlay: bool = True,
        figsize: Tuple[int, int] = (18, 6)
    ):
        """
        Visualize segmentation results.
        
        Args:
            image: HS_image or MS_image object
            predicted_mask (np.ndarray): Predicted segmentation mask
            ground_truth_mask (dict, optional): Manual annotation masks {class_name: mask}
            show_rgb (bool): Show RGB representation of image
            show_overlay (bool): Show overlay of mask on RGB
            figsize (tuple): Figure size
        """
        from .preprocessing import HS_preprocessor
        
        # Calculate number of plots needed
        n_plots = 0
        if ground_truth_mask is not None:
            n_plots += 1  # Ground truth
        elif show_rgb:
            n_plots += 1  # RGB image
        n_plots += 1  # Predicted mask (always shown)
        if show_overlay:
            n_plots += 1  # Overlay
        
        fig, axes = plt.subplots(1, n_plots, figsize=figsize)
        if n_plots == 1:
            axes = [axes]
        
        plot_idx = 0
        
        # Get RGB image if needed
        if show_rgb or show_overlay:
            rgb_img = HS_preprocessor.get_rgb_sample_from_image(
                image, normalize=True, correct=False, show=False, verbose=False
            )
        
        # Plot 1: RGB or manual annotation
        if ground_truth_mask is not None:
            # Create combined manual mask
            rows, cols = predicted_mask.shape
            manual_combined = np.full((rows, cols), -1, dtype=int)
            
            for class_name, mask in ground_truth_mask.items():
                if mask is not None and class_name in self.class_names:
                    class_label = self.label_mapping[class_name]
                    manual_combined[mask == 1] = class_label
            
            im = axes[plot_idx].imshow(manual_combined, cmap='RdYlGn', 
                                      vmin=-1, vmax=len(self.class_names)-1)
            axes[plot_idx].set_title('Manual Annotation', fontsize=14, fontweight='bold')
            axes[plot_idx].axis('off')
            
            cbar = plt.colorbar(im, ax=axes[plot_idx], fraction=0.046, pad=0.04)
            ticks = [-1] + list(range(len(self.class_names)))
            cbar.set_ticks(ticks)
            cbar.set_ticklabels(['Unannotated'] + self.class_names)
            
            plot_idx += 1
        elif show_rgb:
            axes[plot_idx].imshow(rgb_img)
            axes[plot_idx].set_title('Original Image (RGB)', fontsize=14, fontweight='bold')
            axes[plot_idx].axis('off')
            plot_idx += 1
        
        # Plot 2: Predicted segmentation
        im = axes[plot_idx].imshow(predicted_mask, cmap='RdYlGn', 
                                   vmin=0, vmax=len(self.class_names)-1)
        
        model_type = self.study.best_params['model_type'] if self.study else 'Model'
        axes[plot_idx].set_title(f'Predicted Segmentation ({model_type})', 
                                fontsize=14, fontweight='bold')
        axes[plot_idx].axis('off')
        
        cbar = plt.colorbar(im, ax=axes[plot_idx], fraction=0.046, pad=0.04)
        cbar.set_ticks(range(len(self.class_names)))
        cbar.set_ticklabels(self.class_names)
        
        plot_idx += 1
        
        # Plot 3: Overlay
        if show_overlay:
            axes[plot_idx].imshow(rgb_img)
            axes[plot_idx].imshow(predicted_mask, cmap='RdYlGn', alpha=0.5,
                                 vmin=0, vmax=len(self.class_names)-1)
            axes[plot_idx].set_title('RGB + Segmentation Overlay', 
                                    fontsize=14, fontweight='bold')
            axes[plot_idx].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Calculate accuracy if ground truth available
        if ground_truth_mask is not None:
            rows, cols = predicted_mask.shape
            manual_combined = np.full((rows, cols), -1, dtype=int)
            
            for class_name, mask in ground_truth_mask.items():
                if mask is not None and class_name in self.class_names:
                    class_label = self.label_mapping[class_name]
                    manual_combined[mask == 1] = class_label
            
            annotated_pixels = manual_combined >= 0
            n_annotated = np.sum(annotated_pixels)
            
            if n_annotated > 0:
                y_true = manual_combined[annotated_pixels]
                y_pred = predicted_mask[annotated_pixels]
                
                accuracy = np.sum(y_true == y_pred) / n_annotated
                
                print("="*60)
                print("COMPARISON: Predicted vs Manual Annotation")
                print("="*60)
                print(f"Annotated pixels: {n_annotated}")
                print(f"Accuracy: {accuracy*100:.2f}%")
                
                print("\nConfusion Matrix:")
                cm = confusion_matrix(y_true, y_pred)
                print(cm)
                
                print("\nClassification Report:")
                print(classification_report(y_true, y_pred, target_names=self.class_names))
                print("="*60)
    
    def batch_predict(
        self,
        images: List,
        save_masks: bool = True,
        output_dir: Optional[str] = None,
        visualize: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Segment multiple images in batch.
        
        Args:
            images (list): List of HS_image or MS_image objects
            save_masks (bool): Save masks to files
            output_dir (str, optional): Directory to save masks
            visualize (bool): Show visualization for each image
            
        Returns:
            dict: Dictionary mapping image names to segmentation masks
        """
        if self.best_model is None:
            raise ValueError("No trained model. Call optimize_model() first.")
        
        if save_masks and output_dir is None:
            raise ValueError("output_dir must be provided when save_masks=True")
        
        if save_masks:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        if self.verbose:
            print("="*60)
            print(f"BATCH SEGMENTATION: {len(images)} images")
            print("="*60)
        
        for i, image in enumerate(images):
            image_name = getattr(image, 'name', f'image_{i}')
            
            if self.verbose:
                print(f"\n[{i+1}/{len(images)}] Processing: {image_name}")
            
            # Predict
            mask = self.predict_image(image)
            results[image_name] = mask
            
            # Save mask
            if save_masks:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                mask_file = output_path / f"segmentation_{image_name}_{timestamp}.npy"
                np.save(mask_file, mask)
                
                if self.verbose:
                    print(f"  Saved: {mask_file.name}")
            
            # Visualize
            if visualize:
                self.visualize_result(image, mask, show_rgb=True, show_overlay=True)
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"BATCH COMPLETE: {len(results)} images segmented")
            if save_masks:
                print(f"Masks saved to: {output_path}")
            print(f"{'='*60}")
        
        return results
    
    def save_model(self, filepath: str):
        """
        Save trained model and metadata to file.
        
        Args:
            filepath (str): Path to save model (.pkl extension recommended)
        """
        if self.best_model is None:
            raise ValueError("No trained model to save. Call optimize_model() first.")
        
        save_data = {
            'model': self.best_model,
            'wavelengths': self.wavelengths,
            'class_names': self.class_names,
            'label_mapping': self.label_mapping,
            'reverse_label_mapping': self.reverse_label_mapping,
            'study_params': self.study.best_params if self.study else None,
            'study_value': self.study.best_value if self.study else None,
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'n_train_samples': len(self.X_train) if self.X_train is not None else None,
                'n_test_samples': len(self.X_test) if self.X_test is not None else None,
                'n_features': len(self.wavelengths) if self.wavelengths is not None else None
            }
        }
        
        joblib.dump(save_data, filepath)
        
        if self.verbose:
            print(f"Model saved to: {filepath}")
            print(f"  Classes: {self.class_names}")
            print(f"  Features: {len(self.wavelengths)}")
            if self.study:
                print(f"  Best model: {self.study.best_params['model_type']}")
                print(f"  CV Accuracy: {self.study.best_value:.4f}")
    
    def load_model(self, filepath: str):
        """
        Load trained model from file.
        
        Args:
            filepath (str): Path to model file
            
        Returns:
            self: Returns self for method chaining
        """
        save_data = joblib.load(filepath)
        
        self.best_model = save_data['model']
        self.wavelengths = save_data['wavelengths']
        self.class_names = save_data['class_names']
        self.label_mapping = save_data['label_mapping']
        self.reverse_label_mapping = save_data['reverse_label_mapping']
        
        if self.verbose:
            print(f"Model loaded from: {filepath}")
            print(f"  Classes: {self.class_names}")
            print(f"  Features: {len(self.wavelengths)}")
            
            metadata = save_data.get('metadata', {})
            if metadata:
                print(f"  Trained: {metadata.get('timestamp', 'Unknown')}")
                print(f"  Training samples: {metadata.get('n_train_samples', 'Unknown')}")
            
            study_params = save_data.get('study_params')
            if study_params:
                print(f"  Model type: {study_params.get('model_type', 'Unknown')}")
                print(f"  CV Accuracy: {save_data.get('study_value', 'Unknown'):.4f}")
        
        return self
