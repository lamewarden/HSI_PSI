from sklearn.decomposition import PCA
from .core import HS_image
import numpy as np
import copy

from scipy import linalg
from scipy.ndimage import median_filter

class transformer:
    """
    Generic hyperspectral dimensionality reduction transformer.
    Supports PCA and MNF (Minimum Noise Fraction) transformations.
    """

    def __init__(self, method='pca'):
        """
        Initialize the transformer.
        
        Parameters:
        -----------
        method : str, optional (default='pca')
            Transformation method to use. Options: 'pca', 'mnf'
        """
        self.method = method.lower()
        if self.method not in ['pca', 'mnf']:
            raise ValueError("Method must be 'pca' or 'mnf'")
        
        # MNF is now implemented internally - no external dependencies needed
        
        self.fitted = False
        self.n_components = None
        self.original_shape = None
        
        # Method-specific attributes
        if self.method == 'pca':
            self.transformer_obj = None
        elif self.method == 'mnf':
            self.mean_ = None
            self.eigenvalues_ = None
            self.eigenvectors_ = None
            self.noise_cov_ = None
            self.signal_cov_ = None
    

    def HSI_to_X(self, HSI_image, drop_nulls=False):
        """
        Convert hyperspectral image to 2D matrix format.
        
        Parameters:
        -----------
        HSI_image : HS_image or numpy.ndarray
            Input hyperspectral image
        drop_nulls : bool, optional (default=False)
            Whether to drop pixels that are all zeros
            
        Returns:
        --------
        X : numpy.ndarray
            2D array of shape (n_pixels, n_bands)
        """
        # allows it to work both with HS_images and 3d arrays
        try:
            HSI_image.shape
            array = HSI_image.T
        except:
            array = copy.deepcopy(HSI_image.img).T
            
        # getting rid of nans
        array[np.isnan(array)] = 0

        self.original_shape = array.shape

        if drop_nulls:
            X_array = array.reshape(self.original_shape[0], -1).T
            return X_array[np.all(X_array != 0, axis=1)]
        
        return array.reshape(self.original_shape[0], -1).T

    def _estimate_noise_covariance(self, X):
        """
        Estimate noise covariance matrix using improved spatial differences method.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input data matrix (n_pixels, n_bands)
            
        Returns:
        --------
        noise_cov : numpy.ndarray
            Estimated noise covariance matrix (n_bands, n_bands)
        """
        if self.original_shape is None:
            raise RuntimeError("Must call HSI_to_X first to store original shape")
        
        # Reshape to spatial format (height, width, bands)
        height, width, bands = self.original_shape[1], self.original_shape[2], self.original_shape[0]
        spatial_data = X.T.reshape(height, width, bands)
        
        # Apply median filter to reduce strong edges and preserve noise characteristics
        # This helps separate signal from noise more effectively
        filtered_data = np.zeros_like(spatial_data)
        for b in range(bands):
            filtered_data[:, :, b] = median_filter(spatial_data[:, :, b], size=3)
        
        # Calculate differences from filtered version to estimate noise
        noise_estimates = spatial_data - filtered_data
        
        # Also use spatial differences as secondary noise estimate
        diff_data = []
        
        if height > 1:
            # Vertical differences (reduced weight)
            diff_v = np.diff(spatial_data, axis=0) * 0.5
            diff_data.append(diff_v.reshape(-1, bands))
        
        if width > 1:
            # Horizontal differences (reduced weight)
            diff_h = np.diff(spatial_data, axis=1) * 0.5
            diff_data.append(diff_h.reshape(-1, bands))
        
        # Combine noise estimates
        noise_data = noise_estimates.reshape(-1, bands)
        
        if diff_data:
            all_diffs = np.vstack(diff_data)
            # Combine both methods with weighting
            noise_data = np.vstack([noise_data * 0.7, all_diffs * 0.3])
        
        # Estimate noise covariance
        noise_cov = np.cov(noise_data.T)
        
        # More aggressive regularization to ensure proper conditioning
        eigenvals, eigenvecs = linalg.eigh(noise_cov)
        
        # Regularize: set minimum eigenvalue to prevent numerical issues
        min_eigenval = np.maximum(1e-8 * eigenvals.max(), 1e-12)
        eigenvals = np.maximum(eigenvals, min_eigenval)
        
        # Add small diagonal term for extra stability
        noise_cov = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        noise_cov += np.eye(bands) * (np.trace(noise_cov) * 1e-6)
        
        return noise_cov

    def fit(self, X, n_components):
        """
        Fit the transformer to the data.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input data matrix (n_pixels, n_bands)
        n_components : int
            Number of components to retain
        """
        self.n_components = n_components
        
        if self.method == 'pca':
            self.transformer_obj = PCA(n_components=n_components, random_state=42)
            self.transformer_obj.fit(X)
            
        elif self.method == 'mnf':
            # Custom MNF implementation using improved generalized eigenvalue decomposition
            print(f"Fitting MNF with {n_components} components...")
            
            # Center the data first
            self.mnf_mean_ = np.mean(X, axis=0)
            X_centered = X - self.mnf_mean_
            
            # Estimate signal and noise covariance matrices on centered data
            signal_cov = np.cov(X_centered.T)  # Signal covariance
            
            # Try alternative noise estimation: minimum eigenvalue method
            eigenvals_signal = linalg.eigvals(signal_cov)
            min_eigenval = np.min(eigenvals_signal)
            
            # Estimate noise level as fraction of minimum signal eigenvalue
            noise_level = max(min_eigenval * 0.01, np.trace(signal_cov) * 1e-6)
            noise_cov_simple = np.eye(signal_cov.shape[0]) * noise_level
            
            # Also get spatial difference estimate
            noise_cov_spatial = self._estimate_noise_covariance(X)
            
            # Combine both methods: use spatial estimate but regularize with simple estimate
            noise_eigenvals = linalg.eigvals(noise_cov_spatial)
            if np.min(noise_eigenvals) < noise_level:
                print(f"Using hybrid noise estimation (spatial + regularization)")
                # Regularize the spatial noise estimate
                noise_cov_spatial += noise_cov_simple * 0.1
                noise_cov = noise_cov_spatial
            else:
                print(f"Using spatial difference noise estimation")
                noise_cov = noise_cov_spatial
            
            print(f"Signal covariance condition number: {np.linalg.cond(signal_cov):.2e}")
            print(f"Noise covariance condition number: {np.linalg.cond(noise_cov):.2e}")
            
            # Use Cholesky decomposition approach for better numerical stability
            try:
                # Cholesky decomposition of noise covariance
                L = linalg.cholesky(noise_cov, lower=True)
                L_inv = linalg.solve_triangular(L, np.eye(L.shape[0]), lower=True)
                
                # Transform signal covariance: L_inv @ signal_cov @ L_inv.T
                transformed_signal_cov = L_inv @ signal_cov @ L_inv.T
                
                # Standard eigenvalue decomposition on transformed matrix
                eigenvals, eigenvecs = linalg.eigh(transformed_signal_cov)
                
                # Transform eigenvectors back: eigenvecs = L_inv.T @ eigenvecs
                eigenvecs = L_inv.T @ eigenvecs
                
            except linalg.LinAlgError:
                # Fallback to regularized generalized eigenvalue problem
                print("Warning: Cholesky decomposition failed, using regularized approach")
                
                # Add regularization to both matrices
                signal_cov += np.eye(signal_cov.shape[0]) * (np.trace(signal_cov) * 1e-6)
                noise_cov += np.eye(noise_cov.shape[0]) * (np.trace(noise_cov) * 1e-6)
                
                # Solve generalized eigenvalue problem
                eigenvals, eigenvecs = linalg.eigh(signal_cov, noise_cov)
            
            # Sort by eigenvalues in descending order (highest SNR first)
            sort_idx = np.argsort(eigenvals)[::-1]
            eigenvals = eigenvals[sort_idx]
            eigenvecs = eigenvecs[:, sort_idx]
            
            # Normalize eigenvectors
            for i in range(eigenvecs.shape[1]):
                eigenvecs[:, i] = eigenvecs[:, i] / np.linalg.norm(eigenvecs[:, i])
            
            # Store transformation components
            self.mnf_components_ = eigenvecs[:, :n_components].copy()
            self.eigenvalues_ = eigenvals[:n_components].copy()
            
            print(f"Top 5 eigenvalues (SNR): {self.eigenvalues_[:5]}")
            print(f"Eigenvalue range: [{eigenvals.min():.6f}, {eigenvals.max():.6f}]")
        
        self.fitted = True

    def transform(self, X):
        """
        Transform the data using the fitted transformer.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input data matrix (n_pixels, n_bands)
            
        Returns:
        --------
        X_transformed : numpy.ndarray
            Transformed data matrix (n_pixels, n_components)
        """
        if not self.fitted:
            raise RuntimeError("Transformer must be fitted before transforming data")
            
        if self.method == 'pca':
            return self.transformer_obj.transform(X)
            
        elif self.method == 'mnf':
            # Apply MNF transformation using fitted components
            # Center the data (subtract training mean)
            X_centered = X - self.mnf_mean_
            
            # Apply transformation: X_transformed = X_centered @ components
            X_transformed = X_centered @ self.mnf_components_
            
            return X_transformed
    
    def fit_transform(self, X, n_components):
        """
        Fit the transformer and transform the data in one step.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input data matrix (n_pixels, n_bands)
        n_components : int
            Number of components to retain
            
        Returns:
        --------
        X_transformed : numpy.ndarray
            Transformed data matrix (n_pixels, n_components)
        """
        self.fit(X, n_components)
        return self.transform(X)
    
    def X_to_img(self, X_transformed):
        """
        Convert transformed 2D matrix back to image format.
        
        Parameters:
        -----------
        X_transformed : numpy.ndarray
            Transformed data matrix (n_pixels, n_components)
            
        Returns:
        --------
        img : numpy.ndarray
            3D image array (height, width, n_components)
        """
        if self.original_shape is None:
            raise RuntimeError("Must call HSI_to_X first to store original shape")
        return X_transformed.T.reshape(self.n_components, self.original_shape[1], self.original_shape[2]).T
    
    def transformed_img_to_rgb(self, transformed_img):
        """
        Convert first 3 components of transformed image to RGB.
        
        Parameters:
        -----------
        transformed_img : numpy.ndarray
            Transformed image array (height, width, n_components)
            
        Returns:
        --------
        rgb_img : numpy.ndarray
            RGB image array (height, width, 3)
        """
        if transformed_img.shape[2] < 3:
            raise ValueError("Need at least 3 components for RGB conversion")
        return np.dstack((transformed_img[:,:,0], transformed_img[:,:,1], transformed_img[:,:,2]))
    
    def HSI_to_transformed_img(self, HSI_image, n_components=10, transform_only=True):
        """
        Convert hyperspectral image to transformed image in one step.
        
        Parameters:
        -----------
        HSI_image : HS_image or numpy.ndarray
            Input hyperspectral image
        n_components : int, optional (default=10)
            Number of components to retain
        transform_only : bool, optional (default=True)
            If True, use existing fitted transformer. If False, fit new transformer.
            
        Returns:
        --------
        transformed_img : numpy.ndarray
            Transformed image array (height, width, n_components)
        """
        if transform_only and not self.fitted:
            raise RuntimeError("No fitted transformer available. Set transform_only=False to fit new transformer.")
            
        X = self.HSI_to_X(HSI_image)
        
        if transform_only:
            X_transformed = self.transform(X)
        else:
            X_transformed = self.fit_transform(X, n_components)
            
        return self.X_to_img(X_transformed)
    
    def get_explained_variance_ratio(self):
        """
        Get explained variance ratio for the fitted transformer.
        
        Returns:
        --------
        explained_variance_ratio : numpy.ndarray or None
            Explained variance ratio for each component
        """
        if not self.fitted:
            raise RuntimeError("Transformer must be fitted first")
            
        if self.method == 'pca':
            return self.transformer_obj.explained_variance_ratio_
        elif self.method == 'mnf':
            # For MNF, return normalized signal-to-noise ratios
            if self.eigenvalues_ is not None and len(self.eigenvalues_) > 0:
                total_snr = np.sum(self.eigenvalues_)
                return self.eigenvalues_ / total_snr if total_snr > 0 else self.eigenvalues_
        return None
    
    def get_method_info(self):
        """
        Get information about the current transformation method.
        
        Returns:
        --------
        info : dict
            Dictionary containing method information
        """
        info = {
            'method': self.method,
            'fitted': self.fitted,
            'n_components': self.n_components
        }
        
        if self.fitted:
            if self.method == 'pca':
                info['explained_variance_ratio'] = self.get_explained_variance_ratio()
                info['total_explained_variance'] = np.sum(self.get_explained_variance_ratio())
            elif self.method == 'mnf':
                info['signal_to_noise_ratios'] = self.eigenvalues_
                info['mean_snr'] = np.mean(self.eigenvalues_) if self.eigenvalues_ is not None else None
        
        return info
    
    def validate_components(self, X_transformed, n_check=5):
        """
        Validate transformed components to identify potential noise issues.
        
        Parameters:
        -----------
        X_transformed : numpy.ndarray
            Transformed data matrix (n_pixels, n_components)
        n_check : int, optional (default=5)
            Number of components to validate
            
        Returns:
        --------
        validation_info : dict
            Dictionary with component validation information
        """
        if self.original_shape is None:
            raise RuntimeError("Original shape information needed for validation")
        
        height, width = self.original_shape[1], self.original_shape[2]
        n_check = min(n_check, X_transformed.shape[1])
        
        validation_info = {
            'component_quality': [],
            'recommendations': []
        }
        
        for i in range(n_check):
            component = X_transformed[:, i]
            comp_spatial = component.reshape(height, width)
            
            # Calculate quality metrics
            # 1. Spatial correlation (good components should have some spatial structure)
            if height > 1 and width > 1:
                corr_h = np.corrcoef(comp_spatial[:-1, :].flatten(), comp_spatial[1:, :].flatten())[0, 1]
                corr_v = np.corrcoef(comp_spatial[:, :-1].flatten(), comp_spatial[:, 1:].flatten())[0, 1]
                avg_spatial_corr = (corr_h + corr_v) / 2
            else:
                avg_spatial_corr = 0
            
            # 2. High frequency content (noisy components have high frequency content)
            if height > 2 and width > 2:
                grad_x = np.diff(comp_spatial, axis=1)
                grad_y = np.diff(comp_spatial, axis=0)
                gradient_magnitude = np.sqrt(grad_x[:-1, :]**2 + grad_y[:, :-1]**2)
                high_freq_ratio = np.mean(gradient_magnitude) / np.std(component)
            else:
                high_freq_ratio = 0
            
            # 3. Dynamic range
            dynamic_range = (np.max(component) - np.min(component)) / np.std(component)
            
            # Component quality assessment
            quality_score = 0
            quality_flags = []
            
            # Check eigenvalue/SNR
            snr = self.eigenvalues_[i] if hasattr(self, 'eigenvalues_') else 1.0
            if snr > 2.0:
                quality_score += 2
                quality_flags.append("Good SNR")
            elif snr > 1.0:
                quality_score += 1
                quality_flags.append("Moderate SNR")
            else:
                quality_flags.append("Low SNR - likely noise")
            
            # Check spatial correlation
            if avg_spatial_corr > 0.1:
                quality_score += 2
                quality_flags.append("Good spatial structure")
            elif avg_spatial_corr > 0.05:
                quality_score += 1
                quality_flags.append("Moderate spatial structure")
            else:
                quality_flags.append("No spatial structure - likely noise")
            
            # Check high frequency content
            if high_freq_ratio < 0.3:
                quality_score += 1
                quality_flags.append("Low noise content")
            elif high_freq_ratio > 0.7:
                quality_flags.append("High noise content")
            
            # Overall assessment
            if quality_score >= 4:
                overall_quality = "GOOD"
            elif quality_score >= 2:
                overall_quality = "MODERATE"
            else:
                overall_quality = "POOR (likely noise)"
            
            component_info = {
                'component': i + 1,
                'snr': float(snr),
                'spatial_correlation': float(avg_spatial_corr),
                'high_freq_ratio': float(high_freq_ratio),
                'dynamic_range': float(dynamic_range),
                'quality_score': quality_score,
                'overall_quality': overall_quality,
                'flags': quality_flags
            }
            
            validation_info['component_quality'].append(component_info)
        
        # Generate recommendations
        poor_components = [info for info in validation_info['component_quality'] 
                          if info['overall_quality'] == "POOR (likely noise)"]
        
        if poor_components:
            validation_info['recommendations'].append(
                f"Components {[c['component'] for c in poor_components]} appear to be noise-dominated. "
                "Consider using fewer components or checking noise estimation."
            )
        
        if len(poor_components) > 1 and poor_components[0]['component'] <= 3:
            validation_info['recommendations'].append(
                "Early components are noisy - this suggests issues with noise covariance estimation. "
                "The MNF algorithm may not be separating signal from noise effectively."
            )
        
        return validation_info
    


# Backward compatibility alias
HS_PCA_transformer = transformer



    
    