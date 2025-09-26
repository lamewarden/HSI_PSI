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
        Estimate noise covariance matrix using spatial differences method.
        
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
        
        # Calculate spatial differences to estimate noise
        # Use differences in both horizontal and vertical directions
        diff_data = []
        
        if height > 1:
            # Vertical differences
            diff_v = np.diff(spatial_data, axis=0)  # (height-1, width, bands)
            diff_data.append(diff_v.reshape(-1, bands))
        
        if width > 1:
            # Horizontal differences  
            diff_h = np.diff(spatial_data, axis=1)  # (height, width-1, bands)
            diff_data.append(diff_h.reshape(-1, bands))
        
        if not diff_data:
            # Fallback for very small images
            return np.eye(bands) * np.var(X, axis=0).mean() * 0.1
        
        # Combine all differences
        all_diffs = np.vstack(diff_data)
        
        # Estimate noise covariance from differences
        # Divide by 2 because differences amplify noise by sqrt(2)
        noise_cov = np.cov(all_diffs.T) / 2.0
        
        # Regularize to ensure positive definiteness
        eigenvals, eigenvecs = linalg.eigh(noise_cov)
        eigenvals = np.maximum(eigenvals, 1e-10 * eigenvals.max())
        noise_cov = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        
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
            # Custom MNF implementation using generalized eigenvalue decomposition
            print(f"Fitting MNF with {n_components} components...")
            
            # Estimate signal and noise covariance matrices
            signal_cov = np.cov(X.T)  # Signal covariance
            noise_cov = self._estimate_noise_covariance(X)  # Noise covariance
            
            # Solve generalized eigenvalue problem: signal_cov * v = lambda * noise_cov * v
            eigenvals, eigenvecs = linalg.eigh(signal_cov, noise_cov)
            
            # Sort by eigenvalues in descending order (highest SNR first)
            sort_idx = np.argsort(eigenvals)[::-1]
            eigenvals = eigenvals[sort_idx]
            eigenvecs = eigenvecs[:, sort_idx]
            
            # Store transformation components
            self.mnf_components_ = eigenvecs[:, :n_components].copy()
            self.eigenvalues_ = eigenvals[:n_components].copy()
            
            # Store data statistics for inverse transform
            self.mnf_mean_ = np.mean(X, axis=0)
        
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
    


# Backward compatibility alias
HS_PCA_transformer = transformer



    
    