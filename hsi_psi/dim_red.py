from sklearn.decomposition import PCA
from .core import HS_image
import numpy as np
import copy

try:
    from pysptools.noise import MNF
    PYSPTOOLS_AVAILABLE = True
except ImportError:
    PYSPTOOLS_AVAILABLE = False
    from scipy import linalg

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
        
        # Check for MNF availability
        if self.method == 'mnf' and not PYSPTOOLS_AVAILABLE:
            raise ImportError(
                "MNF requires pysptools library. Install with: pip install pysptools\n"
                "Alternatively, use method='pca' for PCA-based dimensionality reduction."
            )
        
        self.fitted = False
        self.n_components = None
        self.original_shape = None
        
        # Method-specific attributes
        if self.method == 'pca':
            self.transformer_obj = None
        elif self.method == 'mnf':
            self.mnf_obj = None
            self.eigenvalues_ = None
    

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
            # PySptools MNF expects data in shape (height, width, bands)
            # Convert from (n_pixels, n_bands) to (height, width, bands)
            if self.original_shape is None:
                raise RuntimeError("Must call HSI_to_X first to store original shape")
            
            height, width, bands = self.original_shape[1], self.original_shape[2], self.original_shape[0]
            mnf_input = X.T.reshape(height, width, bands)
            
            # Initialize and fit MNF using pysptools
            self.mnf_obj = MNF()
            
            # Fit the MNF transformer
            # PySptools MNF.apply() both fits and transforms the data
            mnf_result = self.mnf_obj.apply(mnf_input)
            
            # Store eigenvalues (signal-to-noise ratios) if available
            if hasattr(self.mnf_obj, 'eigenvalues'):
                self.eigenvalues_ = self.mnf_obj.eigenvalues[:n_components]
            else:
                # Fallback: compute from transformed data variance
                mnf_flat = mnf_result.reshape(-1, mnf_result.shape[2])
                self.eigenvalues_ = np.var(mnf_flat, axis=0)[:n_components]
        
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
            # PySptools MNF expects data in shape (height, width, bands)
            if self.original_shape is None:
                raise RuntimeError("Must call HSI_to_X first to store original shape")
            
            height, width, bands = self.original_shape[1], self.original_shape[2], self.original_shape[0]
            mnf_input = X.T.reshape(height, width, bands)
            
            # Apply MNF transformation
            mnf_result = self.mnf_obj.apply(mnf_input)
            
            # Keep only the requested number of components and reshape back to 2D
            mnf_result_truncated = mnf_result[:, :, :self.n_components]
            return mnf_result_truncated.reshape(-1, self.n_components)
    
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
    
    @staticmethod
    def check_dependencies():
        """
        Check availability of optional dependencies.
        
        Returns:
        --------
        info : dict
            Dictionary with dependency availability information
        """
        deps = {
            'pysptools': {
                'available': PYSPTOOLS_AVAILABLE,
                'required_for': ['MNF'],
                'install_command': 'pip install pysptools'
            }
        }
        
        if not PYSPTOOLS_AVAILABLE:
            print("Warning: pysptools not available. MNF functionality will be limited.")
            print("Install with: pip install pysptools")
        
        return deps

# Backward compatibility alias
HS_PCA_transformer = transformer



    
    