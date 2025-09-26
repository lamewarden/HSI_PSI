from sklearn.decomposition import PCA
from .core import HS_image
import numpy as np
import copy

class HS_PCA_transformer:

    def __init__(self):
        pass
    

    def HSI_to_X(self, HSI_image, drop_nulls = False):
        # allows it to work both with HSI_images and 3d arrays
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
        self.n_components = n_components
        self.pca = PCA(n_components=n_components, random_state=42)
        self.pca.fit(X)

    def transform(self, X):
        return self.pca.transform(X)
    
    def fit_transform(self, X, n_components):
        self.fit(X, n_components)
        return self.transform(X)

    
    def X_to_img(self, X_transformed):
        return X_transformed.T.reshape(self.n_components, self.original_shape[1], self.original_shape[2]).T
    
    def pca_img_to_rgb(self, pca_img):
        return np.dstack((pca_img[:,:,0], pca_img[:,:,1], pca_img[:,:,2]))
    
    def HSI_to_pca_img(self, HSI_image, n_components=10, transform_only=True):
        if transform_only:
            return self.X_to_img(self.transform(self.HSI_to_X(HSI_image)))
        return self.X_to_img(self.fit_transform(self.HSI_to_X(HSI_image),n_components))



    
    