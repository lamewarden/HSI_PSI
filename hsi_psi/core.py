
import numpy as np
import bisect
import spectral as sp
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import copy
import warnings
import json  # Add this missing import

# Ignore RuntimeWarning
warnings.filterwarnings("ignore", category=RuntimeWarning)


# some primitive class to read data
class HS_image:
    epsilon = 1e-7
    def __init__(self, data_path):
        self.data_path = data_path
        self.read_hdr(data_path)


    def read_hdr(self, data_path):
        try:
            hdr = sp.open_image(data_path)
        except:
            convert_header_to_envi(data_path)
            hdr = sp.envi.open(data_path)
        self.hdr = hdr.bands.centers
        self.rows, self.cols, self.bands = hdr.nrows, hdr.ncols, hdr.nbands
        self.meta = hdr.metadata
        self.img = hdr.load()
        self.name = os.path.basename(data_path)
        self.ind = [int(float(x)) for x in self.meta['wavelength'][:-1]]
        self.bits = int(self.meta['data type'])
        self.normalized = False
        self.calibrated = False
        # self.bits = hdr.bits
        # self.line = self.name.split('-')[2]

    def __str__(self):
        return str(self.name)

    def calibrate(self, dc=False, clip_to=3):
        white_matrix, dark_matrix = self.upload_calibration(dc)
        # limiting the height of the reflectance
        self.img = np.clip((self.img - dark_matrix) / (white_matrix - dark_matrix), 0, clip_to)
        # filling nans with 0
        self.img[np.isnan(self.img)] = 0
        self.img[np.isinf(self.img)] = clip_to
        self.rgb_sample = get_rgb_sample(self, normalize=True)


    def upload_calibration(self, dc):
        # upload wc
        white_calibration = HS_image(os.path.join(self.data_path[:-8] + "WhiteCalibration.hdr"))
        white_matrix = np.mean(white_calibration.img, axis=0)
        if dc:
            dark_calibration = HS_image(os.path.join(self.data_path[:-8] + "DarkCalibration.hdr"))
            dark_matrix = np.mean(dark_calibration.img, axis=0)
        else:
            dark_matrix = np.zeros_like(white_matrix)
        return white_matrix, dark_matrix

   

    def get_closest_wavelength(self, wl):
        pos = bisect.bisect_left(self.ind, wl)
        if pos == len(self.ind):
            return self.ind[-1]
        if pos == 0:
            return self.ind[0]
        if self.ind[pos] == wl:
            return self.ind[pos]
        return self.ind[pos]


    def __getitem__(self, wl):
        if isinstance(wl, slice):
            # Handle slice of wavelengths
            start, stop, step = wl.start, wl.stop, wl.step
            start = int(float(start)) if start is not None else self.ind[0]
            stop = int(float(stop)) if stop is not None else self.ind[-1]
            step = int(float(step)) if step is not None else 1
            
            start_index = self.ind.index(start)
            stop_index = self.ind.index(stop)
            step = step if step > 0 else 1

            return self.img[:, :, start_index:stop_index:step]
        else:
            # Handle single wavelength 
            wl = int(float(wl))
            closest_wl = self.get_closest_wavelength(wl)
            wl_index = self.ind.index(closest_wl)
            return self.img[:, :, wl_index]
    
        
    def __setitem__(self, wl, value):
        wl = int(float(wl))
        try:
            wl_index = self.ind.index(wl)
            self.img[:,:,wl_index] = value
        except:
            raise IndexError("Entered wavelength is not in spectrum")
        
    @staticmethod
    def divide_arrays(array_3d, array_other, remove_outliers=False, sigma_threshold=2):
        """
        Divide a 3D array by another array (3D or 2D), handling divisions by zero and removing outliers.

        Parameters:
        - array_3d: np.ndarray
            The 3D array to be divided.
        - array_other: np.ndarray
            The array to divide by, can be 3D or 2D.
        - remove_outliers: bool
            Whether to remove outliers based on sigma threshold.
        - sigma_threshold: float
            The number of standard deviations to use as the threshold for outlier removal.

        Returns:
        - np.ndarray
            The resulting 3D array after division and optional outlier removal.
        """
        # Ensure input is a numpy array
        array_3d = np.asarray(array_3d)
        array_other = np.asarray(array_other)
        
        # Check the shape compatibility for broadcasting
        if array_other.ndim == 2:
            if array_3d.shape[:2] != array_other.shape:
                raise ValueError("The 2D array must have the same shape as the first two dimensions of the 3D array.")
            # Expand dimensions to make broadcasting work
            array_other = array_other[:, :, np.newaxis]
        elif array_other.ndim == 3 and array_other.shape[-1] == 1:
            if array_3d.shape[:2] != array_other.shape[:2]:
                raise ValueError("The 3D array with shape (X, Y, 1) must match the first two dimensions of the 3D array.")
        
        # Perform division with handling for divide by zero
        result = np.divide(array_3d, array_other, where=array_other!=0, out=np.zeros_like(array_3d))
        
        # Replace Inf values with 0 (since np.divide can create Inf)
        result[np.isinf(result)] = 0

        if remove_outliers:
            # Remove outliers along the third axis for each slice [:,:,i]
            for i in range(result.shape[2]):
                slice_ = result[:, :, i]
                mean = np.mean(slice_)
                std = np.std(slice_)
                # Identify outliers
                outliers = np.abs(slice_ - mean) > sigma_threshold * std
                # Set outliers to zero
                slice_[outliers] = 0
                # Assign back the slice with removed outliers
                result[:, :, i] = slice_
        
        return result
        
    @staticmethod
    def img_align(img, template, inplace = False):

        sift = cv2.SIFT_create()
        template_bw = cv2.normalize(template, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        img_bw = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        kp_temp, des_temp = sift.detectAndCompute(template_bw, None)
        kp_ch, des_ch = sift.detectAndCompute(img_bw, None)
        # Match descriptors using FLANN (or you can use BFMatcher for brute force matching)
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(des_ch, des_temp, k=2)
        # Store the good matches using Lowe's ratio test
        good_matches = []
        for m,n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        height, width, *_ = template.shape

        # Ensure we have enough matches to compute the homography
        if len(good_matches) > 10:
            src_pts = np.float32([kp_ch[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
            dst_pts = np.float32([kp_temp[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

            # Compute the homography
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            # Warp the image
            if inplace is True:
                img = cv2.warpPerspective(img, M, (width, height))

            return M, width, height
        # if not enough matches found - return M as an identity matrix
        return np.eye(3), width, height
    


    def normalize(self, to_wl = 1000, clip_to = 3):
        if self.normalized == True:
            print("HS image is already normalized. No new transformation will be performed")
        else:
            try:
                self.img = self.img / self[to_wl][:,:,np.newaxis]
            except ValueError:
                self.img = self.img / self[to_wl]
            self.img[np.isnan(self.img)] = 0
            self.img[np.isinf(self.img)] = 0
            self.img = np.clip(self.img, 0, clip_to)
            self.normalized=True
        
    def apply_snv(self):
        """
        Applies SNV transformation to the entire hyperspectral image in a vectorized manner.
        """
        # Reshape the image to (num_pixels, num_bands)
        flat_img = self.img.reshape(-1, self.img.shape[-1])
        
        # Calculate mean and std along the spectral axis
        mean_spectrum = np.mean(flat_img, axis=1, keepdims=True)
        std_spectrum = np.std(flat_img, axis=1, keepdims=True)
        
        # Apply SNV and handle zero std deviation
        snv_image = (flat_img - mean_spectrum) / (std_spectrum + self.epsilon)  # Add epsilon to avoid division by zero
        
        # Reshape back to the original image dimensions
        self.img = snv_image.reshape(self.img.shape)



    def flatten_to_df(self):
        """
        Flattenting whole HS image into 2D DF with separate pixels as rows and WL as columns.
        
        """               
        return pd.DataFrame(self.img[self.img.mean(axis=2) != 0], columns=self.ind)


class MS_image(HS_image):
    def __init__(self, data_path=None):
        super().__init__(data_path)
        self.devignet_counter = 0


    def devignet(self, ref_HS, sigma=10, deblack=False, black_noise=0.0586):
    # Extracting de-vignetting matrix from ref images:
    # Gaussian blur application
        if self.devignet_counter == 1:
            return
        
        for channel in self.ind:

            # Gaussian blur
            ref_frame = cv2.GaussianBlur(ref_HS[channel], (3, 3), sigmaX=sigma)
            # Inverting
            ref_frame = np.round(1 / ref_frame, 9)
            # Dividing by mean
            ref_frame = (ref_frame / np.mean(ref_frame))[:,:,np.newaxis]

            # Applying calculated de-vignetting mask for the every image in the original TS
            if deblack:
                self[channel] = np.clip((self[channel] * ref_frame - black_noise * 4095), 0, 4095).astype(np.uint16)
            else:
                self[channel] = np.clip((self[channel] * ref_frame), 0, 4095).astype(np.uint16)
        
        self.devignet_counter = 1
        return 
    

    
    def devignet_old_school(self, ref_HS, sigma=10, deblack = False, black_noise = 0.0586):
        # extracting de-vignetting matrix from ref images:
        # gaussian blur application
        if self.devignet_counter == 1:
            return ref_HS
        
        ref_HS_copy = copy.deepcopy(ref_HS)
        # ref_list = []
        for channel in range(0,6):
            # gaussian blur
            ref_HS_copy.img[:,:,channel] = cv2.GaussianBlur(ref_HS_copy.img[:,:,channel] , (3, 3), sigmaX=sigma)
            # inverting 
            ref_HS_copy.img[:,:,channel]  = 1/ref_HS_copy.img[:,:,channel] 
            # dividing by mean
            ref_HS_copy.img[:,:,channel]  = ref_HS_copy.img[:,:,channel] /np.mean(ref_HS_copy.img[:,:,channel] )
            # ref_list.append(ref_HS_copy.img[channel])
        # Plying calculated de-vignetting mask for the every image in the original TS:
            if deblack is True:
                self.img[:,:,channel]  = np.clip((self.img[:,:,channel]  * ref_HS_copy.img[:,:,channel][:,:,np.newaxis] - 4095*black_noise), 0, 4095).astype(np.uint16) 
            else:
                self.img[:,:,channel] = np.clip((self.img[:,:,channel] * ref_HS_copy.img[:,:,channel][:,:,np.newaxis]), 0, 4095).astype(np.uint16)
        self.devignet_counter = 1


        return ref_HS_copy



# read data
def get_hdr_images(folder, min_rows = 1, format='hdr'):
    all_images = {}
    for root, dirs, files in os.walk(folder):
        for file in files:
            filepath = os.path.join(root, file)
            # select only files with .hdr extension and with more than 15 rows
            if filepath.endswith('.hdr') and int(HS_image(filepath).rows) > min_rows:
                if format == 'hdr':
                    img = HS_image(filepath)
                else:
                    img = MS_image(filepath)
                all_images[img.name] = img
    return all_images



def get_polygon_masks_from_json(json_file_path):
    """
    Converts polygon annotations from a Labelme JSON file into binary masks.
    
    Args:
        json_file_path (str): Path to the Labelme annotation JSON file.
        
    Returns:
        dict: A dictionary where keys are labels (e.g., "rotten") and values are binary masks (numpy arrays).
    """
    # Load the annotation JSON file
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Get the image dimensions from the JSON data
    image_width = data.get('imageWidth')
    image_height = data.get('imageHeight')
    
    # Initialize a dictionary to store the binary masks for each label
    masks = {}
    
    # Iterate over each shape in the JSON file
    for shape in data['shapes']:
        if shape['shape_type'] == 'polygon':
            label = shape['label']  # Get the label (e.g., "rotten", "fresh")
            
            # Initialize a mask for the label if not already created
            if label not in masks:
                masks[label] = np.zeros((image_height, image_width), dtype=np.uint8)
            
            # Get the polygon points
            polygon = np.array(shape['points'], dtype=np.int32)
            
            # Draw the polygon on the corresponding label's mask
            cv2.fillPoly(masks[label], [polygon], color=1)
    
    return masks


def standardize_image(image: np.ndarray) -> np.ndarray:
    """
    Standardizes a 3D image array (Height x Width x Channels).

    Parameters:
    image (np.ndarray): The input image in the form of a 3D array.
    
    Returns:
    np.ndarray: The standardized image.
    """
    # Ensure the image is in float format
    image = image.astype(np.float32)

    # Calculate mean and standard deviation for each channel
    means = np.mean(image, axis=(0, 1), keepdims=True)
    stds = np.std(image, axis=(0, 1), keepdims=True)

    # Avoid division by zero by setting stds to 1 where std is zero
    stds[stds == 0] = 1

    # Standardize each channel: (value - mean) / std
    standardized_image = (image - means) / stds

    return standardized_image

def convert_header_to_envi(bil_header):
    # Open the BIL header file and read its lines
    with open(bil_header, 'r') as f:
        lines = f.readlines()
        # Save the original BIL header file with a new name
        original_header_path = f'{bil_header}_original'
        with open(original_header_path, 'w') as f:
            f.writelines(lines)

    # Initialize a dictionary to hold the header information
    header_info = {}
    wavelengths = []

    # Parse the BIL header file
    in_wavelengths = False
    for line in lines:
        if line.strip():
            if 'WAVELENGTHS' in line:
                in_wavelengths = True
            elif 'WAVELENGTHS_END' in line:
                in_wavelengths = False
            elif in_wavelengths:
                wavelengths.append(line.strip())
            else:
                key, value = line.split()
                header_info[key] = value

    # Write the ENVI header file
    with open(bil_header, 'w') as f:
        f.write('ENVI\n')
        f.write('file type = ENVI\n')
        f.write(f'interleave = {header_info["LAYOUT"]}\n')
        f.write(f'samples = {header_info["NCOLS"]}\n')
        f.write(f'lines = {header_info["NROWS"]}\n')
        f.write(f'bands = {header_info["NBANDS"]}\n')
        # selecting the data type
        if header_info["NBITS"] == '16':
            f.write(f'data type = 2\n')
        elif header_info["NBITS"] == '32':
            f.write(f'data type = 3\n')
        elif header_info["NBITS"] == '64':
            f.write(f'data type = 14\n')
        elif header_info["NBITS"] == '12':
            f.write(f'data type = 12\n')
            # 64-bit unsigned int
        if header_info["BYTEORDER"] == 'I':
            f.write(f'byte order = 0\n')
        else:
            f.write(f'byte order = 1\n')
            f.write(f'byte order = {header_info["BYTEORDER"]}\n')
            
        f.write('wavelength units = nm\n')
        f.write(f';bit depth = {header_info["NBITS"]}\n')
        f.write(f';chromatic correction = {header_info["CHROMATICCORRECTION"]}\n')
        f.write(f';integration time = {header_info["INTEGRATIONTIME"]}\n')
        try:
            f.write(f';gain = {header_info["GAIN"]}\n')
        except:
            f.write(f';gain = 0\n')
        f.write('wavelength = {\n')
        for wavelength in wavelengths:
            f.write(f'{wavelength},\n')
        f.write('}\n')
