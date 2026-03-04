
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
import xml.etree.ElementTree as ET

# Ignore RuntimeWarning
warnings.filterwarnings("ignore", category=RuntimeWarning)


# some primitive class to read data
class HS_image:
    """Base class for hyperspectral image data.
    
    Encapsulates a hyperspectral image with all associated metadata, providing
    core functionality for loading ENVI format images and accessing spectral data.
    
    Attributes:
        img (np.ndarray): 3D array containing hyperspectral data [rows, columns, bands]
        meta (dict): Metadata from ENVI header file (dimensions, data type, etc.)
        ind (list): Wavelength values (nm) corresponding to each spectral band
        name (str): Original filename of the hyperspectral image
        rows (int): Number of rows in the image
        cols (int): Number of columns in the image
        bands (int): Number of spectral bands
        bits (int): Data type bit depth
        normalized (bool): Whether the image has been normalized
        calibrated (bool): Whether sensor calibration has been applied
    """
    epsilon = 1e-7
    def __init__(self, data_path):
        """Initialize HS_image by loading hyperspectral data from file.
        
        Args:
            data_path (str): Path to the hyperspectral image header file (.hdr)
        """
        self.data_path = data_path
        self.read_hdr(data_path)


    def read_hdr(self, data_path):
        try:
            hdr = sp.open_image(data_path)
        except:
            convert_header_to_envi(data_path)
            hdr = sp.envi.open(data_path)
        
        # Store wavelength information - self.ind is the primary source
        self.rows, self.cols, self.bands = hdr.nrows, hdr.ncols, hdr.nbands
        self.meta = hdr.metadata
        self.img = hdr.load()
        self.name = os.path.basename(data_path)
        self.ind = [int(float(x)) for x in self.meta['wavelength'] if x.strip() != '']
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


    def upload_calibration(self, dc):
        """Upload white and dark calibration matrices, mapped to current image wavelengths."""
        from scipy.interpolate import interp1d
        
        # Get current image wavelengths
        current_wavelengths = np.array(self.ind)
        
        # Upload white calibration
        white_calibration = HS_image(os.path.join(self.data_path[:-8] + "WhiteCalibration.hdr"))
        white_cal_wavelengths = np.array(white_calibration.ind)
        
        # Check if wavelength mapping is needed for white calibration
        # if not np.array_equal(white_cal_wavelengths, current_wavelengths):
        #     print(f"Mapping white calibration: {white_cal_wavelengths[0]}-{white_cal_wavelengths[-1]} nm → {current_wavelengths[0]}-{current_wavelengths[-1]} nm")
            
        #     # Take spatial mean first, then interpolate
        #     white_spectrum = np.mean(white_calibration.img, axis=(0, 1))
            
        #     # Create interpolation function
        #     interp_func = interp1d(
        #         white_cal_wavelengths, 
        #         white_spectrum, 
        #         kind='linear', 
        #         bounds_error=False, 
        #         fill_value=0  # Use 0 for extrapolated values
        #     )
            
        #     # Interpolate to current wavelengths
        #     white_matrix = interp_func(current_wavelengths)
            
        #     # Ensure we have valid data
        #     if np.all(white_matrix == 0):
        #         raise ValueError(f"No overlap between white calibration ({white_cal_wavelengths[0]}-{white_cal_wavelengths[-1]} nm) "
        #                        f"and current image ({current_wavelengths[0]}-{current_wavelengths[-1]} nm)")
        # else:
            # Direct spatial mean if wavelengths match
        white_matrix = np.mean(white_calibration.img, axis=0)
        
        # Handle dark calibration
        if dc:
            dark_calibration = HS_image(os.path.join(self.data_path[:-8] + "DarkCalibration.hdr"))
            dark_cal_wavelengths = np.array(dark_calibration.ind)
            
            # Check if wavelength mapping is needed for dark calibration
            if not np.array_equal(dark_cal_wavelengths, current_wavelengths):
                print(f"Mapping dark calibration: {dark_cal_wavelengths[0]}-{dark_cal_wavelengths[-1]} nm → {current_wavelengths[0]}-{current_wavelengths[-1]} nm")
                
                # Take spatial mean first, then interpolate
                dark_spectrum = np.mean(dark_calibration.img, axis=(0, 1))
                
                # Create interpolation function
                interp_func = interp1d(
                    dark_cal_wavelengths, 
                    dark_spectrum, 
                    kind='linear', 
                    bounds_error=False, 
                    fill_value=0
                )
                
                # Interpolate to current wavelengths
                dark_matrix = interp_func(current_wavelengths)
            else:
                # Direct spatial mean if wavelengths match
                dark_matrix = np.mean(dark_calibration.img, axis=0)
        else:
            dark_matrix = np.zeros_like(white_matrix)
            
        return white_matrix, dark_matrix

   

    def get_closest_wavelength(self, wl):
        idx = int(np.argmin(np.abs(np.array(self.ind) - wl)))
        return self.ind[idx]


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

    def apply_rnv(self, eps=1e-8, nan_safe=True):
        # Expect self.img shape (H, W, B)
        img = self.img.astype(np.float32, copy=False)
        flat = img.reshape(-1, img.shape[-1])  # (N_pixels, B)

        med = (np.nanmedian if nan_safe else np.median)(flat, axis=1, keepdims=True)
        mad = (np.nanmedian if nan_safe else np.median)(np.abs(flat - med), axis=1, keepdims=True)

        # Scale MAD to be comparable to std under normality
        mad_scaled = 1.4826 * mad
        mad_safe = np.where(mad_scaled < eps, 1.0, mad_scaled)  # avoid divide-by-zero only where needed

        rnv = (flat - med) / mad_safe
        self.img = rnv.reshape(img.shape)

    def apply_l2(self):
        """
        Applies L2 normalization to the entire hyperspectral image.
        Each spectrum is scaled to have unit L2 norm (Euclidean norm).
        """
        # Reshape the image to (num_pixels, num_bands)
        flat_img = self.img.reshape(-1, self.img.shape[-1])
        
        # Calculate L2 norm along the spectral axis
        l2_norm = np.linalg.norm(flat_img, axis=1, keepdims=True)
        
        # Apply L2 normalization and handle zero norm
        l2_image = flat_img / (l2_norm + self.epsilon)
        
        # Reshape back to the original image dimensions
        self.img = l2_image.reshape(self.img.shape)

    def crop_spectral_range(self, wl_start=None, wl_end=None, band_start=None, band_end=None):
        """
        Crop the hyperspectral image to a specified spectral range.
        This should be the first step in any processing pipeline to ensure consistent spectral coverage.
        
        Parameters:
        -----------
        wl_start : int, optional
            Starting wavelength in nm. If None, uses the first available wavelength.
        wl_end : int, optional
            Ending wavelength in nm. If None, uses the last available wavelength.
        band_start : int, optional
            Starting band index (alternative to wl_start). Takes precedence over wl_start.
        band_end : int, optional
            Ending band index (alternative to wl_end). Takes precedence over wl_end.
            
        Returns:
        --------
        self : HS_image
            Returns self for method chaining.
            
        Notes:
        ------
        - Updates self.img, self.ind, and self.bands to match the cropped range
        - All wavelengths outside the specified range are removed
        - Band indices are adjusted to maintain consistency
        
        Example:
        --------
        >>> hs_img = HS_image("data.hdr")
        >>> hs_img.crop_spectral_range(wl_start=450, wl_end=900)  # Keep only 450-900 nm
        >>> hs_img.crop_spectral_range(band_start=10, band_end=50)  # Keep only bands 10-50
        """
        
        # Determine the start and end indices
        if band_start is not None:
            start_idx = max(0, band_start)
        elif wl_start is not None:
            # Find closest wavelength index
            start_idx = 0
            for i, wl in enumerate(self.ind):
                if wl >= wl_start:
                    start_idx = i
                    break
        else:
            start_idx = 0
            
        if band_end is not None:
            end_idx = min(len(self.ind), band_end + 1)  # +1 for inclusive end
        elif wl_end is not None:
            # Find closest wavelength index
            end_idx = len(self.ind)
            for i, wl in enumerate(self.ind):
                if wl > wl_end:
                    end_idx = i
                    break
        else:
            end_idx = len(self.ind)
            
        # Validate indices
        if start_idx >= end_idx:
            raise ValueError(f"Invalid spectral range: start_idx ({start_idx}) >= end_idx ({end_idx})")
        if start_idx >= len(self.ind) or end_idx > len(self.ind):
            raise ValueError(f"Indices out of range: available bands 0-{len(self.ind)-1}, requested {start_idx}-{end_idx-1}")
            
        # Store original range for logging
        original_bands = len(self.ind)
        original_wl_range = f"{self.ind[0]}-{self.ind[-1]} nm"
        
        # Crop the image data
        self.img = self.img[:, :, start_idx:end_idx]
        
        # Update wavelength indices
        self.ind = self.ind[start_idx:end_idx]
        
        # Update band count
        self.bands = len(self.ind)
        
        # Update metadata if it exists
        if hasattr(self, 'meta') and self.meta is not None and 'wavelength' in self.meta:
            try:
                # Update wavelength metadata
                cropped_wavelengths = [str(wl) for wl in self.ind]
                cropped_wavelengths.append('')  # ENVI format often has trailing empty string
                self.meta['wavelength'] = cropped_wavelengths
                self.meta['bands'] = str(self.bands)
            except Exception:
                pass  # If metadata update fails, continue without it
        
        # Log the cropping operation
        new_wl_range = f"{self.ind[0]}-{self.ind[-1]} nm"

    
        
        return self



    def visualize_mask(self, rgb_array=None, repeat=1, title='Segmentation Result'):
        """
        Visualize the mask overlay on RGB image.
        
        Args:
            rgb_array (np.ndarray, optional): RGB image to use as background. If None, will be generated.
            repeat (int): Vertical repetition factor for display
            title (str): Figure title
            
        Returns:
            None: Displays the visualization
            
        Raises:
            ValueError: If mask is not set or empty
        """
        if not hasattr(self, 'mask') or self.mask is None:
            raise ValueError("No mask available. Run extract_masks() first.")
        
        if self.mask.size == 0:
            raise ValueError("Mask is empty.")
        
        # Import here to avoid circular imports
        from hsi_psi.preprocessing import HS_preprocessor
        
        # Generate RGB if not provided
        if rgb_array is None:
            try:
                rgb_array = HS_preprocessor.get_rgb_sample_from_image(
                    self, normalize=True, correct=False, show=False, repeat=1
                )
            except Exception as e:
                print(f"Failed to generate RGB: {e}")
                return
        
        # Validate RGB
        is_masked_array = np.ma.is_masked(rgb_array)
        if is_masked_array:
            all_masked = np.ma.getmask(rgb_array).all() if hasattr(np.ma.getmask(rgb_array), 'all') else False
            if all_masked:
                print("Warning: RGB image is fully masked, cannot visualize")
                return
            rgb_array = np.ma.filled(rgb_array, fill_value=0)
        
        if rgb_array.size == 0 or rgb_array.shape[0] == 0 or rgb_array.shape[1] == 0:
            print("Warning: RGB image is empty, cannot visualize")
            return
        
        try:
            # Create figure
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            fig.suptitle(title, fontsize=16, fontweight='bold')
            
            # Original RGB
            rgb_repeated = np.repeat(rgb_array, repeat, axis=0)
            axes[0].imshow(rgb_repeated)
            axes[0].set_title("Original RGB")
            axes[0].axis('off')
            
            # Mask overlay
            axes[1].imshow(rgb_repeated)
            
            # Create turquoise overlay for mask
            mask_2d = self.mask[:, :, 0] if self.mask.ndim == 3 else self.mask
            mask_repeated = np.repeat(mask_2d, repeat, axis=0)
            
            turquoise_overlay = np.full((mask_repeated.shape[0], mask_repeated.shape[1], 3), np.nan)
            turquoise_overlay[mask_repeated == 1, 0] = 64/255
            turquoise_overlay[mask_repeated == 1, 1] = 224/255
            turquoise_overlay[mask_repeated == 1, 2] = 208/255
            
            axes[1].imshow(turquoise_overlay, alpha=0.6)
            
            # Calculate mask statistics
            mask_pixels = np.sum(mask_2d)
            total_pixels = mask_2d.shape[0] * mask_2d.shape[1]
            percentage = 100 * mask_pixels / total_pixels if total_pixels > 0 else 0
            
            axes[1].set_title(f"Mask Overlay\n({mask_pixels}/{total_pixels} pixels = {percentage:.1f}%)")
            axes[1].axis('off')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Visualization failed: {e}")
            import traceback
            traceback.print_exc()
    
    def flatten_to_df(self):
        """
        Flattenting whole HS image into 2D DF with separate pixels as rows and WL as columns.
        
        """               
        return pd.DataFrame(self.img[self.img.mean(axis=2) != 0], columns=self.ind)

    def _get_mask_2d(self):
        """Convert image mask to a 2D boolean mask with shape (H, W)."""
        if not hasattr(self, 'mask') or self.mask is None:
            raise ValueError("No mask found on image. Set image.mask first.")

        if not hasattr(self, 'img') or self.img is None:
            raise ValueError("No image data found.")

        if self.img.ndim != 3:
            raise ValueError(f"Image must be 3D (H, W, B), got shape {self.img.shape}")

        _, _, bands = self.img.shape
        mask = self.mask

        if mask.ndim == 3:
            if mask.shape[2] == 1:
                mask_2d = mask[:, :, 0].astype(bool)
            elif mask.shape[2] == bands:
                mask_2d = np.any(mask, axis=2)
            else:
                mask_2d = np.any(mask != 0, axis=2)
        elif mask.ndim == 2:
            mask_2d = mask.astype(bool)
        else:
            raise ValueError(f"Mask must be 2D or 3D, got shape {mask.shape}")

        if mask_2d.shape != self.img.shape[:2]:
            raise ValueError(
                f"Mask shape {mask_2d.shape} does not match image spatial shape {self.img.shape[:2]}"
            )

        return mask_2d

    def _load_tray_masks_from_xsel(self, tray_mask_path):
        """Load tray sub-area masks from an XSEL file as dict[name -> 2D bool mask]."""
        if tray_mask_path is None:
            return None

        root = ET.parse(tray_mask_path).getroot()

        try:
            height = int(root.attrib["height"])
            width = int(root.attrib["width"])
        except KeyError as exc:
            raise ValueError(f"Missing required XSEL attribute: {exc}") from exc

        if (height, width) != self.img.shape[:2]:
            raise ValueError(
                f"Tray mask shape ({height}, {width}) does not match image shape {self.img.shape[:2]}"
            )

        subarea_masks = {}
        for index, rect in enumerate(root.findall(".//TRectangleShape")):
            name = rect.attrib.get("name", f"tray_{index}")
            left = int(rect.attrib["left"])
            top = int(rect.attrib["top"])
            right = int(rect.attrib["right"])
            bottom = int(rect.attrib["bottom"])

            left = max(0, min(left, width))
            right = max(0, min(right, width))
            top = max(0, min(top, height))
            bottom = max(0, min(bottom, height))

            mask = np.zeros((height, width), dtype=bool)
            if bottom > top and right > left:
                mask[top:bottom, left:right] = True
            subarea_masks[name] = mask

        if not subarea_masks:
            raise ValueError(f"No tray sub-areas found in XSEL file: {tray_mask_path}")

        return subarea_masks

    def extract_masked_spectra(self, include_background=False, as_dataframe=False,
                               foreground_label="FG", background_label="BG", tray_mask=None):
        """
        Extract masked pixel spectra from this image.

        Args:
            include_background (bool): If True, also extract non-masked pixels.
            as_dataframe (bool): If True, returns a DataFrame with wavelength columns and 'label'.
            foreground_label (str): Label for masked pixels when as_dataframe=True.
            background_label (str): Label for background pixels when as_dataframe=True.
            tray_mask (str, optional): Path to XSEL tray mask file. If provided, spectra are
                extracted per tray sub-area and DataFrame includes 'tray_position'.

        Returns:
            If as_dataframe=False:
                - include_background=False: np.ndarray of foreground spectra (N_fg, B)
                - include_background=True: tuple(fg_spectra, bg_spectra)
            If as_dataframe=True:
                pd.DataFrame with wavelength columns, 'label', and 'tray_position'
        """
        if not hasattr(self, 'img') or self.img is None:
            raise ValueError("No image data found.")

        if self.img.ndim != 3:
            raise ValueError(f"Image must be 3D (H, W, B), got shape {self.img.shape}")

        if not hasattr(self, 'ind'):
            raise ValueError("Image wavelengths (ind) are missing.")

        _, _, bands = self.img.shape
        if len(self.ind) != bands:
            raise ValueError(f"Wavelength count ({len(self.ind)}) does not match bands ({bands}).")

        mask_2d = self._get_mask_2d()
        tray_masks = self._load_tray_masks_from_xsel(tray_mask) if tray_mask is not None else None

        if tray_masks is None:
            fg_spectra = self.img[mask_2d]
        else:
            tray_seg_masks = {}
            tray_segmented_hypercubes = {}
            fg_chunks = []

            for tray_name, tray_mask_2d in tray_masks.items():
                tray_seg_mask = mask_2d & tray_mask_2d
                tray_seg_masks[tray_name] = tray_seg_mask
                tray_segmented_hypercubes[tray_name] = np.where(tray_seg_mask[:, :, np.newaxis], self.img, 0)

                tray_fg = self.img[tray_seg_mask]
                if tray_fg.size > 0:
                    fg_chunks.append(tray_fg)

            self.tray_masks = tray_masks
            self.tray_seg_masks = tray_seg_masks
            self.tray_segmented_hypercubes = tray_segmented_hypercubes

            if fg_chunks:
                fg_spectra = np.vstack(fg_chunks)
            else:
                fg_spectra = np.empty((0, bands), dtype=self.img.dtype)

        bg_spectra = self.img[~mask_2d] if include_background else None

        if not as_dataframe:
            if include_background:
                return fg_spectra, bg_spectra
            return fg_spectra

        frames = []

        if tray_masks is None:
            if fg_spectra.size > 0:
                fg_df = pd.DataFrame(fg_spectra, columns=self.ind)
                fg_df['label'] = foreground_label
                fg_df['tray_position'] = ""
                frames.append(fg_df)
        else:
            for tray_name, tray_mask_2d in tray_masks.items():
                tray_seg_mask = mask_2d & tray_mask_2d
                tray_fg = self.img[tray_seg_mask]
                if tray_fg.size == 0:
                    continue

                tray_df = pd.DataFrame(tray_fg, columns=self.ind)
                tray_df['label'] = self.name
                tray_df['tray_position'] = tray_name
                frames.append(tray_df)

        if include_background and bg_spectra is not None and bg_spectra.size > 0:
            bg_df = pd.DataFrame(bg_spectra, columns=self.ind)
            bg_df['label'] = background_label
            bg_df['tray_position'] = ""
            frames.append(bg_df)

        if not frames:
            return pd.DataFrame(columns=list(self.ind) + ['label', 'tray_position'])

        return pd.concat(frames, axis=0, ignore_index=True)


class MS_image(HS_image):
    """Multispectral image class extending HS_image with channel mapping capabilities.
    
    This class is designed for multispectral cameras with fixed wavelength channels.
    It allows custom wavelength mapping and includes vignetting correction functionality.
    
    Attributes:
        Inherits all attributes from HS_image, plus:
        devignet_counter (int): Counter for de-vignetting operations
    """
    def __init__(self, data_path=None, map_channels={0:755, 1:850, 2:420, 3:495, 4:670, 5:595}):
        """Initialize MS_image with custom channel-to-wavelength mapping.
        
        Args:
            data_path (str, optional): Path to the multispectral image header file (.hdr).
                If None, creates an empty MS_image object. Default: None.
            map_channels (dict|list|array, optional): Channel to wavelength mapping.
                - dict: Maps band indices to wavelengths {0: 755, 1: 850, ...}
                - list/array: Wavelengths in channel order [755, 850, 420, ...]
                Default: {0:755, 1:850, 2:420, 3:495, 4:670, 5:595}
        
        Example:
            >>> img = hs.MS_image('data.hdr', map_channels={0:755, 1:850})
            >>> img = hs.MS_image('data.hdr', map_channels=[755, 850, 420])
        """
        super().__init__(data_path)
        self.devignet_counter = 0
        # If the header already contains real wavelengths (all >= 200 nm),
        # trust them and skip the hardcoded channel mapping.
        # Placeholder indices (e.g. 1,2,3,4,5,6) are all < 200 nm, so we
        # apply the default physical mapping only in that case.
        hdr_wls = getattr(self, 'ind', [])
        if hdr_wls and all(w < 200 for w in hdr_wls):
            self.map_channels(map_channels)


    def map_channels(self, target_wavelengths):
            """
            Set channel wavelength mapping.
    
            Accepts:
             - list/tuple/np.ndarray of wavelengths -> preserves order
             - dict mapping band_index -> wavelength -> entries sorted by band_index
             - dict mapping any keys -> wavelength -> preserves insertion order (fallback)
    
            After mapping updates:
             - self.ind (list of int wavelengths)
             - self.bands
             - meta['wavelength'] if present (ENVI style, trailing empty string kept)
            """
            # If dict, try to interpret keys as band indices and sort by key
            if isinstance(target_wavelengths, dict):
                try:
                    # try numeric keys -> sort by index
                    items = sorted(target_wavelengths.items(), key=lambda kv: int(kv[0]))
                    wl_list = [float(v) for _, v in items]
                except Exception:
                    # fallback: keep dict value order (Python 3.7+ preserves insertion order)
                    wl_list = [float(v) for v in target_wavelengths.values()]
            elif hasattr(target_wavelengths, '__iter__'):
                wl_list = [float(x) for x in target_wavelengths]
            else:
                raise TypeError("target_wavelengths must be list/array or dict")
    
            # normalize to ints if your code expects ints (keep floats if needed)
            self.ind = [int(round(w)) for w in wl_list]
    
            # update dependent attributes
            self.bands = len(self.ind)
            if hasattr(self, 'meta') and isinstance(self.meta, dict):
                try:
                    self.meta['wavelength'] = [str(w) for w in self.ind] + ['']
                    self.meta['bands'] = str(self.bands)
                except Exception:
                    pass
            return self


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
