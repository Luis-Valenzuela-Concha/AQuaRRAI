from dataclasses import dataclass
from casatools import image
import os
from astropy.io import fits
import numpy as np
from skimage.metrics import structural_similarity
from math import log10
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.measure import shannon_entropy
import cv2

def normalize_image(data):
    """
    Normalize the image data to the range [0, 255].
    """
    data = (data - data.min()) / (data.max() - data.min()) * 255
    return data

@dataclass
class ImageQualityMetrics:
    image_path: str

    def __init__(self, image_path: str, residual_path: str = None, reference_path: str = None):
        self.image_path = image_path
        self.reference_path = reference_path
        self.residual_path = residual_path

        self.data_image = self.__get_data_image()
        self.data_reference = self.__get_data_reference()
        self.data_residual = self.__get_data_residual()

    def __get_data_image(self):
        if not self.image_path.endswith('.image') and not self.image_path.endswith('.model'):
            raise ValueError("Image must be a CASA image file (.image, .model).")
        
        path = os.path.abspath(self.image_path)
        ia = image()
        ia.open(path)
        data = ia.getchunk()
        ia.close()
        data_image = data[:, :, 0, 0] if data.ndim > 2 else data
        data_image = normalize_image(data_image)
        data_image = data_image.T[::-1, :][::-1, :]
        data_image = data_image.astype(np.float32)
        return data_image
        
    def __get_data_residual(self):
        if self.residual_path is None:
            return None
        
        if not self.residual_path.endswith('.residual'):
            raise ValueError("Residual image must be a CASA image file (.residual).")

        path = os.path.abspath(self.residual_path)
        ia = image()
        ia.open(path)
        data = ia.getchunk()
        ia.close()
        data_residual = data[:, :, 0, 0] if data.ndim > 2 else data
        data_residual = normalize_image(data_residual)
        data_residual = data_residual.T[::-1, :][::-1, :]
        return data_residual


    def __get_data_reference(self):
        if not self.reference_path:
            return None
        
        if not self.reference_path.endswith('.fits'):
            raise ValueError("Reference image must be a FITS file.")

        path = os.path.abspath(self.reference_path)
        with fits.open(path) as hdul:
            data_reference = hdul[0].data
        data_reference = data_reference[:, :, 0, 0] if data_reference.ndim > 2 else data_reference
        data_reference = normalize_image(data_reference)
        return data_reference

    def snr(self, eps=1e-10):
        if self.data_reference is None:
            raise ValueError("Reference image is required to calculate SNR.")

        signal = np.linalg.norm(self.data_image.flatten())
        noise = np.linalg.norm(self.data_image.flatten() - self.data_reference.flatten()) + eps

        snr = 20 * np.log10(signal / noise)
        return snr
    
    def ssim(self) -> float:
        if self.data_reference is None:
            raise ValueError("Reference image is required to calculate SSIM.")
        
        ssim = structural_similarity(self.data_image, self.data_reference, data_range=255)
        return ssim

    def psnr_reference(self, eps = 1e-10) -> float:
        if self.data_reference is None:
            raise ValueError("Reference image is required to calculate PSNR.")
        
        mse = np.mean((self.data_reference - self.data_image) ** 2) + eps
        max_pixel = np.max(self.data_reference)  # Max pixel of reference image

        psnr = 10 * log10((max_pixel ** 2) / mse)
        return psnr
    
    def rms(self) -> float:
        if self.data_residual is None:
            raise ValueError("Residual image is required to calculate RMS.")
        
        rms = np.sqrt(np.mean(self.data_residual ** 2))
        return rms
    
    def peak(self) -> float:
        if self.data_image is None:
            raise ValueError("Image data is required to calculate peak.")
        
        peak_value = np.max(self.data_image)
        return peak_value

    def psnr_no_reference(self, eps = 1e-10) -> float:
        if self.data_image is None:
            raise ValueError("Image data is required to calculate PSNR without reference.")
        if self.data_residual is None:
            raise ValueError("Residual data is required to calculate PSNR without reference.")
        
        peak = self.peak()  # Peak value of the reconstructed image
        rms = self.rms() + eps  # RMS of the residual
        psnr = peak/rms
        return psnr
    
    def sharpness(self) -> float:
        if self.data_image is None:
            raise ValueError("Image data is required to calculate sharpness.")
        
        gradient_x = cv2.Sobel(self.data_image, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(self.data_image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        sharpness_value = np.mean(gradient_magnitude)
        return sharpness_value
    
    def median(self) -> float:
        if self.data_image is None:
            raise ValueError("Image data is required to calculate median.")
        
        median = cv2.medianBlur(self.data_image, ksize=3)
        diff = np.abs(self.data_image.astype(np.float32) - median.astype(np.float32))
        median_value = np.mean(diff)
        return median_value
    
    def entropy(self) -> float:
        if self.data_image is None:
            raise ValueError("Image data is required to calculate entropy.")
        
        entropy_value = shannon_entropy(self.data_image)
        return entropy_value
    
    def noise(self) -> float:
        if self.data_image is None:
            raise ValueError("Image data is required to calculate noise.")
        
        blur = cv2.GaussianBlur(self.data_image, (5, 5), 0)
        noise_value = self.data_image.astype(np.float32) - blur.astype(np.float32)
        noise_value = np.std(noise_value)
        return noise_value


    def plot_images(self):
        titles = ['Reference', 'Reconstructed', 'Residual']
        images = [self.data_reference, self.data_image, self.data_residual]
        count = sum(1 for img in images if img is not None)

        _, axes = plt.subplots(1, count, figsize=(5 * count, 5))

        if count == 1:
            axes = [axes]

        idx = 0
        for i, data in enumerate(images):
            if data is not None:
                ax = axes[idx]
                im = ax.imshow(data, origin='lower', cmap='inferno')
                ax.set_title(titles[i])
                ax.set_xlabel('Pixel X')
                ax.set_ylabel('Pixel Y')

                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.2)
                plt.colorbar(im, cax=cax, label='Intensity')

                idx += 1

        plt.tight_layout()
        plt.show()