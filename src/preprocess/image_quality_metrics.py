from dataclasses import dataclass
import matplotlib.pyplot as plt
from casatools import image
import os
from astropy.io import fits
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

@dataclass
class ImageQualityMetrics:
    image_path: str


    def __init__(self, image_path: str, reference_path: str = None):
        self.image_path = image_path
        self.reference_path = reference_path

        self.data_image = self.__get_data_image()
        self.data_reference = self.__get_data_reference()

    def __get_data_image(self):
        if self.image_path.endswith('.image') or self.image_path.endswith('.model'):
            path = os.path.abspath(self.image_path)
            ia = image()
            ia.open(path)
            data = ia.getchunk()
            ia.close()
            data_image = data[:, :, 0, 0] if data.ndim > 2 else data
            data_image = (data_image - data_image.min()) / (data_image.max() - data_image.min()) * 255
            data_image = data_image.T[::-1, :][::-1, :]
            return data_image

    def __get_data_reference(self):
        if self.reference_path:
            path = os.path.abspath(self.reference_path)
            with fits.open(path) as hdul:
                data_reference = hdul[0].data
            data_reference = data_reference[:, :, 0, 0] if data_reference.ndim > 2 else data_reference

            data_reference = (data_reference - data_reference.min()) / (data_reference.max() - data_reference.min()) * 255
            return data_reference
        return None

    def calculate_snr(self) -> float:
        if self.data_reference is None:
            raise ValueError("Reference image is required to calculate SNR.")
        
        eps=1e-12
        signal = np.linalg.norm(self.data_image.flatten())
        error = np.linalg.norm(self.data_image.flatten() - self.data_reference.flatten())

        snr = 20 * np.log10(signal / (error + eps))
        return snr
    
    def calculate_psnr(self) -> float:
        if self.data_reference is None:
            raise ValueError("Reference image is required to calculate PSNR.")
        
        psnr = peak_signal_noise_ratio(self.data_image, self.data_reference, data_range=255)
        return psnr
    
    def calculate_ssim(self) -> float:
        if self.data_reference is None:
            raise ValueError("Reference image is required to calculate SSIM.")
        
        ssim = structural_similarity(self.data_image, self.data_reference, data_range=255)
        return ssim
    
    def plot_data(self):
        if self.data_reference is not None:
            plt.figure(figsize=(8, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(self.data_image, origin='lower', cmap='inferno')
            plt.colorbar(label='Intensity')
            plt.title('Image')
            plt.xlabel('Pixel X')
            plt.ylabel('Pixel Y')

            plt.subplot(1, 2, 2)
            plt.imshow(self.data_reference, origin='lower', cmap='inferno')
            plt.colorbar(label='Intensity')
            plt.title(f'Reference')
            plt.xlabel('Pixel X')
            plt.ylabel('Pixel Y')

            plt.tight_layout()
        else:
            plt.imshow(self.data_image, origin='lower', cmap='inferno')
            plt.colorbar(label='Intensity')
            plt.title('Image')
            plt.xlabel('Pixel X')
            plt.ylabel('Pixel Y')
            plt.show()
    
    def plot_images(self):
        self.plot_data()