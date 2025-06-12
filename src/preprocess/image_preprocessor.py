from PIL import Image
from astropy.io import fits
import numpy as np
import os
import matplotlib.pyplot as plt

class ImagePreprocessor:
    def __init__(self, image_path: str, width: int = 512, height: int = 512):
        """        
        Args:
            image_path (str): Path to the image file. (JPG or PNG)
        """
        self.image_name = os.path.basename(image_path).split('/')[-1].split('.')[0]
        self.width = width
        self.height = height
        self.image = self.__resize_image(image_path)

    def __resize_image(self, image_path: str) -> Image:
        image = Image.open(image_path).convert('L')

        data = np.array(image)
        
        # Normalize the data to the [0, 255] range
        data_min = np.min(data)
        data_ptp = np.ptp(data)  # same as np.max(data) - np.min(data)
        data_normalized = (data - data_min) / (data_ptp if data_ptp > 0 else 1)
        data_scaled = (255 * data_normalized).astype(np.uint8)
        
        # Resize the image using PIL
        img_pil = Image.fromarray(data_scaled)
        image_resized = img_pil.resize((self.width, self.height), resample=Image.BICUBIC)
        data_resized = np.array(image_resized)

        return data_resized

    def convert_to_fits(self, output_folder: str):        
        # Build the FITS header
        delta = 1.0 / 3600
        header = fits.Header()
        header['BUNIT']    = 'Jy/beam'
        header['TELESCOP'] = 'Simulated Telescope'
        header['CTYPE1']   = 'RA---TAN'
        header['CTYPE2']   = 'DEC--TAN'
        header['CRVAL1']   = 0.0
        header['CRVAL2']   = 0.0
        header['CRPIX1']   = self.width / 2
        header['CRPIX2']   = self.height / 2
        header['CDELT1']   = -delta
        header['CDELT2']   = delta
        header['BMAJ']     = delta
        header['BMIN']     = delta     
        header['BPA']      = 0.0
        
        # Save the final FITS file
        if not os.path.exists(output_folder):
            os.makedirs(output_folder, exist_ok=True)
        output_file = os.path.abspath(f'{output_folder}/{self.image_name}.fits')
        hdu = fits.PrimaryHDU(self.image, header=header)
        hdu.writeto(output_file, overwrite=True)
        
        print(f"Image converted to FITS and saved as {output_file}")

    def show_image(self):
        plt.imshow(self.image, cmap='inferno')
        plt.axis('off')
        plt.show()