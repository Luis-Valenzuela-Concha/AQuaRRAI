import os
from casatools import image
from casatasks import casalog, tclean, exportfits
import matplotlib.pyplot as plt
import shutil
from astropy.io import fits

tclean_temp = 'temp_im'

def plot_image(image_path: str, zoom: int = 1, cmap: str = 'inferno', title = ''):
    path = os.path.abspath(image_path)
    ia = image()
    ia.open(path)
    data = ia.getchunk()
    ia.close()

    if data.ndim == 2:
        plt.imshow(data, origin='lower', cmap=cmap)
    else:
        plt.imshow(data[:, :, 0, 0], origin='lower', cmap=cmap)
    plt.colorbar(label='Intensity')
    
    if title:
        plt.title(title)
    else:
       plt.title(f'{path.split("/")[-1]} - {data.shape[0]}x{data.shape[1]} pixels - Zoom: {zoom}x')

    middle_x = data.shape[1] // 2
    middle_y = data.shape[0] // 2
    plt.xlim(
        middle_x - (middle_x // zoom), 
        middle_x + (middle_x // zoom)
    )
    plt.ylim(
        middle_y - (middle_y // zoom), 
        middle_y + (middle_y // zoom)
    )
    plt.xlabel('Pixel X')
    plt.ylabel('Pixel Y')
    plt.show()

def ms_to_image(ms_path: str):
    path = os.path.abspath(ms_path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Measurement set not found: {path}")
    tclean(
        vis=path,
        imagename=f'{tclean_temp}/{tclean_temp}',
        cell='0.04arcsec',
        imsize=256,
        niter=0,
        interactive=False
    )
    temp_image_path = f'{tclean_temp}/{tclean_temp}.image'
    return temp_image_path

def delete_temp_image():
    if os.path.exists(tclean_temp):
        shutil.rmtree(tclean_temp)

def plot_ms(ms_path: str, zoom: int = 1, cmap: str = 'inferno', title: str = ''):
    temp_image_path = ms_to_image(ms_path)
    plot_image(temp_image_path, zoom, cmap)
    delete_temp_image()

def plot_fits(fits_path: str, zoom: int = 1, cmap: str = 'inferno'):
    path = os.path.abspath(fits_path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"FITS file not found: {path}")

    with fits.open(path) as hdul:
        data = hdul[0].data

    if data.ndim == 2:
        plt.imshow(data, origin='lower', cmap=cmap)
        #plt.title(f'{path.split("/")[-1]} - {data.shape[0]}x{data.shape[1]} pixels - Zoom: {zoom}x')
        middle_x = data.shape[0] // 2
        middle_y = data.shape[1] // 2
    else:
        plt.imshow(data[:, :, 0, 0], origin='lower', cmap=cmap)
        #plt.title(f'{path.split("/")[-1]} - {data.shape[2]}x{data.shape[3]} pixels - Zoom: {zoom}x')
        middle_x = data.shape[3] // 2
        middle_y = data.shape[2] // 2

    #plt.colorbar(label='Intensity')
    plt.xlim(
        middle_x - (middle_x // zoom), 
        middle_x + (middle_x // zoom)
    )
    plt.ylim(
        middle_y - (middle_y // zoom), 
        middle_y + (middle_y // zoom)
    )
    # plt.xlabel('Pixel X')
    # plt.ylabel('Pixel Y')
    # no axis
    plt.axis('off') # quitar
    plt.show()

def get_ms_stats(ms_path: str):
    path = os.path.abspath(ms_path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Measurement set not found: {path}")
    
    temp_image_path = ms_to_image(ms_path)
    
    ia = image()
    ia.open(temp_image_path)
    data = ia.getchunk()
    ia.close()

    delete_temp_image()

    pixel_data = data[:, :, 0, 0]

    # normalize pixel data
    pixel_data = pixel_data[pixel_data != 0]  # Remove zero values to avoid division by zero
    if pixel_data.size == 0:
        raise ValueError("No non-zero pixel data found in the image.")
    
    pixel_data = (pixel_data - pixel_data.min()) / (pixel_data.max() - pixel_data.min())

    return {
        'mean': pixel_data.mean(),
        'std': pixel_data.std()
    }