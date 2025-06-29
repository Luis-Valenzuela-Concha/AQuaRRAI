import os
from casatools import image
from casatasks import casalog, tclean, exportfits
import matplotlib.pyplot as plt
import shutil
from astropy.io import fits

tclean_temp = 'temp_im'

def plot_image(image_path: str, zoom: int = 1, cmap: str = 'inferno', title = ''):
    image_name = image_path.split('/')[-1]

    plot_image_path = os.path.join(image_path, f'{image_name}.image')
    plot_residual_path = os.path.join(image_path, f'{image_name}.residual')
    plot_model_path = os.path.join(image_path, f'{image_name}.model')

    # Load image
    image_path_abs = os.path.abspath(plot_image_path)
    ia = image()
    ia.open(image_path_abs)
    image_data = ia.getchunk()
    ia.close()

    # Load residual
    residual_path_abs = os.path.abspath(plot_residual_path)
    ia = image()
    ia.open(residual_path_abs)
    residual_data = ia.getchunk()
    ia.close()

    model_path_abs = os.path.abspath(plot_model_path)
    ia = image()
    ia.open(model_path_abs)
    model_data = ia.getchunk()
    ia.close()

    fig, axes = plt.subplots(1, 3, figsize=(8, 4))

    # Plot image
    if image_data.ndim == 2:
        axes[0].imshow(image_data, origin='lower', cmap=cmap)
    else:
        axes[0].imshow(image_data[:, :, 0, 0], origin='lower', cmap=cmap)
    axes[0].set_title(f'{image_name}.image' if not title else f'{title} (image)')
    axes[0].axis('off')

    # Plot model
    if model_data.ndim == 2:
        axes[1].imshow(model_data, origin='lower', cmap=cmap)
    else:
        axes[1].imshow(model_data[:, :, 0, 0], origin='lower', cmap=cmap)
    axes[1].set_title(f'{image_name}.model' if not title else f'{title} (model)')
    axes[1].axis('off')

    # Plot residual
    if residual_data.ndim == 2:
        axes[2].imshow(residual_data, origin='lower', cmap=cmap)
    else:
        axes[2].imshow(residual_data[:, :, 0, 0], origin='lower', cmap=cmap)
    axes[2].set_title(f'{image_name}.residual' if not title else f'{title} (residual)')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

def ms_to_image(ms_path: str):
    path = os.path.abspath(ms_path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Measurement set not found: {path}")
    tclean(
        vis=path,
        imagename=f'{tclean_temp}/{tclean_temp}',
        cell='0.02arcsec',
        imsize=256,
        niter=0,
        interactive=False
    )
    temp_image_path = f'{tclean_temp}'
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
    plt.colorbar(label='Intensity')
    plt.axis('off') # quitar
    plt.show()
