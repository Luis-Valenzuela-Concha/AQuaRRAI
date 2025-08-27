import os
import matplotlib.pyplot as plt
import shutil
from astropy.io import fits
from src.utils.casa_log_deletter import delete_casa_logs
import numpy as np

tclean_temp = 'temp_im'

def load_casa_image(file_path: str):
    from casatools import image
    ia = image()
    ia.open(file_path)
    data = ia.getchunk()
    ia.close()
    delete_casa_logs()
    return data


def __plot_panel(ax, data, cmap: str, title: str):
    if data.ndim == 2:
        ax.imshow(data, origin="lower", cmap=cmap)
    else:
        ax.imshow(data[:, :, 0, 0], origin="lower", cmap=cmap)
    ax.set_title(title)
    ax.axis("off")


def plot_image(image_path: str, cmap: str = "inferno", title: str = ""):
    image_name = os.path.basename(image_path)

    files = {
        "image": os.path.join(image_path, f"{image_name}.image"),
        "model": os.path.join(image_path, f"{image_name}.model"),
        "residual": os.path.join(image_path, f"{image_name}.residual"),
    }

    data = {key: load_casa_image(os.path.abspath(path)) for key, path in files.items()}

    for key in data:
        data[key] = np.rot90(data[key], k=1)
        data[key] = np.flip(data[key], axis=0)


    fig, axes = plt.subplots(1, 3, figsize=(9, 4))
    for ax, (key, arr) in zip(axes, data.items()):
        panel_title = f"{image_name}.{key}" if not title else f"{title} ({key})"
        __plot_panel(ax, arr, cmap, panel_title)

    plt.tight_layout()
    plt.show()

def ms_to_image(ms_path: str):
    from casatasks import tclean 

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
    delete_casa_logs()
    return temp_image_path

def delete_temp_image():
    if os.path.exists(tclean_temp):
        shutil.rmtree(tclean_temp)

def plot_ms(ms_path: str, cmap: str = 'inferno', title: str = ''):
    temp_image_path = ms_to_image(ms_path)
    plot_image(temp_image_path, cmap=cmap)
    delete_temp_image()

def plot_fits(fits_path: str, zoom: int = 1, cmap: str = 'inferno'):
    path = os.path.abspath(fits_path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"FITS file not found: {path}")

    with fits.open(path) as hdul:
        data = hdul[0].data

    if data.ndim == 2:
        plt.imshow(data, origin='lower', cmap=cmap)
        middle_x = data.shape[0] // 2
        middle_y = data.shape[1] // 2
    else:
        plt.imshow(data[:, :, 0, 0], origin='lower', cmap=cmap)
        middle_x = data.shape[3] // 2
        middle_y = data.shape[2] // 2

    plt.xlim(
        middle_x - (middle_x // zoom), 
        middle_x + (middle_x // zoom)
    )
    plt.ylim(
        middle_y - (middle_y // zoom), 
        middle_y + (middle_y // zoom)
    )
    
    plt.colorbar(label='Intensity')
    plt.axis('off')
    plt.show()
