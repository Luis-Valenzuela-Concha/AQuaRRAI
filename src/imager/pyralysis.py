import os
from .base import BaseImager
import shutil

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from imager.helpers.pyralysis.setup_client import setup_optimal_client
from imager.helpers.pyralysis.mask_modified import MaskModified

from pyralysis.reconstruction import Image
from pyralysis.convolution import Bicubic, PSWF1, Spline, Pillbox, Gaussian, GaussianSinc, Sinc
from pyralysis.estimators import Degridding, BilinearInterpolation, NearestNeighbor
from pyralysis.optimization.fi import Chi2, L1Norm, Entropy, Tikhonov, TotalFlux, TSV
from pyralysis.optimization import ObjectiveFunction
from pyralysis.optimization.linesearch import Fixed, BacktrackingArmijo, Brent
from pyralysis.io.zarr import ZarrArray
from pyralysis.optimization.projection import LessThan
from pyralysis.optimization.optimizer import PolakRibiere, HagerZhang, LBFGS, FletcherReeves, HestenesStiefel, DaiYuan, SDMM
from pyralysis.io import DaskMS
from pyralysis.transformers import HermitianSymmetry
import dask.array as da
import xarray as xr
import numpy as np
import astropy.units as un
import matplotlib.pyplot as plt

class PyralysisImager(BaseImager):
    def __init__(self, vis, output_folder):
        super().__init__(vis, output_folder)
        self.client = None
        self.n_workers = 0
        self.imsize = 256
        self.image = None
        self.cellsize = None
        self.dataset = None
        self.mask = None
        self.h_symmetry = None
        
    def start_client(self, config):
        backend = config.get("backend", "numba_cpu")
        memory_fraction = config.get("memory_fraction", 0.8)
        cpu_fraction = config.get("cpu_fraction", 0.9)
        verbose = config.get("verbose", False)

        self.client = setup_optimal_client(
            backend=backend,
            memory_fraction=memory_fraction,
            cpu_fraction=cpu_fraction,
            verbose=verbose
        )
        self.n_workers = len(self.client.scheduler_info()['workers'])
        self.client.restart()


    def stop_client(self):
        if self.client is not None:
            self.client.shutdown()
            self.client = None
            self.n_workers = 0

    def setup_dataset(self, config):
        self.imsize = config.get("imsize", 256)
        filter_flag_column = config.get("filter_flag_column", False)
        calculate_psf = config.get("calculate_psf", False)
        cs = config.get("cellsize", None)
        hermitian_symmetry = config.get("hermitian_symmetry", True)
        verbose = config.get("verbose", False)

        
        ds = DaskMS(input_name=self.vis)
        self.dataset = ds.read(filter_flag_column=filter_flag_column, calculate_psf=calculate_psf)  #
        self.dataset.calculate_theoretical_noise(per_field=True, per_spw=True)

        self.cellsize = (
            cs * un.arcsec
            if cs is not None
            else self.dataset.theo_resolution / 7
        )

        if hermitian_symmetry:
            self.h_symmetry = HermitianSymmetry(input_data=self.dataset)
            self.h_symmetry.apply()

        X, Y = np.mgrid[0:self.imsize, 0:self.imsize]
        chunks = {"y": self.imsize // self.n_workers, "x": self.imsize // self.n_workers}
        image_start = da.zeros((self.imsize, self.imsize), chunks=chunks, dtype=np.float32)
        self.image = Image(
            data=xr.DataArray(
                data=image_start,
                dims=["x", "y"],
                coords=dict(X=(["x", "y"], Y), Y=(["x", "y"], X)),
            ),
            cellsize=self.cellsize
        )
        self.image.data = self.image.data.chunk(chunks)

    def set_mask(self, config):
        threshold = config.get("threshold", 0.80)
        plot_mask = config.get("plot_mask", False)

        self.mask = MaskModified(
            dataset=self.dataset,
            imsize=self.image.data.shape,
            threshold=threshold,
            cellsize=self.cellsize
        )
        if plot_mask:
            plt.figure(figsize=(4, 4))
            plt.imshow(self.mask.data.values, origin="lower", cmap="gray")
            plt.title("Generated Mask")
            plt.show()

    def reconstruct(self, config):
            oversampling_factor = config["estimator"].get("kernel", {}).get("oversampling_factor", 1)
            size = config["estimator"].get("kernel", {}).get("size", 7)
            k = config["estimator"].get("kernel", {}).get("algorithm", None)

            kernel_classes = {
                "spline": Spline,
                "bicubic": Bicubic,
                "pswf1": PSWF1,
                "gaussian": Gaussian,
                "gaussian_sinc": GaussianSinc,
                "sinc": Sinc,
                "pillbox": Pillbox
            }

            if k is not None:
                k = k.lower()
                if k not in kernel_classes:
                    print(f"Kernel algorithm '{k}' not recognized. Using default 'pswf1'.")
                    k = "pswf1"

                kernel = kernel_classes[k](
                    size=size,
                    cellsize=self.cellsize,
                    oversampling_factor=oversampling_factor
                )
            else:
                kernel = None

            estimator_name = config["estimator"].get("algorithm", "bilinear_interpolation").lower()
            padding_factor = config["estimator"].get("padding_factor", 1.0)

            estimator_classes = {
                "nearest_neighbour": NearestNeighbor,
                "bilinear_interpolation": BilinearInterpolation,
                "degridding": Degridding
            }

            if estimator_name not in estimator_classes:
                print(f"Estimator '{estimator_name}' not recognized. Using default 'bilinear_interpolation'.")
                estimator_name = "bilinear_interpolation"

            estimator = estimator_classes[estimator_name](
                input_data=self.dataset,
                image=self.image,
                hermitian_symmetry=self.h_symmetry,
                padding_factor=padding_factor,
                ckernel_object=kernel
            )

            chi_squared = Chi2(model_visibility=estimator, normalize=True)
            l1_norm = L1Norm(penalization_factor=1e-3)
            tsv = TSV(penalization_factor=1e-20)
            fi_list = [chi_squared, l1_norm, tsv]

            of = ObjectiveFunction(fi_list=fi_list, image=self.image, persist_gradient=True)

            ls_name = config["linear_search"].get("algorithm", "backtracking_armijo").lower()
            ls_step = config["linear_search"].get("step", 10.0)

            line_search_classes = {
                "fixed": Fixed,
                "backtracking_armijo": BacktrackingArmijo,
                "brent": Brent
            }

            if ls_name not in line_search_classes:
                print(f"Line search method '{ls_name}' not recognized. Using default 'backtracking_armijo'.")
                ls_name = "backtracking_armijo"

            ls = line_search_classes[ls_name](
                objective_function=of,
                step=ls_step
            )
            ioh = ZarrArray()

            proj_name = config.get("projection","less_than").lower()

            projection_classes = {
                "less_than": LessThan,
                # Add other projection classes here if needed
            }
            if proj_name is not None:
                if proj_name not in projection_classes:
                    print(f"Projection algorithm '{proj_name}' not recognized. Using default 'less_than'.")
                    proj_name = "less_than"

                proj = projection_classes[proj_name](
                    compared_value=0.0,
                    replacement_value=0.0
                ) 
            else:
                proj = None

            niter_differentiable = config["differentiable"].get("niter", 1)
            differentiable_algorithm = config["differentiable"].get("algorithm", "lbfgs").lower()

            differentiable_classes = {
                "lbfgs": LBFGS,
                "hager_zhang": HagerZhang
            }

            if differentiable_algorithm not in differentiable_classes:
                print(f"Differentiable algorithm '{differentiable_algorithm}' not recognized. Using default 'lbfgs'.")
                differentiable_algorithm = "lbfgs"

            differentiable = differentiable_classes[differentiable_algorithm](
                image=self.image,
                objective_function=of,
                linesearch=ls,
                mask=self.mask,
                max_iter=niter_differentiable,
                io_handler=ioh,
                projection=proj,
                max_corrections=1
            )

            niter_optimizer = config["optimizer"].get("niter", 3)
            optimizer_algorithm = config["optimizer"].get("algorithm", "sdmm").lower()

            optimizer_classes = {
                "sdmm": SDMM
            }

            if optimizer_algorithm not in optimizer_classes:
                print(f"Optimizer algorithm '{optimizer_algorithm}' not recognized. Using default 'sdmm'.")
                optimizer_algorithm = "sdmm"

            optimizer = optimizer_classes[optimizer_algorithm](
                image=self.image,
                optimizer=differentiable,
                max_iter=niter_optimizer,
                objective_function=of,
                io_handler=ioh,
                persist_z_u=True,
                rho=0.001
            )

            verbose = config.get("verbose", False)
            res = optimizer.optimize(verbose=verbose, partial_image=False)
            res_mem = res.data.compute()

            output_file = f"{self.output_folder}/{optimizer_algorithm}_{niter_optimizer}.jpg"
            plt.imsave(output_file, res_mem, cmap="inferno", origin="lower")

            if verbose:
                print("Optimization completed.")
                print(f"Final image shape: {res_mem.shape}")
                print(f"Reconstructed image saved to {output_file}")