import os
from casatasks import tclean
from .base import BaseImager
import shutil
from src.utils.casa_log_deletter import delete_casa_logs

class TCleanImager(BaseImager):
    delete_casa_logs()
    def reconstruct(self, config: dict):
        deconvolver = config.get('deconvolver', 'hogbom').lower()

        imsize = config.get('imsize', 256)
        cell = config.get('cell', '0.02arcsec')
        niter = config.get('niter', 0)
        weighting = config.get('weighting', 'natural')

        imagename = f"tclean_{deconvolver}_{niter}"
        out_dir = os.path.join(self.output_folder, imagename)
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)

        kwargs = dict(
            vis=self.vis,
            imagename=os.path.join(out_dir, imagename),
            imsize=imsize,
            cell=cell,
            niter=niter,
            weighting=weighting,
        )
        if deconvolver == "hogbom":
            kwargs["deconvolver"] = "hogbom"
        elif deconvolver == "multiscale":
            kwargs["deconvolver"] = "multiscale"
            kwargs["scales"] = config.get("scales", [0, 3, 10, 30])
        elif deconvolver == "mem":
            kwargs["deconvolver"] = "mem"
        else:
            raise ValueError(f"Unsupported deconvolver: {deconvolver}")

        tclean(**kwargs)
        print(f"Imaging completed.")
        