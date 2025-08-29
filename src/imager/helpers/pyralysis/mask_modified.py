import warnings
from dataclasses import InitVar, dataclass, field
from typing import List, Tuple, Union

import astropy.units as un
import dask.array as da
import numpy as np
import xarray as xr

from pyralysis.base.dataset import Dataset
from pyralysis.reconstruction.parameter import Parameter
from pyralysis.utils.decorators import temp_irreg_local_save
from pyralysis.reconstruction.mask import Mask

@dataclass(init=True, repr=True)
class MaskModified(Mask):
    inverse_beam: Union[None, da.Array] = field(init=False, default=None)

    def _default_mask(
        self,
        dataset: Dataset,
        imsize: Union[Tuple[int, int], List[int]],
        imcenter: Union[Tuple[int, int], List[int]],
        threshold: float
    ) -> xr.DataArray:
        freq = [dataset.spws.min_nu.value]
        pb = dataset.antenna.primary_beam
        pointings = dataset.field.phase_direction_cosines[0:2].value
        cellsize = np.array([-1.0, 1.0]) * un.rad

        if self.cellsize is not None:
            cellsize = self.cellsize

        beams = da.array([
            pb.beam(
                frequency=freq,
                imsize=imsize,
                cellsize=cellsize,
                antenna=np.array([0]),
                x_0=pointings[0][i],
                y_0=pointings[1][i],
                imcenter=imcenter,
                imchunks="auto" if self.chunks is None else self.chunks
            ) for i in range(pointings.shape[-1])
        ])

        beam = da.sum(beams, axis=(0, 1, 2))
        normalized_beam = beam / da.max(beam)
        self.inverse_beam = 1 / normalized_beam

        inverse_beam_data = self.inverse_beam.compute()

        mid_x = inverse_beam_data.shape[0] // 2
        mid_y = inverse_beam_data.shape[1] // 2
        
        length = min(mid_x, mid_y)
        line_corner_to_center = da.array([inverse_beam_data[i, i] for i in range(length)])

        min_value = line_corner_to_center.compute()[0]
        max_value = line_corner_to_center.compute()[-1]

        normalized_threshold = threshold * (max_value - min_value) + min_value
        mask = xr.DataArray(self.inverse_beam < normalized_threshold, dims=["x", "y"])
        return mask