from dataclasses import dataclass
import os

@dataclass
class Imager:
    vis: str
    output_path: str

    def __init__(self, vis: str, output_path: str):
        """
        Initialize the Imager with visibility data and output path.
        
        Args:
            vis (str): Path to the visibility file (Measurement Set).
            output_path (str): Path where the reconstructed images will be saved.
        """

        if not vis.endswith('.ms'):
            raise ValueError("Visibility data must be a Measurement Set (.ms) file.")
        
        if not os.path.exists(vis):
            raise FileNotFoundError(f"Visibility file not found: {vis}")

        self.vis = os.path.abspath(vis)
        self.output_path = os.path.abspath(output_path)

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path, exist_ok=True)

    def tclean_hogbom(self, config: dict):
        from casatasks import tclean

        imsize = config.get('imsize', 256)
        cell = config.get('cell', '0.02arcsec')
        niter = config.get('niter', 0)
        weighting = config.get('weighting', 'natural')

        if os.path.exists(f'{self.output_path}/tclean_hogbom_{niter}'):
            os.system(f'rm -rf {self.output_path}/tclean_hogbom_{niter}')

        niter = config.get('niter', 0)
        tclean(
            vis = self.vis,
            imsize = imsize,
            cell = cell,
            niter = niter,
            weighting = weighting,
            imagename= f'{self.output_path}/tclean_hogbom_{niter}/tclean_hogbom_{niter}',
        )
    
    def tclean_multiscale(self, config: dict):
        from casatasks import tclean

        imsize = config.get('imsize', 256)
        cell = config.get('cell', '0.02arcsec')
        niter = config.get('niter', 0)
        weighting = config.get('weighting', 'natural')
        scales = config.get('scales', [0,3,10,30])

        if os.path.exists(f'{self.output_path}/tclean_multiscale_{niter}'):
            os.system(f'rm -rf {self.output_path}/tclean_multiscale_{niter}')

        tclean(
            vis = self.vis,
            imsize = imsize,
            cell = cell,
            niter = niter,
            weighting = weighting,
            deconvolver = 'multiscale',
            scales = scales,
            imagename= f'{self.output_path}/tclean_multiscale_{niter}/tclean_multiscale_{niter}',
        )

    def tclean_mem(self, config: dict):
        from casatasks import tclean

        imsize = config.get('imsize', 256)
        cell = config.get('cell', '0.02arcsec')
        niter = config.get('niter', 0)
        weighting = config.get('weighting', 'natural')

        if os.path.exists(f'{self.output_path}/tclean_mem_{niter}'):
            os.system(f'rm -rf {self.output_path}/tclean_mem_{niter}')

        tclean(
            vis = self.vis,
            imsize = imsize,
            cell = cell,
            niter = niter,
            weighting = weighting,
            deconvolver = 'mem',
            imagename= f'{self.output_path}/tclean_mem_{niter}/tclean_mem_{niter}',
        )

    def gpuvmem(self, config: dict):
        print("GPU VMEM is not implemented yet. Please use tclean instead.")