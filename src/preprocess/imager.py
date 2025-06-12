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
        self.vis = vis
        self.output_path = output_path
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path, exist_ok=True)

    def tclean(self, config: dict):
        from casatasks import tclean

        # eliminar any existing tclean output
        if os.path.exists(f'{self.output_path}/tclean'):
            os.system(f'rm -rf {self.output_path}/tclean')

        tclean(
            vis = self.vis,
            imsize = config.get('imsize', 256),
            cell = config.get('cell', '0.04arcsec'),
            niter = config.get('niter', 0),
            weighting = config.get('weighting', 'natural'),
            imagename= f'{self.output_path}/tclean/tclean',
        )
    def gpuvmem(self, config: dict):
        print("GPU VMEM is not implemented yet. Please use tclean instead.")