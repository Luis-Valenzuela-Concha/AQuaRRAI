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

        self.vis = os.path.abspath(vis)
        self.output_path = os.path.abspath(output_path)

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path, exist_ok=True)

    def tclean(self, config: dict):
        from casatasks import tclean

        if os.path.exists(f'{self.output_path}/tclean'):
            os.system(f'rm -rf {self.output_path}/tclean')

        niter = config.get('niter', 0)
        tclean(
            vis = self.vis,
            imsize = config.get('imsize', 256),
            cell = config.get('cell', '0.04arcsec'),
            niter = niter,
            weighting = config.get('weighting', 'natural'),
            imagename= f'{self.output_path}/tclean_{niter}/tclean_{niter}',
        )
    def gpuvmem(self, config: dict):
        print("GPU VMEM is not implemented yet. Please use tclean instead.")