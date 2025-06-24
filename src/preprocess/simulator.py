from dataclasses import dataclass
import os
from casatasks import simobserve
import shutil

@dataclass
class Simulator:
    image: str
    simobserve_config: dict
    output_folder: str

    def __init__(self, image: str, simobserve_config: dict, output_folder: str = '../data/processed/simulations'):
        if not isinstance(image, str):
            raise ValueError("Image must be a string.")
        if not isinstance(simobserve_config, dict):
            raise ValueError("Config must be a dictionary.")
        if not isinstance(output_folder, str):
            raise ValueError("Output folder must be a string.")

        self.image = os.path.abspath(image)
        self.image_name = os.path.basename(self.image).split('.')[0]
        self.simobserve_config = simobserve_config
        self.output_folder = os.path.abspath(output_folder) + '/' + self.image_name + '/ms'

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder, exist_ok=True)

    def simulate(self):
        temp_file = self.image_name
        antenna_name = self.simobserve_config.get('antenna', 'alma.out08.cfg').split('.cfg')[0]
        ms_file = f'{temp_file}/{temp_file}.{self.simobserve_config.get("antenna", "alma.out08.cfg").split(".cfg")[0]}.ms'
        if os.path.exists(os.path.join(self.output_folder, os.path.basename(ms_file))):
            shutil.rmtree(os.path.join(self.output_folder, os.path.basename(ms_file)))

        if os.path.exists(temp_file):
            shutil.rmtree(temp_file)  
        simobserve(
            project=temp_file,
            skymodel=self.image,
            indirection = 'J2000 03h30m00 1d00m00', #0 y -90
            inbright= '1Jy',
            incell = '0.04arcsec',
            incenter = '230GHz',
            inwidth = '50MHz',
            setpointings = True,
            integration = '300s',
            mapsize = ['0.2arcsec','0.2arcsec'],
            maptype = 'ALMA',
            pointingspacing = '1arcsec',
            graphics = 'both',
            obsmode = 'int',
            antennalist = self.simobserve_config.get('antenna', 'alma.out08.cfg'),
            totaltime = str(self.simobserve_config.get('totaltime', '12000')) + 's',
        )
        
        noise = self.simobserve_config.get('noise', '')
        if noise:
            from casatools import simulator

            antennalist = self.simobserve_config.get('antenna', 'alma.out08.cfg')
            antenna_name = antennalist.split('.cfg')[0]
            sim = simulator()
            sim.openfromms(f'{temp_file}/{temp_file}.{antenna_name}.ms')
            sim.setnoise(simplenoise=str(noise) + 'Jy')
            sim.corrupt()
            sim.close()
        else:
            print("No noise added to the simulation.")
    
        
        if os.path.exists(ms_file):
            shutil.move(ms_file, self.output_folder)
            shutil.rmtree(temp_file)
