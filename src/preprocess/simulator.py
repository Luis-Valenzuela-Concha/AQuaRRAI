from dataclasses import dataclass
import os
from casatasks import simobserve
import shutil

@dataclass
class Simulator:
    image: str
    simobserve_config: dict
    output_folder: str

    def __init__(
            self, 
            image: str, 
            simobserve_config: dict, 
            output_folder: str = '../data/processed/',
            output_name: str = None
        ):
        if not isinstance(simobserve_config, dict):
            raise ValueError("Config must be a dictionary.")

        self.image = os.path.abspath(image)
        self.image_name = os.path.basename(self.image).split('.')[0]
        self.simobserve_config = simobserve_config
        self.output_folder = os.path.abspath(output_folder)
        self.output_name = output_name if output_name else f'{self.image_name}_visibilities'

    def simulate(self):
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder, exist_ok=True)

        temp_file = self.image_name
        antenna_name = self.simobserve_config.get('antenna', 'alma.out08.cfg').split('.cfg')[0]
        
        ms_file = f'{temp_file}/{temp_file}.{antenna_name}.ms'

        if os.path.exists(os.path.join(self.output_folder, os.path.basename(ms_file))):
            shutil.rmtree(os.path.join(self.output_folder, os.path.basename(ms_file)))

        if os.path.exists(temp_file):
            shutil.rmtree(temp_file)

        arcsec = str(self.simobserve_config.get('arcsec', 0.02)) + 'arcsec'
        antenna = self.simobserve_config.get('antenna', 'alma.out08.cfg')
        total_time = str(self.simobserve_config.get('totaltime', 12000)) + 's'

        simobserve(
            project=temp_file,
            skymodel=self.image,
            indirection = 'J2000 03h30m00 1d00m00', #0 y -90
            inbright= '1Jy',
            incell = arcsec,
            incenter = '230GHz',
            inwidth = '50MHz',
            setpointings = True,
            integration = '300s',
            mapsize = [arcsec,arcsec],
            maptype = 'ALMA',
            pointingspacing = '1arcsec',
            graphics = 'both',
            obsmode = 'int',
            antennalist = antenna,
            totaltime = total_time
        )
        
        noise = str(self.simobserve_config.get('noise', ''))
        if noise:
            from casatools import simulator
            sim = simulator()
            sim.openfromms(f'{temp_file}/{temp_file}.{antenna_name}.ms')
            sim.setnoise(simplenoise=noise+'Jy')
            sim.corrupt()
            sim.done()
            sim.close()
        else:
            print("No noise added to the simulation.")
    
        
        if os.path.exists(ms_file):
            new_ms_file = os.path.join(self.output_folder, f'{self.output_name}.ms')
            
            if os.path.exists(new_ms_file):
                shutil.rmtree(new_ms_file)
                
            os.rename(ms_file, new_ms_file)
            shutil.rmtree(temp_file)
