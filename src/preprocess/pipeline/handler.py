import pandas as pd
import os
from src.preprocess.image_extractor import ImageExtractor

class PipelineHandler:
    def __init__(self, data_folder=None):
        self.data_folder = data_folder if data_folder else 'data'
        if not os.path.exists(self.data_folder):
            print(f"Folder {self.data_folder} does not exist, creating it.")
            os.makedirs(self.data_folder)
        pass

    def get_gt_images(self, decals_file=None, eso_names_file=None):
        extractor = ImageExtractor(output_path=self.data_folder / 'raw')
        if decals_file:
            extractor.extract_decals()
        if eso_names_file:
            extractor.extract_eso(names_file=eso_names_file)
        print("Reference images extraction completed.")

    def set_fits_images(self):
        from src.preprocess.image_preprocessor import ImagePreprocessor

        image_folder = self.data_folder / 'processed'

        if not os.path.exists(image_folder):
            print(f"Creating directory: {image_folder}")
            os.makedirs(image_folder)

        raw_folder = self.data_folder / 'raw'
        processed_folder = self.data_folder / 'processed'

        decals_folder = raw_folder / 'galaxy10_decals' # Galaxies
        eso_folder = raw_folder / 'eso' # Nebulae

        size = 256

        df = pd.DataFrame(columns=['image_name', 'source'])

        img = 0
        for folder in [decals_folder, eso_folder]:
            # if folder doesnt exist
            if not os.path.exists(folder):
                continue
            print(f"Formatting FITS images of folder: {folder}")
            source_type = 'galaxy' if 'decals' in str(folder) else 'nebula'
            for reference in os.listdir(folder):
                img_str = str(img).zfill(4)
                reference_path = os.path.join(folder, reference)

                preprocessor = ImagePreprocessor(
                    image_path=reference_path,
                    width=size, 
                    height=size
                )
                
                preprocessor.convert_to_fits(
                    output_folder=processed_folder / f'img_{img_str}', 
                    output_name=f'groundtruth',
                    verbose=False
                )
                reference_name = reference.split('.')[0]
                df = pd.concat([df, pd.DataFrame({
                    'image_name': [f'img_{img_str}'], 
                    'source': [reference_name], 
                    'source_type': [source_type]})
                ], ignore_index=True)

                img += 1
        df.to_csv(self.data_folder / 'id_raw.csv', index=False)

    def simulate_observations(self, sim_conditions):
        from src.preprocess.simulator import Simulator

        proccesed_folder = self.data_folder / 'processed'
        proccesed_images = os.listdir(proccesed_folder)
        proccesed_images.sort()
        print(f"Found {len(proccesed_images)} processed images for simulation.")

        sim_numbers = [f'sim{index + 1}' for index in range(len(sim_conditions))]
        sim_conditions_df = pd.DataFrame(sim_conditions)
        sim_conditions_df['sim'] = sim_numbers
        sim_conditions_df = sim_conditions_df[['sim'] + list(sim_conditions_df.columns[:-1])]
        sim_conditions_df.to_csv(self.data_folder / 'sim_conditions.csv', index=False)

        for image in proccesed_images:
            if not image.startswith('img_'):
                continue
            
            gt_path = proccesed_folder / image / 'groundtruth.fits'
            
            for sim_number, simobserve_config in enumerate(sim_conditions):
                simulator = Simulator(
                    image=gt_path, 
                    output_folder= proccesed_folder / image / f'sim{sim_number + 1}',
                    output_name='visibilities'
                )
                simulator.simobserve_simulate(simobserve_config=simobserve_config)
            
        print("Simulation of observations completed.")
        
    def reconstruct_images(self, rec_conditions):
        from src.imager.tclean import TCleanImager
        
        processed_folder = self.data_folder / 'processed'
        proccesed_images = os.listdir(processed_folder)
        proccesed_images.sort()
        print(f"Found {len(proccesed_images)} processed images for reconstruction.")

        for image in proccesed_images:
            if not image.startswith('img_'):
                continue

            sim_folders = [f for f in os.listdir(processed_folder / image) if 'sim' in f]
            sim_folders.sort()

            for sim in sim_folders:
                sim_folder = processed_folder / image / sim
                visibilities = sim_folder / 'visibilities.ms'
                imager = TCleanImager(vis=str(visibilities), output_folder=str(sim_folder))
                for config in rec_conditions:
                    imager.reconstruct(config=rec_conditions[config])
             
    def generate_data_description(self):
        id_file = self.data_folder / 'id_raw.csv'
        df = pd.read_csv(id_file)

        data_descriptions = []
        processed_folder = self.data_folder / 'processed'
        for image in os.listdir(processed_folder):
            if not image.startswith('img_'):
                continue

            sim_folders = [f for f in os.listdir(processed_folder / image) if 'sim' in f]
            sim_folders.sort()

            for sim in sim_folders:
                recs_list = [f for f in os.listdir(processed_folder / image / sim) if 'tclean' in f]
                for rec in recs_list:
                    rec_path = processed_folder / image / sim / rec
                    if not os.path.exists(rec_path):
                        print(f"Reconstruction path {rec_path} does not exist, skipping.")
                        continue
                    
                    source = df[df['image_name'] == image]['source'].values[0]
                    source_type = df[df['image_name'] == image]['source_type'].values[0]
                    
                    data_description = {
                        'path': str(rec_path / f'{rec}.model'),
                        'id': image,
                        'object': source,
                        'sim': sim,
                        'reconstruction': rec,
                        'type': source_type
                    }
                    data_descriptions.append(data_description)
        data_description_df = pd.DataFrame(data_descriptions)
        data_description_df.to_csv(self.data_folder / 'rec_data_description.csv', index=False)
        print(f"Data description saved to {self.data_folder / 'rec_data_description.csv'}")