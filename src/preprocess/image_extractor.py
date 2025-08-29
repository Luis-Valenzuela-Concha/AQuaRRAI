import os
import numpy as np
import pandas as pd

class ImageExtractor:
    def __init__(self, output_path=None):
        self.output_path = output_path if output_path else '../../data/raw'
        if not os.path.exists(self.output_path):
            print(f"Creating directory: {self.output_path}")
            os.makedirs(self.output_path)
        pass

    def extract_decals(self):
        print("Extracting DECaLS images...")
        import h5py
        from PIL import Image
        from tensorflow.keras import utils

        # Galaxy10 dataset (17736 images)
        # ├── Class 0 (1081 images): Disturbed Galaxies
        # ├── Class 1 (1853 images): Merging Galaxies
        # ├── Class 2 (2645 images): Round Smooth Galaxies
        # ├── Class 3 (2027 images): In-between Round Smooth Galaxies
        # ├── Class 4 ( 334 images): Cigar Shaped Smooth Galaxies
        # ├── Class 5 (2043 images): Barred Spiral Galaxies
        # ├── Class 6 (1829 images): Unbarred Tight Spiral Galaxies
        # ├── Class 7 (2628 images): Unbarred Loose Spiral Galaxies
        # ├── Class 8 (1423 images): Edge-on Galaxies without Bulge
        # └── Class 9 (1873 images): Edge-on Galaxies with Bulge

        with h5py.File('../../data/Galaxy10_DECals.h5', 'r') as F:
            images = np.array(F['images'])
            labels = np.array(F['ans'])

        labels = utils.to_categorical(labels, 10)

        labels = labels.astype(np.uint8)
        images = images.astype(np.uint8)

        df = pd.DataFrame({'images': list(images), 'labels': list(labels)})

        df['label'] = df['labels'].apply(lambda x: np.argmax(x))
        df = df[['images', 'label']]
        
        label_counts = df['label'].value_counts().reset_index()
        label_counts.columns = ['label', 'count']

        df_grouped = (
            df[['images', 'label']]
            .groupby('label', group_keys=False)
            .apply(lambda x: x.sample(20, random_state=42))
            .reset_index(drop=True)
        )
        
        label_counts_grouped = df_grouped['label'].value_counts().reset_index()
        label_counts_grouped.columns = ['label', 'count']
    
        columns_excel = ['name', 'type']

        output_directory = f'{self.output_path}/galaxy10_decals'
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        def save_images(df, output_dir):
            rows = []
            for index, row in df.iterrows():
                img = Image.fromarray(row['images'])
                label = row['label']

                rows.append({
                    'name': f"image_{index}_label_{label}.png",
                    'type': label
                })
                img.save(f"{output_dir}/image_{index}_label_{label}.png")
            df_excel = pd.DataFrame(rows, columns=columns_excel)
            df_excel.to_csv(f'{self.output_path}/galaxy10_decals.csv', index=False)

        save_images(df_grouped, output_directory)
        print(f"Saved {len(df_grouped)} images to {output_directory}")

    def extract_eso(self, names_file=None):
        if not names_file:
            raise ValueError("ESO names file is required for extraction.")
        
        import requests
        from io import BytesIO
        from PIL import Image as PILImage

        print("Extracting ESO images...")

        DOWNLOAD_BASE_URL = 'https://cdn.eso.org/images/screen/'

        output_directory = f'{self.output_path}/eso'

        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        df = pd.read_csv(names_file)
        source_names = df['source_name'].tolist()

        count = 0
        for source_name in source_names:
            image_url = f'{DOWNLOAD_BASE_URL}{source_name}.jpg'
            try:
                resp = requests.get(image_url)
                img = PILImage.open(BytesIO(resp.content))
                save_path = os.path.join(output_directory, f'{source_name}.jpg')
                img.save(save_path)
                count += 1
            except Exception as e:
                print(f"Failed to download {image_url}: {e}")
        print(f"Downloaded {count} ESO images to {output_directory}")
        