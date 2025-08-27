import numpy as np
from skimage.transform import resize
from src.preprocess.image_quality_metrics import ImageQualityMetrics 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from skimage import filters
import os

import sys
from pathlib import Path

ROOT = Path().resolve().parent.parent.parent

def sobel_image(image):
    sobel_image = filters.sobel(image)
    sobel_image = (sobel_image - sobel_image.min()) / (sobel_image.max() - sobel_image.min())
    return sobel_image

class Metric:
    def __init__(self, raw=None, value=None):
        self.raw = raw
        self.value = value

class Metrics:
    def __init__(self):
        self.ssim = Metric()
        self.psnr = Metric()
        self.residual_rms = Metric()

class ReconstructionData:
    def __init__(self, path, sim, algorithm, data=None, residual_data=None):
        self.path = path
        self.sim = sim
        self.algorithm = algorithm
        self.data = data
        self.residual_data = residual_data
        self.filtered_data = None
        self.residual_rms = None
        self.metrics = Metrics()

class GroundtruthData:
    def __init__(self, data):
        self.data = data
        self.filtered_data = None

class SetsData:
    def __init__(self, object_name, object_type, reconstructions, ground_truth):
        self.object_name = object_name
        self.object_type = object_type
        self.ground_truth : GroundtruthData = ground_truth
        self.reconstructions : list[ReconstructionData] = reconstructions

class Dataset:
    def __init__(self, csv_data, IMAGE_SIZE=(64, 64), include_filter=False):
        self.df = pd.read_csv(csv_data)
        self.IMAGE_SIZE = IMAGE_SIZE
        self.sets : list[SetsData] = []
        self._create_sets(include_filter=include_filter)

    @staticmethod
    def __to_gray(img: np.ndarray) -> np.ndarray:
        if img.ndim == 3 and img.shape[-1] == 3:
            img = np.mean(img, axis=-1)
        elif img.ndim == 3 and img.shape[-1] == 1:
            img = img.squeeze(-1)
        return img

    @staticmethod
    def __prep(img: np.ndarray, size) -> np.ndarray:
        img = resize(img, size, anti_aliasing=True, preserve_range=True)
        img_min, img_max = np.min(img), np.max(img)
        if img_max == img_min:
            if img_max == 0.0:
                return np.zeros_like(img)
            val = (img_max / 255.0) if img_max > 1.0 else img_max
            return np.clip(np.ones_like(img) * val, 0.0, 1.0)
        return (img - img_min) / (img_max - img_min)

    def __load_image(self, image_path, residual_path, reference_path):
        iqm = ImageQualityMetrics(
            image_path=str(image_path),
            residual_path=str(residual_path),
            reference_path=str(reference_path),
        )

        image = self.__to_gray(iqm.data_image)
        residual_image = self.__to_gray(iqm.data_residual)
        reference_image = self.__to_gray(iqm.data_reference)

        image = self.__prep(image, self.IMAGE_SIZE)
        residual_image = self.__prep(residual_image, self.IMAGE_SIZE)
        reference_image = self.__prep(reference_image, self.IMAGE_SIZE)

        return image, residual_image, reference_image

    def _create_sets(self, include_filter=False):
        self.df['image'], self.df['residual'], self.df['ground_truth'] = zip(*self.df['path'].apply(
            lambda x: self.__load_image(
                image_path=ROOT / x, 
                residual_path = Path(str(ROOT / x).replace('.model', '.residual')),
                reference_path=(ROOT / x).parent.parent.parent / 'groundtruth.fits', 
            )
        ))

        df_grouped = self.df.groupby('object')
        
        for img, rows in df_grouped:
            object_name = img
            object_type = rows.iloc[0]['type']

            reconstructionDatas = []
            for _, row in rows.iterrows():

                if row['reconstruction'] == 'tclean_mem_200' and \
                   row['sim'] == 'sim1':
                    continue

                reconstructionData = ReconstructionData(
                    path=row['path'],
                    sim=row['sim'],
                    algorithm=row['reconstruction'],
                    data=row['image'],
                    residual_data = row['residual']
                )

                reconstructionDatas.append(reconstructionData)

            setData = SetsData(
                object_name=object_name,
                object_type=object_type,
                ground_truth=GroundtruthData(data=rows.iloc[0]['ground_truth']),
                reconstructions=reconstructionDatas
            )
            self.sets.append(setData)

        if include_filter:
            for set_data in self.sets:
                gt_image = set_data.ground_truth.data
                gt_image = sobel_image(gt_image)
                set_data.ground_truth.filtered_data = gt_image

                for recon in set_data.reconstructions:
                    img = recon.data
                    img = sobel_image(img)
                    recon.filtered_data = img

    def get_splits(self, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, object_types=['galaxy', 'nebula'] ,random_state=42, normalize_metrics=False):
        objects = [s.object_name for s in self.sets if s.object_type in object_types]
        unique_objects = np.unique(objects)
        n = len(unique_objects)

        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError("Las proporciones deben sumar 1.0")

        ratios = np.array([train_ratio, val_ratio, test_ratio], dtype=float)
        raw = ratios * n
        base = np.floor(raw).astype(int)
        resto = n - base.sum()

        order = np.argsort(-(raw - base))
        base[order[:resto]] += 1
        _, n_val, n_test = base.tolist()

        rem, test_objects = train_test_split(
            unique_objects, test_size=n_test, random_state=random_state
        )
        train_objects, val_objects = train_test_split(
            rem, test_size=n_val, random_state=random_state
        )

        train_sets = [s for s in self.sets if s.object_name in train_objects]
        val_sets   = [s for s in self.sets if s.object_name in val_objects]
        test_sets  = [s for s in self.sets if s.object_name in test_objects]

        scalers = None

        if not normalize_metrics:
            def set_all_metrics_to_raw(dataset):
                for set_data in dataset:
                    for recon in set_data.reconstructions:
                        for metric_name, metric_obj in vars(recon.metrics).items():
                            if isinstance(metric_obj, Metric) and metric_obj.raw is not None:
                                metric_obj.value = metric_obj.raw
            set_all_metrics_to_raw(train_sets)
            set_all_metrics_to_raw(val_sets)
            set_all_metrics_to_raw(test_sets)
    
        else:
            metrics_dict = {}

            for set_data in train_sets:
                for recon in set_data.reconstructions:
                    for metric_name, metric_obj in vars(recon.metrics).items():
                        if isinstance(metric_obj, Metric) and metric_obj.raw is not None:
                            metrics_dict.setdefault(metric_name, []).append(metric_obj.raw)

            scalers = {}
            for metric_name, values in metrics_dict.items():
                scaler = MinMaxScaler()
                scalers[metric_name] = scaler.fit([[v] for v in values])

            def normalize_set(dataset):
                for set_data in dataset:
                    for recon in set_data.reconstructions:
                        for metric_name, scaler in scalers.items():
                            metric_obj = getattr(recon.metrics, metric_name, None)
                            if isinstance(metric_obj, Metric) and metric_obj.raw is not None:
                                metric_obj.value = scaler.transform([[metric_obj.raw]])[0, 0]

            normalize_set(train_sets)
            normalize_set(val_sets)
            normalize_set(test_sets)

        return train_sets, val_sets, test_sets, scalers
    
    def calculate_metrics(self, metrics=['ssim', 'psnr', 'residual_rms'], to_filter=False):
        from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
        for set_data in self.sets:
            gt_image = set_data.ground_truth.data if not to_filter else set_data.ground_truth.filtered_data
            for recon in set_data.reconstructions:
                recon_image = recon.data if not to_filter else recon.filtered_data
                if 'ssim' in metrics:
                    recon.metrics.ssim.raw = ssim(gt_image, recon_image, data_range=recon_image.max() - recon_image.min())
                if 'psnr' in metrics:
                    recon.metrics.psnr.raw = psnr(gt_image, recon_image, data_range=recon_image.max() - recon_image.min())
                if 'residual_rms' in metrics:
                    residual_image = recon.residual_data
                    recon.metrics.residual_rms.raw = np.sqrt(np.mean((residual_image) ** 2))
            