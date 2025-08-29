from abc import ABC, abstractmethod
from dataclasses import dataclass
import os

@dataclass
class BaseImager(ABC):
    vis: str
    output_folder: str

    def __post_init__(self):
        if not os.path.exists(self.vis):
            raise FileNotFoundError(f"Visibility data file '{self.vis}' does not exist.")
        
        if not self.vis.endswith('.ms'):
            raise ValueError("Visibility data must be a Measurement Set (.ms) file.")

        if not os.path.exists(self.output_folder):
            print(f"Output folder '{self.output_folder}' does not exist. Creating it.")
            os.makedirs(self.output_folder, exist_ok=True)

    @abstractmethod
    def reconstruct(self, config: dict):
        pass
