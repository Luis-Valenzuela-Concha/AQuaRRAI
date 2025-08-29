# Automatic Quality Ranking of Reconstructed Astronomical Images (AQuaRRAI)

AQuaRRAI is modular system to simulate, reconstruct, evaluate, and rank the quality of reconstructed astronomical images **without requiring a reference image**, with a focus on radio interferometry. It includes CNN models with and without group context that predict SSIM or PSNR values, enabling the ranking of reconstructions from the same observation.

# Key Features

- End-to-end pipeline: from preprocessing → simulation → reconstruction → metric evaluation → ranking.
- Reference-free evaluation: CNN models predict SSIM / PSNRr values without a ground-truth image.
- Flexible modules: plug-in simulators, imagers and quality metrics.
- Two ranking modes:
  - Without context: per-image prediction.
  - With context: incorporates group-level information for more consistent rankings.

## Requirements

- Python >= 3.11

## Installation

### 1. Clone this repository

```bash
git clone https://github.com/Luis-Valenzuela-Concha/AQUARRAI.git
cd AQUARRAI
```

### 2. Create virtual environments

```bash
make install-base     # base environment (Linux/macOS)
make install-pyralysis   # optional, extended environment (Linux — includes imaging via pyralysis)
```

### 3. Activate a virtual environment

```bash
source venv_base/bin/activate     # activate base environment
source venv_pyralysis/bin/activate  # activate pyralysis environment (if reconstructing with pyralysis)
```

## Author

Luis Valenzuela, Computer Engineering student, University of Concepción.

This work was submitted in fulfillment of degree requirements.
