# RL-FM: Reinforcement Learning-Driven Flow Matching for Multi-Modal Remote Sensing Image Classification

<div align="center">

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**Official implementation of the RL-FM paper for multi-modal remote sensing image classification**

[Dataset Setup](#dataset-setup) • [Installation](#installation) • [Usage](#usage) • [Citation](#citation)

</div>

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset Setup](#dataset-setup)
  - [Dataset Directory Structure](#dataset-directory-structure)
  - [Dataset Requirements](#dataset-requirements)
  - [Dataset Path Configuration](#dataset-path-configuration)
  - [Dataset Selection](#dataset-selection)
- [Installation](#installation)
  - [Requirements](#requirements)
  - [Key Dependencies](#key-dependencies)
- [Usage](#usage)
  - [Basic Usage](#basic-usage)
  - [Output](#output)
- [Project Structure](#project-structure)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

---

## Overview

RL-FM is a reinforcement learning-based approach for multi-modal remote sensing image classification, which combines **Hyperspectral Imaging (HSI)** and **LiDAR** data for improved classification performance. The method uses reinforcement learning to optimize the fusion of multi-modal features through flow matching techniques.

---

## Features

- ✅ **Multi-modal Fusion**: Seamlessly combines HSI and LiDAR data
- ✅ **Reinforcement Learning**: Uses PPO (Proximal Policy Optimization) for optimal feature fusion
- ✅ **Flow Matching**: Advanced flow matching techniques for feature alignment
- ✅ **Multiple Datasets**: Supports Houston (HS) and MUUFL datasets
- ✅ **Auto Download**: Automatic dataset and model download on first run
- ✅ **Easy Configuration**: Simple configuration via `main.py`

---

## Dataset Setup

### Dataset Directory Structure

Place your datasets in the following directory structure:

```
../dataset/
├── HS/
│   ├── Houston.mat
│   ├── Houston_LR.mat
│   └── Houston_gt.mat
└── MUUFL/
    ├── MUF_HSI.mat
    ├── MUF_LiDAR.mat
    └── MUF_gt.mat
```

> **Note:** Datasets will be automatically downloaded on the first run. To improve download speed, it is recommended to use a VPN or proxy service.

### Dataset Requirements

#### Houston (HS) Dataset

| File | Description | Shape |
|------|-------------|-------|
| `Houston.mat` | HSI data | (H, W, B) |
| `Houston_LR.mat` | LiDAR data | (H, W, 1) |
| `Houston_gt.mat` | Ground truth labels | (H, W) |

#### MUUFL Dataset

| File | Description | Shape |
|------|-------------|-------|
| `MUF_HSI.mat` | HSI data | (H, W, B) |
| `MUF_LiDAR.mat` | LiDAR data | (H, W, 1) |
| `MUF_gt.mat` | Ground truth labels | (H, W) |

> All files should be in MATLAB `.mat` format.

### Dataset Path Configuration

The default dataset path is set to `../dataset` in the code. If you need to use a different path, you can modify the `dataset_root` variable in `main.py`:

```python
dataset_root = os.path.join('C:', os.sep, 'code', 'dataset')
```

### Dataset Selection

You can switch between different datasets by modifying the `dataset_name` variable in `main.py`:

```python
dataset_name = 'HS'  # Options: 'HS' or 'MUUFL'
```

> **Download Tip:** Since files are hosted on GitHub, if download speed is slow, it is recommended to use a VPN or proxy service to accelerate the download process.

---

## Installation

### Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Key Dependencies

| Package | Version | Description |
|---------|---------|-------------|
| **Python** | 3.7+ | Programming language |
| **PyTorch** | 1.10+ | Deep learning framework |
| **NumPy** | Latest | Numerical computing |
| **scikit-learn** | Latest | Machine learning utilities |
| **SciPy** | Latest | Scientific computing |

---

## Usage

### Basic Usage

Run the main script:

```bash
python main.py
```

The code will automatically:

1. **Load** the dataset from the configured path
2. **Process** and extract features from HSI and LiDAR data
3. **Train** the reinforcement learning policy for feature fusion
4. **Perform** classification and output evaluation metrics

### Output

The script will output classification metrics including:

| Metric | Description |
|--------|-------------|
| **Overall Accuracy (OA)** | Overall classification accuracy |
| **Average Accuracy (AA)** | Average per-class accuracy |
| **Kappa coefficient** | Cohen's kappa coefficient |
| **Class-wise accuracy** | Accuracy for each individual class |

---

## Project Structure

```
RL-FM/
├── main.py                 # Main entry point
├── data_loader.py          # Dataset loading
├── data_processor.py       # Data preprocessing
├── patch_extractor.py      # Patch extraction
├── data_splitter.py        # Train/test split
├── feature_extractor.py    # Feature extraction models
├── trainer.py              # Training logic
├── ppo_agent.py            # CPPO reinforcement learning agent network
├── model.py                # Model definitions
└── requirements.txt        # Python dependencies
```

---

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{RL-FM2024,
  title={RL-FM: Reinforcement Learning-Driven Flow Matching for Multi-Modal Remote Sensing Image Classification},
  author={[Authors]},
  journal={[Journal]},
  year={2025}
}
```

---

## License

This project is licensed under the **MIT License**.

---

## Contact

For questions or issues, please open an issue on GitHub or contact the authors.

<div align="center">

**⭐ Star this repository if you find it helpful! ⭐**

</div>
