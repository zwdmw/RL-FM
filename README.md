# RL-FM: Reinforcement Learning-Driven Flow Matching for Multi-Modal Remote Sensing Image Classification

This repository contains the official implementation of the paper **"RL-FM: Reinforcement Learning-Driven Flow Matching for Multi-Modal Remote Sensing Image Classification"**.

## Overview

RL-FM is a reinforcement learning-based approach for multi-modal remote sensing image classification, which combines Hyperspectral Imaging (HSI) and LiDAR data for improved classification performance. The method uses reinforcement learning to optimize the fusion of multi-modal features through flow matching techniques.

## Dataset Setup

### Dataset Directory Structure

Place your datasets in the following directory structure:

```
../dataset/
└── HS/
    ├── Houston.mat
    ├── Houston_LR.mat
    └── Houston_gt.mat
```
The dataset is automatically downloaded on the first run.
### Dataset Requirements

**Houston (HS) Dataset:**
- `Houston.mat`: HSI data with shape (H, W, B)
- `Houston_LR.mat`: LiDAR data with shape (H, W, 1 or 2)
- `Houston_gt.mat`: Ground truth labels with shape (H, W)

All files should be in MATLAB `.mat` format.

### Dataset Path Configuration

The default dataset path is set to `../dataset` in the code. If you need to use a different path, you can modify the `dataset_root` variable in `main.py`:

```python
dataset_root = os.path.join('C:', os.sep, 'code', 'dataset')
```

## Installation

### Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Key Dependencies

- Python 3.7+
- PyTorch 1.10+
- NumPy
- scikit-learn
- SciPy

## Usage

### Basic Usage

Run the main script:

```bash
python main.py
```

The code will automatically:
1. Load the dataset from the configured path
2. Process and extract features from HSI and LiDAR data
3. Train the reinforcement learning policy for feature fusion
4. Perform classification and output evaluation metrics

### Output

The script will output classification metrics including:
- Overall Accuracy (OA)
- Average Accuracy (AA)
- Kappa coefficient
- Class-wise accuracy

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

## License

This project is licensed under the MIT License.

## Contact

For questions or issues, please open an issue on GitHub or contact the authors.

