# AI vs Real Image and Video Classification

MLOps project for binary classification of AI-generated vs real images and videos using deep learning.

## Introduction

The proliferation of AI-generated content has created a need for automated detection systems. This project implements a deep learning solution to distinguish between authentic and AI-generated images and videos, addressing challenges in content verification and media authenticity.

## Team

- Muhammad Basil
- Taha Zahid

## MLOps Stack

- **Airflow** - Workflow orchestration
- **DagsHub** - Data versioning and collaboration
- **MLflow** - Experiment tracking and model registry
- **MongoDB** - Data storage
- **Git** - Version control
- **Deep Learning** - Model training

## Technology

- **EfficientNet** - Transfer learning model architecture
- **FFmpeg** - Video frame extraction
- **TensorFlow/Keras** - Deep learning framework

## Proposed Methodology

1. **Data Preprocessing**
   - Extract video frames using FFmpeg
   - Clean and resize images to 256x256
   - Balance dataset classes
   - Split into train/validation/test sets

2. **Model Architecture**
   - EfficientNetB0 with transfer learning (ImageNet weights)
   - Two-phase training: frozen backbone then fine-tuning
   - Data augmentation: rotation, flipping, zoom, brightness, contrast

3. **MLOps Pipeline**
   - Airflow orchestrates preprocessing and training workflows
   - MLflow tracks experiments and model versions
   - DagsHub manages data versioning
   - MongoDB stores metadata and results
   - Git for code version control

4. **Training Process**
   - Phase 1: Train with frozen EfficientNetB0 backbone (10 epochs)
   - Phase 2: Fine-tune last 20 layers (10 epochs)
   - Callbacks: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

## Initial Results

- **Test Accuracy: 86%**
- Model successfully distinguishes between real and AI-generated content
- EfficientNetB0 transfer learning approach shows strong performance

## Code

Training code available in:
- `train.py` - Main training script
- `train.ipynb` - Jupyter notebook with full pipeline

Key components:
- Data preprocessing and augmentation
- EfficientNetB0 model implementation
- Training and evaluation loops
- Model saving and inference

## Project Structure

```
mlops/
├── train.py
├── train.ipynb
└── README.md
```

## Dataset

Expected structure:
```
data/
├── train/real/
├── train/fake/
├── val/real/
├── val/fake/
├── test/real/
└── test/fake/
```

Videos are processed using FFmpeg to extract frames before training.

## Usage

```bash
python train.py
```

Or use the Jupyter notebook `train.ipynb`.

## Requirements

```bash
pip install tensorflow keras numpy pillow matplotlib
```

**Note**: Originally developed in Google Colab. Update paths and remove Colab-specific imports for local execution.
