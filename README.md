# AI vs Real Image Classification

Binary image classification project to distinguish between AI-generated images and real photographs. Implements EfficientNetB0 transfer learning and custom CNN architectures.

## Overview

Trains deep learning models to classify images as:
- **REAL**: Authentic photographs
- **FAKE**: AI-generated images

Includes data preprocessing, model training, and inference pipelines.

## Features

- Data preprocessing: extraction, cleaning, resizing, balancing
- Model architectures: EfficientNetB0 and custom CNNs
- Data augmentation during training
- Training pipeline with callbacks and checkpointing
- Model evaluation and inference

## Project Structure

```
mlops/
├── train.py          # Training script
├── train.ipynb       # Jupyter notebook
└── README.md
```

## Requirements

```bash
pip install tensorflow keras numpy pillow matplotlib
```

**Note**: Originally developed in Google Colab. For local execution:
1. Remove Google Colab imports (`from google.colab import drive`)
2. Update file paths to match local directory structure
3. GPU support recommended for training

## Dataset Structure

Expected directory structure:

```
data/
├── train/
│   ├── real/
│   └── fake/
├── val/
│   ├── real/
│   └── fake/
└── test/
    ├── real/
    └── fake/
```

### Preprocessing

Scripts handle:
- Dataset extraction from zip files
- Removing corrupted images
- Resizing to 256x256 pixels
- Balancing real/fake classes
- Creating train/validation/test splits

## Model Architectures

### EfficientNetB0

- Base model: EfficientNetB0 with ImageNet weights
- Input size: 256x256x3
- Training: Frozen backbone (10 epochs), then fine-tune last 20 layers (10 epochs)
- Features: Data augmentation, Global Average Pooling, Dropout (0.3), label smoothing (0.1)

### Custom CNN (Basic)

- 4 convolutional blocks: Conv2D → BatchNorm → MaxPooling
- Binary classification with sigmoid activation
- Dropout: 0.5

### Custom CNN (Upgraded)

- Enhanced 4-block CNN with double convolutions per block
- Global Average Pooling instead of Flatten
- Dropout: 0.6, label smoothing: 0.05
- Training: 20 epochs

## Usage

### Training

**Python script:**
```bash
python train.py
```

Modify the script to remove Colab dependencies and update paths.

**Jupyter notebook:**
Open `train.ipynb` and run cells sequentially.

### Configuration

Key parameters:
```python
IMG_SIZE = (256, 256)
BATCH_SIZE = 32
EPOCHS = 15-20
LEARNING_RATE = 1e-4
```

## Training Process

1. Data loading using `ImageDataGenerator` or `image_dataset_from_directory`
2. Data augmentation: rotation, flipping, zoom, brightness, contrast
3. Model compilation with Adam optimizer
4. Callbacks: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
5. Training and validation
6. Test set evaluation

### Model Files

- `custom_cnn_best.keras` - Best model by validation accuracy
- `custom_cnn_last.keras` - Last epoch model
- `custom_cnn_final.keras` - Final saved model
- `custom_cnn_interrupted.keras` - Saved on interruption

## Inference

```python
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np

model = keras.models.load_model("path/to/model.keras")

img = image.load_img("path/to/image.jpg", target_size=(256, 256))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

pred = model.predict(img_array)
label = "REAL" if pred[0][0] > 0.5 else "FAKE"
confidence = pred[0][0] if pred[0][0] > 0.5 else 1 - pred[0][0]

print(f"Prediction: {label} (Confidence: {confidence:.4f})")
```

## Workflow

1. Data preparation: extract, clean, resize, balance, split
2. Model development: choose architecture, configure hyperparameters
3. Training: train with callbacks, monitor metrics, save checkpoints
4. Evaluation: test set evaluation
5. Inference: load model and make predictions

## Notes

- Originally developed in Google Colab
- Code paths reference Colab directories (`/content/data`, `/content/drive`)
- Update paths for local execution
- GPU acceleration recommended
- Dataset should be balanced
