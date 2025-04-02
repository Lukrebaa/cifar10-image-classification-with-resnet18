# CIFAR-10 Image Classification with ResNet18

This repository contains a deep learning project for image classification on the CIFAR-10 dataset using a pre-trained ResNet18 model with PyTorch.

## Project Overview

The project implements transfer learning using a pre-trained ResNet18 model to classify images from the CIFAR-10 dataset into 10 different categories: planes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks.

## Repository Structure

```
CIFAR10-CLASSIFICATION/
├── data/                      # Dataset directory
│   └── cifar-10-batches-py/   # CIFAR-10 Python batches
│       └── batches.meta               # Metadata for batches
│       └── data_batch_1               # Training data batch 1
│       └── data_batch_2               # Training data batch 2
│       └── data_batch_3               # Training data batch 3
│       └── data_batch_4               # Training data batch 4
│       └── data_batch_5               # Training data batch 5
│       └── readme.html                # Original CIFAR-10 readme
│       └── test_batch                 # Test data batch
├── models/                    # Saved model directory
│   └── resnet18_cifar10.pth   # Trained ResNet18 model
├── results/                   # Results directory
│   ├── 1117/                  # Results from run 1117
│   └── 1120/                  # Results from run 1120
│        
└── cifar10.ipynb              # Jupyter notebook with results
```

## Features

- **Data Preprocessing**: Resizing, data augmentation, and normalization
- **Transfer Learning**: Fine-tuning a pre-trained ResNet18 model
- **Training**: Full training pipeline with optimizer and learning rate scheduler
- **Performance Metrics**: Tracking of accuracy and loss for both training and validation
- **Visualization**: Plots for accuracy, loss, and confusion matrix
- **Learning Rate Finding**: Implementation of learning rate finder

## Model Architecture

The project uses a ResNet18 model pre-trained on ImageNet, with the final fully connected layer modified to output 10 classes (for CIFAR-10 categories) instead of the original 1000 ImageNet classes.

## Requirements

- Python 3.6+
- PyTorch 1.7+
- torchvision
- NumPy
- Matplotlib
- scikit-learn
- seaborn
- opencv-python
- torch_lr_finder

## Usage

### Training the Model

You can open and run the Jupyter notebook:

```python

jupyter notebook cifar10.ipynb

```

### Using Pre-trained Weights
The repository includes pre-trained model weights in the `models` directory. You can load and use these weights for inference without retraining:

```python
import torch
import torchvision.models as models

# Define model architecture
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 10)  # 10 classes for CIFAR-10

# Load pre-trained weights
model.load_state_dict(torch.load('models/resnet18_cifar10.pth'))
model.eval()  # Set to evaluation mode

# Now you can use the model for inference
```

### Hyperparameters

- Batch size: 256
- Learning rate: 5.59e-4
- Optimizer: Adam
- Scheduler: StepLR (step_size=7, gamma=0.1)
- Epochs: 20

## Results

The model achieves competitive accuracy on the CIFAR-10 test set. Detailed results, including confusion matrices, accuracy plots, and loss plots, can be found in the `results` directory.

## License

MIT License

Copyright (c) 2025 Rebeka Rachel Lukacs

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.