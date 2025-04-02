# CIFAR-10 Image Classification with PyTorch and TensorFlow

This repository contains deep learning projects for image classification on the CIFAR-10 dataset using both PyTorch (with ResNet18) and TensorFlow/Keras (with a custom ResNet18 implementation).

## Project Overview

The project implements image classification models to classify the CIFAR-10 dataset into 10 different categories: planes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks. It provides two separate implementations:

1. **PyTorch Implementation**: Uses transfer learning with a pre-trained ResNet18 model
2. **TensorFlow/Keras Implementation**: Uses a custom-built ResNet18 architecture

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
├── models/                       # Saved model directory
│   └── resnet18_cifar10.pth      # Trained PyTorch ResNet18 model weights (included)
├── results/                      # Results directory
│   ├── 1117/                     # Results from run 1117
│   ├── 1120/                     # Results from run 1120
├── cifar10_pytorch.ipynb         # PyTorch implementation notebook
└── cifar10_tensorflow.ipynb      # TensorFlow implementation notebook
```

## Features

### Common Features
- **Classification**: Classification of CIFAR-10 images into 10 categories
- **Data Preprocessing and Augmentation**:Data preprocessing and augmentation techniques
-  **Performance Metrics**: Performance evaluation with accuracy metrics and visualizations
- **Pre-trained Model Weights**: Pre-trained model weights included for immediate inference

### PyTorch Implementation Features
- **Transfer Learning**: Fine-tuning a pre-trained ResNet18 model
- **Learning Rate Finding**: Implementation of learning rate finder for optimal training
- **Data Augmentation** : Advanced data augmentation with RandomResizedCrop and RandomHorizontalFlip
- **Visualization**: Plots for accuracy, loss, and confusion matrix
- **Training**: Full training pipeline with optimizer and learning rate scheduler

### TensorFlow Implementation Features
- **Custom ResNet18**: Custom ResNet18 architecture built from scratch
- **Block Class Implementation**: Block class implementation for ResNet structure
- **Early Stopping**: Early stopping callback for training optimization
- **Data Augmentation**: ImageDataGenerator for on-the-fly data augmentation
- **Visualization**: Plots for accuracy and loss

## Model Architectures

### PyTorch ResNet18
The PyTorch implementation uses a ResNet18 model pre-trained on ImageNet, with the final fully connected layer modified to output 10 classes (for CIFAR-10 categories) instead of the original 1000 ImageNet classes.

### TensorFlow Custom ResNet18
The TensorFlow implementation features a custom-built ResNet18 architecture with:
- Custom Block class implementing residual connections
- Initial 7x7 convolution followed by batch normalization
- Four stages of residual blocks with increasing channel dimensions (64, 128, 256, 512)
- Global average pooling and fully connected output layer

## Requirements

- Python 3.6+
- PyTorch 1.7+ (for PyTorch implementation)
- TensorFlow 2.x and Keras (for TensorFlow implementation)
- NumPy
- Matplotlib
- scikit-learn
- seaborn (for PyTorch visualization)
- opencv-python
- torch_lr_finder (for PyTorch implementation)

## Usage

### PyTorch Implementation

To run the PyTorch implementation:

1. Open the Jupyter notebook:
```
jupyter notebook cifar10_pytorch.ipynb
```

2. Run all cells in the notebook to train the model or load the pre-trained weights

#### Using Pre-trained PyTorch Weights

The notebook contains code to load the pre-trained weights:

```python
import torch
import torchvision.models as models

# Define model architecture
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 10)  # 10 classes for CIFAR-10

# Load pre-trained weights
model.load_state_dict(torch.load('models/resnet18_cifar10.pth'))
model.eval()  # Set to evaluation mode

# Now you can use the model for inference
```

### TensorFlow Implementation

To run the TensorFlow implementation:

1. Open the Jupyter notebook:
```
jupyter notebook cifar10_tensorflow.ipynb
```

2. Run all cells in the notebook to train the model or load the pre-trained model

#### Using Pre-trained TensorFlow Model

The notebook contains code to load the pre-trained model:

```python
import tensorflow as tf

# Load the saved model
model = tf.keras.models.load_model('models/resnet18')

# Use the model for inference
# Example:
predictions = model.predict(images)
```

## Hyperparameters 

### PyTorch Implementation

- **Architecture**: ResNet18 (pre-trained on ImageNet)
- **Optimizer**: Adam with learning rate 5.59E-04
- **Loss Function**: CrossEntropyLoss
- **Batch Size**: 256
- **Epochs**: 20
- **Learning Rate Scheduler**: StepLR (step_size=7, gamma=0.1)
- **Data Augmentation**: 
  - RandomResizedCrop(224)
  - RandomHorizontalFlip
  - Normalization ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
- **Input Size**: 224×224×3

### TensorFlow Implementation
- **Architecture**: Custom ResNet18
- **Optimizer**: Adam (default learning rate)
- **Loss Function**: Categorical CrossEntropy
- **Batch Size**: 256
- **Epochs**: 10 (with early stopping)
- **Early Stopping**: Patience=8, monitored on validation accuracy
- **Data Augmentation**:
  - Horizontal flip
  - Width shift range (0.05)
  - Height shift range (0.05)
- **Input Size**: 32×32×3 (native CIFAR-10 resolution)
- **Preprocessing**: Normalization (divide by 255)

## Results

Both implementations achieve competitive accuracy on the CIFAR-10 test set. Detailed results, including accuracy plots, loss plots, and (for PyTorch) confusion matrices, can be found in the `results` directory.

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















## Future Work

- Experiment with different model architectures beyond ResNet18
- Implement more sophisticated data augmentation techniques
- Explore different hyperparameter optimization methods
- Deploy the models as a web service
- Add ensemble methods combining both implementations

## License

© 2025 [Your Name]

**All Rights Reserved**

This project is currently not licensed for public use, modification, or distribution. The code and associated documentation are provided for demonstration and educational purposes only. 

If you're interested in using, modifying, or distributing this code, please contact the repository owner for permission.

*Note: I'm considering adding an open-source license to this project in the future. If you have suggestions or would like to contribute, please open an issue to discuss.*