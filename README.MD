# AlexNet Implementation

This is a PyTorch implementation of AlexNet, based on the seminal paper "ImageNet Classification with Deep Convolutional Neural Networks" by Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton.

## Model Description

AlexNet consists of five convolutional layers, followed by three fully connected layers. It uses ReLU as the activation function and dropout for regularization. The model is designed to process 224x224 RGB images.

Key features of this implementation:
- 5 convolutional layers
- 3 fully connected layers
- ReLU activation functions
- Dropout for regularization
- AdaptiveAvgPool2d to ensure fixed input size to the classifier

## Dataset

This project uses the ImageNet dataset, a large-scale dataset for object recognition consisting of over 1 million images across 1000 classes.

## Requirements

- Python 3.7+
- PyTorch 1.7+
- torchvision
- datasets (Hugging Face)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/quant-eagle/alexnet.git
   cd alexnet
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv alexnet
   source alexnet/bin/activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. To train the model:
   ```
   python train.py
   ```

## Model Architecture

The model architecture is defined in `model.py`. It closely follows the original AlexNet architecture with some minor modifications to work with PyTorch and modern GPUs.

## Training

The model is trained on the ImageNet dataset using the following setup:
- **Optimizer**: SGD with momentum
- **Loss function**: Cross-Entropy Loss
- **Learning rate**: 0.01
- **Momentum**: 0.9
- **Number of epochs**: 90

I use data augmentation techniques including random resized crops and horizontal flips during training.

## Evaluation

While I don't currently have a separate evaluation script, the training process includes validation steps.

After each epoch, the model's performance is evaluated on a validation set, providing accuracy metrics.

In the future, I plan to add a standalone `evaluate.py` script.

This script will:

- Load a trained model
- Run it on a separate test set
- Compute various metrics such as:
    - Top-1 and Top-5 accuracy
    - Confusion matrix
    - Per-class precision and recall
    - F1 score

This will allow for a more comprehensive evaluation of the model's performance post-training.

## Results

TBD when training is completed.

## Future Work

- Implement evaluate.py for comprehensive model evaluation
- Add `infer.py` for running inference on single images
- Experiment with different hyperparameters and data augmentation techniques
- Implement visualization tools for model interpretability

## Acknowledgments

- The original AlexNet paper: Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet classification with deep convolutional neural networks. In Advances in neural information processing systems (pp. 1097-1105).
- PyTorch team for their excellent deep learning framework
- Hugging Face for their datasets and transformers libraries