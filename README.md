# MNIST Digit Recognizer

This project demonstrates the use of Convolutional Neural Networks (CNNs) to classify handwritten digits (0-9) using the MNIST dataset. The model is built using Keras with TensorFlow backend and trained on the famous MNIST dataset available on [Kaggle](https://www.kaggle.com/competitions/digit-recognizer).

## Technologies

* [x] **Python:** Core programming language used for data processing and model training.
* [x] **Keras:** High-level neural networks API, used to build the CNN model.
* [x] **TensorFlow:** Backend engine for Keras, used for training the model.
* [x] **NumPy:** Library for numerical computations and data manipulation.
* [x] **Pandas:** Library for numerical computations and data manipulation.


## Dataset

- **Training Set**: 42,000 images of 28x28 grayscale pixels with labels (digits 0-9).
- **Test Set**: 28,000 images of 28x28 grayscale pixels without labels.
- **Source**: [Kaggle MNIST Digit Recognizer](https://www.kaggle.com/competitions/digit-recognizer).

## Project Structure

- **data/**: Contains the training and test data (CSV files with pixel values).
- **notebooks/**: Jupyter notebooks with model training and evaluation steps.
- **README.md**: Project documentation (this file).

## Model Architecture
The core of the model uses a combination of residual blocks and convolutional layers. The architecture is designed to capture both high-level and low-level features through a series of convolutional layers followed by residual connections. The key components are:
- **Initial Convolutional Layers**: Capture early-stage features.
- **Parallel Branches**
  - **Standard Convolutional Branch**: This branch consists of additional convolutional layers, expanding the feature map with higher-dimensional filters (64 filters) and progressively capturing more complex features. Batch normalization and ReLU activation are applied after each layer to maintain efficient learning, followed by a dropout layer for regularization.
  - **Residual Blocks**: A parallel residual block applies two convolutional layers with skip connections, which help avoid the vanishing gradient problem and allow the model to learn deeper representations. The residual connection allows information to bypass these layers, leading to more stable and efficient training. After the residual block, another convolution layer down-samples the feature map with strides.
- **Global Average Pooling**: Used before the fully connected layers to reduce overfitting.
- **Fully Connected Layers**: Final classification layers.
  
### Model:
![image](https://github.com/user-attachments/assets/4b419529-fc61-4946-b11d-f0c50a6c2871)


## Data Preprocessing
To optimize the input images for the model, a custom preprocessing function is applied. The goal is to enhance the clarity of the digit in each image by removing background noise, centering the digit, and normalizing pixel values. This ensures that the model receives consistently processed images for better generalization and improved accuracy.

1. Conversion to 2D Array
2. Otsu's Thresholding
3. Bounding Box Calculation
4. Centering the Digit
5. Padding
6. Normalization

### Before Preprocessing:
![image](https://github.com/user-attachments/assets/3803e4d9-4d38-4e7c-84d4-c47ccb97471d)
### After Preprocessing:
![image](https://github.com/user-attachments/assets/6576cddc-393c-4917-864a-2838e326c478)

## Data Augmentation Parameters:

- **zoom_range**: 0.2
- **width_shift_range**: 0.2
- **height_shift_range**: 0.2
- **shear_range**: 0.2

## Results

| Metric                     | Value    |
|----------------------------|----------|
| **Training Accuracy**      | 99.49%   |
| **Training Loss**          | 0.0156   |
| **Validation Accuracy**    | 99.54%   |
| **Validation Loss**        | 0.0134   |
| **Submisson Accuracy**     | 99.60%   |

### Prerequisites

- Python 3.8+
- Kaggle account (for dataset access)
- Jupyter Notebook or Google Colab


