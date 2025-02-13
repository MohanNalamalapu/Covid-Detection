# COVID-19 Detection using Deep Learning

This repository contains a deep learning model for detecting COVID-19 from chest X-ray images. The model is built using TensorFlow and Keras, leveraging the VGG16 architecture for image classification. The goal of this project is to classify chest X-ray images into two categories: **COVID** and **bacterial_viral**.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [License](#license)

## Project Overview
The project aims to build a deep learning model that can classify chest X-ray images into two categories:
- **COVID**: Images of patients infected with COVID-19.
- **bacterial_viral**: Images of patients with bacterial or viral pneumonia.

The model is based on the **VGG16** architecture, a pre-trained convolutional neural network (CNN) fine-tuned for this specific task.

## Dataset
The dataset used in this project is divided into two main directories:
- **Train**: Contains **175 images** belonging to 2 classes (COVID and bacterial_viral).
- **Test**: Contains **50 images** belonging to 2 classes (COVID and bacterial_viral).

The dataset is preprocessed using **ImageDataGenerator** from Keras, which applies various transformations such as rotation, shifting, and flipping to augment the data and improve model generalization.

## Model Architecture
The model is built using the following architecture:
- **VGG16 Base Model**: Used as a feature extractor with the top layers excluded.
- **Fine-Tuning**: The last four layers of the VGG16 model are fine-tuned, while the rest of the layers are frozen.
- **Additional Layers**:
  - Fully connected layer with **64 units** and **ReLU** activation.
  - Softmax layer for binary classification.

## Training
The model is trained using the following parameters:
- **Image Size**: 150x150 pixels
- **Batch Size**: 10
- **Epochs**: 100
- **Learning Rate**: 0.0001
- **Optimizer**: Adam

The training process uses data augmentation techniques to improve the model's ability to generalize to unseen data.

## Evaluation
The model is evaluated on both the training and test datasets using the following metrics:
- **Loss**: Categorical cross-entropy
- **Accuracy**: Classification accuracy

### Training Results:
- **Train Loss**: 0.1130
- **Train Accuracy**: 95.99%

### Test Results:
- **Test Loss**: 0.1060
- **Test Accuracy**: 95.99%

### Confusion Matrix
#### Training Data:
```
[[72  3]
 [ 4 96]]
```
#### Test Data:
```
[[23  2]
 [ 0 25]]
```

## Usage
To use this project, follow these steps:

### Clone the Repository
```bash
git clone https://github.com/your-username/covid-detection.git
cd covid-detection
```

### Install Dependencies
Ensure you have the required dependencies installed. You can install them using:
```bash
pip install -r requirements.txt
```

### Run the Jupyter Notebook
Open the `covidDetection.ipynb` notebook and run the cells to train and evaluate the model.

### Modify Parameters
You can modify hyperparameters such as batch size, learning rate, and number of epochs in the notebook to experiment with different configurations.

## Dependencies
The project requires the following Python libraries:
- TensorFlow
- Keras
- NumPy
- scikit-learn

You can install the dependencies using:
```bash
pip install tensorflow numpy scikit-learn
```


