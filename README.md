# Vision AI Fundamentals: Building Image Classifiers from Scratch

This repository contains comprehensive implementations of deep learning models for image classification tasks, demonstrating the fundamentals of computer vision and neural network architectures. The project explores how model complexity impacts performance across different datasets and challenges.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Contents](#contents)
- [Prerequisites](#prerequisites)
- [Project Structure](#project-structure)
- [Key Findings](#key-findings)


## 🎯 Project Overview

This repository implements and compares different deep learning architectures for image classification:

### Main Project: Fashion-MNIST Classification
- **Dataset**: Fashion-MNIST (28×28 grayscale images, 10 clothing classes)
- **Models Implemented**:
  - Basic Artificial Neural Network (ANN)
  - Basic Convolutional Neural Network (CNN)
  - Deeper CNN with regularization techniques
- **Techniques**: Early stopping, model checkpointing, data preprocessing, performance evaluation

### Assignment: CIFAR-100 Classification
- **Dataset**: CIFAR-100 (32×32 color images, 100 object classes)
- **Models Adapted**:
  - Enhanced ANN for color images
  - Advanced CNN with batch normalization and dropout
- **Challenge**: Higher complexity with more classes and color channels

## 📁 Contents

### Notebooks
- `Vision_AI_Fundamentals_Building_a_Digit_Recognizer_from_Scratch.ipynb` - Main project implementing Fashion-MNIST classification
- `Assignment_Solution_CIFAR_100.ipynb` - Complete solution for CIFAR-100 classification assignmen

### Model Weights
- `best_ann_model_weights.weights.h5` - Best ANN model weights (Fashion-MNIST)
- `best_basic_cnn_model_weights.weights.h5` - Best Basic CNN model weights (Fashion-MNIST)
- `best_deeper_cnn_model_weights.weights.h5` - Best Deeper CNN model weights (Fashion-MNIST)

## 🔧 Prerequisites

- Python 3.7+
- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook



## 🏗️ Project Structure

```
vision-ai-fundamentals/
│
├── Vision_AI_Fundamentals_Building_a_Digit_Recognizer_from_Scratch.ipynb
├── Assignment_Solution_CIFAR_100.ipynb
│
├── best_ann_model_weights.weights.h5
├── best_basic_cnn_model_weights.weights.h5
├── best_deeper_cnn_model_weights.weights.h5
│
└── README.md
```

## 📊 Key Findings

### Fashion-MNIST Results
- **Basic CNN** achieved the best performance with highest accuracy and lowest loss
- CNN architectures significantly outperformed ANN on image data
- Deeper CNN with regularization showed mixed results, highlighting the importance of appropriate complexity

### CIFAR-100 Results
- **Enhanced CNN** achieved 52.95% accuracy (383.56% improvement over ANN)
- Demonstrated the power of convolutional architectures for complex image classification
- Showed how model complexity should match dataset complexity

### contact
- **Email** : rohanpaswan001782@gmail.com 

