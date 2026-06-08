# Neural Network Optimization and Generalization Experiments

This repository contains three TensorFlow/Keras experiments focused on improving neural network performance through hyperparameter tuning, regularization, and data augmentation.

## Contents

### 1. Hyperparameter Tuning (Fashion-MNIST)
A dense neural network is trained on the Fashion-MNIST dataset using different combinations of:
- Learning rate
- Batch size
- Number of hidden layers

The script evaluates each configuration, visualizes validation accuracy, identifies the best hyperparameters, and measures final performance on the test set.

**Output:** `hyperparameter_tuning.png`

---

### 2. Regularization and Overfitting
This experiment demonstrates the impact of L2 regularization and Dropout on a regression problem with limited and noisy data.

Two high-capacity neural networks are compared:
- Without regularization
- With L2 regularization and Dropout

Training curves, predictions, and validation errors are visualized to illustrate how regularization improves generalization.

**Output:** `regularization_effect.png`

---

### 3. Data Augmentation and CNN Generalization (CIFAR-10)
A convolutional neural network is trained on a reduced CIFAR-10 dataset to emphasize overfitting.

Two approaches are compared:
- CNN without regularization or augmentation
- CNN with L2 regularization, Dropout, and image augmentation

The experiment evaluates training behavior and final test accuracy to simulate real-world deployment performance.

**Output:** `real_world_training_comparison.png`

---

## Requirements

Install the required dependencies:

```bash
pip install tensorflow numpy matplotlib
```

---

## Concepts Explored

- Hyperparameter optimization
- Learning rate selection
- Batch size effects
- Network architecture tuning
- Overfitting and underfitting
- L2 regularization
- Dropout
- Data augmentation
- Model generalization
- Convolutional Neural Networks (CNNs)

---

## Educational Purpose

These scripts are designed as practical demonstrations of fundamental deep learning optimization techniques and provide visual comparisons that help understand how model design choices affect performance and generalization.
