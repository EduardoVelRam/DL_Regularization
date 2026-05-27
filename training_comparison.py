import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load and prepare CIFAR-10 
(x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Use only 2000 training samples to make overfitting more apparent
x_train = x_train_full[:2000] / 255.0
y_train = y_train_full[:2000]
x_val = x_train_full[2000:3000] / 255.0
y_val = y_train_full[2000:3000]

# keep test set ad is
x_test = x_test / 255.0

# Build two models

def build_model(regularize=False, dropout_rate=0.0):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,323)))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(128, (3,3) activation='relu'))
    model.add(layers.Flatten())

    if regularize:
        # L2 regularization on the dense layer
        model.add(layers.Dense(256, activation='relu',
                               ))
    