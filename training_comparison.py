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

# keep test set as is
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
                               kernel_regularizer=regularizers.l2(0.001)))
        model.add(layers.Dropout(dropout_rate))
        model.add(layers.Dense(10, activation='softmax'))
    else:
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(10, activation='softmax'))
    return model

# model A: No regularization, no augmentation
model_a = build_model(regularize=False)
model_a.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])                               

# model B: With L2, dropout, and data augmentation
model_b = build_model(regularize=True, dropout_rate=0.5)
model_b.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])  

# Data augmentation generator (for model B)
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)
datagen.fit(x_train)

# Train both models









#