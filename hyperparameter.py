import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from intertools import product

# Load data
(x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# normalize
x_train_full = x_train_full / 255.0
x_test = x_test / 255.0

# subset for faster training
x_train = x_train_full[:3000]
y_train = y_train_full[:3000]
x_val = x_train_full[3000:4000]
y_val = y_train_full[3000:4000]

# Flatten images for a simple dense network
x_train = x_train.reshape(-1, 784)
x_val = x_val.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

# Define a model builder with variable architecture

def build_model(num_layers, units_per_layer=64, learning_rate=0.001):
    model = models.Sequential()
    # input layer
    model.add(layers.Dense(units_per_layer, activation='relu', input_shape=(784,)))
    # hidden layers
    for _ in range(num_layers - 1):
        model.add(layers.Dense(units_per_layer, activation='relu'))
    # output layer
    model.add(layers.Dense(10, activation='softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


#