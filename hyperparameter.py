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









#