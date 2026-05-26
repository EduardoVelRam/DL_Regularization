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
    
