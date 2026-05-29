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
    model.add(layers.Conv2D(128, (3,3), activation='relu'))
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
history_a = model_a.fit(
    x_train, y_train,
    validation_data = (x_val, y_val),
    epochs=50, batch_size=64, verbose=0
)

history_b = model_b.fit(
    datagen.flow(x_train, y_train, batch_size=64),
    validation_data=(x_val, y_val),
    steps_per_epoch=len(x_train) // 64,
    epochs=50, verbose=0
)

# Visualize training comparison

fig, axes = plt.subplots(1,2, figsize=(14,5))

# Accuracy over epochs
ax = axes[0]
ax.plot(history_a.history['accuracy'], label='Model A (no reg, no aug) train')
ax.plot(history_a.history['val_accuracy'], label='Model A val')
ax.plot(history_b.history['accuracy'], '--', label='Model B train')
ax.plot(history_b.history['val_accuracy'], '--',label='Model B val')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend()
ax.set_title('Training vs Validation Loss')
ax.grid(True)

plt.tight_layout()
plt.savefig("real_world_training_comparison.png", dpi=150)
plt.show()

# Final evaluation on test set (simulating "deployed" performance)

test_acc_a = model_a.evaluate(x_test, y_test, verbose=0)[1]
test_acc_b = model_b.evaluate(x_test, y_test, verbose=0)[1]

print("\n" + "="*50)
print("REAL-WORLD DEPLOYMENT PERFORMANCE")
print("="*50)
print(f"Model A (overfitted) -> Test accuracy: {test_acc_a:.2%}")
print(f"Model B (regularized+aug) -> Test accuracy: {test_acc_b:.2%}")
print("\nVoiceover summary:")
print(" ''  Models that use these optimization techniques (regularization and augmentation) ")
print(" typically perform better when deployed in real-world applications.'' ")      

#