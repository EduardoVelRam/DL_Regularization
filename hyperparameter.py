import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from itertools import product

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

# Hyperparameter grid

learning_rates = [0.001, 0.01]
batch_sizes = [32, 128]
num_layers_list = [1,2]

results = []

for lr, bs, layers_ in product(learning_rates, batch_sizes, num_layers_list):
    print(f"Training: lr={lr}, batch={bs} layers={layers_}")
    model = build_model(num_layers=layers_, learning_rate=lr)
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=20,
        batch_size=bs,
        verbose=0
    )
    val_acc = max(history.history['val_accuracy'])
    results.append((val_acc, (lr, bs, layers_)))

# Visualize results

results.sort(key=lambda x: x[0], reverse=True)

# bar plot
configs = [f"lr{lr}\nbs={bs}\nlayers={layers_}" for _, (lr, bs, layers_) in results]
accuracies = [acc for acc, _ in results]

plt.figure(figsize=(10, 6))
bars = plt.bar(range(len(configs)), accuracies, color='blue')
plt.xticks(range(len(configs)), configs, rotation=0, ha='center')
plt.ylabel('Validation Accuracy')
plt.title('Hyperparameter Tuning: Impact on validation performance')
plt.ylim(0, 1)
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{acc:.3f}', ha='center', va='bottom')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("hyperparameter_tuning.png", dpi=150)
plt.show()

# Print best configuration
best_acc, (best_lr, best_bs, best_layers) = results[0]
print("\n" + "="*50)
print("BEST HYPERPARAMETERS FOUND")
print("="*50)
print(f"Learning rate: {best_lr}")
print(f"Batch size: {best_bs}")
print(f"Hiden layers: {best_layers}")
print(f"Validation accuracy: {best_acc:.3f}")

# Evaluate on test set with best model

best_model = build_model(num_layers=best_layers, learning_rate=best_lr)
best_model.fit(x_train, y_train, epochs=20, batch_size=best_bs, verbose=0)
test_loss, test_acc = best_model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy with tuned hyperparameters: {test_acc:.3f}")

#