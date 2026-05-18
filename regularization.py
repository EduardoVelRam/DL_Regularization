
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

# Small dataset with noise
np.random.seed(2)
X = np.linspace(-3, 3, 50).reshape(-1,1)
y_true = np.sin(X) + 0.1 * X**2
y = y_true + 0.3 * np.random.randn(*X.shape)

# Train / validation
X_train, X_val = X[:40], X[40:]
y_train, y_val = y[:40], y[40:]

# Build two high-capacity models
def create_model(regularization=None, dropout_rate=0.0):
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(1,),kernel_regularization=regularization))
    model.add(layers.Dropout(dropout_rate ))
    model.add(layers.Dense(64, activation='relu', kernel_regularizer=regularization))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(64, activation='relu', kernel_regularizer=regularization))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(1))
    return model

# model without regularization
model_no_reg = create_model(regularization=None)
model_no_reg.compile(optimizer='adam', loss='mse')

#model with L2 regularization (and optional dropout)
model_reg = create_model(
    regularization=regularizers.l2(0.01),
    dropout_rate=0.3
)
model_reg.compile(optimizer='adam', loss='mse')

# Train both models

history_no_reg = model_no_reg.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=500, batch_size=16, verbose=0
)

history_reg = model_reg.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=500, batch_size=16, verbose=0
)

# Plotting
fig, axes = plt.subplots(1, 3, figsize=(15,4))

# Plot training vs validation loss
ax = axes[0]
ax.plot(history_no_reg.history['loss'], label='Train loss (no reg)')
ax.plot(history_no_reg.history['val_loss'], label='Val loss (no reg)')
ax.plot(history_reg.history['loss'], label='Train loss (with reg)')
ax.plot(history_reg.history['val_loss'], label='Val loss (with reg)')
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE')
ax.legend()
ax.set_title('Loss Curves')
ax.grid(True)

# Plot model predictions
X_plot = np.linspace(-3, 3, 200).reshape(-1,1)
y_plot_true = np.sin(X_plot) + 0.1 * X_plot**2
y_plot_no_reg = model_no_reg.predict(X_plot, verbose=0)
y_plot_reg = model_reg.predict(X_plot, verbose=0)

ax = axes[1]
ax.scatter(X_train, y_train, alpha=0.6, label='Training data')
ax.plot(X_plot, y_plot_true, 'k-', label='True function')
ax.plot(X_plot, y_plot_no_reg, 'r--', label='No regularization')
ax.plot(X_plot, y_plot_reg, 'g--', label='With L2 + dropout')
ax.set_xlabel('X')
ax.set_ylabel('y')
ax.legend()
ax.set_title('Model predictions')
ax.grid(True)

# Compare validation errors
val_mse_no_reg = model_no_reg.evaluate(X_val, y_val, verbose=0)
val_mse_reg = model_reg.evaluate(X_val, y_val, verbose=0)

ax = axes[2]
ax.bar(['No regularizattion', 'With L2 + dropout'],
       [val_mse_no_reg, val_mse_reg],
       color=['red', 'green'])
ax.set_ylabel('Validation MSE')
ax.set_title('Generalization Performance')
ax.grid(True, axis='y')

plt.tight_layout()
plt.savefig("regularization_effect.png", dpi=150)
plt.show()

print(f"Validation MSE without regularization: {val_mse_no_reg:.4f}")
print(f"Validation MSE with L2 + dropout: {val_mse_reg:.4f}")

# Writed by my niece:
#  .oo--o-oki9i0oo0oo00ooooo0o0pp+++-o