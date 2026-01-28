import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Define a custom layer instead of Lambda for better serialization
class AbsoluteDifference(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, inputs):
        return tf.abs(inputs[0] - inputs[1])
    
    def compute_output_shape(self, input_shape):
        return input_shape[0]
    
    def get_config(self):
        return super().get_config()

print("Loading data...")
# Load user traits
traits_df = pd.read_csv("user_traits.csv")
print(f"Loaded {len(traits_df)} users with traits")

# Load user pair scores
pairs_df = pd.read_csv("user_pair_score.csv")
print(f"Loaded {len(pairs_df)} user pairs with scores")

# Create a dictionary for quick trait lookup
traits_dict = {}
for _, row in traits_df.iterrows():
    user_id = row['id']
    traits = row[['t1', 't2', 't3', 't4', 't5']].values.astype(np.float32)
    traits_dict[user_id] = traits

print("\nPreparing training data...")
# Prepare training data
traits_A = []
traits_B = []
scores = []

for _, row in pairs_df.iterrows():
    user_id_1 = row['user_id_1']
    user_id_2 = row['user_id_2']
    score = row['score']
    
    # Get traits for both users
    if user_id_1 in traits_dict and user_id_2 in traits_dict:
        traits_A.append(traits_dict[user_id_1])
        traits_B.append(traits_dict[user_id_2])
        scores.append(score)

traits_A = np.array(traits_A, dtype=np.float32)
traits_B = np.array(traits_B, dtype=np.float32)
scores = np.array(scores, dtype=np.float32)

print(f"Training data shape: {traits_A.shape}, {traits_B.shape}")
print(f"Scores shape: {scores.shape}")
print(f"Score range: [{scores.min():.2f}, {scores.max():.2f}]")

# Normalize traits to [0, 1] range (they appear to be in 0-100 range)
traits_A = traits_A / 100.0
traits_B = traits_B / 100.0

# Split into train and validation sets
X_train_A, X_val_A, X_train_B, X_val_B, y_train, y_val = train_test_split(
    traits_A, traits_B, scores, test_size=0.2, random_state=42
)

print(f"\nTraining set size: {len(X_train_A)}")
print(f"Validation set size: {len(X_val_A)}")

# Build the model with custom layer instead of Lambda
print("\nBuilding model...")
inpA = keras.Input(shape=(5,), name="traits_A")
inpB = keras.Input(shape=(5,), name="traits_B")

# Use custom layer instead of Lambda
abs_diff = AbsoluteDifference(name="abs_diff")([inpA, inpB])
prod     = layers.Multiply(name="prod")([inpA, inpB])

x = layers.Concatenate(name="x")([abs_diff, prod])  # 10 dims

# Tiny head
x = layers.Dense(32, activation="relu")(x)
x = layers.Dense(16, activation="relu")(x)
out = layers.Dense(1, activation="linear", name="score")(x)

model = keras.Model([inpA, inpB], out)
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="mse",
    metrics=[keras.metrics.MeanAbsoluteError(name="mae")]
)

print("\nModel Summary:")
model.summary()

# Train the model
print("\nTraining model...")
history = model.fit(
    [X_train_A, X_train_B], y_train,
    validation_data=([X_val_A, X_val_B], y_val),
    epochs=80,
    batch_size=64,
    verbose=1,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-11,
            verbose=1
        )
    ]
)

# Evaluate the model
print("\nEvaluating model...")
train_loss, train_mae = model.evaluate([X_train_A, X_train_B], y_train, verbose=0)
val_loss, val_mae = model.evaluate([X_val_A, X_val_B], y_val, verbose=0)

print(f"\nFinal Results:")
print(f"Training - Loss (MSE): {train_loss:.4f}, MAE: {train_mae:.4f}")
print(f"Validation - Loss (MSE): {val_loss:.4f}, MAE: {val_mae:.4f}")

# Save the model
print("\nSaving model...")
model.save("matchmaking_model.keras")
print("Model saved as 'matchmaking_model.keras'")

# Plot training history
print("\nGenerating training plots...")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Plot loss
axes[0].plot(history.history['loss'], label='Training Loss')
axes[0].plot(history.history['val_loss'], label='Validation Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('MSE Loss')
axes[0].set_title('Training and Validation Loss')
axes[0].legend()
axes[0].grid(True)

# Plot MAE
axes[1].plot(history.history['mae'], label='Training MAE')
axes[1].plot(history.history['val_mae'], label='Validation MAE')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Mean Absolute Error')
axes[1].set_title('Training and Validation MAE')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('training_history.png', dpi=150)
print("Training plots saved as 'training_history.png'")

# Test predictions on a few examples
print("\nSample Predictions:")
sample_indices = np.random.choice(len(X_val_A), size=5, replace=False)
sample_A = X_val_A[sample_indices]
sample_B = X_val_B[sample_indices]
sample_true = y_val[sample_indices]
sample_pred = model.predict([sample_A, sample_B], verbose=0).flatten()

for i in range(5):
    print(f"  True: {sample_true[i]:.2f}, Predicted: {sample_pred[i]:.2f}, Error: {abs(sample_true[i] - sample_pred[i]):.2f}")

print("\nTraining complete!")
