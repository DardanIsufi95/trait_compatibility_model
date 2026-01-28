import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Custom layer definition (needed for loading the pretrained model)
class AbsoluteDifference(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, inputs):
        return tf.abs(inputs[0] - inputs[1])
    
    def compute_output_shape(self, input_shape):
        return input_shape[0]
    
    def get_config(self):
        return super().get_config()

print("="*70)
print("FINE-TUNING PRETRAINED MODEL")
print("="*70)

# Check if pretrained model exists
if not os.path.exists("matchmaking_model_pretrained.keras"):
    print("\nERROR: Pretrained model not found!")
    print("Please ensure 'matchmaking_model_pretrained.keras' exists.")
    exit(1)

print("\nLoading pretrained model...")
model = keras.models.load_model(
    "matchmaking_model_pretrained.keras",
    custom_objects={'AbsoluteDifference': AbsoluteDifference}
)
print("✓ Model loaded successfully!")

print("\nLoading data...")
# Load user traits
traits_df = pd.read_csv("user_traits.csv")
print(f"✓ Loaded {len(traits_df)} users with traits")

# Load user pair scores
pairs_df = pd.read_csv("user_pair_score.csv")
print(f"✓ Loaded {len(pairs_df)} user pairs with scores")

print("\nPreparing training data...")
# Create a dictionary for quick trait lookup
traits_dict = {}
for _, row in traits_df.iterrows():
    user_id = row['id']
    traits = row[['t1', 't2', 't3', 't4', 't5']].values.astype(np.float32)
    traits_dict[user_id] = traits

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

print(f"✓ Training data shape: {traits_A.shape}, {traits_B.shape}")
print(f"✓ Scores shape: {scores.shape}")
print(f"✓ Score range: [{scores.min():.2f}, {scores.max():.2f}]")

# Normalize traits to [0, 1] range
traits_A = traits_A / 100.0
traits_B = traits_B / 100.0

# Split into train and validation sets
X_train_A, X_val_A, X_train_B, X_val_B, y_train, y_val = train_test_split(
    traits_A, traits_B, scores, test_size=0.2, random_state=42
)

print(f"\n✓ Training set size: {len(X_train_A)}")
print(f"✓ Validation set size: {len(X_val_A)}")

# Evaluate pretrained model performance
print("\n" + "="*70)
print("PRETRAINED MODEL PERFORMANCE")
print("="*70)
pretrain_loss, pretrain_mae = model.evaluate([X_val_A, X_val_B], y_val, verbose=0)
print(f"Validation Loss (MSE): {pretrain_loss:.4f}")
print(f"Validation MAE: {pretrain_mae:.4f}")

# Fine-tuning configuration
print("\n" + "="*70)
print("FINE-TUNING CONFIGURATION")
print("="*70)

# Option 1: Lower learning rate for fine-tuning
LEARNING_RATE = 1e-4  # Much lower than initial training (1e-3)
EPOCHS = 50           # Fewer epochs for fine-tuning
BATCH_SIZE = 64

print(f"Learning Rate: {LEARNING_RATE}")
print(f"Epochs: {EPOCHS}")
print(f"Batch Size: {BATCH_SIZE}")

# Recompile with lower learning rate
model.compile(
    optimizer=keras.optimizers.Adam(LEARNING_RATE),
    loss="mse",
    metrics=[keras.metrics.MeanAbsoluteError(name="mae")]
)

print("\nModel Summary:")
model.summary()

# Fine-tune the model
print("\n" + "="*70)
print("STARTING FINE-TUNING")
print("="*70)

history = model.fit(
    [X_train_A, X_train_B], y_train,
    validation_data=([X_val_A, X_val_B], y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
)

# Evaluate fine-tuned model
print("\n" + "="*70)
print("FINE-TUNED MODEL PERFORMANCE")
print("="*70)
train_loss, train_mae = model.evaluate([X_train_A, X_train_B], y_train, verbose=0)
val_loss, val_mae = model.evaluate([X_val_A, X_val_B], y_val, verbose=0)

print(f"\nTraining - Loss (MSE): {train_loss:.4f}, MAE: {train_mae:.4f}")
print(f"Validation - Loss (MSE): {val_loss:.4f}, MAE: {val_mae:.4f}")

# Show improvement
print("\n" + "="*70)
print("IMPROVEMENT SUMMARY")
print("="*70)
mae_improvement = pretrain_mae - val_mae
improvement_pct = (mae_improvement / pretrain_mae) * 100
print(f"Pretrained MAE: {pretrain_mae:.4f}")
print(f"Fine-tuned MAE: {val_mae:.4f}")
print(f"Improvement: {mae_improvement:.4f} ({improvement_pct:+.2f}%)")

if mae_improvement > 0:
    print("✓ Model improved!")
else:
    print("⚠ Model did not improve (this is normal if already well-trained)")

# Save the fine-tuned model
print("\n" + "="*70)
print("SAVING FINE-TUNED MODEL")
print("="*70)
model.save("matchmaking_model_pretrained_finetuned.keras")
print("✓ Fine-tuned model saved as 'matchmaking_model_pretrained_finetuned.keras'")

# Plot training history
print("\nGenerating fine-tuning plots...")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Plot loss
axes[0].plot(history.history['loss'], label='Training Loss')
axes[0].plot(history.history['val_loss'], label='Validation Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('MSE Loss')
axes[0].set_title('Fine-Tuning: Loss')
axes[0].legend()
axes[0].grid(True)

# Plot MAE
axes[1].plot(history.history['mae'], label='Training MAE')
axes[1].plot(history.history['val_mae'], label='Validation MAE')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Mean Absolute Error')
axes[1].set_title('Fine-Tuning: MAE')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('training_history.png', dpi=150)
print("✓ Fine-tuning plots saved as 'training_history_pretrained_finetuned.png'")

# Test predictions on a few examples
print("\n" + "="*70)
print("SAMPLE PREDICTIONS (Fine-tuned Model)")
print("="*70)
sample_indices = np.random.choice(len(X_val_A), size=5, replace=False)
sample_A = X_val_A[sample_indices]
sample_B = X_val_B[sample_indices]
sample_true = y_val[sample_indices]
sample_pred = model.predict([sample_A, sample_B], verbose=0).flatten()

for i in range(5):
    print(f"  True: {sample_true[i]:.2f}, Predicted: {sample_pred[i]:.2f}, Error: {abs(sample_true[i] - sample_pred[i]):.2f}")

print("\n" + "="*70)
print("FINE-TUNING COMPLETE!")
print("="*70)
print("\nNext steps:")
print("1. Use 'matchmaking_model_pretrained_finetuned.keras' for predictions (improved model)")
print("2. Run 'python predict.py' to test the fine-tuned model")
print("3. Compare 'training_history_pretrained_finetuned.png' with 'training_history_pretrained.png'")
