import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import sys

# Define the custom layer (must match the one used in training)
class AbsoluteDifference(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, inputs):
        return tf.abs(inputs[0] - inputs[1])
    
    def compute_output_shape(self, input_shape):
        return input_shape[0]
    
    def get_config(self):
        return super().get_config()

# Load the trained model
print("Loading trained model...")
model = tf.keras.models.load_model(
    "matchmaking_model.keras",
    custom_objects={'AbsoluteDifference': AbsoluteDifference}
)
print("Model loaded successfully!")

# Load user traits for easy lookup
print("\nLoading user traits database...")
traits_df = pd.read_csv("user_traits.csv")
traits_dict = {}
for _, row in traits_df.iterrows():
    user_id = row['id']
    traits = row[['t1', 't2', 't3', 't4', 't5']].values.astype(np.float32) / 100.0
    traits_dict[user_id] = traits

print(f"Loaded traits for {len(traits_dict)} users")

def predict_compatibility(user_id_1, user_id_2):
    """Predict compatibility score between two users."""
    if user_id_1 not in traits_dict:
        print(f"Error: User {user_id_1} not found in database")
        return None
    if user_id_2 not in traits_dict:
        print(f"Error: User {user_id_2} not found in database")
        return None
    
    traits_1 = traits_dict[user_id_1].reshape(1, -1)
    traits_2 = traits_dict[user_id_2].reshape(1, -1)
    
    score = model.predict([traits_1, traits_2], verbose=0)[0][0]
    return score

def evaluate_model_samples(num_samples=20):
    """Evaluate model on random sample pairs from the actual pair data."""
    print(f"\nEvaluating model on {num_samples} random pairs from the dataset...")
    
    # Load the actual pair scores for comparison
    pairs_df = pd.read_csv("user_pair_score.csv")
    
    # Sample random pairs
    sample_pairs = pairs_df.sample(n=num_samples, random_state=42)
    
    results = []
    for _, row in sample_pairs.iterrows():
        user_id_1 = row['user_id_1']
        user_id_2 = row['user_id_2']
        actual_score = row['score']
        
        if user_id_1 in traits_dict and user_id_2 in traits_dict:
            traits_1 = traits_dict[user_id_1].reshape(1, -1)
            traits_2 = traits_dict[user_id_2].reshape(1, -1)
            
            predicted_score = model.predict([traits_1, traits_2], verbose=0)[0][0]
            error = abs(actual_score - predicted_score)
            
            results.append({
                'user_1': user_id_1,
                'user_2': user_id_2,
                'actual': actual_score,
                'predicted': predicted_score,
                'error': error
            })
    
    return results

# Main execution
if __name__ == "__main__":
    print("\n" + "="*60)
    print("MATCHMAKING MODEL - PREDICTION TOOL")
    print("="*60)
    
    # Example 1: Predict compatibility between two specific users
    print("\n--- Example 1: Predict compatibility between two users ---")
    user_1 = 194
    user_2 = 240
    
    score = predict_compatibility(user_1, user_2)
    if score is not None:
        print(f"Compatibility score between User {user_1} and User {user_2}: {score:.2f}")
    
    # Example 2: Evaluate model on sample pairs
    print("\n--- Example 2: Evaluate model on sample pairs ---")
    
    results = evaluate_model_samples(num_samples=20)
    
    if results:
        print(f"\nModel Evaluation Results:")
        print("-" * 70)
        print(f"{'User 1':>8} {'User 2':>8} {'Actual':>10} {'Predicted':>10} {'Error':>10}")
        print("-" * 70)
        
        total_error = 0
        for r in results:
            print(f"{r['user_1']:>8} {r['user_2']:>8} {r['actual']:>10.2f} {r['predicted']:>10.2f} {r['error']:>10.2f}")
            total_error += r['error']
        
        avg_error = total_error / len(results)
        print("-" * 70)
        print(f"Average Error (MAE): {avg_error:.2f}")
        print(f"This means on average, predictions are off by ~{avg_error:.1f} points")
    
    # Interactive mode
    print("\n" + "="*60)
    print("INTERACTIVE MODE")
    print("="*60)
    print("\nCommands:")
    print("  1. predict <user_id_1> <user_id_2> - Get compatibility score")
    print("  2. evaluate [num_samples]          - Evaluate model on random pairs (default: 20)")
    print("  3. quit                             - Exit")
    
    while True:
        try:
            print("\n" + "-"*60)
            command = input("Enter command: ").strip().lower()
            
            if command == "quit" or command == "q" or command == "exit":
                print("Goodbye!")
                break
            
            parts = command.split()
            
            if len(parts) == 0:
                continue
            
            if parts[0] == "predict" and len(parts) == 3:
                try:
                    uid1 = int(parts[1])
                    uid2 = int(parts[2])
                    score = predict_compatibility(uid1, uid2)
                    if score is not None:
                        print(f"\nCompatibility score: {score:.2f}")
                except ValueError:
                    print("Error: User IDs must be integers")
            
            elif parts[0] == "evaluate" or parts[0] == "eval":
                try:
                    num_samples = int(parts[1]) if len(parts) > 1 else 20
                    results = evaluate_model_samples(num_samples)
                    if results:
                        print(f"\nModel Evaluation ({num_samples} samples):")
                        print("-" * 70)
                        total_error = sum(r['error'] for r in results)
                        avg_error = total_error / len(results)
                        print(f"Average Error (MAE): {avg_error:.2f}")
                        print(f"\nShowing first 10 predictions:")
                        print(f"{'User 1':>8} {'User 2':>8} {'Actual':>10} {'Predicted':>10} {'Error':>10}")
                        print("-" * 70)
                        for r in results[:10]:
                            print(f"{r['user_1']:>8} {r['user_2']:>8} {r['actual']:>10.2f} {r['predicted']:>10.2f} {r['error']:>10.2f}")
                except ValueError:
                    print("Error: num_samples must be an integer")
                except Exception as e:
                    print(f"Error: {e}")
            
            else:
                print("Invalid command. Use 'predict', 'matches', or 'quit'")
        
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
