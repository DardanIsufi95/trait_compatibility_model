# Trait Compatibility Model

A neural network model for predicting compatibility scores between user pairs based on their trait profiles.

## 📊 Data Source

### Database: `user_matchmaking` (MySQL/MariaDB)

The data comes from a MySQL database with two main tables:

#### 1. **user_traits** Table
- **Size**: ~4,000 users
- **Generation**: Created using **Navicat Data Generator**
- **Schema**:
  ```sql
  CREATE TABLE user_traits (
    id INT PRIMARY KEY AUTO_INCREMENT,
    t1 FLOAT,  -- Trait 1 (0-100 scale)
    t2 FLOAT,  -- Trait 2 (0-100 scale)
    t3 FLOAT,  -- Trait 3 (0-100 scale)
    t4 FLOAT,  -- Trait 4 (0-100 scale)
    t5 FLOAT   -- Trait 5 (0-100 scale)
  );
  ```
- **Traits**: Each user has 5 personality/compatibility traits randomly generated in the range [0, 100]

#### 2. **user_pair_score** Table
- **Size**: ~100,000 user pairs
- **Generation**: Created using the `generate_scores()` stored procedure
- **Schema**:
  ```sql
  CREATE TABLE user_pair_score (
    user_id_1 INT,
    user_id_2 INT,
    score FLOAT,
    PRIMARY KEY (user_id_1, user_id_2)
  );
  ```

### 🔧 Score Generation Function: `match_score_traits()`

The compatibility scores were generated using a sophisticated SQL function that implements a non-linear matchmaking algorithm:

#### **Key Features:**

1. **Symmetry**: `score(A, B) = score(B, A)` - Uses lexicographic ordering to ensure consistency

2. **Closeness Calculation**: 
   - Measures similarity for each trait: `closeness = 1 - (|trait_A - trait_B| / 100)`
   - Range: [0, 1] where 1 = identical, 0 = maximum difference

3. **Non-Linear Aggregation**:
   - **Geometric Mean** (70% weight): Punishes mismatches severely
     ```
     geo = (c1 × c2 × c3 × c4 × c5)^(1/5)
     ```
   - **Synergy Score** (30% weight): Amplifies specific trait combinations
     ```
     synergy = 0.6 × (c2 × c4)^0.65 + 0.4 × (c1 × c5)^0.65
     ```

4. **Dealbreaker Penalty**: 
   - Applies smooth sigmoid penalty if any single trait is very mismatched (< 0.30)
   - Uses logistic function to create sharp drop-off for incompatible pairs

5. **Final Mapping**: 
   - Maps raw score [0, 1] to output range [0, 100] using logistic curve
   - Creates natural spread with ability to hit both high and low scores

#### **Algorithm Summary:**
```
raw_score = 0.7 × geometric_mean + 0.3 × synergy_score
raw_score = raw_score × (1 - 0.5 × dealbreaker_penalty)
final_score = 100 / (1 + exp(-11 × (raw_score - 0.43)))
```

This creates a realistic matchmaking score that:
- ✅ Rewards trait similarity
- ✅ Amplifies synergistic trait combinations
- ✅ Penalizes severe mismatches
- ✅ Produces natural score distribution

## 🧠 Model Architecture

The neural network learns to approximate the complex `match_score_traits()` function:

- **Inputs**: Two sets of 5 traits each (traits_A and traits_B)
- **Feature Engineering**: 
  - Absolute difference: `|traits_A - traits_B|`
  - Element-wise product: `traits_A × traits_B`
- **Network**: Dense layers (32 → 16 → 1) with ReLU activations
- **Output**: A single compatibility score [0, 100] (linear activation)
- **Total Parameters**: 897 (3.50 KB) - Very lightweight!

## 📁 Data Format

### user_traits.csv
```csv
id,t1,t2,t3,t4,t5
1,19.42,38.11,82.63,79.86,90.97
2,21.09,23.00,78.26,3.11,15.86
...
```

### user_pair_score.csv
```csv
user_id_1,user_id_2,score
194,210,58.2682
194,216,29.1494
...
```

## 🚀 Setup

### 1. Create Virtual Environment 
```bash
py -m venv venv
```

### 2. Activate Virtual Environment
```bash
# Git Bash / WSL
source venv/Scripts/activate

# Command Prompt
venv\Scripts\activate.bat

# PowerShell
venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

**Dependencies:**
- TensorFlow >= 2.15.0
- Pandas >= 2.0.0
- NumPy >= 1.24.0
- Scikit-learn >= 1.3.0
- Matplotlib >= 3.7.0

## 🎯 Training the Model

### Run Training Script
```bash
python train_model.py
```

### What It Does:
1. ✅ Loads ~4,000 users and ~100,000 pair scores from CSV files
2. ✅ Normalizes traits to [0, 1] range
3. ✅ Splits data: 80% training, 20% validation
4. ✅ Builds the dual-input neural network
5. ✅ Trains with:
   - **Epochs**: 80 (with early stopping)
   - **Batch size**: 64
   - **Optimizer**: Adam (learning rate: 1e-3)
   - **Loss**: MSE (Mean Squared Error)
   - **Callbacks**: Early stopping (patience=15), Learning rate reduction
6. ✅ Saves trained model as `matchmaking_model.keras`
7. ✅ Generates training plots in `training_history.png`
8. ✅ Displays evaluation metrics and sample predictions

### Expected Training Time:
- **CPU**: ~2-5 minutes (80 epochs)
- **GPU**: ~2-3 minutes

### Expected Performance:
- **Training MAE**: ~2.05
- **Validation MAE**: ~2.07
- **Meaning**: Model predicts scores with average error of ±2 points on 0-100 scale

### Training Output Files:
- `matchmaking_model.keras` - Trained model (can be loaded with TensorFlow/Keras)
- `training_history.png` - Plots showing loss and MAE over epochs

## 🔮 Using the Model for Predictions

### Run Prediction Script
```bash
python predict.py
```

### What It Does:
1. ✅ Loads the trained model
2. ✅ Loads user traits database for quick lookup
3. ✅ Runs two examples:
   - **Example 1**: Predicts compatibility between two specific users
   - **Example 2**: Evaluates model on 20 random pairs from the dataset
4. ✅ Enters interactive mode for manual predictions

### Interactive Commands:
```
predict <user_id_1> <user_id_2>  - Get compatibility score between two users
evaluate [num_samples]            - Evaluate model on random pairs (default: 20)
quit                              - Exit
```

### Example Session:
```
============================================================
MATCHMAKING MODEL - PREDICTION TOOL
============================================================

--- Example 1: Predict compatibility between two users ---
Compatibility score between User 194 and User 240: 77.65

--- Example 2: Evaluate model on sample pairs ---

Model Evaluation Results:
----------------------------------------------------------------------
  User 1   User 2     Actual  Predicted      Error
----------------------------------------------------------------------
     194      240      77.62      76.58       1.04
     210      320      45.23      47.15       1.92
     ...
----------------------------------------------------------------------
Average Error (MAE): 2.07
This means on average, predictions are off by ~2.1 points

Enter command: predict 100 200
Compatibility score: 65.43

Enter command: evaluate 50
Average Error (MAE): 2.13

Enter command: quit
Goodbye!
```

## 📈 Model Performance

Based on 100K training samples:
- **MSE Loss**: 8.44 (validation)
- **MAE**: 2.07 (validation)
- **R² Score**: ~0.95 (excellent fit)
- **Generalization**: Very good - training and validation metrics are close

The model successfully learns the complex non-linear patterns from the `match_score_traits()` SQL function!

## 🔧 Programmatic Usage

```python
import tensorflow as tf
import numpy as np

# Custom layer definition (required for model loading)
class AbsoluteDifference(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.abs(inputs[0] - inputs[1])
    def get_config(self):
        return super().get_config()

# Load the model
model = tf.keras.models.load_model(
    "matchmaking_model.keras",
    custom_objects={'AbsoluteDifference': AbsoluteDifference}
)

# Prepare user traits (normalized to 0-1 range)
user_A_traits = np.array([[0.19, 0.38, 0.83, 0.80, 0.91]])
user_B_traits = np.array([[0.21, 0.23, 0.78, 0.03, 0.16]])

# Predict compatibility score
score = model.predict([user_A_traits, user_B_traits])
print(f"Compatibility score: {score[0][0]:.2f}")
```

## 📚 Files Overview

```
trait_compatibility_model/
├── user_traits.csv                      # 4K users with 5 traits each
├── user_pair_score.csv                  # 100K user pairs with compatibility scores
├── user_matchmaking.sql                 # Database schema and scoring function
├── train_model.py                       # Training script
├── predict.py                           # Prediction and evaluation script
├── requirements.txt                     # Python dependencies
├── .gitignore                           # Git ignore file
├── README.md                            # This file
├── matchmaking_model_pretrained.keras   # Pre-trained model (ready to use!)
└── training_history_pretrained.png      # Training plots from pre-trained model

# Local files (not in repo - see .gitignore):
└── venv/                                # Virtual environment (created locally)
```

## 🎓 Technical Notes

### Why This Architecture Works:
1. **Feature Engineering**: The combination of absolute difference and element-wise product captures both similarity and interaction patterns
2. **Dual Input**: Processes both users' traits simultaneously, learning relationship patterns
3. **Compact Network**: Only 897 parameters prevents overfitting on 100K samples
4. **Non-linear Learning**: ReLU activations and multiple layers approximate the complex SQL function

### Potential Improvements:
- Add more traits (currently 5)
- Experiment with attention mechanisms
- Try ensemble methods
- Incorporate user feedback for online learning
