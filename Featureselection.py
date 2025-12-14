import pandas as pd
import numpy as np
from skrebate import ReliefF
import random

# =========================
# Input / Output Files
# =========================
input_csv  = ""          # Input feature file
output_csv = ""       # Output selected features file

# =========================
# Parameters
# =========================
NUM_FEATURES = 10      # Number of features to select
HMS = 10               # Harmony Memory Size
HMCR = 0.9             # Harmony Memory Consideration Rate
PAR = 0.3              # Pitch Adjustment Rate
ITERATIONS = 30        # HSA iterations

# =========================
# Read CSV File
# =========================
df = pd.read_csv(input_csv)

# Assume last column is label
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
feature_names = df.columns[:-1]

# =========================
# ReliefF Feature Ranking
# =========================
relief = ReliefF(n_neighbors=100)
relief.fit(X, y)

relief_scores = relief.feature_importances_

# Rank features (descending)
ranked_indices = np.argsort(relief_scores)[::-1]

# =========================
# Harmony Search Algorithm
# =========================
def initialize_harmony():
    return sorted(random.sample(range(X.shape[1]), NUM_FEATURES))

def fitness(solution):
    return np.sum(relief_scores[solution])

# Initialize Harmony Memory
harmony_memory = [initialize_harmony() for _ in range(HMS)]

# =========================
# HSA Optimization Loop
# =========================
for _ in range(ITERATIONS):
    new_harmony = []

    for _ in range(NUM_FEATURES):
        if random.random() < HMCR:
            feature = random.choice(random.choice(harmony_memory))
            if random.random() < PAR:
                feature = random.choice(ranked_indices[:NUM_FEATURES * 2])
        else:
            feature = random.randint(0, X.shape[1] - 1)

        if feature not in new_harmony:
            new_harmony.append(feature)

    # Ensure correct size
    new_harmony = sorted(new_harmony[:NUM_FEATURES])

    # Replace worst harmony if better
    harmony_memory.sort(key=fitness)
    if fitness(new_harmony) > fitness(harmony_memory[0]):
        harmony_memory[0] = new_harmony

# =========================
# Best Feature Subset
# =========================
best_features_idx = max(harmony_memory, key=fitness)
selected_feature_names = feature_names[best_features_idx]

# =========================
# Save Selected Features
# =========================
selected_df = df[list(selected_feature_names) + [df.columns[-1]]]
selected_df.to_csv(output_csv, index=False)

print("HAS-ReliefF Feature Selection Completed")
print("Selected Features:")
print(selected_feature_names.tolist())
print("Saved to:", output_csv)
