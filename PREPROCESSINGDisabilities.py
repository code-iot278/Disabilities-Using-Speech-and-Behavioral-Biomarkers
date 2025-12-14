import numpy as np
import scipy.io.wavfile as wav
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

# =========================
# Load Speech Signal
# =========================
fs, signal = wav.read("")

# Convert to float if needed
signal = signal.astype(np.float64)

# =========================
# Enhanced Gaussian-Based Noise Filtering
# =========================
def enhanced_gaussian_filter(signal, sigma=1.2):
    """
    Enhanced Gaussian-Based Noise Filtering
    Args:
        signal: input speech signal
        sigma : standard deviation of Gaussian kernel
    Returns:
        filtered speech signal
    """
    # Mean removal (enhancement step)
    signal_mean = np.mean(signal)
    signal_centered = signal - signal_mean

    # Gaussian smoothing
    filtered_signal = gaussian_filter1d(signal_centered, sigma=sigma)

    # Restore mean
    enhanced_signal = filtered_signal + signal_mean

    return enhanced_signal

# Apply filtering
filtered_signal = enhanced_gaussian_filter(signal, sigma=1.2)

# =========================
# Save Filtered Output
# =========================
wav.write("", fs, filtered_signal.astype(np.int16))

# =========================
# Visualization (Optional)
# =========================
plt.figure(figsize=(12, 4))
plt.plot(signal[:5000], label="Original Speech", alpha=0.6)
plt.plot(filtered_signal[:5000], label="Filtered Speech", linewidth=2)
plt.legend()
plt.title("Enhanced Gaussian-Based Noise Filtering")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.show()
import pandas as pd
import numpy as np

# =========================
# Input / Output Files
# =========================
input_csv  = ""      # <-- your input file
output_csv = ""   # <-- normalized output

# =========================
# Read Input CSV
# =========================
df = pd.read_csv(input_csv)

# If first column is label, separate it
labels = None
if not np.issubdtype(df.iloc[:, 0].dtype, np.number):
    labels = df.iloc[:, 0]
    features = df.iloc[:, 1:]
else:
    features = df

# =========================
# Z-Score Normalization
# =========================
mean = features.mean(axis=0)
std  = features.std(axis=0) + 1e-8  # avoid division by zero

normalized_features = (features - mean) / std

# =========================
# Combine Labels (if any)
# =========================
if labels is not None:
    normalized_df = pd.concat([labels, normalized_features], axis=1)
else:
    normalized_df = normalized_features

# =========================
# Save Output CSV
# =========================
normalized_df.to_csv(output_csv, index=False)

print("Z-score normalization completed.")
print("Saved to:", output_csv)
