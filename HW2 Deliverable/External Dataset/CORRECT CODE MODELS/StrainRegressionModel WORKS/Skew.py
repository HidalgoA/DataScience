import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------
csv_folder = r"C:\Users\hidal\Desktop\Classes\GRAD\Data Sci Mech of Mat\HW2 Deliverable\External Dataset\TEST\All data_Stress"
TARGET_LENGTH = 241  # Fixed length for resampling

# ---------------------------------------------------------
# 2. Helper Functions
# ---------------------------------------------------------
def resample_to_fixed_length(x, target_length):
    """Resamples 1D array x to target_length points using linear interpolation."""
    original_length = len(x)
    if original_length == target_length:
        return x
    original_idx = np.linspace(0, 1, original_length)
    target_idx = np.linspace(0, 1, target_length)
    return np.interp(target_idx, original_idx, x)

def flatten_csv(csv_path, target_length=241):
    """
    Reads a CSV file (assumed 2 columns), resamples each column to target_length,
    and flattens the result into a 1D array of length 2*target_length without any transformation.
    """
    df_csv = pd.read_csv(csv_path, header=None)
    col1 = df_csv.iloc[:, 0].values
    col2 = df_csv.iloc[:, 1].values

    col1_resampled = resample_to_fixed_length(col1, target_length)
    col2_resampled = resample_to_fixed_length(col2, target_length)

    combined = np.concatenate([col1_resampled, col2_resampled])
    return combined

# ---------------------------------------------------------
# 3. Process Each CSV File and Compute Skewness on Raw Data
# ---------------------------------------------------------
skew_results = []
csv_files = [f for f in os.listdir(csv_folder) if f.endswith(".csv")]

for file in csv_files:
    file_path = os.path.join(csv_folder, file)
    try:
        raw_vec = flatten_csv(file_path, TARGET_LENGTH)
        
        # Compute skewness of the raw data
        skew_value = pd.Series(raw_vec).skew()
        
        # Store results
        skew_results.append({"File": file, "Skewness": skew_value})

    except Exception as e:
        print(f"Error processing {file}: {e}")

# Convert results into a DataFrame
df_skew = pd.DataFrame(skew_results)

# Display the first few rows of skewness analysis
print("Skewness Analysis Summary (Raw Data):")
print(df_skew.head())

# Optionally, save the summary to a CSV file
df_skew.to_csv("skewness_summary_raw.csv", index=False)
print("Skewness analysis saved to 'skewness_summary_raw.csv'.")

# ---------------------------------------------------------
# 4. Identify Highly Skewed Files
# ---------------------------------------------------------
df_high_skew = df_skew[abs(df_skew["Skewness"]) > 1]
print("Files with high skew (|skewness| > 1):")
print(df_high_skew)

# ---------------------------------------------------------
# 5. Visualization: Distribution of Skewness Across CSV Files
# ---------------------------------------------------------



fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: Full range
sns.histplot(df_skew["Skewness"], kde=True, bins=30, ax=axes[0])
axes[0].axvline(x=1, color="r", linestyle="--")
axes[0].axvline(x=-1, color="r", linestyle="--")
axes[0].set_title("Full Range of Skewness")
axes[0].set_xlabel("Skewness")

# Right: Zoomed range
sns.histplot(df_skew["Skewness"], kde=True, bins=30, ax=axes[1])
axes[1].set_xlim(-1, 1)
axes[1].axvline(x=1, color="r", linestyle="--")
axes[1].axvline(x=-1, color="r", linestyle="--")
axes[1].set_title("Zoomed to -1 to 1")
axes[1].set_xlabel("Skewness")

plt.tight_layout()
plt.show()
exact_zero = df_skew[df_skew["Skewness"] == 0]
print("Number of files with exactly 0 skew:", len(exact_zero))

near_zero = df_skew[df_skew["Skewness"].abs() < 1e-3]
print("Number of files with skewness within ±0.001:", len(near_zero))