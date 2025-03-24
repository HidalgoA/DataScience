import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

#############################################
# 1. Configuration
#############################################
csv_folder = r"C:\Users\hidal\Desktop\Classes\GRAD\Data Sci Mech of Mat\HW2 Deliverable\External Dataset\TEST\All data_Strain"
master_file = r"C:\Users\hidal\Desktop\Classes\GRAD\Data Sci Mech of Mat\HW2 Deliverable\External Dataset\data_all_strain-controlled.xls"

TARGET_LENGTH = 241
EPSILON = 1e-6

#############################################
# 2. Flatten CSV with Log Transform
#############################################
def resample_to_fixed_length(x, target_len):
    orig_len = len(x)
    if orig_len == target_len:
        return x
    orig_idx = np.linspace(0, 1, orig_len)
    tgt_idx = np.linspace(0, 1, target_len)
    return np.interp(tgt_idx, orig_idx, x)

def flatten_and_log_transform_csv(csv_path, target_length=241):
    df_csv = pd.read_csv(csv_path, header=None)
    col1 = df_csv.iloc[:, 0].values
    col2 = df_csv.iloc[:, 1].values
    
    col1_res = resample_to_fixed_length(col1, target_length)
    col2_res = resample_to_fixed_length(col2, target_length)
    combined = np.concatenate([col1_res, col2_res])
    min_val = np.min(combined)
    shift = abs(min_val) + EPSILON if min_val <= 0 else 0
    shifted = combined + shift
    return np.log(shifted)

#############################################
# 3. Load Master Data
#############################################
df_master = pd.read_excel(master_file)
df_master.drop(columns=['Unnamed: 5'], inplace=True, errors='ignore')
print("Master data columns:", df_master.columns.tolist())

material_cols = ['E(Gpa)', 'TS(Mpa)', 'ss£¨Mpa£©', 'm']
y_all = df_master['Nf(label)'].values

# Shift target if needed
if np.any(y_all <= 0):
    shift_amt = abs(np.min(y_all)) + EPSILON
    y_all = y_all + shift_amt
else:
    shift_amt = 0
print(f"Shifted target by {shift_amt:.6f} if needed.")

#############################################
# 4. Process Each CSV
#############################################
csv_files = df_master['load'].values
raw_features_list = []
for csv_file in csv_files:
    full_path = os.path.join(csv_folder, csv_file)
    try:
        flattened = flatten_and_log_transform_csv(full_path, TARGET_LENGTH)
    except Exception as e:
        print(f"Error reading {csv_file}: {e}")
        flattened = [np.nan]*(2*TARGET_LENGTH)
    raw_features_list.append(flattened)

num_feats = 2 * TARGET_LENGTH
raw_cols = [f"raw_{i}" for i in range(num_feats)]
df_raw = pd.DataFrame(raw_features_list, columns=raw_cols)

#############################################
# 5. (Optional) Add Material Props
#############################################
df_material = df_master[material_cols].copy()  # raw or minimal transform

#############################################
# 6. Combine Into df_features
#############################################
df_features = pd.concat([df_material.reset_index(drop=True),
                         df_raw.reset_index(drop=True)], axis=1)

df_features.dropna(inplace=True)
y_all = y_all[df_features.index]  # align target

print("df_features shape:", df_features.shape)
print("y_all shape:", y_all.shape)

#############################################
# 7. Compute Pearson & Spearman for All Features
#############################################
df_numeric = df_features.select_dtypes(include=[np.number])
feature_names = df_numeric.columns
X_numeric = df_numeric.values

correlations = []
for i, feat in enumerate(feature_names):
    pear_r, pear_p = pearsonr(X_numeric[:, i], y_all)
    spear_r, spear_p = spearmanr(X_numeric[:, i], y_all)
    correlations.append((feat, pear_r, pear_p, spear_r, spear_p))

corr_df = pd.DataFrame(correlations, columns=[
    "feature","pearson_r","pearson_p","spearman_r","spearman_p"
])
corr_df["abs_pearson"]  = corr_df["pearson_r"].abs()
corr_df["abs_spearman"] = corr_df["spearman_r"].abs()

# Sort by absolute Pearson
corr_pearson = corr_df.sort_values(by="abs_pearson", ascending=False)
# Also sort by absolute Spearman if needed
corr_spearman = corr_df.sort_values(by="abs_spearman", ascending=False)

#############################################
# 8. Print top 10 or top 20 if you want
#############################################
print("\n=== Top 10 by abs Pearson ===")
print(corr_pearson.head(10))
print("\n=== Top 10 by abs Spearman ===")
print(corr_spearman.head(10))

#############################################
# 9. Merge & Plot All Features Side-by-Side
#############################################
# We'll plot all features, sorted by abs Pearson, 
# showing Pearson & Spearman side-by-side.

df_merged = corr_pearson[["feature","pearson_r"]].merge(
    corr_spearman[["feature","spearman_r"]],
    on="feature",
    how="left"
)
# Now df_merged is in the order of highest abs Pearson first,
# but also includes Spearman values for each feature.
n_features = len(df_merged)

print(f"\nPlotting side-by-side bars for ALL {n_features} features.")

x = np.arange(n_features)
width = 0.4

plt.figure(figsize=(max(10, n_features/10), 5))  # dynamic width if you have many columns
# Plot Pearson bars at x - width/2
plt.bar(x - width/2, df_merged["pearson_r"], width=width, alpha=0.6, color='blue', label='Pearson')
# Plot Spearman bars at x + width/2
plt.bar(x + width/2, df_merged["spearman_r"], width=width, alpha=0.6, color='red', label='Spearman')

plt.axhline(0, color='k', linestyle='--')
plt.xticks(x, df_merged["feature"], rotation=90)
plt.title(f"All {n_features} Features: Pearson vs. Spearman (Sorted by abs Pearson)")
plt.ylabel("Correlation Coefficient")
plt.legend()
plt.tight_layout()
plt.show()
