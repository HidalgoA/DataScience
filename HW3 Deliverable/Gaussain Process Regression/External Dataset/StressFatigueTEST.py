import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.interpolate import interp1d
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, ExpSineSquared, WhiteKernel, ConstantKernel

# ---------------------------------------------------------
# 1. Configuration and Paths
# ---------------------------------------------------------
csv_path = r"C:\Users\hidal\Desktop\Classes\GRAD\Data Sci Mech of Mat\HW2 Deliverable\External Dataset\All data_Stress"
file_path = r"C:\Users\hidal\Desktop\Classes\GRAD\Data Sci Mech of Mat\HW2 Deliverable\External Dataset\data_all_stress-controlled.xls"

# ---------------------------------------------------------
# 2. Load Master Data and CSV Files
# ---------------------------------------------------------
# Load material properties dataset (master file)
pd_train = pd.read_excel(file_path, nrows=100000)
pd_train.drop(columns=['Unnamed: 5'], inplace=True, errors='ignore')  # Drop unnecessary column

# Extract CSV file names from dataset and remove NaNs
csv_files = pd_train['load'].dropna().values  

# Read CSV files (for stress, we use raw values without log transformation)
value_list = []
valid_indices = []
for i, csv_file in enumerate(csv_files):
    try:
        full_csv_path = os.path.join(csv_path, csv_file)
        one_df = pd.read_csv(full_csv_path, header=None).iloc[:, :2]
        # Flatten the two columns
        value_list.append(one_df.values.flatten())
        valid_indices.append(i)
    except Exception as e:
        print(f"Skipping {csv_file}: {e}")

# Convert to numpy array
csv_value_array = np.array(value_list)

# Filter the main dataset to match valid CSV indices
pd_train = pd_train.iloc[valid_indices]

# ---------------------------------------------------------
# 3. Process Material Properties
# ---------------------------------------------------------
# Exclude CSV and target columns
exclude_cols = ['load', 'Nf(label)']
num_cols = pd_train.select_dtypes(exclude=['object']).columns.tolist()
num_cols = [col for col in num_cols if col not in exclude_cols]

# Define pipeline for numerical columns (material properties)
num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
# Transform material properties
x_all = num_pipe.fit_transform(pd_train[num_cols])
y_all = pd_train['Nf(label)'].values  # Target variable (stress, in this context)

# Ensure y_all has no NaNs
valid_y_indices = ~np.isnan(y_all)
x_all = x_all[valid_y_indices]
y_all = y_all[valid_y_indices]

# ---------------------------------------------------------
# 4. Scale CSV Data and Combine with Material Properties
# ---------------------------------------------------------
csv_scaler = MinMaxScaler()  # Scale CSV data separately
csv_value_array_scaled = csv_scaler.fit_transform(csv_value_array)

# Combine the scaled CSV data with the transformed material properties
combined_data = np.hstack((csv_value_array_scaled, x_all))
y_all = y_all[valid_y_indices]

print(f"Shape of combined_data: {combined_data.shape}")
print(f"Shape of y_all: {y_all.shape}")

# ---------------------------------------------------------
# 5. Train/Test Split
# ---------------------------------------------------------
x_train, x_test, y_train, y_test = train_test_split(combined_data, y_all, test_size=0.2, random_state=42)

# ---------------------------------------------------------
# 6. Define GPR Kernels
# ---------------------------------------------------------
kernels = {
    "RBF": 1.0 * RBF(length_scale=1.0) + WhiteKernel(noise_level=2.0),
    "Matern (ν=0.5)": 1.0 * Matern(length_scale=1.0, nu=0.5) + WhiteKernel(noise_level=2.0),
    "Matern (ν=1.5)": 1.0 * Matern(length_scale=1.0, nu=1.0) + WhiteKernel(noise_level=1.0),
    "Matern (ν=2.5)": 1.0 * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=2.0),
    "Rational Quadratic": 1.0 * RationalQuadratic(length_scale=1.0, alpha=1.0) + WhiteKernel(noise_level=2.0),
    "Constant + RBF": ConstantKernel(constant_value=1.0) + RBF(length_scale=1.0) + WhiteKernel(noise_level=2.0)
}

# ---------------------------------------------------------
# 7. Train and Evaluate GPR Models
# ---------------------------------------------------------
results = []
predictions = {}

# For this stress dataset, we won't use a full pipeline with a ColumnTransformer
# since combined_data is already numeric (CSV features + material properties).
for kernel_name, kernel in kernels.items():
    print(f"\n🔹 Training GPR with Kernel: {kernel_name}")
    
    gpr_model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
    gpr_model.fit(x_train, y_train)
    
    # Predict on the test set (with uncertainty)
    y_pred, y_std = gpr_model.predict(x_test, return_std=True)
    predictions[kernel_name] = y_pred
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results.append([kernel_name, mse, mae, r2])

# Convert results to DataFrame and sort by R² Score
results_df = pd.DataFrame(results, columns=["Kernel", "MSE", "MAE", "R2 Score"])
results_df = results_df.sort_values(by="R2 Score", ascending=False)

# ---------------------------------------------------------
# 8. Display Performance Results
# ---------------------------------------------------------
print("\n **GPR Model Performance Comparison** ")
print("=" * 80)
print(results_df.to_string())
print("=" * 80)

# Identify the best-performing kernel
best_model = results_df.iloc[0]
best_kernel = best_model["Kernel"]
best_r2 = best_model["R2 Score"]

print(f"\n **Best Performing GPR Model:**")
print(f"   - Kernel: {best_kernel}")
print(f"   - Best R² Score: {best_r2:.4f}")
print("=" * 80)

# ---------------------------------------------------------
# 9. Visualization: Actual vs. Predicted Values
# ---------------------------------------------------------
plt.figure(figsize=(8, 6))
for kernel_name, y_pred in predictions.items():
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, label=kernel_name)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--k', label="Perfect Fit")
plt.xlabel("Actual Stress")
plt.ylabel("Predicted Stress")
plt.title("GPR Model Predictions vs. Actual Stress")
plt.legend()
plt.show()
