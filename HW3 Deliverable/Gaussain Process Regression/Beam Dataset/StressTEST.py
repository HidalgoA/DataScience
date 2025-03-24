import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, ExpSineSquared, WhiteKernel

# Define file path for the dataset
transformed_data_file = r"C:\Users\hidal\Desktop\Classes\GRAD\Data Sci Mech of Mat\HW2 Deliverable\Beam Dataset SANDBOX Correct Code\Stress Analysis\Transformed_Stress_Data.xlsx"

# Load the dataset
df = pd.read_excel(transformed_data_file, sheet_name="Sheet1")
df.columns = df.columns.str.strip()
df["Stress_Values"] = pd.to_numeric(df["Stress_Values"], errors="coerce")
df = df.dropna(subset=["Stress_Values"])

# Apply MinMaxScaler to independent variables and target variable
X = df.drop(columns=["Stress_Values"])
y = df[["Stress_Values"]]  # Keep y as a DataFrame to apply scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
y_scaled = scaler.fit_transform(y)  # Scale Stress_Values

# Store Original y
y_raw = df["Stress_Values"].values

# Apply Log Transformation
y_log = np.log1p(y_raw)

# Apply Box-Cox Transformation (Shift if needed)
if np.any(y_raw <= 0):
    shift_value = np.abs(y_raw.min()) + 1
    y_shifted = y_raw + shift_value
    y_boxcox, lambda_boxcox = boxcox(y_shifted)
else:
    y_boxcox, lambda_boxcox = boxcox(y_raw)
    shift_value = 0

# Train-Test Split
X_train, X_test, y_train_raw, y_test_raw = train_test_split(X_scaled, y_raw, test_size=0.2, random_state=42)
_, _, y_train_log, y_test_log = train_test_split(X_scaled, y_log, test_size=0.2, random_state=42)
_, _, y_train_boxcox, y_test_boxcox = train_test_split(X_scaled, y_boxcox, test_size=0.2, random_state=42)

# Define multiple kernel options
kernels = {
    "RBF": 1.0 * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0),
    "Matern (ν=1.5)": 1.0 * Matern(length_scale=1.0, nu=1.5) + WhiteKernel(noise_level=1.0),
    "Matern (ν=2.5)": 1.0 * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1.0),
    "Rational Quadratic": 1.0 * RationalQuadratic(length_scale=1.0, alpha=1.0) + WhiteKernel(noise_level=1.0),
    "Periodic + RBF": 1.0 * ExpSineSquared(length_scale=1.0, periodicity=1.0) + RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0),
}

# Train and Predict for Each Transformation and Kernel
predictions = {}
results = []
for kernel_name, kernel in kernels.items():
    print(f"\n🔹 Testing Kernel: {kernel_name}")
    
    for y_train, y_test, label in zip(
        [y_train_raw, y_train_log, y_train_boxcox],
        [y_test_raw, y_test_log, y_test_boxcox],
        ["Raw", "Log", "Box-Cox"]
    ):
        # Train GPR Model
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
        gpr.fit(X_train, y_train)

        # Make Predictions (Convert Back for Log & Box-Cox)
        y_pred, y_std = gpr.predict(X_test, return_std=True)

        if label == "Log":
            y_pred = np.expm1(y_pred)  # Reverse log transform
        elif label == "Box-Cox":
            y_pred = inv_boxcox(y_pred, lambda_boxcox) - shift_value  # Reverse Box-Cox

        predictions[(kernel_name, label)] = y_pred

        # Evaluate Performance
        mse = mean_squared_error(y_test_raw, y_pred)
        mae = mean_absolute_error(y_test_raw, y_pred)
        r2 = r2_score(y_test_raw, y_pred)
        results.append([kernel_name, label, mse, mae, r2])

# Convert results to DataFrame
results_df = pd.DataFrame(results, columns=["Kernel", "Transformation", "MSE", "MAE", "R2 Score"])

# Separate metrics for better readability
df_mse = results_df.pivot(index="Transformation", columns="Kernel", values="MSE").round(4)
df_mae = results_df.pivot(index="Transformation", columns="Kernel", values="MAE").round(4)
df_r2 = results_df.pivot(index="Transformation", columns="Kernel", values="R2 Score").round(4)

# Display MSE table
print("\n **Mean Squared Error (MSE) Table** ")
print("=" * 80)
print(df_mse.to_string())
print("=" * 80)

# Display MAE table
print("\n **Mean Absolute Error (MAE) Table** ")
print("=" * 80)
print(df_mae.to_string())
print("=" * 80)

# Display R² Score table
print("\n **R² Score Table** ")
print("=" * 80)
print(df_r2.to_string())
print("=" * 80)

# Identify the best performing kernel and transformation based on highest R² Score
best_r2_score = results_df.loc[results_df["R2 Score"].idxmax()]
best_kernel = best_r2_score["Kernel"]
best_transformation = best_r2_score["Transformation"]
best_r2 = best_r2_score["R2 Score"]

# Print the best model information clearly
print(f"\n **Best Performing Model:**")
print(f"   - Kernel: {best_kernel}")
print(f"   - Transformation: {best_transformation}")
print(f"   - Best R² Score: {best_r2:.4f}")
print("=" * 80)


# Extract the best model's predictions
best_pred = predictions[(best_kernel, best_transformation)]

# Plot predicted vs observed for the best model
plt.figure(figsize=(6, 6))
plt.scatter(y_test_raw, best_pred, alpha=0.5)
plt.plot([min(y_test_raw), max(y_test_raw)], [min(y_test_raw), max(y_test_raw)], 'r--', lw=2)  # 45-degree line
plt.xlabel("Observed Values")
plt.ylabel("Predicted Values")
plt.title(f"Best Model: {best_kernel} ({best_transformation})")
plt.grid(True)
plt.show()