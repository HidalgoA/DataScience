import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis, boxcox
from scipy.special import inv_boxcox
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Define file path for the transformed dataset
transformed_data_file = r"C:\Users\hidal\Desktop\Classes\GRAD\Data Sci Mech of Mat\HW2 Deliverable\Beam Dataset SANDBOX Correct Code\Displacement Analysis\Transformed_Displ_Data.xlsx"

# Load the dataset
df = pd.read_excel(transformed_data_file, sheet_name="Sheet1")
df.columns = df.columns.str.strip()  # Remove extra spaces in column names
df["Displacement_Values"] = pd.to_numeric(df["Displacement_Values"], errors="coerce")
df = df.dropna(subset=["Displacement_Values"])

# Compute and print Skewness & Kurtosis Before Transformation
print(f"🔹 Original Skewness: {skew(df['Displacement_Values']):.3f}")
print(f"🔹 Original Kurtosis: {kurtosis(df['Displacement_Values']):.3f}")

# Store Original y
y_raw = df["Displacement_Values"].values

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

print(f"🔹 Box-Cox Lambda: {lambda_boxcox:.3f}")

# Compute and print Skewness & Kurtosis After Transformation
print(f"🔹 Log Transformed Skewness: {skew(y_log):.3f}")
print(f"🔹 Log Transformed Kurtosis: {kurtosis(y_log):.3f}")
print(f"🔹 Box-Cox Transformed Skewness: {skew(y_boxcox):.3f}")
print(f"🔹 Box-Cox Transformed Kurtosis: {kurtosis(y_boxcox):.3f}")

# Visualization of Transformations
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].hist(y_raw, bins=30, color='blue', alpha=0.6, edgecolor='black')
axes[0].set_title('Original Distribution')
axes[1].hist(y_log, bins=30, color='green', alpha=0.6, edgecolor='black')
axes[1].set_title('Log Transformed Distribution')
axes[2].hist(y_boxcox, bins=30, color='red', alpha=0.6, edgecolor='black')
axes[2].set_title('Box-Cox Transformed Distribution')
plt.tight_layout()
plt.show()

# Define Features
X = df.drop(columns=["Displacement_Values"])
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train_raw, y_test_raw = train_test_split(X_scaled, y_raw, test_size=0.2, random_state=42)
_, _, y_train_log, y_test_log = train_test_split(X_scaled, y_log, test_size=0.2, random_state=42)
_, _, y_train_boxcox, y_test_boxcox = train_test_split(X_scaled, y_boxcox, test_size=0.2, random_state=42)

# Initialize models
linear_model = LinearRegression()
alpha_range = np.logspace(-3, 3, 50)

# Hyperparameter tuning
ridge_search = GridSearchCV(Ridge(), param_grid={'alpha': alpha_range}, scoring='r2', cv=5)
lasso_search = GridSearchCV(Lasso(max_iter=10000), param_grid={'alpha': alpha_range}, scoring='r2', cv=5)

# Train and Predict for Each Transformation
predictions = {}
best_alphas = {}
for y_train, y_test, label in zip(
    [y_train_raw, y_train_log, y_train_boxcox],
    [y_test_raw, y_test_log, y_test_boxcox],
    ["Raw", "Log", "Box-Cox"]
):
    # Tune Hyperparameters
    ridge_search.fit(X_train, y_train)
    lasso_search.fit(X_train, y_train)
    best_alphas[label] = {'Ridge': ridge_search.best_params_['alpha'], 'Lasso': lasso_search.best_params_['alpha']}
    
    # Train Models
    ridge_model = Ridge(alpha=ridge_search.best_params_['alpha']).fit(X_train, y_train)
    lasso_model = Lasso(alpha=lasso_search.best_params_['alpha'], max_iter=10000).fit(X_train, y_train)
    
    # Make Predictions (Convert Back for Log & Box-Cox)
    preds = {
        "Linear": linear_model.fit(X_train, y_train).predict(X_test),
        "Ridge": ridge_model.predict(X_test),
        "Lasso": lasso_model.predict(X_test),
    }
    if label == "Log":
        preds = {k: np.expm1(v) for k, v in preds.items()}
    elif label == "Box-Cox":
        preds = {k: inv_boxcox(v, lambda_boxcox) - shift_value for k, v in preds.items()}
    
    predictions[label] = preds

# Print Best Alpha Values
print("\n🔹 Best Alpha Values for Each Transformation:")
for label, alphas in best_alphas.items():
    print(f"  {label}: Ridge Alpha = {alphas['Ridge']:.6f}, Lasso Alpha = {alphas['Lasso']:.6f}")

# Scatter Plots for All Models
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
fig.suptitle("Actual vs Predicted Displacement - All Transformations & Models", fontsize=14)

for i, (method, preds) in enumerate(predictions.items()):
    for j, (model_name, y_pred) in enumerate(preds.items()):
        ax = axes[i, j]
        ax.scatter(y_test_raw, y_pred, alpha=0.6, edgecolors="k")
        ax.plot([y_test_raw.min(), y_test_raw.max()], [y_test_raw.min(), y_test_raw.max()], "r--")
        ax.set_xlabel("Actual Displacement")
        ax.set_ylabel("Predicted Displacement")
        ax.set_title(f"{model_name} ({method})")
        ax.grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit title
plt.show()