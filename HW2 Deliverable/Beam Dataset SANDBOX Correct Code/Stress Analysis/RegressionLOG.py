import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis, boxcox
from scipy.special import inv_boxcox
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Define file path for the dataset
transformed_data_file = r"C:\Users\hidal\Desktop\Classes\GRAD\Data Sci Mech of Mat\HW2 Deliverable\Beam Dataset SANDBOX Correct Code\Stress Analysis\Transformed_Stress_Data.xlsx"

# Load the dataset
df = pd.read_excel(transformed_data_file, sheet_name="Sheet1")
df.columns = df.columns.str.strip()
df["Stress_Values"] = pd.to_numeric(df["Stress_Values"], errors="coerce")
df = df.dropna(subset=["Stress_Values"])

# Apply StandardScaler to independent variables and target variable
X = df.drop(columns=["Stress_Values"])
y = df[["Stress_Values"]]  # Keep y as a DataFrame to apply scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
y_scaled = scaler.fit_transform(y)  # Scale Stress_Values

# Add scaled Stress_Values back to compute correlations
df_scaled = pd.DataFrame(X_scaled, columns=X.columns)
df_scaled["Stress_Values"] = y_scaled.flatten()

# Generate a correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df_scaled.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

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

# Initialize models
linear_model = LinearRegression()
alpha_range = np.logspace(-3, 3, 50)

# Hyperparameter tuning
ridge_search = GridSearchCV(Ridge(), param_grid={'alpha': alpha_range}, scoring='r2', cv=5)
lasso_search = GridSearchCV(Lasso(max_iter=10000), param_grid={'alpha': alpha_range}, scoring='r2', cv=5)

# Train and Predict for Each Transformation
predictions = {}
best_alphas = {}
results = []
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
    linear_model.fit(X_train, y_train)
    
    # Make Predictions (Convert Back for Log & Box-Cox)
    preds = {
        "Linear": linear_model.predict(X_test),
        "Ridge": ridge_model.predict(X_test),
        "Lasso": lasso_model.predict(X_test),
    }
    if label == "Log":
        preds = {k: np.expm1(v) for k, v in preds.items()}
    elif label == "Box-Cox":
        preds = {k: inv_boxcox(v, lambda_boxcox) - shift_value for k, v in preds.items()}
    
    predictions[label] = preds
    
    # Evaluate Performance
    for model_name, y_pred in preds.items():
        mse = mean_squared_error(y_test_raw, y_pred)
        mae = mean_absolute_error(y_test_raw, y_pred)
        r2 = r2_score(y_test_raw, y_pred)
        results.append([label, model_name, mse, mae, r2])

# Convert results to DataFrame
results_df = pd.DataFrame(results, columns=["Transformation", "Model", "MSE", "MAE", "R2 Score"])
print("\n📊 **Model Performance Summary (Train & Test)** 📊")
print("=" * 80)
print(results_df.pivot(index="Transformation", columns="Model", values=["MSE", "MAE", "R2 Score"]).round(4))
print("=" * 80)