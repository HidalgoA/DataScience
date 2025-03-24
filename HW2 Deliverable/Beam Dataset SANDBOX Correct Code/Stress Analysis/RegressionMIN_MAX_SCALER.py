import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Define file path for the transformed dataset
transformed_data_file = r"C:\Users\hidal\Desktop\Classes\GRAD\Data Sci Mech of Mat\HW2 Deliverable\Beam Dataset SANDBOX\Stress Analysis\Transformed_Stress_Data.xlsx"

# Load the transformed dataset
df = pd.read_excel(transformed_data_file, sheet_name="Sheet1")

# Generate a correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

# Define the independent variables (features) and response variable (stress)
X = df.drop(columns=["Stress_Values"])  # Features (everything except stress)
y = df["Stress_Values"]  # Response variable (stress)

# Choose scaler (Switch between MinMaxScaler and StandardScaler)
scaler = StandardScaler()  # Change to MinMaxScaler() if needed

# Apply scaling to independent variables
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize the base models
linear_model = LinearRegression()

# Define a range of alpha values to test
alpha_range = np.logspace(-3, 3, 50)  # 50 values from 0.001 to 1000

# Hyperparameter tuning using GridSearchCV (with cross-validation)
ridge_search = GridSearchCV(Ridge(), param_grid={'alpha': alpha_range}, scoring='r2', cv=5)
lasso_search = GridSearchCV(Lasso(max_iter=10000), param_grid={'alpha': alpha_range}, scoring='r2', cv=5)

# Fit GridSearchCV to find the best alpha
ridge_search.fit(X_train, y_train)
lasso_search.fit(X_train, y_train)

# Get the best alpha values
best_alpha_ridge = ridge_search.best_params_['alpha']
best_alpha_lasso = lasso_search.best_params_['alpha']

print("=" * 50)
print(f"🔹 Best Alpha for Ridge Regression: {best_alpha_ridge}")
print(f"🔹 Best Alpha for Lasso Regression: {best_alpha_lasso}")
print("=" * 50)

# Train models with the optimized alpha values
ridge_model = Ridge(alpha=best_alpha_ridge)
lasso_model = Lasso(alpha=best_alpha_lasso, max_iter=10000)

ridge_model.fit(X_train, y_train)
lasso_model.fit(X_train, y_train)
linear_model.fit(X_train, y_train)

# Predictions on test data
y_pred_linear_test = linear_model.predict(X_test)
y_pred_ridge_test = ridge_model.predict(X_test)
y_pred_lasso_test = lasso_model.predict(X_test)

# Predictions on train data
y_pred_linear_train = linear_model.predict(X_train)
y_pred_ridge_train = ridge_model.predict(X_train)
y_pred_lasso_train = lasso_model.predict(X_train)

# Evaluate models (New Vertical Format for Readability)
results = {
    "Metric": ["MSE", "MAE", "R2 Score"],
    "Linear Regression (Train)": [
        mean_squared_error(y_train, y_pred_linear_train),
        mean_absolute_error(y_train, y_pred_linear_train),
        r2_score(y_train, y_pred_linear_train),
    ],
    "Linear Regression (Test)": [
        mean_squared_error(y_test, y_pred_linear_test),
        mean_absolute_error(y_test, y_pred_linear_test),
        r2_score(y_test, y_pred_linear_test),
    ],
    "Ridge Regression (Train)": [
        mean_squared_error(y_train, y_pred_ridge_train),
        mean_absolute_error(y_train, y_pred_ridge_train),
        r2_score(y_train, y_pred_ridge_train),
    ],
    "Ridge Regression (Test)": [
        mean_squared_error(y_test, y_pred_ridge_test),
        mean_absolute_error(y_test, y_pred_ridge_test),
        r2_score(y_test, y_pred_ridge_test),
    ],
    "Lasso Regression (Train)": [
        mean_squared_error(y_train, y_pred_lasso_train),
        mean_absolute_error(y_train, y_pred_lasso_train),
        r2_score(y_train, y_pred_lasso_train),
    ],
    "Lasso Regression (Test)": [
        mean_squared_error(y_test, y_pred_lasso_test),
        mean_absolute_error(y_test, y_pred_lasso_test),
        r2_score(y_test, y_pred_lasso_test),
    ],
}

# Convert results to DataFrame and print in a clear format
results_df = pd.DataFrame(results)

# Print neatly formatted results
print("\n📊 **Model Performance Summary (Train & Test)** 📊")
print("=" * 80)
print(results_df.to_string(index=False))
print("=" * 80)

# Visualization - Scatter Plots of Actual vs Predicted Values
models = {
    "Linear Regression": y_pred_linear_test,
    "Ridge Regression (Optimized)": y_pred_ridge_test,
    "Lasso Regression (Optimized)": y_pred_lasso_test
}

plt.figure(figsize=(15, 5))

for i, (model_name, y_pred) in enumerate(models.items(), 1):
    plt.subplot(1, 3, i)
    plt.scatter(y_test, y_pred, alpha=0.6, edgecolors="k")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")  # Perfect prediction line
    plt.xlabel("Actual Stress Values")
    plt.ylabel("Predicted Stress Values")
    plt.title(f"{model_name} Predictions")
    plt.grid(True)

# Show plots
plt.tight_layout()
plt.show()
