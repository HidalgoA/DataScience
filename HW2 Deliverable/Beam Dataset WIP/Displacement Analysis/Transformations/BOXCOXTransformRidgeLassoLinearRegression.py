import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.preprocessing import PowerTransformer

# Load the dataset
file_path = "IndpAndResponseTable.xlsx"  # Update with actual file path
df = pd.read_excel(file_path)

# Compute and display skewness before transformation
print("Skewness before Box-Cox transformation:")
print(df.skew())

# Ensure all values are positive for Box-Cox transformation
X = df[['Volume Fraction', 'Loads', 'Youngs_Modulus', 'Poisson_Ratio']]
X_shift = np.abs(X.min()) + 1e-6  # Shift values to be positive
X += X_shift
X_transformer = PowerTransformer(method='box-cox', standardize=True)
X_transformed = X_transformer.fit_transform(X)

# Apply Box-Cox transformation to the dependent variable
y = df['Stress_Values'].values.reshape(-1, 1)
y_shift = np.abs(y.min()) + 1e-6  # Shift values to be positive
y += y_shift
y_transformer = PowerTransformer(method='box-cox', standardize=True)
y_transformed = y_transformer.fit_transform(y).flatten()

# Compute and display skewness after transformation
df_transformed = pd.DataFrame(X_transformed, columns=['Volume Fraction', 'Loads', 'Youngs_Modulus', 'Poisson_Ratio'])
df_transformed['Stress_Values'] = y_transformed
print("\nSkewness after Box-Cox transformation:")
print(df_transformed.skew())

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y_transformed, test_size=0.2, random_state=42)

# Define hyperparameter grid
alpha_range = np.logspace(-3, 3, 10)
param_grid = {'alpha': alpha_range}

# Train Ridge Regression with GridSearchCV
ridge_grid = GridSearchCV(Ridge(), param_grid, cv=5, scoring='neg_mean_squared_error')
ridge_grid.fit(X_train, y_train)
best_ridge_alpha = ridge_grid.best_params_['alpha']
ridge_model = Ridge(alpha=best_ridge_alpha).fit(X_train, y_train)

# Train Lasso Regression with GridSearchCV
lasso_grid = GridSearchCV(Lasso(), param_grid, cv=5, scoring='neg_mean_squared_error')
lasso_grid.fit(X_train, y_train)
best_lasso_alpha = lasso_grid.best_params_['alpha']
lasso_model = Lasso(alpha=best_lasso_alpha).fit(X_train, y_train)

# Make predictions in the transformed space
ridge_pred_transformed = ridge_model.predict(X_test)
lasso_pred_transformed = lasso_model.predict(X_test)

# Inverse transform predictions back to original scale
y_test_original = y_transformer.inverse_transform(y_test.reshape(-1, 1)).flatten()
ridge_pred_original = y_transformer.inverse_transform(ridge_pred_transformed.reshape(-1, 1)).flatten()
lasso_pred_original = y_transformer.inverse_transform(lasso_pred_transformed.reshape(-1, 1)).flatten()

# Compute evaluation metrics in transformed space
ridge_mse_transformed = mean_squared_error(y_test, ridge_pred_transformed)
ridge_r2_transformed = r2_score(y_test, ridge_pred_transformed)
lasso_mse_transformed = mean_squared_error(y_test, lasso_pred_transformed)
lasso_r2_transformed = r2_score(y_test, lasso_pred_transformed)

# Compute evaluation metrics in original space
ridge_mse_original = mean_squared_error(y_test_original, ridge_pred_original)
ridge_r2_original = r2_score(y_test_original, ridge_pred_original)
lasso_mse_original = mean_squared_error(y_test_original, lasso_pred_original)
lasso_r2_original = r2_score(y_test_original, lasso_pred_original)

# Print results
print("\nRidge Regression - Best Alpha:", best_ridge_alpha)
print("Ridge - MSE (Transformed):", ridge_mse_transformed)
print("Ridge - R² (Transformed):", ridge_r2_transformed)
print("Ridge - MSE (Original Scale):", ridge_mse_original)
print("Ridge - R² (Original Scale):", ridge_r2_original)

print("\nLasso Regression - Best Alpha:", best_lasso_alpha)
print("Lasso - MSE (Transformed):", lasso_mse_transformed)
print("Lasso - R² (Transformed):", lasso_r2_transformed)
print("Lasso - MSE (Original Scale):", lasso_mse_original)
print("Lasso - R² (Original Scale):", lasso_r2_original)

# Scatter plot of actual vs predicted stress values in original scale
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test_original, y=ridge_pred_original, color='blue', label='Ridge Predicted Stress', alpha=0.7)
sns.scatterplot(x=y_test_original, y=lasso_pred_original, color='red', label='Lasso Predicted Stress', alpha=0.7)
plt.plot([min(y_test_original), max(y_test_original)], [min(y_test_original), max(y_test_original)], 'k--', lw=2, label="Perfect Prediction Line")
plt.xlabel('Actual Stress Values')
plt.ylabel('Predicted Stress Values')
plt.title('Actual vs Predicted Stress Values (Original Scale)')
plt.legend()
plt.tight_layout()
plt.show()

# Heatmap of transformed correlations
plt.figure(figsize=(8, 6))
sns.heatmap(df_transformed.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap (Box-Cox Transformed Data)")
plt.show()

# Predict stress for a new data point
new_data = pd.DataFrame({
    'Volume Fraction': [0.4],
    'Loads': [3],
    'Youngs_Modulus': [10000],
    'Poisson_Ratio': [0.34]
})

# Apply the same shift and transformation
new_data += X_shift
new_data_transformed = X_transformer.transform(new_data)

# Make predictions
ridge_new_pred_transformed = ridge_model.predict(new_data_transformed)
lasso_new_pred_transformed = lasso_model.predict(new_data_transformed)

# Inverse transform predictions back to original scale
ridge_new_pred_original = y_transformer.inverse_transform(ridge_new_pred_transformed.reshape(-1, 1)).flatten()
lasso_new_pred_original = y_transformer.inverse_transform(lasso_new_pred_transformed.reshape(-1, 1)).flatten()

# Print predictions
print(f"\nPredicted Stress (Ridge, Original Scale): {ridge_new_pred_original[0]}")
print(f"Predicted Stress (Lasso, Original Scale): {lasso_new_pred_original[0]}")
