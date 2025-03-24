import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path = "IndpAndResponseTable.xlsx"  # Update with actual file path
df = pd.read_excel(file_path)

# Compute and display skewness before standardization
print("Skewness before Standardization:")
print(df.skew())

# Apply Standardization
scaler = StandardScaler()
df_standardized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# Compute and display skewness after standardization
print("\nSkewness after Standardization:")
print(df_standardized.skew())

# Define independent and dependent variables
X = df_standardized[['Volume Fraction', 'Loads', 'Youngs_Modulus', 'Poisson_Ratio']]
y = df_standardized['Stress_Values']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid for alpha
alpha_range = np.logspace(-3, 3, 10)  # Values from 0.001 to 1000
param_grid = {'alpha': alpha_range}

# Ridge Regression with GridSearchCV
ridge_model = Ridge()
ridge_grid_search = GridSearchCV(ridge_model, param_grid, cv=5, scoring='neg_mean_squared_error')
ridge_grid_search.fit(X_train, y_train)

# Lasso Regression with GridSearchCV
lasso_model = Lasso()
lasso_grid_search = GridSearchCV(lasso_model, param_grid, cv=5, scoring='neg_mean_squared_error')
lasso_grid_search.fit(X_train, y_train)

# Best alpha values from GridSearchCV
best_ridge_alpha = ridge_grid_search.best_params_['alpha']
best_lasso_alpha = lasso_grid_search.best_params_['alpha']

# Train the best models with optimized alpha values
ridge_best_model = Ridge(alpha=best_ridge_alpha)
ridge_best_model.fit(X_train, y_train)

lasso_best_model = Lasso(alpha=best_lasso_alpha)
lasso_best_model.fit(X_train, y_train)

# Make predictions on the test set
ridge_pred_test = ridge_best_model.predict(X_test)
lasso_pred_test = lasso_best_model.predict(X_test)

# Make predictions on the training set
ridge_pred_train = ridge_best_model.predict(X_train)
lasso_pred_train = lasso_best_model.predict(X_train)

# Evaluate the models
ridge_mse_test = mean_squared_error(y_test, ridge_pred_test)
ridge_r2_test = r2_score(y_test, ridge_pred_test)
lasso_mse_test = mean_squared_error(y_test, lasso_pred_test)
lasso_r2_test = r2_score(y_test, lasso_pred_test)

ridge_mse_train = mean_squared_error(y_train, ridge_pred_train)
ridge_r2_train = r2_score(y_train, ridge_pred_train)
lasso_mse_train = mean_squared_error(y_train, lasso_pred_train)
lasso_r2_train = r2_score(y_train, lasso_pred_train)

# Print the evaluation metrics
print("\nRidge Regression - Best Alpha:", best_ridge_alpha)
print("Ridge Regression - Test MSE:", ridge_mse_test)
print("Ridge Regression - Test R²:", ridge_r2_test)
print("Ridge Regression - Train MSE:", ridge_mse_train)
print("Ridge Regression - Train R²:", ridge_r2_train)

print("\nLasso Regression - Best Alpha:", best_lasso_alpha)
print("Lasso Regression - Test MSE:", lasso_mse_test)
print("Lasso Regression - Test R²:", lasso_r2_test)
print("Lasso Regression - Train MSE:", lasso_mse_train)
print("Lasso Regression - Train R²:", lasso_r2_train)

# Scatter plot for Actual vs Predicted Stress Values for Test Data
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=ridge_pred_test, color='blue', label='Ridge Predicted Stress', alpha=0.7)
sns.scatterplot(x=y_test, y=lasso_pred_test, color='red', label='Lasso Predicted Stress', alpha=0.7)
plt.plot([min(y), max(y)], [min(y), max(y)], 'k--', lw=2, label="Perfect Prediction Line")
plt.xlabel('Actual Stress Values')
plt.ylabel('Predicted Stress Values')
plt.title('Actual vs Predicted Stress Values - Test Data')
plt.legend()
plt.tight_layout()
plt.show()

# Heatmap of correlations
corr_matrix = df_standardized.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap of Standardized Features and Stress Values")
plt.show()

# Predict stress for a new data point
new_data = pd.DataFrame({
    'Volume Fraction': [0.4],
    'Loads': [3],
    'Youngs_Modulus': [10000],
    'Poisson_Ratio': [0.34]
})

# Apply StandardScaler to the new data
new_data = pd.DataFrame(scaler.transform(new_data), columns=['Volume Fraction', 'Loads', 'Youngs_Modulus', 'Poisson_Ratio'])

# Make predictions using the trained models
ridge_prediction = ridge_best_model.predict(new_data)
lasso_prediction = lasso_best_model.predict(new_data)

# Print the predicted stress values
print(f"\nPredicted Stress (Ridge Regression): {ridge_prediction[0]}")
print(f"Predicted Stress (Lasso Regression): {lasso_prediction[0]}")
