import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load the dataset
file_path = "IndpAndResponseTable.xlsx"  # Update with actual file path
df = pd.read_excel(file_path)

# Prepare the data
X = df[['Volume Fraction', 'Loads', 'Youngs_Modulus', 'Poisson_Ratio']]  # Independent variables
y = df['Stress_Values']  # Dependent variable

# Split the data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Try different polynomial degrees and compare performance
degrees = [1, 2, 3, 4]  # Candidate polynomial degrees
best_degree = 1  # Default
best_score = float('inf')

for d in degrees:
    poly = PolynomialFeatures(degree=d, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    ridge_model = Ridge(alpha=1.0)  # Initial Ridge model
    scores = cross_val_score(ridge_model, X_train_poly, y_train, cv=5, scoring='neg_mean_squared_error')
    avg_mse = -np.mean(scores)  # Convert to positive MSE

    print(f"Degree {d}: Avg Cross-Validation MSE = {avg_mse}")

    if avg_mse < best_score:
        best_score = avg_mse
        best_degree = d

print(f"Best Polynomial Degree: {best_degree}")

# Apply the best polynomial degree
poly = PolynomialFeatures(degree=best_degree, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Define alpha range for regularization
alpha_range = np.logspace(-3, 3, 10)  # 0.001 to 1000
param_grid = {'alpha': alpha_range}

# Ridge Regression with GridSearchCV
ridge_grid_search = GridSearchCV(Ridge(), param_grid, cv=5, scoring='neg_mean_squared_error')
ridge_grid_search.fit(X_train_poly, y_train)
best_ridge_alpha = ridge_grid_search.best_params_['alpha']

# Lasso Regression with GridSearchCV
lasso_grid_search = GridSearchCV(Lasso(), param_grid, cv=5, scoring='neg_mean_squared_error')
lasso_grid_search.fit(X_train_poly, y_train)
best_lasso_alpha = lasso_grid_search.best_params_['alpha']

# Train the best models with the optimized alpha values
ridge_best_model = Ridge(alpha=best_ridge_alpha)
ridge_best_model.fit(X_train_poly, y_train)

lasso_best_model = Lasso(alpha=best_lasso_alpha)
lasso_best_model.fit(X_train_poly, y_train)

# Make predictions
ridge_pred_test = ridge_best_model.predict(X_test_poly)
lasso_pred_test = lasso_best_model.predict(X_test_poly)

ridge_pred_train = ridge_best_model.predict(X_train_poly)
lasso_pred_train = lasso_best_model.predict(X_train_poly)

# Evaluate the models
ridge_mse_test = mean_squared_error(y_test, ridge_pred_test)
ridge_r2_test = r2_score(y_test, ridge_pred_test)

lasso_mse_test = mean_squared_error(y_test, lasso_pred_test)
lasso_r2_test = r2_score(y_test, lasso_pred_test)

ridge_mse_train = mean_squared_error(y_train, ridge_pred_train)
ridge_r2_train = r2_score(y_train, ridge_pred_train)

lasso_mse_train = mean_squared_error(y_train, lasso_pred_train)
lasso_r2_train = r2_score(y_train, lasso_pred_train)

# Print evaluation metrics
print("\nRidge Regression:")
print(f"Best Alpha: {best_ridge_alpha}")
print(f"Train MSE: {ridge_mse_train}, Test MSE: {ridge_mse_test}")
print(f"Train R²: {ridge_r2_train}, Test R²: {ridge_r2_test}")

print("\nLasso Regression:")
print(f"Best Alpha: {best_lasso_alpha}")
print(f"Train MSE: {lasso_mse_train}, Test MSE: {lasso_mse_test}")
print(f"Train R²: {lasso_r2_train}, Test R²: {lasso_r2_test}")

# Plot Actual vs Predicted
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=ridge_pred_test, color='blue', label='Ridge Predicted', alpha=0.7)
sns.scatterplot(x=y_test, y=lasso_pred_test, color='red', label='Lasso Predicted', alpha=0.7)
plt.plot([min(y), max(y)], [min(y), max(y)], 'k--', lw=2, label="Perfect Prediction Line")
plt.xlabel('Actual Displ Values')
plt.ylabel('Predicted Displ Values')
plt.title('Actual vs Predicted Displ Values (Test Data)')
plt.legend()
plt.show()
