import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedKFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np

# Load the dataset
file_path = "IndpAndResponseTable.xlsx"  # Update with actual file path
df = pd.read_excel(file_path)

# Prepare the data  
X = df[['Volume Fraction', 'Displacement', 'Youngs_Modulus', 'Poisson_Ratio']]  # Independent variables
scaler = StandardScaler()
X = scaler.fit_transform(X)  # Apply MinMax normalization transformation
y = df['Stress_Values']  # Dependent variable

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid for alpha
alpha_range = np.logspace(-3, 3, 10)  # Values from 0.001 to 1000
param_grid = {'alpha': alpha_range}

# Define K-Fold Cross-Validation
cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=1)

# Ridge Regression with GridSearchCV
ridge_model = Ridge()
ridge_grid_search = GridSearchCV(ridge_model, param_grid, cv=cv, scoring='neg_mean_squared_error')
ridge_grid_search.fit(X_train, y_train)

# Lasso Regression with GridSearchCV
lasso_model = Lasso()
lasso_grid_search = GridSearchCV(lasso_model, param_grid, cv=cv, scoring='neg_mean_squared_error')
lasso_grid_search.fit(X_train, y_train)

# Best alpha values from GridSearchCV
best_ridge_alpha = ridge_grid_search.best_params_['alpha']
best_lasso_alpha = lasso_grid_search.best_params_['alpha']

# Train the best models with the optimized alpha values
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

# Print the evaluation metrics for both training and test data
print("Ridge Regression - Best Alpha:", best_ridge_alpha)
print("Ridge Regression - Test Data MSE:", ridge_mse_test)
print("Ridge Regression - Test Data R-squared:", ridge_r2_test)
print("Ridge Regression - Training Data MSE:", ridge_mse_train)
print("Ridge Regression - Training Data R-squared:", ridge_r2_train)

print("\nLasso Regression - Best Alpha:", best_lasso_alpha)
print("Lasso Regression - Test Data MSE:", lasso_mse_test)
print("Lasso Regression - Test Data R-squared:", lasso_r2_test)
print("Lasso Regression - Training Data MSE:", lasso_mse_train)
print("Lasso Regression - Training Data R-squared:", lasso_r2_train)

# Plotting correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Feature Correlation Heatmap")
plt.show()

# Plotting Actual vs Predicted Stress Values for Test Data
plt.figure(figsize=(10, 6))

# Plot Actual Stress vs Ridge Predicted Stress (Test Data)
sns.scatterplot(x=y_test, y=ridge_pred_test, color='blue', label='Ridge Predicted Stress (Test)', alpha=0.7)

# Plot Actual Stress vs Lasso Predicted Stress (Test Data)
sns.scatterplot(x=y_test, y=lasso_pred_test, color='red', label='Lasso Predicted Stress (Test)', alpha=0.7)

# Plot the line of perfect prediction (diagonal line)
plt.plot([min(y), max(y)], [min(y), max(y)], 'k--', lw=2, label="Perfect Prediction Line")

# Adding labels and title
plt.xlabel('Actual Stress Values')
plt.ylabel('Predicted Stress Values')
plt.title('Actual vs Predicted Stress Values - Test Data')
plt.legend()
plt.show()
