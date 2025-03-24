import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedKFold
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load the dataset
file_path = r"C:\Users\hidal\Desktop\Classes\GRAD\Data Sci Mech of Mat\HW2 Deliverable\External Dataset\StressRegressionModel WIP\Processed_Material_Data_All_Loads.xlsx"
df = pd.read_excel(file_path)

# Prepare the data  
X = df[['Stress', 'Strain', 'Youngs_Modulus', 'Poisson_Ratio', 'Tensile_Strength', 'ss(MPa)']]  # Independent variables
y = df['Fatigue_Life']  # Dependent variable (response)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid for alpha
alpha_range = np.logspace(-3, 3, 10)  # Values from 0.001 to 1000

param_grid = {'alpha': alpha_range}

# Define K-Fold Cross-Validation
cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=1)

# Ridge Regression with GridSearchCV
ridge_model = Ridge()
ridge_grid_search = GridSearchCV(ridge_model, param_grid, cv=cv, scoring='neg_mean_squared_error')  # 5-fold cross-validation
ridge_grid_search.fit(X_train, y_train)

# Lasso Regression with GridSearchCV
lasso_model = Lasso()
lasso_grid_search = GridSearchCV(lasso_model, param_grid, cv=cv, scoring='neg_mean_squared_error')  # 5-fold cross-validation
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



import seaborn as sns
import matplotlib.pyplot as plt

# Prepare the data
X = df[['Stress', 'Strain', 'Youngs_Modulus', 'Poisson_Ratio', 'Tensile_Strength', 'ss(MPa)']]  # Independent variables
y = df['Fatigue_Life']  # Dependent variable (response)

# Combine X and y into a single DataFrame for correlation analysis
data = pd.concat([X, y], axis=1)

# Calculate the correlation matrix
corr_matrix = data.corr()

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt='.2f', cbar=True)
plt.title('Correlation Heatmap of Independent Variables and Fatigue Life')
plt.show()