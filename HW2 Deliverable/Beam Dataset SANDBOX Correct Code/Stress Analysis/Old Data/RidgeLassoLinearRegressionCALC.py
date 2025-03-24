import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedKFold
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
# Load the dataset
file_path = "IndpAndResponseTable.xlsx"  # Update with actual file path
df = pd.read_excel(file_path)

# Prepare the data  

X = df[['Volume Fraction', 'Displacement', 'Youngs_Modulus', 'Poisson_Ratio']]  # Independent variables
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

# Printing headers for Test and Training Data
print(f"{'Actual Stress (Test)':<25}{'Ridge Predicted Stress (Test)':<30}{'Lasso Predicted Stress (Test)':<30}")
print("-" * 85)

# Print Actual vs Predicted values for Test data
for actual, ridge_pred, lasso_pred in zip(y_test, ridge_pred_test, lasso_pred_test):
    print(f"{actual:<25}{ridge_pred:<30}{lasso_pred:<30}")

# Line separator for better readability
print("\n" + "-" * 85)

# Printing headers for Training Data
print(f"{'Actual Stress (Train)':<25}{'Ridge Predicted Stress (Train)':<30}{'Lasso Predicted Stress (Train)':<30}")
print("-" * 85)

# Print Actual vs Predicted values for Training data
for actual, ridge_pred, lasso_pred in zip(y_train, ridge_pred_train, lasso_pred_train):
    print(f"{actual:<25}{ridge_pred:<30}{lasso_pred:<30}")

print("\n" + "-" * 85)  # Line separator for better readability



import matplotlib.pyplot as plt
import seaborn as sns

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

# Show the plot
plt.tight_layout()
plt.show()



#Heat MaP
from sklearn.preprocessing import StandardScaler


# Normalize the data (excluding the response variable 'Stress_Values')
scaler = StandardScaler()

# Normalize the independent variables (everything except 'Stress_Values')
X = df.drop(columns=['Stress_Values'])  # Drop the response variable
X_normalized = scaler.fit_transform(X)

# Create a DataFrame for the normalized data
X_normalized_df = pd.DataFrame(X_normalized, columns=X.columns)

# Add the 'Stress_Values' back to the DataFrame for correlation calculation
df_normalized = pd.concat([X_normalized_df, df['Stress_Values']], axis=1)

# Compute the correlation matrix
corr_matrix = df_normalized.corr()

# Plot the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)

# Add title
plt.title("Correlation Heatmap of Normalized Features and Stress Values")

# Show plot
plt.show()


import pandas as pd
import numpy as np

# 

# Define the new data point 
new_data = pd.DataFrame({
    'Volume Fraction': [0.4],  #  adjust as needed
    'Displacement': [3],       # Set displacement to 2mm
    'Youngs_Modulus': [10000], # Example for MatA, adjust as needed
    'Poisson_Ratio': [0.34]    # Example for MatA, adjust as needed
})

# Make the prediction for Ridge or Lasso model
ridge_prediction = ridge_best_model.predict(new_data)
lasso_prediction = lasso_best_model.predict(new_data)

# Print the predicted stress values
print(f"Predicted Stress (Ridge Regression): {ridge_prediction[0]}")
print(f"Predicted Stress (Lasso Regression): {lasso_prediction[0]}")
