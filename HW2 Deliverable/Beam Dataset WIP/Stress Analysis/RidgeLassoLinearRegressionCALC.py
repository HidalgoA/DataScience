import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the dataset
file_path = "IndpAndResponseTable.xlsx"  # Update with actual file path
df = pd.read_excel(file_path)

# **Use all available features except "Stress_Values"**
X = df.drop(columns=["Stress_Values"])  # Use all independent variables
y = df["Stress_Values"]  # Response variable (stress)

# **Apply Feature Scaling (Same as First Model)**
scaler = StandardScaler()  # Change to MinMaxScaler() if needed
X_scaled = scaler.fit_transform(X)  # Scale all independent variables

# **Train-test split (80% train, 20% test) AFTER scaling**
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# **Hyperparameter tuning (Same as First Model)**
alpha_range = np.logspace(-3, 3, 50)  # 50 values from 0.001 to 1000
param_grid = {'alpha': alpha_range}

# **Use GridSearchCV with 5-fold cross-validation (Same as First Model)**
ridge_search = GridSearchCV(Ridge(), param_grid=param_grid, cv=5, scoring='r2')
lasso_search = GridSearchCV(Lasso(max_iter=10000), param_grid=param_grid, cv=5, scoring='r2')

# Fit models to find the best alpha
ridge_search.fit(X_train, y_train)
lasso_search.fit(X_train, y_train)

# Retrieve best alpha values
best_ridge_alpha = ridge_search.best_params_['alpha']
best_lasso_alpha = lasso_search.best_params_['alpha']

# **Train final Ridge and Lasso models using best alpha values**
ridge_best_model = Ridge(alpha=best_ridge_alpha)
lasso_best_model = Lasso(alpha=best_lasso_alpha, max_iter=10000)

ridge_best_model.fit(X_train, y_train)
lasso_best_model.fit(X_train, y_train)

# **Train a Linear Regression model (for baseline comparison)**
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# **Make predictions**
y_pred_linear = linear_model.predict(X_test)
y_pred_ridge = ridge_best_model.predict(X_test)
y_pred_lasso = lasso_best_model.predict(X_test)

# **Evaluate models (Same format as First Model)**
results = {
    "Model": ["Linear Regression", "Ridge Regression (Optimized)", "Lasso Regression (Optimized)"],
    "MSE": [
        mean_squared_error(y_test, y_pred_linear),
        mean_squared_error(y_test, y_pred_ridge),
        mean_squared_error(y_test, y_pred_lasso),
    ],
    "MAE": [
        mean_absolute_error(y_test, y_pred_linear),
        mean_absolute_error(y_test, y_pred_ridge),
        mean_absolute_error(y_test, y_pred_lasso),
    ],
    "R2 Score": [
        r2_score(y_test, y_pred_linear),
        r2_score(y_test, y_pred_ridge),
        r2_score(y_test, y_pred_lasso),
    ],
}

# **Print model performance**
results_df = pd.DataFrame(results)
print(results_df)

# **Scatter Plots of Actual vs Predicted Values (Same Layout as First Model)**
models = {
    "Linear Regression": y_pred_linear,
    "Ridge Regression (Optimized)": y_pred_ridge,
    "Lasso Regression (Optimized)": y_pred_lasso
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
