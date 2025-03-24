import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import csv
import os

# Path to CSV files
csv_path = r"C:\Users\hidal\Desktop\Classes\GRAD\Data Sci Mech of Mat\HW2 Deliverable\External Dataset\All data_Strain"

# Load the material properties dataset
file_path = r"C:\Users\hidal\Desktop\Classes\GRAD\Data Sci Mech of Mat\HW2 Deliverable\External Dataset\data_all_strain-controlled.xls"
pd_train = pd.read_excel(file_path, nrows=100000)

# Drop unnecessary column
pd_train.drop(columns=['Unnamed: 5'], inplace=True, errors='ignore')

# Extract CSV file names from dataset
csv_files = pd_train['load'].values

# Read CSV files and extract relevant data
value_list = []
for csv_file in csv_files:
    try:
        file_path = os.path.join(csv_path, csv_file)
        one_df = pd.read_csv(file_path, header=None).iloc[:, :2]
        value_list.append(one_df.values)
    except Exception as e:
        print(f"Skipping {csv_file}: {e}")

# Convert to numpy arrays
csv_value_array = np.array(value_list)
flatten_value_list = [item.flatten() for item in value_list]

# Data processing pipeline
num_cols = pd_train.select_dtypes(exclude=['object']).columns.tolist()[:-1]
num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
ct = ColumnTransformer(transformers=[('num', num_pipe, num_cols)])

x_all = ct.fit_transform(pd_train)
y_all = pd_train['Nf(label)'].values

# Combine processed CSV data with material properties
dim_num = x_all.shape[-1]
combined_data = np.hstack((flatten_value_list, x_all))

# Split dataset into training and testing
x_train, x_test, y_train, y_test = train_test_split(combined_data, y_all, test_size=0.2, random_state=42)

# Plot distributions of x_train and x_test
plt.figure(figsize=(12, 5))
plt.hist(x_train.flatten(), bins=50, alpha=0.5, label='x_train', color='blue')
plt.hist(x_test.flatten(), bins=50, alpha=0.5, label='x_test', color='red')
plt.xlabel("Feature Values")
plt.ylabel("Frequency")
plt.title("Distribution of x_train and x_test")
plt.legend()
plt.show()
print("Min y_test:", min(y_test))
print("Max y_test:", max(y_test))
# Scatter plot to check negative values
plt.figure(figsize=(12, 5))
plt.scatter(range(len(x_train.flatten())), x_train.flatten(), alpha=0.5, label='x_train', color='blue')
plt.scatter(range(len(x_test.flatten())), x_test.flatten(), alpha=0.5, label='x_test', color='red')
plt.axhline(0, color='black', linestyle='--')
plt.xlabel("Index")
plt.ylabel("Feature Values")
plt.title("Scatter Plot of x_train and x_test")
plt.legend()
plt.show()

# Hyperparameter tuning for Ridge and Lasso
alpha_values = np.logspace(-3, 3, 10)  # Alpha values from 0.001 to 1000

ridge_param_grid = {'alpha': alpha_values}
ridge_cv = GridSearchCV(Ridge(), ridge_param_grid, cv=5)
ridge_cv.fit(x_train, y_train)

lasso_param_grid = {'alpha': alpha_values}
lasso_cv = GridSearchCV(Lasso(), lasso_param_grid, cv=5)
lasso_cv.fit(x_train, y_train)

# Best hyperparameters
best_ridge_alpha = ridge_cv.best_params_['alpha']
best_lasso_alpha = lasso_cv.best_params_['alpha']

# Initialize models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=best_ridge_alpha),
    "Lasso Regression": Lasso(alpha=best_lasso_alpha)
}

# Train and evaluate models
results = []
model_predictions = {}

for name, model in models.items():
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    model_predictions[name] = y_pred
    
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    print(f"{name} Results:")
    if hasattr(model, "alpha"):
        print(f"  Best Alpha: {model.alpha}")
    print(f"  R² Score: {r2:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  MSE: {mse:.4f}\n")
    
    results.append([name, model.alpha if hasattr(model, "alpha") else "N/A", r2, mae, mse])

# Visualization
plt.figure(figsize=(8, 6))

# Scatter plot for Ridge
sns.scatterplot(x=y_test, y=model_predictions["Ridge Regression"], color='blue', label='Ridge Prediction', alpha=0.6)

# Scatter plot for Lasso
sns.scatterplot(x=y_test, y=model_predictions["Lasso Regression"], color='red', label='Lasso Prediction', alpha=0.6)

# 1:1 Line (Perfect Fit)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--k', label="Perfect Fit")

plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Regression Model Predictions vs Actual Values")
plt.legend()
plt.show()

