import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import csv
import os

# Paths
csv_path = r"C:\Users\hidal\Desktop\Classes\GRAD\Data Sci Mech of Mat\HW2 Deliverable\External Dataset\All data_Strain"
file_path = r"C:\Users\hidal\Desktop\Classes\GRAD\Data Sci Mech of Mat\HW2 Deliverable\External Dataset\data_all_strain-controlled.xls"

# Load material properties dataset
pd_train = pd.read_excel(file_path, nrows=100000)
pd_train.drop(columns=['Unnamed: 5'], inplace=True, errors='ignore')  # Drop unnecessary column

# Extract CSV file names from dataset
csv_files = pd_train['load'].values

# Read CSV files and apply MinMax Scaling
value_list = []
scaler = MinMaxScaler(feature_range=(0, 1))  # MinMax Scaler instance

for csv_file in csv_files:
    try:
        file_path = os.path.join(csv_path, csv_file)
        one_df = pd.read_csv(file_path, header=None).iloc[:, :2]  # Extract first two columns
        
        # Normalize using MinMaxScaler
        one_df = scaler.fit_transform(one_df)
        
        value_list.append(one_df.flatten())
    except Exception as e:
        print(f"Skipping {csv_file}: {e}")

# Convert to numpy arrays
flatten_value_list = value_list

# Data processing pipeline
num_cols = pd_train.select_dtypes(exclude=['object']).columns.tolist()[:-1]
num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', MinMaxScaler(feature_range=(0, 1)))
])
ct = ColumnTransformer(transformers=[('num', num_pipe, num_cols)])

x_all = ct.fit_transform(pd_train)
y_all = pd_train['Nf(label)'].values

# Combine processed CSV data with material properties
combined_data = np.hstack((flatten_value_list, x_all))

# Split dataset into training and testing
x_train, x_test, y_train, y_test = train_test_split(combined_data, y_all, test_size=0.2, random_state=42)

# Define KFold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Hyperparameter tuning for Ridge and Lasso
alpha_values = np.logspace(-3, 3, 10)

ridge_cv = GridSearchCV(Ridge(), {'alpha': alpha_values}, cv=kf)
ridge_cv.fit(x_train, y_train)

lasso_cv = GridSearchCV(Lasso(), {'alpha': alpha_values}, cv=kf)
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

# Train, evaluate, and store results
results_dict = {"Model": [], "Best Alpha": [], "R² Score": [], "MAE": [], "MSE": [], "Predictions": []}
model_predictions = {}

for name, model in models.items():
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    # Compute Metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    # Store results
    results_dict["Model"].append(name)
    results_dict["Best Alpha"].append(model.alpha if hasattr(model, "alpha") else "N/A")
    results_dict["R² Score"].append(r2)
    results_dict["MAE"].append(mae)
    results_dict["MSE"].append(mse)
    results_dict["Predictions"].append(y_pred)

    # Store predictions for plotting
    model_predictions[name] = y_pred

    # Print results
    print(f"{name} Results:")
    if hasattr(model, "alpha"):
        print(f"  Best Alpha: {model.alpha}")
    print(f"  R² Score: {r2:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  MSE: {mse:.4f}\n")

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
plt.title("Ridge & Lasso Predictions vs Actual")
plt.legend()
plt.show()
