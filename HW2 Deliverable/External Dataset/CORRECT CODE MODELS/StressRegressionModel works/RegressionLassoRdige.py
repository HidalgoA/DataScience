import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler  # StandardScaler for scaling
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import os

# Paths
csv_path = r"C:\Users\hidal\Desktop\Classes\GRAD\Data Sci Mech of Mat\HW2 Deliverable\External Dataset\All data_Stress"
file_path = r"C:\Users\hidal\Desktop\Classes\GRAD\Data Sci Mech of Mat\HW2 Deliverable\External Dataset\data_all_stress-controlled.xls"

# Load material properties dataset
pd_train = pd.read_excel(file_path, nrows=100000)
pd_train.drop(columns=['Unnamed: 5'], inplace=True, errors='ignore')  # Drop unnecessary column

# Extract CSV file names from dataset and remove NaNs
csv_files = pd_train['load'].dropna().values  

# Read CSV files without log transformation
value_list = []
valid_indices = []
for i, csv_file in enumerate(csv_files):
    try:
        file_path = os.path.join(csv_path, csv_file)
        one_df = pd.read_csv(file_path, header=None).iloc[:, :2]

        # Flatten data and store it in the list
        value_list.append(one_df.values.flatten())  # Flatten for consistency
        valid_indices.append(i)  # Track valid indices
    except Exception as e:
        print(f"Skipping {csv_file}: {e}")

# Convert to numpy array
csv_value_array = np.array(value_list)

# Filter the main dataset to match valid indices
pd_train = pd_train.iloc[valid_indices]

# Data processing pipeline for material properties (exclude CSV and 'Nf(label)' column)
exclude_cols = ['load', 'Nf(label)']
num_cols = pd_train.select_dtypes(exclude=['object']).columns.tolist()
num_cols = [col for col in num_cols if col not in exclude_cols]  # Exclude 'Fatigue Life' column and CSV-related columns

# Define pipeline for numerical columns (material properties)
num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())  # Standard scaling for material properties
])

# Apply transformation only to the selected numerical columns (material properties)
ct = ColumnTransformer(transformers=[('num', num_pipe, num_cols)])

# Transform material properties data (exclude CSV columns)
x_all = ct.fit_transform(pd_train)
y_all = pd_train['Nf(label)'].values

# Ensure y_all has no NaNs
valid_y_indices = ~np.isnan(y_all)

# Apply valid indices to all datasets
x_all = x_all[valid_y_indices]
y_all = y_all[valid_y_indices]

# Scale the CSV data separately
csv_scaler = MinMaxScaler()  # Separate scaling for CSV data
csv_value_array_scaled = csv_scaler.fit_transform(csv_value_array)

# Combine the scaled CSV data with the transformed material properties
combined_data = np.hstack((csv_value_array_scaled, x_all))
y_all = y_all[valid_y_indices]

# Check shapes
print(f"Shape of combined_data: {combined_data.shape}")
print(f"Shape of y_all: {y_all.shape}")

# Split dataset into training and testing
x_train, x_test, y_train, y_test = train_test_split(combined_data, y_all, test_size=0.2, random_state=42)

# Hyperparameter tuning for Ridge and Lasso
alpha_values = np.logspace(-3, 3, 10)

ridge_cv = GridSearchCV(Ridge(), {'alpha': alpha_values}, cv=5)
ridge_cv.fit(x_train, y_train)

lasso_cv = GridSearchCV(Lasso(), {'alpha': alpha_values}, cv=5)
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
