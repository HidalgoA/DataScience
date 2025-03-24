import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import csv
import os


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

# Hyperparameter tuning for SVM and RandomForest
svm_param_grid = {'C': [0.1, 1, 10, 100], 'epsilon': [0.01, 0.1, 1]}
svm_cv = GridSearchCV(SVR(kernel='rbf'), svm_param_grid, cv=5)
svm_cv.fit(x_train, y_train)

rf_param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
rf_cv = GridSearchCV(RandomForestRegressor(), rf_param_grid, cv=5)
rf_cv.fit(x_train, y_train)

# Best hyperparameters
best_svm_params = svm_cv.best_params_
best_rf_params = rf_cv.best_params_

# Initialize models with optimized hyperparameters
models = {
    "SVM Regression": SVR(kernel='rbf', C=best_svm_params['C'], epsilon=best_svm_params['epsilon']),
    "Random Forest Regression": RandomForestRegressor(n_estimators=best_rf_params['n_estimators'], max_depth=best_rf_params['max_depth'])
}

# Train and evaluate models
results = []
plt.figure(figsize=(12, 6))
for name, model in models.items():
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    print(f"{name} Results:")
    print(f"  R² Score: {r2:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  MSE: {mse:.4f}\n")
    
    results.append([name, r2, mae, mse])
    
    # Visualization
    sns.regplot(x=y_test, y=y_pred, label=name)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Regression Model Predictions vs Actual Values")

plt.legend()
plt.show()

# Save results to CSV
output_file = "Regression_Results_SVM_RF.csv"
with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Model", "R² Score", "MAE", "MSE"])
    writer.writerows(results)

print(f"Results saved to {output_file}")
