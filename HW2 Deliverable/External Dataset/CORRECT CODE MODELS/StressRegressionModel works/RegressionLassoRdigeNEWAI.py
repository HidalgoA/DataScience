import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import csv

##############################################
# 1. Configuration and Paths
##############################################
csv_folder = r"C:\Users\hidal\Desktop\Classes\GRAD\Data Sci Mech of Mat\HW2 Deliverable\External Dataset\CORRECT CODE MODELS\All data_Stress"
master_file = r"C:\Users\hidal\Desktop\Classes\GRAD\Data Sci Mech of Mat\HW2 Deliverable\External Dataset\CORRECT CODE MODELS\data_all_stress-controlled.xls"

TARGET_LENGTH = 241  # Desired fixed length for resampling CSV data

MATERIAL_MAP = {
    "1Cr18Ni9T": "Stainless Steel",
    "S347": "Stainless Steel",
    "X5CrNi18-10": "Stainless Steel",
    "316L": "Stainless Steel",
    "AISI316": "Stainless Steel",
    "304": "Stainless Steel",
    "410": "Stainless Steel",
    "7075": "Aluminum Alloy",
    "2024": "Aluminum Alloy",
    "2198": "Aluminum Alloy",
    "6061": "Aluminum Alloy",
    "6082": "Aluminum Alloy",
    "LY12CZ": "Aluminum Alloy",
    "Al5083": "Aluminum Alloy",
    "PA38": "Aluminum Alloy",
    "CP-Ti": "Titanium Alloy",
    "TC4": "Titanium Alloy",
    "PureTi": "Titanium Alloy",
    "BT9": "Titanium Alloy",
    "AZ61A": "Magnesium Alloy",
    "AZ31B": "Magnesium Alloy",
    "ZK60": "Magnesium Alloy",
    "16MnR": "Carbon and Alloy Steel",
    "S45C": "Carbon and Alloy Steel",
    "30CrMnSiA": "Carbon and Alloy Steel",
    "Q235b": "Carbon and Alloy Steel",
    "HRB335": "Carbon and Alloy Steel",
    "E235": "Carbon and Alloy Steel",
    "E355": "Carbon and Alloy Steel",
    "SA333": "Carbon and Alloy Steel",
    "45#": "Carbon and Alloy Steel",
    "S460N": "Carbon and Alloy Steel",
    "1045HR": "Carbon and Alloy Steel",
    "MildSteel": "Carbon and Alloy Steel",
    "SM45C": "Carbon and Alloy Steel",
    "SNCM630": "Carbon and Alloy Steel",
    "CuZn37": "Copper Alloy",
    "GH4169": "Nickel-Based Alloy",
    "Hayes188": "Nickel-Based Alloy",
    "Inconel718": "Nickel-Based Alloy"
}

ZERO_TOL = 1e-6
RATIO_STD_TOL = 1e-3
EPSILON = 1e-6

##############################################
# 2. Helper Functions
##############################################
def infer_metal_type(filename):
    lower_fn = filename.lower()
    for key, metal in MATERIAL_MAP.items():
        if key.lower() in lower_fn:
            return metal
    return "Unknown"

def detect_path_type(col1, col2):
    max_abs_col1 = np.max(np.abs(col1))
    max_abs_col2 = np.max(np.abs(col2))
    if max_abs_col2 < ZERO_TOL and max_abs_col1 >= ZERO_TOL:
        return "Uniaxial"
    if max_abs_col1 < ZERO_TOL and max_abs_col2 >= ZERO_TOL:
        return "Pure Shear"
    valid_mask = np.abs(col1) > ZERO_TOL
    if not np.any(valid_mask):
        return "Pure Shear"
    ratio = col2[valid_mask] / col1[valid_mask]
    return "Proportional" if np.std(ratio) < RATIO_STD_TOL else "Nonproportional"

def resample_to_fixed_length(x, target_length):
    original_length = len(x)
    if original_length == target_length:
        return x
    original_idx = np.linspace(0, 1, original_length)
    target_idx = np.linspace(0, 1, target_length)
    return np.interp(target_idx, original_idx, x)

def flatten_and_log_transform_csv(csv_path, target_length=241):
    df_csv = pd.read_csv(csv_path, header=None)
    col1 = df_csv.iloc[:, 0].values
    col2 = df_csv.iloc[:, 1].values

    col1_resampled = resample_to_fixed_length(col1, target_length)
    col2_resampled = resample_to_fixed_length(col2, target_length)
    combined = np.concatenate([col1_resampled, col2_resampled])

    min_val = np.min(combined)
    shift = abs(min_val) + EPSILON if min_val <= 0 else 0
    shifted = combined + shift
    log_transformed = np.log(shifted)
    return log_transformed

##############################################
# 3. Load Master Data and Yeo–Johnson Target
##############################################
df_master = pd.read_excel(master_file)
df_master.drop(columns=['Unnamed: 5'], inplace=True, errors='ignore')
print("Master data columns:", df_master.columns.tolist())

material_cols = ['E(Gpa)', 'TS(Mpa)', 'ss£¨Mpa£©', 'm']
y_all = df_master['Nf(label)'].values

if np.any(y_all <= 0):
    shift_target = abs(np.min(y_all)) + EPSILON
    y_all = y_all + shift_target
else:
    shift_target = 0

pt_target = PowerTransformer(method='yeo-johnson')
y_all_transformed = pt_target.fit_transform(y_all.reshape(-1, 1)).ravel()

##############################################
# 4. Process Each CSV (Log Transform)
##############################################
raw_features_list = []
category_list = []
metal_type_list = []

for csv_file in df_master['load']:
    full_path = os.path.join(csv_folder, csv_file)
    metal = infer_metal_type(csv_file)
    try:
        raw_vec = flatten_and_log_transform_csv(full_path, TARGET_LENGTH)
        half = len(raw_vec) // 2
        col1 = raw_vec[:half]
        col2 = raw_vec[half:]
        path_cat = detect_path_type(col1, col2)
        
        raw_features_list.append(raw_vec)
        category_list.append(path_cat)
        metal_type_list.append(metal)
    except Exception as e:
        print(f"Error processing {csv_file}: {e}")
        raw_features_list.append([np.nan]*(2*TARGET_LENGTH))
        category_list.append("Error")
        metal_type_list.append("Unknown")

num_raw_features = 2 * TARGET_LENGTH
raw_feature_cols = [f"raw_{i}" for i in range(num_raw_features)]
df_raw = pd.DataFrame(raw_features_list, columns=raw_feature_cols)

df_master['category'] = category_list
df_master['metal_type'] = metal_type_list

##############################################
# 5. Material Properties (Standardized)
##############################################
material_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
X_material = material_pipeline.fit_transform(df_master[material_cols])
df_material = pd.DataFrame(X_material, columns=material_cols)

##############################################
# 6. Combine All Features
##############################################
df_features = pd.concat([
    df_material.reset_index(drop=True),
    df_raw.reset_index(drop=True),
    df_master[['category','metal_type']].reset_index(drop=True)
], axis=1)
print("Combined features shape:", df_features.shape)
print(df_features.head())

df_features.dropna(inplace=True)
y_all_transformed = y_all_transformed[df_features.index]

numerical_cols = material_cols + raw_feature_cols
categorical_cols = ['category', 'metal_type']

X = df_features[numerical_cols + categorical_cols]
y = y_all_transformed

##############################################
# 7. Preprocessor and Model Pipelines
##############################################
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
    ]
)

ridge_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', Ridge())
])

lasso_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', Lasso(max_iter=10000))
])

elasticnet_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', ElasticNet(max_iter=10000))
])

##############################################
# 8. Train/Test Split
##############################################
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

##############################################
# 9. Hyperparameter Tuning with GridSearchCV
##############################################
alpha_values = np.logspace(-3, 3, 10)

# Ridge
ridge_param_grid = {'regressor__alpha': alpha_values}
ridge_cv = GridSearchCV(ridge_pipeline, ridge_param_grid, cv=5)
ridge_cv.fit(X_train, y_train)
print("Ridge best params:", ridge_cv.best_params_)
y_pred_ridge = ridge_cv.predict(X_test)

# Lasso
lasso_param_grid = {'regressor__alpha': alpha_values}
lasso_cv = GridSearchCV(lasso_pipeline, lasso_param_grid, cv=5)
lasso_cv.fit(X_train, y_train)
print("Lasso best params:", lasso_cv.best_params_)
y_pred_lasso = lasso_cv.predict(X_test)

# ElasticNet
enet_param_grid = {
    'regressor__alpha': alpha_values,
    'regressor__l1_ratio': [0.2, 0.5, 0.8]
}
enet_cv = GridSearchCV(elasticnet_pipeline, enet_param_grid, cv=5)
enet_cv.fit(X_train, y_train)
print("ElasticNet best params:", enet_cv.best_params_)
y_pred_enet = enet_cv.predict(X_test)

##############################################
# 10. Inverse Transform + Evaluate
##############################################
# Invert Yeo–Johnson transformation on predictions and on the test target
y_test_orig = pt_target.inverse_transform(y_test.reshape(-1,1)).ravel()

y_pred_ridge_orig = pt_target.inverse_transform(y_pred_ridge.reshape(-1,1)).ravel()
y_pred_lasso_orig = pt_target.inverse_transform(y_pred_lasso.reshape(-1,1)).ravel()
y_pred_enet_orig = pt_target.inverse_transform(y_pred_enet.reshape(-1,1)).ravel()

def evaluate_model(name, y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    print(f"{name} Results (Original Scale):")
    print(f"  R² Score: {r2:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  MSE: {mse:.4f}\n")

# Evaluate each model
evaluate_model("Ridge Regression", y_test_orig, y_pred_ridge_orig)
evaluate_model("Lasso Regression", y_test_orig, y_pred_lasso_orig)
evaluate_model("ElasticNet Regression", y_test_orig, y_pred_enet_orig)

##############################################
# 12. Prediction vs Actual Plot
##############################################
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test_orig, y=y_pred_ridge_orig, color='blue', alpha=0.6, label='Ridge')
sns.scatterplot(x=y_test_orig, y=y_pred_lasso_orig, color='red', alpha=0.6, label='Lasso')
sns.scatterplot(x=y_test_orig, y=y_pred_enet_orig, color='green', alpha=0.6, label='ElasticNet')
plt.plot([min(y_test_orig), max(y_test_orig)], [min(y_test_orig), max(y_test_orig)], '--k', label="Perfect Fit")
plt.xlabel("Actual Fatigue Life (Orig Scale)")
plt.ylabel("Predicted Fatigue Life (Orig Scale)")
plt.title("Model Predictions vs Actual (Inverted Target)")
plt.legend()
plt.show()
