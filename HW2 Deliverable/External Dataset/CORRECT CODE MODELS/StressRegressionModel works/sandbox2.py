import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

from scipy.special import inv_boxcox
from scipy.stats import boxcox
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, RobustScaler, 
                                    PowerTransformer, QuantileTransformer)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

############################################
# 1. Configuration & Paths
############################################
csv_folder = r"C:\Users\hidal\Desktop\Classes\GRAD\Data Sci Mech of Mat\HW2 Deliverable\External Dataset\CORRECT CODE MODELS\All data_Stress"
master_file = r"C:\Users\hidal\Desktop\Classes\GRAD\Data Sci Mech of Mat\HW2 Deliverable\External Dataset\CORRECT CODE MODELS\data_all_stress-controlled.xls"
TARGET_LENGTH = 241  # Fixed length for CSV resampling

# Minimal MATERIAL_MAP based on stress filenames
MATERIAL_MAP = {
    "5% chrome work roll steel": "Carbon and Alloy Steel",
    "30CrMnSiA": "Carbon and Alloy Steel",
    "SM45C": "Carbon and Alloy Steel",
    "2024": "Aluminum Alloy",
    "2198": "Aluminum Alloy",
    "6082": "Aluminum Alloy",
    "7075": "Aluminum Alloy",
    "LY12CZ": "Aluminum Alloy"
}

ZERO_TOL = 1e-6
RATIO_STD_TOL = 1e-3
EPSILON = 1e-6

############################################
# 2. Helper Functions (common to all)
############################################
def infer_metal_type(filename):
    """Infers metal type from filename based on MATERIAL_MAP."""
    lower_fn = filename.lower()
    for key, metal in MATERIAL_MAP.items():
        if key.lower() in lower_fn:
            return metal
    return "Unknown"

def detect_path_type(col1, col2):
    """Determines loading path type (for diagnostics)."""
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
    """Resamples 1D array x to target_length points using linear interpolation."""
    orig_len = len(x)
    if orig_len == target_length:
        return x
    orig_idx = np.linspace(0, 1, orig_len)
    tgt_idx = np.linspace(0, 1, target_length)
    return np.interp(tgt_idx, orig_idx, x)

############################################
# 3. CSV Transformation Functions
############################################
# Option 1: No additional transform (just flatten)
def flatten_and_csv_none(csv_path, target_length=241):
    df_csv = pd.read_csv(csv_path, header=None)
    col1 = df_csv.iloc[:, 0].values
    col2 = df_csv.iloc[:, 1].values
    return np.concatenate([resample_to_fixed_length(col1, target_length),
                           resample_to_fixed_length(col2, target_length)])

# Option 2: Shift then log transform
def flatten_and_log_transform_csv(csv_path, target_length=241):
    df_csv = pd.read_csv(csv_path, header=None)
    col1 = df_csv.iloc[:, 0].values
    col2 = df_csv.iloc[:, 1].values
    col1_res = resample_to_fixed_length(col1, target_length)
    col2_res = resample_to_fixed_length(col2, target_length)
    combined = np.concatenate([col1_res, col2_res])
    shift = abs(np.min(combined)) + EPSILON if np.min(combined) <= 0 else 0
    shifted = combined + shift
    return np.log(shifted)

# Option 3: Min–max normalization then log transform
def flatten_and_normalize_then_log_transform_csv(csv_path, target_length=241):
    df_csv = pd.read_csv(csv_path, header=None)
    col1 = df_csv.iloc[:, 0].values
    col2 = df_csv.iloc[:, 1].values
    col1_res = resample_to_fixed_length(col1, target_length)
    col2_res = resample_to_fixed_length(col2, target_length)
    combined = np.concatenate([col1_res, col2_res])
    shift = abs(np.min(combined)) + EPSILON if np.min(combined) <= 0 else 0
    shifted = combined + shift
    norm = (shifted - shifted.min()) / (shifted.max() - shifted.min())
    norm += EPSILON  # avoid log(0)
    return np.log(norm)

# Option 4: Standardize (z-score) then log transform
def flatten_and_standardize_then_log_transform_csv(csv_path, target_length=241):
    df_csv = pd.read_csv(csv_path, header=None)
    col1 = df_csv.iloc[:, 0].values
    col2 = df_csv.iloc[:, 1].values
    col1_res = resample_to_fixed_length(col1, target_length)
    col2_res = resample_to_fixed_length(col2, target_length)
    combined = np.concatenate([col1_res, col2_res])
    mean_val = np.mean(combined)
    std_val = np.std(combined)
    standardized = (combined - mean_val) / std_val if std_val != 0 else combined - mean_val
    shift = abs(np.min(standardized)) + EPSILON if np.min(standardized) <= 0 else 0
    shifted = standardized + shift
    return np.log(shifted)

# Option 5: Standardize then normalize then log transform
def flatten_and_std_then_norm_log_transform_csv(csv_path, target_length=241):
    df_csv = pd.read_csv(csv_path, header=None)
    col1 = df_csv.iloc[:, 0].values
    col2 = df_csv.iloc[:, 1].values
    col1_res = resample_to_fixed_length(col1, target_length)
    col2_res = resample_to_fixed_length(col2, target_length)
    combined = np.concatenate([col1_res, col2_res])
    # Standardize
    mean_val = np.mean(combined)
    std_val = np.std(combined)
    standardized = (combined - mean_val) / std_val if std_val != 0 else combined - mean_val
    shift = abs(np.min(standardized)) + EPSILON if np.min(standardized) <= 0 else 0
    shifted = standardized + shift
    # Normalize the standardized values
    norm = (shifted - shifted.min()) / (shifted.max() - shifted.min())
    norm += EPSILON
    return np.log(norm)

# Option 6: Normalize then standardize then log transform
def flatten_and_norm_then_std_log_transform_csv(csv_path, target_length=241):
    df_csv = pd.read_csv(csv_path, header=None)
    col1 = df_csv.iloc[:, 0].values
    col2 = df_csv.iloc[:, 1].values
    col1_res = resample_to_fixed_length(col1, target_length)
    col2_res = resample_to_fixed_length(col2, target_length)
    combined = np.concatenate([col1_res, col2_res])
    shift = abs(np.min(combined)) + EPSILON if np.min(combined) <= 0 else 0
    shifted = combined + shift
    norm = (shifted - shifted.min()) / (shifted.max() - shifted.min())
    norm += EPSILON
    mean_val = np.mean(norm)
    std_val = np.std(norm)
    standardized = (norm - mean_val) / std_val if std_val != 0 else norm - mean_val
    # Ensure positivity before log transform
    shift_std = abs(np.min(standardized)) + EPSILON if np.min(standardized) <= 0 else 0
    shifted_std = standardized + shift_std
    return np.log(shifted_std)

# Option 7: Quantile transformation then log transform
def flatten_and_quantile_then_log_transform_csv(csv_path, target_length=241):
    df_csv = pd.read_csv(csv_path, header=None)
    col1 = df_csv.iloc[:, 0].values
    col2 = df_csv.iloc[:, 1].values
    col1_res = resample_to_fixed_length(col1, target_length)
    col2_res = resample_to_fixed_length(col2, target_length)
    combined = np.concatenate([col1_res, col2_res])
    shift = abs(np.min(combined)) + EPSILON if np.min(combined) <= 0 else 0
    shifted = combined + shift
    qt = QuantileTransformer(output_distribution="uniform", random_state=42)
    qtrans = qt.fit_transform(shifted.reshape(-1, 1)).ravel()
    qtrans += EPSILON
    return np.log(qtrans)

csv_transform_options = {
    "csv_none": flatten_and_csv_none,
    "csv_log": flatten_and_log_transform_csv,
    "csv_norm_log": flatten_and_normalize_then_log_transform_csv,
    "csv_std_log": flatten_and_standardize_then_log_transform_csv,
    "csv_std_then_norm_log": flatten_and_std_then_norm_log_transform_csv,
    "csv_norm_then_std_log": flatten_and_norm_then_std_log_transform_csv,
    "csv_quantile_log": flatten_and_quantile_then_log_transform_csv
}

############################################
# 4. Material Property Transformation Functions
############################################
def material_none(X):
    return X

def material_standardize(X):
    return StandardScaler().fit_transform(X)

def material_normalize(X):
    return MinMaxScaler().fit_transform(X)

def material_robust(X):
    return RobustScaler().fit_transform(X)

def material_std_then_norm(X):
    X_std = StandardScaler().fit_transform(X)
    return MinMaxScaler().fit_transform(X_std)

def material_norm_then_std(X):
    X_norm = MinMaxScaler().fit_transform(X)
    return StandardScaler().fit_transform(X_norm)

def material_quantile(X):
    qt = QuantileTransformer(output_distribution="uniform", random_state=42)
    return qt.fit_transform(X)

material_transform_options = {
    "material_none": material_none,
    "material_standardize": material_standardize,
    "material_normalize": material_normalize,
    "material_robust": material_robust,
    "material_std_then_norm": material_std_then_norm,
    "material_norm_then_std": material_norm_then_std,
    "material_quantile": material_quantile
}

############################################
# 5. Target Transformation Functions
############################################
def target_none(y):
    return y, None, lambda x: x

def target_log(y):
    if np.any(y <= 0):
        y = y + abs(np.min(y)) + 1e-6
    return np.log(y), None, lambda x: np.exp(x)

def target_boxcox(y):
    if np.any(y <= 0):
        y = y + abs(np.min(y)) + 1e-6
    y_trans, lam = boxcox(y)
    return y_trans, lam, lambda x: inv_boxcox(x, lam)

def target_yeojohnson(y):
    pt = PowerTransformer(method="yeo-johnson")
    y_trans = pt.fit_transform(y.reshape(-1,1)).ravel()
    return y_trans, pt, lambda x: pt.inverse_transform(x.reshape(-1,1)).ravel()

def target_quantile(y):
    qt = QuantileTransformer(output_distribution="uniform", random_state=42)
    y_trans = qt.fit_transform(y.reshape(-1,1)).ravel()
    return y_trans, qt, lambda x: qt.inverse_transform(x.reshape(-1,1)).ravel()

target_transform_options = {
    "target_none": target_none,
    "target_log": target_log,
    "target_boxcox": target_boxcox,
    "target_yeojohnson": target_yeojohnson,
    "target_quantile": target_quantile
}

############################################
# 6. Load Master Data and Filter CSV Files
############################################
pd_train = pd.read_excel(master_file, nrows=100000)
pd_train.drop(columns=['Unnamed: 5'], inplace=True, errors='ignore')
print("Master data columns:", pd_train.columns.tolist())

csv_files_ref = set(pd_train['load'].dropna().values)
folder_files = os.listdir(csv_folder)
csv_files = [f for f in folder_files if f in csv_files_ref]
pd_train = pd_train[pd_train['load'].isin(csv_files)].reset_index(drop=True)

material_cols = ['E(Gpa)', 'TS(Mpa)', 'ss£¨Mpa£©', 'm']
y_all = pd_train['Nf(label)'].values
if np.any(y_all <= 0):
    y_all = y_all + abs(np.min(y_all)) + 1e-6

############################################
# 7. Experiment Loop Over All Combinations
############################################
results = []
total_combos = (len(csv_transform_options) *
                len(material_transform_options) *
                len(target_transform_options) *
                3)  # 3 models: Ridge, Lasso, ElasticNet

combo_count = 0
for csv_key, csv_transform_func in csv_transform_options.items():
    print(f"\nCSV Transform: {csv_key}")
    # Process CSV files using current CSV transformation function
    raw_features_list = []
    for csv_file in pd_train['load']:
        full_path = os.path.join(csv_folder, csv_file)
        try:
            raw_vec = csv_transform_func(full_path, TARGET_LENGTH)
        except Exception as e:
            print(f"  Error processing {csv_file} with {csv_key}: {e}")
            raw_vec = np.full((2*TARGET_LENGTH,), np.nan)
        raw_features_list.append(raw_vec)
    raw_feature_cols = [f"raw_{i}" for i in range(2*TARGET_LENGTH)]
    df_csv_features = pd.DataFrame(raw_features_list, columns=raw_feature_cols)
    
    # Loop over material transformation options
    for mat_key, mat_transform_func in material_transform_options.items():
        print(f"  Material Transform: {mat_key}")
        X_material = mat_transform_func(pd_train[material_cols])
        df_material = pd.DataFrame(X_material, columns=material_cols)
        
        # Combine CSV features and material properties
        df_features = pd.concat([df_material.reset_index(drop=True), 
                                 df_csv_features.reset_index(drop=True)], axis=1)
        df_features.dropna(inplace=True)
        y_subset = y_all[df_features.index]
        
        # Loop over target transformation options
        for targ_key, targ_transform_func in target_transform_options.items():
            try:
                y_trans, targ_obj, inv_transform = targ_transform_func(y_subset.copy())
            except Exception as e:
                print(f"    Target transform {targ_key} failed: {e}")
                continue
            
            # Define features and target for modeling
            numerical_cols_final = list(df_features.columns)
            X = df_features[numerical_cols_final]
            y_model = y_trans
            
            # Build preprocessor: here just pass through numerical features
            preprocessor = ColumnTransformer(
                transformers=[('num', 'passthrough', numerical_cols_final)]
            )
            
            # Split data (fixed random_state for consistency)
            X_train, X_test, y_train, y_test = train_test_split(X, y_model, test_size=0.2, random_state=42)
            
            # Loop over models
            for model_name, model_class in [("Ridge", Ridge), ("Lasso", Lasso), ("ElasticNet", ElasticNet)]:
                combo_count += 1
                print(f"    Combo {combo_count}/{total_combos}: csv={csv_key}, mat={mat_key}, targ={targ_key}, model={model_name}")
                pipeline = Pipeline([
                    ('preprocessor', preprocessor),
                    ('regressor', model_class())
                ])
                param_grid = {}
                if model_name in ["Ridge", "Lasso", "ElasticNet"]:
                    param_grid['regressor__alpha'] = np.logspace(-3, 3, 10)
                if model_name == "ElasticNet":
                    param_grid['regressor__l1_ratio'] = [0.2, 0.5, 0.8]
                    
                grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2', error_score='raise')
                try:
                    grid.fit(X_train, y_train)
                except Exception as e:
                    print(f"      {model_name} failed with combination: {e}")
                    continue
                cv_score = grid.best_score_
                best_params = grid.best_params_
                y_pred = grid.predict(X_test)
                test_r2 = r2_score(y_test, y_pred)
                try:
                    y_test_inv = inv_transform(y_test)
                    y_pred_inv = inv_transform(y_pred)
                    test_r2_inv = r2_score(y_test_inv, y_pred_inv)
                except Exception:
                    test_r2_inv = test_r2
                results.append({
                    "csv_transform": csv_key,
                    "material_transform": mat_key,
                    "target_transform": targ_key,
                    "model": model_name,
                    "best_params": best_params,
                    "cv_score": cv_score,
                    "test_r2": test_r2,
                    "test_r2_inv": test_r2_inv
                })
                print(f"      {model_name}: cv_score={cv_score:.4f}, test_r2_inv={test_r2_inv:.4f}")

results_df = pd.DataFrame(results)
print("\nAll Results (sorted by test R² on original scale):")
print(results_df.sort_values(by="test_r2_inv", ascending=False))

results_df.to_excel("experiment_results.xlsx", index=False)

