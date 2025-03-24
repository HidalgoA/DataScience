import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def run_bayesian_regression_log(intercept_mu=1.328, intercept_sigma=0.75, 
                                coefs_sigma=0.35, error_sigma_scale=1.5, 
                                n_cores=4):
    # ---------------------------------------------------------
    # 1. Configuration and Paths
    # ---------------------------------------------------------
    csv_folder = r"C:\Users\hidal\Desktop\Classes\GRAD\Data Sci Mech of Mat\HW2 Deliverable\External Dataset\All data_Strain"
    master_file = r"C:\Users\hidal\Desktop\Classes\GRAD\Data Sci Mech of Mat\HW2 Deliverable\External Dataset\data_all_strain-controlled.xls"
    TARGET_LENGTH = 241  # fixed length for each CSV
    material_cols = ['E(Gpa)', 'TS(Mpa)', 'ss£¨Mpa£©', 'm']

    # ---------------------------------------------------------
    # 2. Helper Functions
    # ---------------------------------------------------------
    def infer_metal_type(filename):
        lower_fn = filename.lower()
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
        for key, metal in MATERIAL_MAP.items():
            if key.lower() in lower_fn:
                return metal
        return "Unknown"

    def detect_path_type(col1, col2):
        ZERO_TOL = 1e-6
        RATIO_STD_TOL = 1e-3
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
        if np.std(ratio) < RATIO_STD_TOL:
            return "Proportional"
        else:
            return "Nonproportional"

    def resample_to_fixed_length(x, target_length):
        original_length = len(x)
        if original_length == target_length:
            return x
        original_idx = np.linspace(0, 1, original_length)
        target_idx = np.linspace(0, 1, target_length)
        return np.interp(target_idx, original_idx, x)

    def flatten_and_log_transform_csv(csv_path, target_length=241):
        EPSILON = 1e-6
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

    # ---------------------------------------------------------
    # 3. Load Master Data and Target
    # ---------------------------------------------------------
    df_master = pd.read_excel(master_file)
    df_master.columns = df_master.columns.str.strip()
    print("Master data columns:", df_master.columns.tolist())
    # Assume fatigue life is in 'Nf(label)' and provided on log10 scale.
    y_all = df_master['Nf(label)'].values

    # ---------------------------------------------------------
    # 4. Process Each CSV File: Flatten, Resample, and Log Transform
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # 5. Process Material Properties: Standardize Them
    # ---------------------------------------------------------
    from sklearn.impute import SimpleImputer
    material_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    X_material = material_pipeline.fit_transform(df_master[material_cols])
    df_material = pd.DataFrame(X_material, columns=material_cols)

    # ---------------------------------------------------------
    # 6. Combine All Features into One DataFrame
    # ---------------------------------------------------------
    df_features = pd.concat([
        df_material.reset_index(drop=True),
        df_raw.reset_index(drop=True),
        df_master[['category','metal_type']].reset_index(drop=True)
    ], axis=1)
    print("Combined features shape:", df_features.shape)
    df_features.dropna(inplace=True)
    y_all = y_all[df_features.index]

    # ---------------------------------------------------------
    # 7. Define Independent (X) and Dependent (y) Variables
    # ---------------------------------------------------------
    numerical_cols = material_cols + raw_feature_cols
    categorical_cols = ['category', 'metal_type']
    X = df_features[numerical_cols + categorical_cols]
    y = y_all  # Fatigue life in log10 scale

    # ---------------------------------------------------------
    # 8. Build a ColumnTransformer for Preprocessing
    # ---------------------------------------------------------
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_cols),
            ('cat', OneHotEncoder(drop='first'), categorical_cols)
        ]
    )
    X_processed = preprocessor.fit_transform(X)
    
    # ---------------------------------------------------------
    # 8.1. PCA Step to Reduce Dimensionality
    # ---------------------------------------------------------
    from sklearn.decomposition import PCA
    # Retain 95% of the variance in the data
    pca = PCA(n_components=0.99999, svd_solver='full')
    X_processed = pca.fit_transform(X_processed)
    print("Reduced feature shape after PCA:", X_processed.shape)
    
    # ---------------------------------------------------------
    # 9. Train-Test Split
    # ---------------------------------------------------------
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42
    )
    correlation_matrix = pd.DataFrame(X_train).corr()
    print(correlation_matrix)

    # ---------------------------------------------------------
    # 10. Bayesian Linear Regression Model using PyMC
    '''
    intercept_pm = pm.Normal("intercept", mu=3.5, sigma=0.8)
        coefs_pm = pm.Normal("coefs", mu=0, sigma=0.3, shape=X_train.shape[1])
        sigma_pm = pm.HalfNormal("sigma", sigma=0.1)
    '''
    # ---------------------------------------------------------
    # We assume y (fatigue life) is provided on log10 scale.
    with pm.Model() as bayes_model:
        intercept_pm = pm.Normal("intercept", mu=3.50, sigma=0.50)
        coefs_pm = pm.Normal("coefs", mu=0, sigma=0.2, shape=X_train.shape[1])  

        sigma_pm = pm.HalfNormal("sigma", sigma=0.1)

        mu_pm = intercept_pm + pm.math.dot(X_train, coefs_pm)
        y_obs = pm.Normal("y_obs", mu=mu_pm, sigma=sigma_pm, observed=y_train)
        trace = pm.sample(2000, tune=4000, chains = 4, target_accept=0.95, random_seed=42, cores=n_cores, init="jitter+adapt_diag")

    # ---------------------------------------------------------
    # 11. Posterior Summary and Predictions on Test Set
    # ---------------------------------------------------------
    print(az.summary(trace, round_to=2))
    # Flatten the intercept and reshape the coefficients so they align correctly:
    posterior_intercept = trace.posterior["intercept"].values.flatten()  # shape: (n_draws,)
    posterior_coefs = trace.posterior["coefs"].values.reshape(-1, X_train.shape[1])  # shape: (n_draws, n_features)

    y_pred_samples = posterior_intercept[:, None] + np.dot(posterior_coefs, X_test.T)
    y_pred_log_mean = y_pred_samples.mean(axis=0)
    y_pred =  (y_pred_log_mean)  # back-transform from log10 scale to fatigue life
    y_test_orig =  (y_test)
    
    # ---------------------------------------------------------
    # 12. Evaluation Metrics
    # ---------------------------------------------------------
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    mse = mean_squared_error(y_test_orig, y_pred)
    mae = mean_absolute_error(y_test_orig, y_pred)
    r2 = r2_score(y_test_orig, y_pred)
    print("Bayesian Regression Performance on Test Set:")
    print("MSE:", mse)
    print("MAE:", mae)
    print("R2 Score:", r2)
    
    # ---------------------------------------------------------
    # 13. Plot: Actual vs Predicted Fatigue Life
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test_orig, y_pred, alpha=0.7, edgecolor="k")
    plt.plot([y_test_orig.min(), y_test_orig.max()],
             [y_test_orig.min(), y_test_orig.max()], "r--")
    plt.xlabel("Actual Fatigue Life")
    plt.ylabel("Predicted Fatigue Life")
    plt.title("Bayesian Regression: Actual vs Predicted Fatigue Life")
    plt.grid(True)
    plt.show()
    
    # Optional: Plot Trace and Posterior Distributions
    pm.plot_trace(trace, figsize=(12, 12))
    plt.show()
    az.plot_posterior(trace)
    plt.show()
    
    return trace

if __name__ == '__main__':
    run_bayesian_regression_log(intercept_mu=3.50, intercept_sigma=0.80, 
                                coefs_sigma=0.30, error_sigma_scale=0.5, 
                                n_cores=4)
