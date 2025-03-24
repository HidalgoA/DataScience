import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score
import itertools
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def load_data():
    # Update this file path as needed.
    transformed_data_file = r"C:\Users\hidal\Desktop\Classes\GRAD\Data Sci Mech of Mat\HW2 Deliverable\Beam Dataset SANDBOX\Stress Analysis\Transformed_Stress_Data.xlsx"
    df = pd.read_excel(transformed_data_file, sheet_name="Sheet1")
    df.columns = df.columns.str.strip()
    df["Stress_Values"] = pd.to_numeric(df["Stress_Values"], errors="coerce")
    df = df.dropna(subset=["Stress_Values"])
    return df

def run_model_with_hyperparams_advi(X_train, X_test, y_train_log, y_test_log,
                                    intercept_sigma, coefs_sigma, error_sigma_scale):
    with pm.Model() as model:
        # Intercept prior: center at mean log-stress (assumed around 6)
        intercept = pm.Normal("intercept", mu=np.mean(y_train_log), sigma=intercept_sigma)
        # Coefficient priors for scaled predictors
        coefs = pm.Normal("coefs", mu=0, sigma=coefs_sigma, shape=X_train.shape[1])
        # Prior for residual error on log scale
        sigma = pm.HalfNormal("sigma", sigma=error_sigma_scale)
        # Linear predictor on log scale
        mu = intercept + pm.math.dot(X_train, coefs)
        # Likelihood with Normal (no Student's T)
        y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_train_log)
        
        # Use ADVI for fast approximate inference.
        approx = pm.fit(n=10000, method='advi', progressbar=False)
        trace = approx.sample(draws=200)  # sample 200 draws from the approximate posterior

    # Extract posterior samples (stack chains and draws, then transpose for proper dimensions)
    posterior_coefs = trace.posterior["coefs"].stack(sample=("chain", "draw")).values.T
    posterior_intercept = trace.posterior["intercept"].stack(sample=("chain", "draw")).values

    # Compute predictions on log scale for each posterior sample
    y_pred_log_samples = np.dot(posterior_coefs, X_test.T) + posterior_intercept[:, np.newaxis]
    y_pred_log_mean = y_pred_log_samples.mean(axis=0)
    # Back-transform predictions using the inverse of log1p.
    y_pred = np.expm1(y_pred_log_mean)
    y_test_orig = np.expm1(y_test_log)
    
    return r2_score(y_test_orig, y_pred)

def master_optimizer():
    df = load_data()
    X = df.drop(columns=["Stress_Values"])
    y = df["Stress_Values"].values

    # Use log1p transformation
    y_log = np.log1p(y)

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train_log, y_test_log = train_test_split(X_scaled, y_log, test_size=0.2, random_state=42)

    # Define grid of hyperparameters
    intercept_sigmas = [1.0, 1.5, 2.0]
    coefs_sigmas = [0.5, 1.0, 2.0]
    error_sigma_scales = [1.0, 2.0, 5.0]

    best_r2 = -np.inf
    best_params = None
    results = []
    
    for intercept_sigma, coefs_sigma, error_sigma_scale in itertools.product(intercept_sigmas, coefs_sigmas, error_sigma_scales):
        print(f"Testing: intercept_sigma={intercept_sigma}, coefs_sigma={coefs_sigma}, error_sigma_scale={error_sigma_scale} ...", end=" ")
        try:
            r2 = run_model_with_hyperparams_advi(X_train, X_test, y_train_log, y_test_log,
                                                 intercept_sigma, coefs_sigma, error_sigma_scale)
            print(f"R2 = {r2:.4f}")
            results.append((intercept_sigma, coefs_sigma, error_sigma_scale, r2))
            if r2 > best_r2:
                best_r2 = r2
                best_params = (intercept_sigma, coefs_sigma, error_sigma_scale)
        except Exception as e:
            print("failed:", e)
    
    print("\nOptimization complete!")
    print("Best R²:", best_r2)
    print("Best hyperparameters:")
    print("  intercept_sigma:", best_params[0])
    print("  coefs_sigma:", best_params[1])
    print("  error_sigma_scale:", best_params[2])
    return results

if __name__ == '__main__':
    master_optimizer()
