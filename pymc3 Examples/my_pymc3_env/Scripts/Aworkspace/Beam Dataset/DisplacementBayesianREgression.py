import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def run_bayesian_regression_log(intercept_mu=1.3283, intercept_sigma=0.1, 
                                                coefs_sigma=0.5, 
                                                error_sigma_scale=5.0, 
                                                n_cores=4):
    
    '''
    # Intercept prior based on log-displacement statistics
    intercept = pm.Normal("intercept", mu=2.0, sigma=1.0)
    
    # Coefficients for scaled predictors
    coefs = pm.Normal("coefs", mu=0, sigma=2.0, shape=X_train.shape[1])
    
    # Residual error prior
    sigma = pm.HalfNormal("sigma", sigma=1.0)
    
    # Linear predictor on the log scale
    mu_pred = intercept + pm.math.dot(X_train, coefs)
    y       = b         + xValues * slope
    
    '''
    """
    Runs a Bayesian linear regression on log-transformed displacement data using a Normal likelihood.
    
    Parameters:
      intercept_mu: float or None. The prior mean for the intercept. If None, it is set to the mean of the log-transformed training target.
      intercept_sigma: float. The prior standard deviation for the intercept.
      coefs_sigma: float. The prior standard deviation for the coefficients.
      error_sigma_scale: float. The scale parameter for the HalfNormal prior on the residual error.
      n_cores: int. The number of cores to use for sampling.
    """
    # 1. Load and Preprocess the Data
    transformed_data_file = r"C:\Users\hidal\Desktop\Classes\GRAD\Data Sci Mech of Mat\HW2 Deliverable\Beam Dataset SANDBOX Correct Code\Displacement Analysis\Transformed_Displ_Data.xlsx"
    
    df = pd.read_excel(transformed_data_file, sheet_name="Sheet1")
    df.columns = df.columns.str.strip()
    df["Displacement_Values"] = pd.to_numeric(df["Displacement_Values"], errors="coerce")
    df = df.dropna(subset=["Displacement_Values"])
    
    # 2. Define Features and Target, then Scale Features
    X = df.drop(columns=["Displacement_Values"])
    y = df["Displacement_Values"].values
    
    # Log-transform the target to stabilize variance (same approach as stress)
    y_log = np.log1p(y)
    
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. Train-Test Split
    X_train, X_test, y_train_log, y_test_log = train_test_split(
        X_scaled, y_log, test_size=0.2, random_state=42
    )
    
    # If no intercept_mu is provided, default to the mean of the training log-target
    if intercept_mu is None:
        intercept_mu = np.mean(y_train_log)
    
    print("Intercept prior mean set to:", intercept_mu)
    
    # 4. Build the Bayesian Linear Regression Model
    with pm.Model() as bayes_model:
        # Same priors as before
        intercept = pm.Normal("intercept", mu=intercept_mu, sigma=intercept_sigma)
        coefs = pm.Normal("coefs", mu=0, sigma=coefs_sigma, shape=X_train.shape[1])
        sigma = pm.HalfNormal("sigma", sigma=error_sigma_scale)
        
        # Linear predictor on the log scale
        mu_pred = intercept + pm.math.dot(X_train, coefs)
        
        # Normal likelihood (no StudentT)
        y_obs = pm.Normal("y_obs", mu=mu_pred, sigma=sigma, observed=y_train_log)
        
        # MCMC sampling
        trace = pm.sample(2000, tune=1000, target_accept=0.95, random_seed=42, cores=n_cores)
    
    # Summary
    print(az.summary(trace, round_to=2))
    
    # 5. Predictions on Test Set
    posterior_coefs = trace.posterior["coefs"].stack(sample=("chain", "draw")).values.T
    posterior_intercept = trace.posterior["intercept"].stack(sample=("chain", "draw")).values
    
    y_pred_log_samples = np.dot(posterior_coefs, X_test.T) + posterior_intercept[:, np.newaxis]
    y_pred_log_mean = y_pred_log_samples.mean(axis=0)
    
    # Back-transform predictions
    y_pred = np.expm1(y_pred_log_mean)
    y_test_orig = np.expm1(y_test_log)
    
    # 6. Evaluation Metrics
    mse = mean_squared_error(y_test_orig, y_pred)
    mae = mean_absolute_error(y_test_orig, y_pred)
    r2 = r2_score(y_test_orig, y_pred)
    
    print("Bayesian Regression Performance on Test Set (log-transformed model):")
    print("MSE:", mse)
    print("MAE:", mae)
    print("R2 Score:", r2)
    
    # 7. Plot: Actual vs Predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test_orig, y_pred, alpha=0.7, edgecolor="k")
    plt.plot([y_test_orig.min(), y_test_orig.max()], [y_test_orig.min(), y_test_orig.max()], "r--")
    plt.xlabel("Actual Displacement Values")
    plt.ylabel("Predicted Displacement Values")
    plt.title("Bayesian Regression (Log Model): Actual vs Predicted Displacement")
    plt.grid(True)
    plt.show()
    pm.plot_trace(trace, figsize = (12, 12));
    plt.show()
    az.plot_posterior(trace)
    plt.show()

if __name__ == '__main__':
    run_bayesian_regression_log(intercept_mu=1.4, intercept_sigma=0.3, 
                                                coefs_sigma=0.60, 
                                                error_sigma_scale=0.10, 
                                                n_cores=4)
"""
 run_bayesian_regression_log(intercept_mu=1.30, intercept_sigma=0.80, 
                                                coefs_sigma=0.35, 
                                                error_sigma_scale=2.0, 
                                                n_cores=4)
                                                0.7919

    run_bayesian_regression_log(intercept_mu=1.3283, intercept_sigma=0.1, 
                                                coefs_sigma=0.5, 
                                                error_sigma_scale=5.0, 
                                                n_cores=4)
                                                0.799

 run_bayesian_regression_log(intercept_mu=1.0, intercept_sigma=0.75, 
                                                coefs_sigma=0.8, 
                                                error_sigma_scale=2.0, zz
                                                n_cores=4)
                                                0.7964

                                                run_bayesian_regression_log(intercept_mu=1.30, intercept_sigma=6.0, 
                                                coefs_sigma=0.60, 
                                                error_sigma_scale=5.5, 
                                                n_cores=4)
                                                0.8001
    
"""