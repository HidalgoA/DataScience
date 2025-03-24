import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def run_bayesian_regression_log(intercept_mu=20, intercept_sigma=2.0, 
                                                coefs_sigma=1.0, 
                                                error_sigma_scale=1.0, 
                                                 n_cores=4):
    
    '''
    # Intercept prior based on log-displacement statistics
    intercept = pm.Normal("intercept", mu=2.0, sigma=1.0) 7
    
    # Coefficients for scaled predictors
    coefs = pm.Normal("coefs", mu=0, sigma=2.0, shape=X_train.shape[1])
    
    # Residual error prior
    sigma = pm.HalfNormal("sigma", sigma=1.0)
    
    # Linear predictor on the log scale
    mu_pred = intercept + pm.math.dot(X_train, coefs)
    y       = b         + xValues * slope
    '''
    """
    Runs a Bayesian linear regression on log-transformed stress data using a Normal likelihood.
    
    Parameters:
      intercept_mu: float or None. The prior mean for the intercept. If None, it is set to the mean of the log-transformed training target.
      intercept_sigma: float. The prior standard deviation for the intercept.
      coefs_sigma: float. The prior standard deviation for the coefficients.
      error_sigma_scale: float. The scale parameter for the HalfNormal prior on the residual error.
      n_cores: int. The number of cores to use for sampling.
    """
    # -----------------------------
    # 1. Load and Preprocess the Data
    # -----------------------------
    transformed_data_file = r"C:\Users\hidal\Desktop\Classes\GRAD\Data Sci Mech of Mat\HW2 Deliverable\Beam Dataset SANDBOX Correct Code\Stress Analysis\Transformed_Stress_Data.xlsx"
    df = pd.read_excel(transformed_data_file, sheet_name="Sheet1")
    df.columns = df.columns.str.strip()
    df["Stress_Values"] = pd.to_numeric(df["Stress_Values"], errors="coerce")
    df = df.dropna(subset=["Stress_Values"])
    
    # -----------------------------
    # Define Features and Target, then Scale Features
    # -----------------------------
    X = df.drop(columns=["Stress_Values"])
    y = df["Stress_Values"].values
    
    # Transform target using log1p to handle any zero values
    y_log = np.log1p(y)
    
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # -----------------------------
    #  Train-Test Split
    # -----------------------------
    X_train, X_test, y_train_log, y_test_log = train_test_split(X_scaled, y_log, test_size=0.2, random_state=42)
    
    # If no intercept_mu is provided, use the mean of the training log-targets.
    if intercept_mu is None:
        intercept_mu = np.mean(y_train_log)
    
    print("Intercept prior mean set to:", intercept_mu)
    
    # -----------------------------
    #  Build the Bayesian Linear Regression Model
    # -----------------------------
    with pm.Model() as bayes_model:
        # Prior for the intercept: using user-specified (or computed) mean.
        intercept = pm.Normal("intercept", mu=intercept_mu, sigma=intercept_sigma)
        
        # Prior for the coefficients for scaled predictors.
        coefs = pm.Normal("coefs", mu=0, sigma=coefs_sigma, shape=X_train.shape[1])
        
        # Prior for the residual error on the log scale.
        sigma = pm.HalfNormal("sigma", sigma=error_sigma_scale)
        
        # Linear predictor on the log scale.
        mu_pred = intercept + pm.math.dot(X_train, coefs)
        
        # Likelihood with Normal distribution.
        y_obs = pm.Normal("y_obs", mu=mu_pred, sigma=sigma, observed=y_train_log)
        
        # Sample from the posterior using multiple cores.
        trace = pm.sample(2000, tune=1000, target_accept=0.95, random_seed=42, cores=n_cores)
    
    print(az.summary(trace, round_to=2))
    
    # -----------------------------
    #  Predictions on Test Set
    # -----------------------------
    # Stack the chains and draws:
    posterior_coefs = trace.posterior["coefs"].stack(sample=("chain", "draw")).values.T  # shape: (n_samples, n_features)
    posterior_intercept = trace.posterior["intercept"].stack(sample=("chain", "draw")).values  # shape: (n_samples,)
    
    # Compute predictions on the log scale for each posterior sample.
    y_pred_log_samples = np.dot(posterior_coefs, X_test.T) + posterior_intercept[:, np.newaxis]
    # Take the mean over the posterior samples.
    y_pred_log_mean = y_pred_log_samples.mean(axis=0)
    
    # Back-transform predictions using expm1 (inverse of log1p).
    y_pred = np.expm1(y_pred_log_mean)
    y_test_orig = np.expm1(y_test_log)
    
    # -----------------------------
    #  Evaluation Metrics and Plotting
    # -----------------------------
    mse = mean_squared_error(y_test_orig, y_pred)
    mae = mean_absolute_error(y_test_orig, y_pred)
    r2 = r2_score(y_test_orig, y_pred)
    
    print("Bayesian Regression Performance on Test Set (log-transformed model):")
    print("MSE:", mse)
    print("MAE:", mae)
    print("R2 Score:", r2)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test_orig, y_pred, alpha=0.7, edgecolor="k")
    plt.plot([y_test_orig.min(), y_test_orig.max()], [y_test_orig.min(), y_test_orig.max()], "r--")
    plt.xlabel("Actual Stress Values")
    plt.ylabel("Predicted Stress Values")
    plt.title("Bayesian Regression (Log Model): Actual vs Predicted Stress")
    plt.grid(True)
    plt.show()

    pm.plot_trace(trace, figsize = (12, 12));
    plt.show()
    az.plot_posterior(trace)
    plt.show()

if __name__ == '__main__':
    # controlling the intercept mean and using 4 cores.0.2
    run_bayesian_regression_log(intercept_mu=6.5, intercept_sigma=0.7, 
                                                    coefs_sigma=0.2, 
                                                    error_sigma_scale=0.10, 
                                                    n_cores=4)

"""
   run_bayesian_regression_log(intercept_mu=6.5, intercept_sigma=0.3, 
                                                    coefs_sigma=0.2, 
                                                    error_sigma_scale=0.1, 
                                                    n_cores=4)



  run_bayesian_regression_log(intercept_mu=6.5, intercept_sigma=0.8, 
                                                    coefs_sigma=0.5, 
                                                    error_sigma_scale=1.0, 
                                                    n_cores=4)
Intercept (αα):

    Prior Mean:
    Use the computed mean of the log‑transformed response (or a value supported by domain knowledge).
    For example, if the mean of ylogylog​ is about 6.5, you might set:
    α∼N(6.5,σ2)
    α∼N(6.5,σ2)
    Prior Standard Deviation:
    If you are moderately confident about the baseline value but want to allow for some uncertainty, start with a moderate standard deviation (e.g., 1.0 or 2.0). A smaller value (like 1.0) indicates you believe the intercept is well-known, while a larger value (like 2.0 or 3.0) means you are more uncertain.

Coefficients (βiβi​):

    Prior Mean:
    Typically, set this to 0 if you have no strong reason to expect a directional effect.
    Prior Standard Deviation:
    For standardized predictors, a common starting point is a Normal(0, 1) or Normal(0, 2) prior.
    βi∼N(0,τ2)
    βi​∼N(0,τ2) Here, ττ (e.g., 1 or 2) reflects how much effect you think each predictor might have. If you believe many predictors should have little influence (as in sparse settings), you might choose a smaller ττ or even consider a Laplace (double exponential) or hierarchical (e.g., horseshoe) prior.

Residual Standard Deviation (σσ):

    This parameter represents the noise or unexplained variability.
    Use a HalfNormal (or HalfCauchy) prior because σσ must be positive.
    For example, if your log‑transformed outcome has a standard deviation around 1.0, you might choose:
    σ∼HalfNormal(1.0)
    σ∼HalfNormal(1.0)
    You can adjust this if you believe the noise is higher or lower.
"""

'''
 run_bayesian_regression_log(intercept_mu=6.0, intercept_sigma=0.1, 
                                                    coefs_sigma=3.0, 
                                                    error_sigma_scale=3.0, 
                                                    n_cores=4)

                                                    good r squared



                                                      run_bayesian_regression_log(intercept_mu=6.5, intercept_sigma=0.1, 
                                                    coefs_sigma=0.5, 
                                                    error_sigma_scale=1.0, 
                                                    n_cores=4)
                                                    good r squared

                                                       run_bayesian_regression_log(intercept_mu=6.5, intercept_sigma=0.8, 
                                                    coefs_sigma=0.5, 
                                                    error_sigma_scale=1.0, 
                                                    n_cores=4)
                                                    moderate r squared
'''