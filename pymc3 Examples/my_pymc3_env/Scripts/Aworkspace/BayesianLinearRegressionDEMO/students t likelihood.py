import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def run_bayesian_regression_log():
    # -----------------------------
    # 1. Load and Preprocess the Data
    # -----------------------------
    transformed_data_file = r"C:\Users\hidal\Desktop\Classes\GRAD\Data Sci Mech of Mat\HW2 Deliverable\Beam Dataset SANDBOX\Stress Analysis\Transformed_Stress_Data.xlsx"
    df = pd.read_excel(transformed_data_file, sheet_name="Sheet1")
    df.columns = df.columns.str.strip()
    df["Stress_Values"] = pd.to_numeric(df["Stress_Values"], errors="coerce")
    df = df.dropna(subset=["Stress_Values"])
    
    # -----------------------------
    # 2. Define Features and Target, then Scale Features
    # -----------------------------
    X = df.drop(columns=["Stress_Values"])
    y = df["Stress_Values"].values
    
    # Transform target using log1p (log(1+y)) to handle 0 values if any, else use np.log(y)
    y_log = np.log1p(y)
    
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # -----------------------------
    # 3. Train-Test Split
    # -----------------------------
    X_train, X_test, y_train_log, y_test_log = train_test_split(X_scaled, y_log, test_size=0.2, random_state=42)
    print(np.mean(y_train_log))
    # -----------------------------
    # 4. Bayesian Linear Regression on log-transformed target with a Student's T likelihood
    # -----------------------------
    with pm.Model() as bayes_model:
        # Prior for the intercept: center around the mean of y_log (e.g., np.mean(y_log))
        intercept = pm.Normal("intercept", mu=np.mean(y_train_log), sigma=1.0)
        
        # For scaled predictors, a weaker prior on slopes:
        coefs = pm.Normal("coefs", mu=0, sigma=1, shape=X_train.shape[1])
        
        # Robust likelihood parameters:
        sigma = pm.HalfNormal("sigma", sigma=1)
        nu = pm.Exponential("nu", 1/30)  # degrees of freedom, larger nu approximates a Normal
        
        # Expected value on the log scale:
        mu = intercept + pm.math.dot(X_train, coefs)
        
        # Likelihood with Student's T distribution:
        y_obs = pm.StudentT("y_obs", mu=mu, sigma=sigma, nu=nu, observed=y_train_log)
        
        trace = pm.sample(2000, tune=1000, target_accept=0.95, random_seed=42)
    
    print(az.summary(trace, round_to=2))
    
    # -----------------------------
    # 5. Predictions on Test Set
    # -----------------------------
    # Stack the chains and draws:
    posterior_coefs = trace.posterior["coefs"].stack(sample=("chain", "draw")).values.T  # shape: (n_samples, n_features)
    posterior_intercept = trace.posterior["intercept"].stack(sample=("chain", "draw")).values  # shape: (n_samples,)
    
    # Compute predictions on the log scale for each posterior sample
    y_pred_log_samples = np.dot(posterior_coefs, X_test.T) + posterior_intercept[:, np.newaxis]
    
    # Take the mean over the posterior samples
    y_pred_log_mean = y_pred_log_samples.mean(axis=0)
    
    # Back-transform predictions to the original scale using the inverse of log1p, i.e. expm1:
    y_pred = np.expm1(y_pred_log_mean)
    
    # Convert y_test_log back to original scale:
    y_test_orig = np.expm1(y_test_log)
    
    # -----------------------------
    # 6. Evaluation Metrics
    # -----------------------------
    mse = mean_squared_error(y_test_orig, y_pred)
    mae = mean_absolute_error(y_test_orig, y_pred)
    r2 = r2_score(y_test_orig, y_pred)
    
    print("Bayesian Regression Performance on Test Set (log-transformed model):")
    print("MSE:", mse)
    print("MAE:", mae)
    print("R2 Score:", r2)
    
    # -----------------------------
    # 7. Plot: Actual vs Predicted
    # -----------------------------
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test_orig, y_pred, alpha=0.7, edgecolor="k")
    plt.plot([y_test_orig.min(), y_test_orig.max()], [y_test_orig.min(), y_test_orig.max()], "r--")
    plt.xlabel("Actual Stress Values")
    plt.ylabel("Predicted Stress Values")
    plt.title("Bayesian Regression (Log Model): Actual vs Predicted Stress")
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    run_bayesian_regression_log()
