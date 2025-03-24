import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
from sklearn.preprocessing import RobustScaler

# -----------------------------
# 1. Load and Preprocess the Data
# -----------------------------
file_path = r"C:\Users\hidal\Desktop\Classes\GRAD\Data Sci Mech of Mat\HW2 Deliverable\Beam Dataset SANDBOX Correct Code\Stress Analysis\Transformed_Stress_Data.xlsx"
df = pd.read_excel(file_path, sheet_name="Sheet1")
df.columns = df.columns.str.strip()
df["Stress_Values"] = pd.to_numeric(df["Stress_Values"], errors="coerce")
df = df.dropna(subset=["Stress_Values"])

# Extract stress values and apply the log1p transformation (i.e., log(1 + y))
y = df["Stress_Values"].values
y_log = np.log1p(y)

prior_intercept_mu = 6.5
prior_intercept_sigma = 0.70


# -----------------------------
# 2. Define Features and Target, then Transform and Scale
# -----------------------------
X = df.drop(columns=["Stress_Values"])
y = df["Stress_Values"].values

# Transform the target using log1p (log(1 + y))
y_log = np.log1p(y)


# Scale predictors using RobustScaler
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# 3. Build the Model for Prior Predictive Checking
# -----------------------------
with pm.Model() as prior_model:
    # Priors for parameters (example values)
    intercept = pm.Normal("intercept", mu=prior_intercept_mu, sigma=prior_intercept_sigma)
    coefs = pm.Normal("coefs", mu=0, sigma=0.2, shape=X_scaled.shape[1])
    sigma = pm.HalfNormal("sigma", sigma = 0.10)
    
    # Linear predictor on the log scale using actual X values
    mu_sim = intercept + pm.math.dot(X_scaled, coefs)
    
    # Likelihood: simulate y values from the priors (no observed data)
    y_sim = pm.Normal("y_sim", mu=mu_sim, sigma=sigma)
    
    # Draw samples from the prior predictive distribution
    prior_samples = pm.sample_prior_predictive(samples=100, random_seed=42, return_inferencedata=True)

# -----------------------------
# 4. Extract and Visualize Prior Predictive Samples
# -----------------------------
# Access the simulated y values from the "prior" group
y_sim_log = prior_samples.prior["y_sim"].values



fig, axes = plt.subplots(1, 4, figsize=(12, 6))

# Left plot: Prior Predictive Distribution (Log Scale)
axes[0].hist(y_sim_log.ravel(), bins=30, color="skyblue", edgecolor="k", alpha=0.7)
# Add vertical lines for the intercept prior:

axes[0].axvline(prior_intercept_mu, color="red", linestyle="--", linewidth=2, label="Intercept prior mean")
axes[0].axvline(prior_intercept_mu - prior_intercept_sigma, color="green", linestyle="--", linewidth=2, label=f"Intercept ± {prior_intercept_sigma} SD")
axes[0].axvline(prior_intercept_mu + prior_intercept_sigma, color="green", linestyle="--", linewidth=2)
axes[0].set_xlabel("Simulated log(1+Stress)")
axes[0].set_ylabel("Frequency")
axes[0].set_title("Prior Predictive (Log Scale)")
axes[0].legend()

# Right plot: Actual Log-Transformed Stress Data
axes[1].hist(y_log, bins=30, color="skyblue", edgecolor="k", alpha=0.7)
axes[1].set_xlabel("log(1 + Stress)")
axes[1].set_ylabel("Frequency")
axes[1].set_title("Observed Log(1+Stress) Data")

# Back-transform simulated outcomes to the original stress scale
y_sim_orig = np.expm1(y_sim_log)

# Back-transform simulated outcomes to the original displacement scale
y_sim_orig = np.expm1(y_sim_log)

axes[2].hist(y_sim_orig.ravel(), bins=50, color="lightgreen", edgecolor="k", alpha=0.7)
axes[2].set_xlabel("Simulated Stress")
axes[2].set_ylabel("Frequency")
axes[2].set_title("Prior Predictive Distribution (Original Scale)")

axes[3].hist(y, bins=50, color="lightgreen", edgecolor="k", alpha=0.7)
axes[3].set_xlabel("Actual Stress")
axes[3].set_ylabel("Frequency")
axes[3].set_title("Actual Stress Values")

plt.tight_layout()
plt.show()
