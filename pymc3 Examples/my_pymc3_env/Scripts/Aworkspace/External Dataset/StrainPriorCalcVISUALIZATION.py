import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
from sklearn.preprocessing import RobustScaler

# -----------------------------
# 1. Load and Preprocess the Fatigue Data
# -----------------------------
# Path to the master Excel file for strain-controlled fatigue data
file_path = r"C:\Users\hidal\Desktop\Classes\GRAD\Data Sci Mech of Mat\HW2 Deliverable\External Dataset\data_all_strain-controlled.xls"
df = pd.read_excel(file_path)
df.columns = df.columns.str.strip()

# Assume fatigue life is in the column 'Nf(label)' on log10 scale
y_all = df['Nf(label)'].values

# Use some material properties (if available) as predictors.
# For example, we use the columns defined in material_cols.
material_cols = ['E(Gpa)', 'TS(Mpa)', 'ss£¨Mpa£©', 'm']
X_material = df[material_cols].copy()

# Scale predictors using RobustScaler
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_material)

# -----------------------------
# 2. Set Prior Parameters for Fatigue Model
# -----------------------------
# These are example values—adjust based on your domain knowledge.
prior_intercept_mu = 3.50    # e.g., typical log10(fatigue life)
prior_intercept_sigma = 0.50
prior_coefs_sigma = 0.2
prior_error_sigma_scale = 0.1

# -----------------------------
# 3. Build the Bayesian Model for Prior Predictive Checking
# -----------------------------
with pm.Model() as fatigue_prior_model:
    # Priors for model parameters
    intercept = pm.Normal("intercept", mu=prior_intercept_mu, sigma=prior_intercept_sigma)
    coefs = pm.Normal("coefs", mu=0, sigma=prior_coefs_sigma, shape=X_scaled.shape[1])
    sigma = pm.HalfNormal("sigma", sigma=prior_error_sigma_scale)
    
    # Linear predictor on the log10 scale (fatigue life is assumed in log10 scale)
    mu_sim = intercept + pm.math.dot(X_scaled, coefs)
    
    # Likelihood: simulate fatigue life (on log10 scale)
    y_sim = pm.Normal("y_sim", mu=mu_sim, sigma=sigma)
    
    # Draw samples from the prior predictive distribution
    fatigue_prior_samples = pm.sample_prior_predictive(samples=100, random_seed=42, return_inferencedata=True)

# -----------------------------
# 4. Extract and Visualize Prior Predictive Samples
# -----------------------------
# Extract simulated fatigue life on the log10 scale
y_sim_log = fatigue_prior_samples.prior["y_sim"].values

fig, axes = plt.subplots(1, 4, figsize=(12, 6))

# Left plot: Simulated (prior predictive) fatigue life on log10 scale
axes[0].hist(y_sim_log.ravel(), bins=30, color="skyblue", edgecolor="k", alpha=0.7)
axes[0].axvline(prior_intercept_mu, color="red", linestyle="--", linewidth=2, label="Intercept prior mean")
axes[0].axvline(prior_intercept_mu - prior_intercept_sigma, color="green", linestyle="--", linewidth=2, 
              label=f"Intercept ± {prior_intercept_sigma} SD")
axes[0].axvline(prior_intercept_mu + prior_intercept_sigma, color="green", linestyle="--", linewidth=2)
axes[0].set_xlabel("Simulated log10(Fatigue Life)")
axes[0].set_ylabel("Frequency")
axes[0].set_title("Prior Predictive (Log10 Scale)")
axes[0].legend()

# Right plot: Observed fatigue life (log10 scale) from master file
axes[1].hist(y_all, bins=30, color="skyblue", edgecolor="k", alpha=0.7)
axes[1].set_xlabel("log10(Fatigue Life)")
axes[1].set_ylabel("Frequency")
axes[1].set_title("Observed log10(Fatigue Life) Data")

# Back-transform simulated fatigue life from log10 scale to original scale
y_sim_orig =  (y_sim_log)

axes[2].hist(y_sim_orig.ravel(), bins=50, color="lightgreen", edgecolor="k", alpha=0.7)
axes[2].set_xlabel("Simulated Fatigue Life")
axes[2].set_ylabel("Frequency")
axes[2].set_title("Prior Predictive Distribution (Original Scale)")

axes[3].hist(y_all, bins=50, color="lightgreen", edgecolor="k", alpha=0.7)
axes[3].set_xlabel("Actual Fatigue Life")
axes[3].set_ylabel("Frequency")
axes[3].set_title("Prior Predictive Distribution (Actual Data)")

plt.tight_layout()
plt.show()
