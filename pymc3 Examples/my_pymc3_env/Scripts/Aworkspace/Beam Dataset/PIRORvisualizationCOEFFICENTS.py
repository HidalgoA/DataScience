import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
from sklearn.preprocessing import RobustScaler

# -----------------------------
# 1. Load and Preprocess the Data
# -----------------------------
file_path = r"C:\Users\hidal\Desktop\Classes\GRAD\Data Sci Mech of Mat\HW2 Deliverable\Beam Dataset SANDBOX Correct Code\Displacement Analysis\Transformed_Displ_Data.xlsx"
df = pd.read_excel(file_path, sheet_name="Sheet1")
df.columns = df.columns.str.strip()
df["Displacement_Values"] = pd.to_numeric(df["Displacement_Values"], errors="coerce")
df = df.dropna(subset=["Displacement_Values"])

# Extract displacement values and apply the log1p transformation (i.e., log(1 + y))
y = df["Displacement_Values"].values
y_log = np.log1p(y)
print(y_log)
# -----------------------------
# 2. Define Features and Target, then Transform and Scale
# -----------------------------
X = df.drop(columns=["Displacement_Values"])
# y is already defined above as displacement values; we already computed y_log

# Scale predictors using RobustScaler
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# Set the prior parameters (example values based on displacement data)
prior_intercept_mu = 1.4  # 1.30
prior_intercept_sigma = 0.6 #0.5
prior_coefs_sigma = 0.10 #0.30
prior_error_sigma_scale = 0.10 #0.10

# -----------------------------
# 3. Build the Model for Prior Predictive Checking
# -----------------------------
with pm.Model() as prior_model:
    # Priors for parameters (using displacement-specific values)
    intercept = pm.Normal("intercept", mu=prior_intercept_mu, sigma=prior_intercept_sigma)
    coefs = pm.Normal("coefs", mu=0, sigma=prior_coefs_sigma, shape=X_scaled.shape[1])
    sigma = pm.HalfNormal("sigma", sigma=prior_error_sigma_scale)
    
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
# Add vertical lines for the intercept prior (mean and ± 1 SD)
axes[0].axvline(prior_intercept_mu, color="red", linestyle="--", linewidth=2, label="Intercept prior mean")
axes[0].axvline(prior_intercept_mu - prior_intercept_sigma, color="green", linestyle="--", linewidth=2, label=f"Intercept ± {prior_intercept_sigma} SD")
axes[0].axvline(prior_intercept_mu + prior_intercept_sigma, color="green", linestyle="--", linewidth=2)
axes[0].set_xlabel("Simulated log(1+Displacement)")
axes[0].set_ylabel("Frequency")
axes[0].set_title("Prior Predictive (Log Scale)")
axes[0].legend()

# Right plot: Actual Log-Transformed Displacement Data
axes[1].hist(y_log, bins=30, color="skyblue", edgecolor="k", alpha=0.7)
axes[1].set_xlabel("log(1 + Displacement)")
axes[1].set_ylabel("Frequency")
axes[1].set_title("Observed log(1+Displacement) Data")



# Back-transform simulated outcomes to the original displacement scale
y_sim_orig = np.expm1(y_sim_log)

axes[2].hist(y_sim_orig.ravel(), bins=50, color="lightgreen", edgecolor="k", alpha=0.7)
axes[2].set_xlabel("Simulated Displacement")
axes[2].set_ylabel("Frequency")
axes[2].set_title("Prior Predictive Distribution (Original Scale)")

axes[3].hist(y, bins=50, color="lightgreen", edgecolor="k", alpha=0.7)
axes[3].set_xlabel("Actual Displacement")
axes[3].set_ylabel("Frequency")
axes[3].set_title("Actual Displacement Values")

plt.tight_layout()
plt.show()