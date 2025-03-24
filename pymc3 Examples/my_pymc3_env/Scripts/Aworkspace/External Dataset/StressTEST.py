import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def run_bayesian_regression_stress(intercept_mu=4.80, intercept_sigma=0.8, 
                                   coefs_sigma=0.3, sigma_scale=0.1, n_cores=4):
    # ---------------------------------------------------------
    # 1. Configuration and Paths
    # ---------------------------------------------------------
    csv_folder = r"C:\Users\hidal\Desktop\Classes\GRAD\Data Sci Mech of Mat\HW2 Deliverable\External Dataset\All data_Stress"
    master_file = r"C:\Users\hidal\Desktop\Classes\GRAD\Data Sci Mech of Mat\HW2 Deliverable\External Dataset\data_all_stress-controlled.xls"
    
    # ---------------------------------------------------------
    # 2. Load Material Properties and CSV File Names
    # ---------------------------------------------------------
    # Load stress master data (material properties + CSV file names)
    df_master = pd.read_excel(master_file, nrows=100000)
    df_master.drop(columns=['Unnamed: 5'], inplace=True, errors='ignore')  # Drop unnecessary column
    print("Master data columns:", df_master.columns.tolist())
    
    # Extract CSV file names from the 'load' column and drop NaNs
    csv_files = df_master['load'].dropna().values  
    
    # ---------------------------------------------------------
    # 3. Process CSV Files: Read and Flatten Data
    # ---------------------------------------------------------
    value_list = []
    valid_indices = []
    for i, csv_file in enumerate(csv_files):
        try:
            full_csv_path = os.path.join(csv_folder, csv_file)
            one_df = pd.read_csv(full_csv_path, header=None).iloc[:, :2]
            # Flatten the CSV data (both columns) into a single vector
            value_list.append(one_df.values.flatten())
            valid_indices.append(i)
        except Exception as e:
            print(f"Skipping {csv_file}: {e}")
    
    # Convert list to numpy array
    csv_value_array = np.array(value_list)
    
    # Filter master data to only include valid CSV entries
    df_master = df_master.iloc[valid_indices].reset_index(drop=True)
    
    # ---------------------------------------------------------
    # 4. Process Material Properties Features
    # ---------------------------------------------------------
    # Exclude CSV file and target columns from processing
    exclude_cols = ['load', 'Nf(label)']
    num_cols = df_master.select_dtypes(exclude=['object']).columns.tolist()
    num_cols = [col for col in num_cols if col not in exclude_cols]
    
    # Build a pipeline for numerical material properties
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    ct_material = ColumnTransformer(transformers=[('num', num_pipe, num_cols)])
    X_material = ct_material.fit_transform(df_master)
    
    # Extract target variable (fatigue life)
    y_all = df_master['Nf(label)'].values
    # Ensure no NaNs in y_all
    valid_y_indices = ~np.isnan(y_all)
    X_material = X_material[valid_y_indices]
    y_all = y_all[valid_y_indices]
    
    # ---------------------------------------------------------
    # 5. Scale and Combine CSV Data with Material Properties
    # ---------------------------------------------------------
    csv_scaler = MinMaxScaler()  # Scale CSV features separately
    csv_value_array_scaled = csv_scaler.fit_transform(csv_value_array)
    
    # Combine the scaled CSV data with the processed material properties
    combined_data = np.hstack((csv_value_array_scaled, X_material))
    
    print(f"Shape of combined data: {combined_data.shape}")
    print(f"Shape of target y_all: {y_all.shape}")
    
    # ---------------------------------------------------------
    # 6. Train-Test Split
    # ---------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(combined_data, y_all, 
                                                        test_size=0.2, random_state=42)
    
    # ---------------------------------------------------------
    # 7. Bayesian Linear Regression Model using PyMC
    # ---------------------------------------------------------
    with pm.Model() as bayes_model:
        # Priors for intercept and coefficients
        intercept_pm = pm.Normal("intercept", mu=intercept_mu, sigma=intercept_sigma)
        coefs_pm = pm.Normal("coefs", mu=0, sigma=coefs_sigma, shape=X_train.shape[1])
        sigma_pm = pm.HalfNormal("sigma", sigma=sigma_scale)
        
        # Linear model
        mu_pm = intercept_pm + pm.math.dot(X_train, coefs_pm)
        # Likelihood
        y_obs = pm.Normal("y_obs", mu=mu_pm, sigma=sigma_pm, observed=y_train)
        
        # Sample from the posterior
        trace = pm.sample(2000, tune=2000, target_accept=0.95, random_seed=42, cores=n_cores, chains = 4)

    
    # ---------------------------------------------------------
    # 8. Posterior Summary and Predictions on Test Set
    # ---------------------------------------------------------
    print(az.summary(trace, round_to=2))
    
    # Extract posterior samples and compute predictions for test set
    posterior_intercept = trace.posterior["intercept"].values.flatten()
    posterior_coefs = trace.posterior["coefs"].values.reshape(-1, X_train.shape[1])
    
    # Compute predictions from posterior draws
    y_pred_samples = posterior_intercept[:, None] + np.dot(posterior_coefs, X_test.T)
    y_pred_mean = y_pred_samples.mean(axis=0)
    y_pred = y_pred_mean  # predicted fatigue life
    
    # ---------------------------------------------------------
    # 9. Evaluation Metrics
    # ---------------------------------------------------------
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("Bayesian Regression Performance on Test Set:")
    print("MSE:", mse)
    print("MAE:", mae)
    print("R2 Score:", r2)
    
    # ---------------------------------------------------------
    # 10. Plot: Actual vs Predicted Fatigue Life
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.7, edgecolor="k")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
    plt.xlabel("Actual Fatigue Life")
    plt.ylabel("Predicted Fatigue Life")
    plt.title("Bayesian Regression: Actual vs Predicted Fatigue Life")
    plt.grid(True)
    plt.show()
    
    # ---------------------------------------------------------
    # 11. Plot Trace and Posterior Distributions (Optional)
    # ---------------------------------------------------------
    pm.plot_trace(trace, figsize=(12, 12))
    plt.show()
    az.plot_posterior(trace)
    plt.show()

    
    return trace

if __name__ == '__main__':
    run_bayesian_regression_stress(intercept_mu=4.8, intercept_sigma=0.4, 
                                   coefs_sigma=1.0, sigma_scale=0.1, n_cores=4)






'''
 run_bayesian_regression_stress(intercept_mu=4.8, intercept_sigma=0.4, 
                                   coefs_sigma=5.0, sigma_scale=0.001, n_cores=4)


  run_bayesian_regression_stress(intercept_mu=5.0, intercept_sigma=1.0, 
                                   coefs_sigma=1.0, sigma_scale=2.0, n_cores=4)
                                   0.3605


                                    run_bayesian_regression_stress(intercept_mu=4.8, intercept_sigma=1.0, 
                                   coefs_sigma=10.0, sigma_scale=0.5, n_cores=4)
                                   0.3635

'''
