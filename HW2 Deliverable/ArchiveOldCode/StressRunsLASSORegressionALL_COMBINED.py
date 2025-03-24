import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Load Data from Excel
file_path = "StressRuns.xlsx"  # Update to the correct file path
df = pd.read_excel(file_path)

# Reshape into long format
df_long = df.melt(id_vars=["Dataset", "Volume Fraction"], var_name="Condition", value_name="Stress")

# Extract displacement (1mm or 5mm)
df_long["Displacement"] = df_long["Condition"].str.extract(r'(\d+)mm')
df_long["Displacement"] = pd.to_numeric(df_long["Displacement"], errors='coerce')

# Extract material type
df_long["Material"] = df_long["Condition"].str.replace(r' \dmm', '', regex=True).str.strip()

# Debugging: Print unique values to check correctness
print("Unique Displacement values:", df_long["Displacement"].unique())
print("Unique Material values:", df_long["Material"].unique())

# Function to perform Lasso Regression on combined datasets
def perform_lasso_combined(df, displacement):
    """Performs Lasso Regression on all datasets combined for Holes, MatA, and MatB."""
    results = {}
    plt.figure(figsize=(10, 6))

    for material in ["Holes", "MatA", "MatB"]:
        # Filter for current material and displacement
        df_subset = df[(df["Displacement"] == displacement) & (df["Material"] == material)].copy()

        # Debugging: Print shape of filtered data
        print(f"\nProcessing {material} ({displacement}mm) - Found {df_subset.shape[0]} rows")

        # Check if data exists
        if df_subset.empty:
            print(f"Warning: No data found for {material} at {displacement}mm. Skipping.")
            continue

        # Define X (Volume Fraction) and y (Stress)
        X = df_subset[["Volume Fraction"]].values
        y = df_subset["Stress"].values

        # Standardize X
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Define range of alpha values
        alphas = np.logspace(-3, 3, 100)

        # Perform Lasso Regression with Cross-Validation
        lasso_cv = LassoCV(alphas=alphas, cv=5, max_iter=10000)  # Using 5-fold CV
        lasso_cv.fit(X_scaled, y)
        best_alpha = lasso_cv.alpha_

        # Train Lasso model with the best alpha
        lasso = LassoCV(alphas=[best_alpha], max_iter=10000, cv=5)
        lasso.fit(X_scaled, y)

        # Predict Stress values
        df.loc[df_subset.index, "Predicted Stress"] = lasso.predict(X_scaled)

        # Evaluate Model
        r2 = r2_score(y, df.loc[df_subset.index, "Predicted Stress"])
        mse = mean_squared_error(y, df.loc[df_subset.index, "Predicted Stress"])
        print(f"{material} ({int(displacement)}mm) - Best Alpha: {best_alpha:.4f}, R²: {r2:.3f}, MSE: {mse:.3f}")

        # Store results
        results[material] = {"best_alpha": best_alpha, "r2": r2, "mse": mse, "coefficients": lasso.coef_}

        # Plot actual vs predicted
        sns.scatterplot(x=df_subset["Volume Fraction"], y=df_subset["Stress"], label=f"{material} Actual")
        sns.lineplot(x=df_subset["Volume Fraction"], y=df.loc[df_subset.index, "Predicted Stress"], label=f"{material} Predicted")

    plt.xlabel("Volume Fraction")
    plt.ylabel("Stress")
    plt.title(f"Stress vs. Volume Fraction ({int(displacement)}mm Runs) - Lasso Regression")
    plt.legend()
    plt.grid(True)
    plt.show()

    return results

# Run Lasso Regression for 1mm and 5mm displacement levels
results_1mm = perform_lasso_combined(df_long, displacement=1.0)
results_5mm = perform_lasso_combined(df_long, displacement=5.0)
