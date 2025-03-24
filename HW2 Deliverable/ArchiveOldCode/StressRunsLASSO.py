import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Create DataFrame
data = {
    "Volume Fraction": [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7],
    "Holes_1mm": [447.3, 270.9, 513.7, 481.6, 435.5, 472.1, 461.6, 871.7, 278.2, 485.6, 284.1, 195.5, 455.4],
    "MatA_1mm": [357, 233.6, 377.3, 349.2, 325.5, 438, 376.3, 580.1, 258.8, 347.2, 293, 399, 415.5],
    "MatB_1mm": [226, 217.9, 233.1, 226.1, 227.4, 238.2, 253.3, 256, 235.6, 273.7, 265.1, 309, 313.8],
    "Holes_5mm": [2236, 1355, 2568, 2408, 2178, 2361, 2308, 4358, 1391, 2428, 1421, 977.4, 2277],
    "MatA_5mm": [1785, 1168, 1887, 1746, 1628, 1740, 1881, 2900, 1294, 1736, 1465, 1995, 415.5],
    "MatB_5mm": [1130, 1090, 1165, 1133, 1137, 1191, 1266, 1280, 1178, 1369, 1326, 1545, 1569],
}

df = pd.DataFrame(data)

# Reshape into long format for easier plotting & analysis
df_long = df.melt(id_vars=["Volume Fraction"], var_name="Condition", value_name="Stress")

# Extract displacement type (1mm or 5mm)
df_long["Displacement"] = df_long["Condition"].str.extract(r'(\d+)mm').astype(float)

# Extract Material Type
df_long["Material"] = df_long["Condition"].str.replace(r'_\dmm', '', regex=True)

# Function to perform Lasso Regression with alpha tuning
def perform_lasso_with_tuning(df, displacement):
    """Performs Lasso Regression with alpha tuning for Holes, Mat A, and Mat B for a given displacement (1mm or 5mm)."""
    results = {}
    plt.figure(figsize=(10, 6))

    for material in ["Holes", "MatA", "MatB"]:
        df_subset = df[(df["Displacement"] == displacement) & (df["Material"] == material)].copy()  # Fix: Use `.copy()`

        # Define X (Volume Fraction) and y (Stress)
        X = df_subset[["Volume Fraction"]].values
        y = df_subset["Stress"].values

        # Standardize X
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Define alphas to test
        alphas = np.logspace(-3, 3, 100)  # Testing 100 alphas from 0.001 to 1000

        # Use LassoCV to find the best alpha
        lasso_cv = LassoCV(alphas=alphas, cv=5, max_iter=10000)  # 5-fold cross-validation
        lasso_cv.fit(X_scaled, y)
        best_alpha = lasso_cv.alpha_

        # Train Lasso model with the best alpha
        lasso = Lasso(alpha=best_alpha, max_iter=10000)
        lasso.fit(X_scaled, y)

        # Predict Stress
        df_long.loc[df_subset.index, "Predicted Stress"] = lasso.predict(X_scaled)  # Fix: Use `.loc[]`

        # Evaluate Model
        r2 = r2_score(y, df_long.loc[df_subset.index, "Predicted Stress"])
        mse = mean_squared_error(y, df_long.loc[df_subset.index, "Predicted Stress"])
        print(f"{material} ({int(displacement)}mm) - Best Alpha: {best_alpha:.4f}, R²: {r2:.3f}, MSE: {mse:.3f}")

        # Store results
        results[material] = {"best_alpha": best_alpha, "r2": r2, "mse": mse, "coefficients": lasso.coef_}

        # Plot actual vs predicted
        sns.scatterplot(x=df_subset["Volume Fraction"], y=df_subset["Stress"], label=f"{material} Actual")
        sns.lineplot(x=df_subset["Volume Fraction"], y=df_long.loc[df_subset.index, "Predicted Stress"], label=f"{material} Predicted")

    plt.xlabel("Volume Fraction")
    plt.ylabel("Stress")
    plt.title(f"Stress vs. Volume Fraction ({int(displacement)}mm Runs) - Lasso Regression")
    plt.legend()
    plt.grid(True)
    plt.show()

    return results

# Run for 1mm and 5mm displacement separately with alpha tuning
results_1mm = perform_lasso_with_tuning(df_long, displacement=1.0)
results_5mm = perform_lasso_with_tuning(df_long, displacement=5.0)
