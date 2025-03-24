import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Read the Excel file
file_path = "StressRuns.xlsx"  # Change to your actual file path
df = pd.read_excel(file_path)

# Reshape DataFrame from wide to long format
df_long = df.melt(id_vars=["Dataset", "Volume Fraction"], var_name="Condition", value_name="Stress")

# Extract displacement (1mm or 5mm)
df_long["Displacement"] = df_long["Condition"].str.extract(r'(\d+)mm').astype(float)

# Extract material type (Holes, MatA, MatB)
df_long["Material"] = df_long["Condition"].str.replace(r'\s*\d+mm', '', regex=True)


def perform_lasso_with_tuning(df, dataset, displacement):
    results = {}
    plt.figure(figsize=(10, 6))

    for material in ["Holes", "MatA", "MatB"]:
        df_subset = df[(df["Dataset"] == dataset) & 
                       (df["Displacement"] == displacement) & 
                       (df["Material"] == material)].copy()

        print(f"Processing {dataset}: {material} ({displacement}mm)")
        if df_subset.empty:
            print(f"⚠️ No data found for {material} in {dataset} at {displacement}mm!")
            continue

        # Define inputs and outputs
        X = df_subset[["Volume Fraction"]].values
        y = df_subset["Stress"].values

        # Standardize X
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Lasso Regression with Cross-Validation
        alphas = np.logspace(-3, 3, 100)
        lasso_cv = LassoCV(alphas=alphas, cv=5, max_iter=5000)
        lasso_cv.fit(X_scaled, y)
        best_alpha = lasso_cv.alpha_

        lasso = Lasso(alpha=best_alpha)
        lasso.fit(X_scaled, y)

        # Predict and append predictions to df_subset
        predictions = lasso.predict(X_scaled)
        df_subset["Predicted Stress"] = predictions  

        # Append back into df_long
        global df_long  # Ensure we modify the original dataframe
        df_long = pd.concat([df_long, df_subset], ignore_index=False).drop_duplicates(
            subset=["Dataset", "Volume Fraction", "Displacement", "Material"], keep="last"
        )

        print(f"\nPredictions for {dataset} - {material} ({displacement}mm):\n", 
              df_subset[["Dataset", "Volume Fraction", "Condition", "Displacement", "Material", "Stress", "Predicted Stress"]])

        # Calculate Metrics
        r2 = r2_score(y, predictions)
        mse = mean_squared_error(y, predictions)
        print(f"{dataset} - {material} ({int(displacement)}mm) - Best Alpha: {best_alpha:.4f}, R²: {r2:.3f}, MSE: {mse:.3f}")

        results[material] = {"best_alpha": best_alpha, "r2": r2, "mse": mse, "coefficients": lasso.coef_}

        # Plotting
        sns.scatterplot(x=df_subset["Volume Fraction"], y=df_subset["Stress"], label=f"{material} Actual")
        sns.lineplot(x=df_subset["Volume Fraction"], y=predictions, label=f"{material} Predicted")

    plt.xlabel("Volume Fraction")
    plt.ylabel("Stress")
    plt.title(f"Stress vs. Volume Fraction ({dataset}, {int(displacement)}mm) - Lasso Regression")
    plt.legend()
    plt.grid(True)
    plt.show()

    return results


datasets = df_long["Dataset"].unique()  # Get unique dataset names

for dataset in datasets:
    print(f"\nRunning Lasso Regression for {dataset} (1mm Displacement)")
    results_1mm = perform_lasso_with_tuning(df_long, dataset, displacement=1.0)

    print(f"\nRunning Lasso Regression for {dataset} (5mm Displacement)")
    results_5mm = perform_lasso_with_tuning(df_long, dataset, displacement=5.0)
