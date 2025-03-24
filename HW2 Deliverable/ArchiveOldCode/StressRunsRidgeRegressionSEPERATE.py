import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, RepeatedKFold

# Load Data from Excel
file_path = "StressRuns.xlsx"  # Update to the correct file path
df = pd.read_excel(file_path)

# Reshape into long format
df_long = df.melt(id_vars=["Dataset", "Volume Fraction"], var_name="Condition", value_name="Stress")

# Extract displacement (1mm or 5mm)
df_long["Displacement"] = df_long["Condition"].str.extract(r'(\d+)mm').astype(float)

# Extract material type (Holes, MatA, MatB)
df_long["Material"] = df_long["Condition"].str.replace(r'\s*\d+mm', '', regex=True).str.strip()

# Debugging: Print unique values
print("Unique Datasets:", df_long["Dataset"].unique())
print("Unique Displacement values:", df_long["Displacement"].unique())
print("Unique Material values:", df_long["Material"].unique())

# Function to perform Ridge Regression on each dataset individually
def perform_ridge_per_dataset(df, dataset, displacement):
    """Performs Ridge Regression separately for each dataset (Alex, DS1, DS2)."""
    results = {}
    plt.figure(figsize=(10, 6))

    for material in ["Holes", "MatA", "MatB"]:
        # Filter for the current dataset, displacement, and material
        df_subset = df[(df["Dataset"] == dataset) & (df["Displacement"] == displacement) & (df["Material"] == material)].copy()

        print(f"\nProcessing {dataset} - {material} ({displacement}mm) - Found {df_subset.shape[0]} rows")

        if df_subset.empty:
            print(f"⚠️ Warning: No data found for {dataset} - {material} at {displacement}mm. Skipping.")
            continue

        # Define X (Volume Fraction) and y (Stress) - NO SCALING
        X = df_subset[["Volume Fraction"]].values
        y = df_subset["Stress"].values

        # Split into training (80%) and test (20%) sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define range of alpha values
        alphas = np.logspace(-3, 3, 100)

        # Train Ridge Regression with Cross-Validation
        cv = RepeatedKFold(n_splits=2, n_repeats=3, random_state=1)
        ridge_cv = RidgeCV(alphas=alphas, cv=cv, scoring='neg_mean_absolute_error')
        ridge_cv.fit(X_train, y_train)
        best_alpha = ridge_cv.alpha_

        # Predict on test data
        y_pred = ridge_cv.predict(X_test)

        # Evaluate model
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        print(f"{dataset} - {material} ({int(displacement)}mm) - Best Alpha: {best_alpha:.4f}, R²: {r2:.3f}, MSE: {mse:.3f}")

        # Store results
        results[material] = {
            "best_alpha": best_alpha,
            "r2": r2,
            "mse": mse,
            "coefficients": ridge_cv.coef_,
        }

        # Create results DataFrame
        df_results = pd.DataFrame({"Volume Fraction": X_test.flatten(), "Actual Stress": y_test, "Predicted Stress": y_pred})
        df_results.sort_values(by="Volume Fraction", inplace=True)

        # Plot actual vs predicted values
        sns.scatterplot(x=df_results["Volume Fraction"], y=df_results["Actual Stress"], label=f"{material} Actual")
        sns.lineplot(x=df_results["Volume Fraction"], y=df_results["Predicted Stress"], label=f"{material} Predicted")

    plt.xlabel("Volume Fraction")
    plt.ylabel("Stress")
    plt.title(f"Stress vs. Volume Fraction ({dataset}, {int(displacement)}mm) - Ridge Regression")
    plt.legend()
    plt.grid(True)
    plt.show()

    return results

# Run Ridge Regression for each dataset individually
datasets = ["Alex", "DS1", "DS2"]
displacements = [1.0, 5.0]

results = {}
for dataset in datasets:
    for displacement in displacements:
        results[(dataset, displacement)] = perform_ridge_per_dataset(df_long, dataset, displacement)

