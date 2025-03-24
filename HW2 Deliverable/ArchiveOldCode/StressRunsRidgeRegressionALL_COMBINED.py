import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Load Data from Excel
file_path = "StressRuns.xlsx"  # Update to the correct file path
df = pd.read_excel(file_path)

# Reshape into long format
df_long = df.melt(id_vars=["Dataset", "Volume Fraction"], var_name="Condition", value_name="Stress")

# Extract displacement (1mm or 5mm)
df_long["Displacement"] = df_long["Condition"].str.extract(r'(\d+)mm').astype(float)

# Extract material type (Holes, MatA, MatB)
df_long["Material"] = df_long["Condition"].str.replace(r'\s*\d+mm', '', regex=True).str.strip()

# Debugging: Print unique values to check correctness
print("Unique Displacement values:", df_long["Displacement"].unique())
print("Unique Material values:", df_long["Material"].unique())

# Function to perform Ridge Regression with train-test split
def perform_ridge_with_train_test(df, displacement):
    """Performs Ridge Regression using train-test split for each material."""
    results = {}
    plt.figure(figsize=(10, 6))

    for material in ["Holes", "MatA", "MatB"]:
        # Filter for current material and displacement
        df_subset = df[(df["Displacement"] == displacement) & (df["Material"] == material)].copy()

        print(f"\nProcessing {material} ({displacement}mm) - Found {df_subset.shape[0]} rows")

        if df_subset.empty:
            print(f"⚠️ Warning: No data found for {material} at {displacement}mm. Skipping.")
            continue

        # Define X (Volume Fraction) and y (Stress)
        X = df_subset[["Volume Fraction"]].values
        y = df_subset["Stress"].values.reshape(-1, 1)  # Reshape for normalization

        # Split into training (80%) and test (20%) sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardize X and y
        x_scaler = StandardScaler()
        y_scaler = StandardScaler()

        X_train_scaled = x_scaler.fit_transform(X_train)
        X_test_scaled = x_scaler.transform(X_test)

        y_train_scaled = y_scaler.fit_transform(y_train).ravel()  # Flatten to 1D for Ridge
        y_test_scaled = y_scaler.transform(y_test).ravel()

        # Define range of alpha values
        alphas = np.logspace(-3, 3, 100)

        # Train Ridge Regression with Cross-Validation
        ridge_cv = RidgeCV(alphas=alphas)
        ridge_cv.fit(X_train_scaled, y_train_scaled)
        best_alpha = ridge_cv.alpha_

        # Predict on test data
        y_pred_scaled = ridge_cv.predict(X_test_scaled)
        y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

        # Evaluate model
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        print(f"{material} ({int(displacement)}mm) - Best Alpha: {best_alpha:.4f}, R²: {r2:.3f}, MSE: {mse:.3f}")

        # Store results
        results[material] = {
            "best_alpha": best_alpha,
            "r2": r2,
            "mse": mse,
            "coefficients": ridge_cv.coef_,
        }

        # Plot actual vs predicted values
        df_results = pd.DataFrame({"Volume Fraction": X_test.flatten(), "Actual Stress": y_test.flatten(), "Predicted Stress": y_pred})
        df_results.sort_values(by="Volume Fraction", inplace=True)

        sns.scatterplot(x=df_results["Volume Fraction"], y=df_results["Actual Stress"], label=f"{material} Actual")
        sns.lineplot(x=df_results["Volume Fraction"], y=df_results["Predicted Stress"], label=f"{material} Predicted")

    plt.xlabel("Volume Fraction")
    plt.ylabel("Stress")
    plt.title(f"Stress vs. Volume Fraction ({int(displacement)}mm Runs) - Ridge Regression")
    plt.legend()
    plt.grid(True)
    plt.show()

    return results

# Run Ridge Regression for 1mm and 5mm displacement levels
results_1mm = perform_ridge_with_train_test(df_long, displacement=1.0)
results_5mm = perform_ridge_with_train_test(df_long, displacement=5.0)
