import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Read the Excel file
file_path = "StressRuns.xlsx"  # Change this to your actual file path
df = pd.read_excel(file_path)

# Reshape DataFrame from wide to long format
df_long = df.melt(id_vars=["Dataset", "Volume Fraction"], var_name="Condition", value_name="Stress")

# Extract displacement (1mm or 5mm)
df_long["Displacement"] = df_long["Condition"].str.extract(r'(\d+)mm', expand=False).astype(float)

# Extract material type (Holes, MatA, MatB)
df_long["Material"] = df_long["Condition"].str.replace(r'\s*\d+mm', '', regex=True)

# Debugging: Check if the transformation was successful
print(df_long.head())
print(df_long.columns)

def optimize_ridge_alpha(df, dataset, displacement):
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

        # Split into training and test sets (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardize X (fit on train, transform both train and test)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Optimize Ridge alpha
        alphas = np.logspace(-2, 100, 1000)  # Test values from 10^-4 to 10^2
        best_alpha = None
        best_score = -np.inf
        mse_scores = []

        for alpha in alphas:
            ridge = Lasso(alpha=alpha)
            ridge.fit(X_train_scaled, y_train)
            y_pred = ridge.predict(X_test_scaled)
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            mse_scores.append(mse)

            if r2 > best_score:
                best_score = r2
                best_alpha = alpha

        print(f"Best alpha for {material}: {best_alpha}, R²: {best_score:.4f}")

        # Final model with best alpha
        ridge_final = Lasso(alpha=best_alpha)
        ridge_final.fit(X_train_scaled, y_train)
        y_pred_final = ridge_final.predict(X_test_scaled)

        # Plot results
        plt.plot(alphas, mse_scores, label=f'{material} (best α={best_alpha:.4f})')

        results[material] = {
            "Best Alpha": best_alpha,
            "R2 Score": best_score,
            "MSE": min(mse_scores),
        }

    plt.xscale('log')
    plt.xlabel("Alpha (log scale)")
    plt.ylabel("Mean Squared Error")
    plt.title(f"Ridge Regression Alpha Optimization for {dataset} ({displacement}mm)")
    plt.legend()
    plt.show()

    return results

# Run the optimization for a specific dataset and displacement
results = optimize_ridge_alpha(df_long, "Alex", 5)
print(results)
