import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, Ridge, Lasso 
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold
from numpy import arange

#/////////////INPUTS HERE/////////////
DatasetName = ["DS2"] # "Alex", "DS1", "DS2"; pick multiple or one set within array 
MaterialDisplacement = "1mm" # Alternate between "1mm" and "5mm" only
RidgeAlpha = 0.25
LassoAlpha = 0.25
#////////////INPUTS HERE/////////////

holeTarget = f"Holes {MaterialDisplacement}" 
MaterialATarget = f"MatA {MaterialDisplacement}" 
MaterialBTarget  = f"MatB {MaterialDisplacement}" 

# Load dataset from Excel file
file_path = "StressRuns.xlsx"  # Update with actual file path, use Excel file name 
df = pd.read_excel(file_path)

# Ensure column names are correct, removes leading and trailing spaces from column names. .str apply string ops, .strip() remvoes leading/trailing spaces
df.columns = df.columns.str.strip()

# Converts numeric coumns from text to actual nubers, ignores Dataset column 
for col in df.columns: #df.columns is the name of all columns in df
    if col != "Dataset":  # Avoid modifying categorical column
        df[col] = pd.to_numeric(df[col], errors="coerce") #Converts values in df[col] into a numerical float value, gives NaN if column value isn't a number

# Filter dataset for specified user inputs, both DatasetName and MaterialDisplacement
df["Dataset"] = df["Dataset"].str.strip()  # Remove leading/trailing spaces along the Dataset column, " Alex ", or " DS1" or "DS2 "
df = df[df["Dataset"].isin(DatasetName)].drop(columns=["Dataset"])  # Filter for DatasetName inputs 
                    # df["Dataset"] selects the "Dataset" column from DataFrame
                    # .isin(DatasetName) checks the rows in the "Dataset" column matches any values in DatasetName
                        #Returns True if there's matches, False otherwise
                    # df[..] boolean filters True only 
                    # .drop(columns=["Dataset"]) Dataset column not needed, drops entire column

# Creates the Indepdent Var, X, to be Volume Fraction 2D column values, double brackets to extract column values as a table 
X = df[["Volume Fraction"]]

targets = [holeTarget, MaterialATarget, MaterialBTarget]  # List that stores 3 variables/3 column names defined by user

# Ridge & Lasso Alpha Grid for optimization 
alphas = np.linspace(0.001, 1, 100)

# Creates for loop,to loop through the 3 variables in "targets" to calculate Regression data/plots 
for target in targets:

    """INDEPENDENT AND DEPDENT VARIABLE SECTION"""
    Y = df[target] #Extracts from the df Datatable, the current variable in "targets" that is being worked on

    # Ensure consistent train-test splits
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
    """INDEPENDENT AND DEPDENT VARIABLE SECTION"""

    #Creates Cross Validation strategey, using RepeatedKFold. 
        # n_splits; split into k folds for each iteration
        # n_repeats; repeated n times, new random split
            # n * k number different train test splits
   
    # Fit Linear, Ridge and Lasso Regression models to the data
    lin_reg = LinearRegression().fit(X_train, Y_train) #Linear Regression
    ridge = Ridge(alpha=RidgeAlpha).fit(X_train, Y_train) #Ridge Regression
    lasso = Lasso(alpha=LassoAlpha, random_state=1).fit(X_train, Y_train) #Lasso Regression


    # Predictions on test data
    y_pred_lin = lin_reg.predict(X_test)
    y_pred_ridge = ridge.predict(X_test)
    y_pred_lasso = lasso.predict(X_test)

    # Compute Metrics, MSE
    mse_lin = mean_squared_error(Y_test, y_pred_lin)
    mse_ridge = mean_squared_error(Y_test, y_pred_ridge)
    mse_lasso = mean_squared_error(Y_test, y_pred_lasso)
    # Compute Metrics, R^2 
    r2_lin = lin_reg.score(X_test, Y_test)
    r2_ridge = ridge.score(X_test, Y_test)
    r2_lasso = lasso.score(X_test, Y_test)

    print(f"\n{target} {DatasetName}")
    print(f"  Linear - R²: {r2_lin:.3f}, MSE: {mse_lin:.3f}")
    print(f"  Ridge  - Best Alpha: {ridge.alpha:.4f}, R²: {r2_ridge:.3f}, MSE: {mse_ridge:.3f}")
    print(f"  Lasso  - Best Alpha: {lasso.alpha:.4f}, R²: {r2_lasso:.3f}, MSE: {mse_lasso:.3f}")

    # Plot results
    plt.figure(figsize=(8, 5))
    plt.scatter(X_train, Y_train, color='gray', alpha=0.5, label="Train Data")
    plt.scatter(X_test, Y_test, color='black', alpha=0.7, label="Test Data")

    # Plots all 3 Regression models to predict on all values of X 
    plt.plot(X, lin_reg.predict(X), color="blue", label="Linear")
    plt.plot(X, ridge.predict(X), color="red", label=f"Ridge (α={ridge.alpha:.4f})")
    plt.plot(X, lasso.predict(X), color="green", label=f"Lasso (α={lasso.alpha:.4f})")

    plt.xlabel("Volume Fraction")
    plt.ylabel(target)
    plt.title(f"Regression Models for {target} {DatasetName}")
    plt.legend()
    plt.show()