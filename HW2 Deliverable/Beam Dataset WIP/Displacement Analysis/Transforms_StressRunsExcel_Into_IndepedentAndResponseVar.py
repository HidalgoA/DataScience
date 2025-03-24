import pandas as pd

# Load the dataset
file_path = "DisplacementRuns.xlsx"  # Update with actual file path
df = pd.read_excel(file_path)

# Print column names to verify them
print(df.columns)

# Convert the data from wide to long format
df_long = df.melt(id_vars=["Dataset", "Volume Fraction"],  # Adjust 'Dataset' if needed
                  var_name="Material_Displacement",
                  value_name="Stress_Values")

# Extract Displacement (1mm or 5mm) from the column name
df_long["Loads"] = df_long["Material_Displacement"].apply(lambda x: 100 if "100N" in x else 500)

# Extract Material type from column names and assign Young's Modulus values
df_long["Youngs_Modulus"] = df_long["Material_Displacement"].apply(lambda x: 
                                                                  0 if "Holes" in x else 
                                                                  10000 if "MatA" in x else 
                                                                  150000)

# Add the Poisson Ratio column based on the material type
df_long["Poisson_Ratio"] = df_long["Material_Displacement"].apply(lambda x: 
                                                                  0 if "Holes" in x else 
                                                                  0.34 if "MatA" in x else 
                                                                  0.4)

# Drop the original Material_Displacement column
df_long = df_long.drop(columns=["Material_Displacement"])

# Reorder columns to desired format
df_long = df_long[["Volume Fraction", "Loads", "Youngs_Modulus", "Poisson_Ratio", "Stress_Values"]]

# Optionally, save the cleaned and transformed data to a new Excel file
df_long.to_excel("IndpAndResponseTable.xlsx", index=False)
