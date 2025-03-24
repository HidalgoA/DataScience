import pandas as pd
import os

# Define file paths for input datasets
stress_results_file = r"C:\Users\hidal\Desktop\Classes\GRAD\Data Sci Mech of Mat\HW2 Deliverable\Beam Dataset SANDBOX\RawDataStressResults.xlsx"
hole_info_file = r"C:\Users\hidal\Desktop\Classes\GRAD\Data Sci Mech of Mat\HW2 Deliverable\Beam Dataset SANDBOX\RawDataHoleInfo.xlsx"

# Get the directory where the Python script is running
script_directory = os.path.dirname(os.path.abspath(__file__))

# Output file path (save in the same directory as the Python script)
output_file = os.path.join(script_directory, "Transformed_Stress_Data.xlsx")

# Read the Excel files
stress_df = pd.read_excel(stress_results_file, sheet_name="Sheet1")
hole_info_df = pd.read_excel(hole_info_file, sheet_name="Sheet1")

# Drop non-numeric columns (like 'Dataset') from stress dataset
stress_df = stress_df.select_dtypes(include=["number"])

# Merge stress data with hole geometry data based on Volume Fraction
df_merged = stress_df.merge(hole_info_df, on="Volume Fraction", how="left")

# Convert from wide to long format to align stress values in a single column
df_long = df_merged.melt(id_vars=["Volume Fraction", "Number of Largest Holes", "Largest Hole Radius",
                                   "Number of Smallest Holes", "Smallest Hole Radius", "Shortest Distance to Edge"],
                         var_name="Material_Displacement",
                         value_name="Stress_Values")

# Extract Displacement (1mm or 5mm) from the column name
df_long["Displacement"] = df_long["Material_Displacement"].apply(lambda x: 1 if "1mm" in x else 5)

# Extract Material type and assign Young's Modulus values
df_long["Youngs_Modulus"] = df_long["Material_Displacement"].apply(lambda x: 
                                                                  0 if "Holes" in x else 
                                                                  10000 if "MatA" in x else 
                                                                  150000)

# Assign Poisson Ratio based on material type
df_long["Poisson_Ratio"] = df_long["Material_Displacement"].apply(lambda x: 
                                                                  0 if "Holes" in x else 
                                                                  0.34 if "MatA" in x else 
                                                                  0.4)

# Drop the original Material_Displacement column since its information is now extracted
df_long = df_long.drop(columns=["Material_Displacement"])

# Reorder columns so that independent variables are on the left and stress response is on the right
df_long = df_long[["Volume Fraction", "Displacement", "Youngs_Modulus", "Poisson_Ratio",
                   "Number of Largest Holes", "Largest Hole Radius", "Number of Smallest Holes",
                   "Smallest Hole Radius", "Shortest Distance to Edge", "Stress_Values"]]

# Save the transformed dataset in the same directory as the Python script
df_long.to_excel(output_file, index=False)

print(f"Transformed dataset saved as: {output_file}")
