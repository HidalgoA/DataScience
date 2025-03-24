import pandas as pd
import os

# Path to the folder containing CSV files
csv_path = r"C:\Users\hidal\Desktop\Classes\GRAD\Data Sci Mech of Mat\HW2 Deliverable\External Dataset\All data_Stress"

# Path to the master Excel file
excel_path = r"C:\Users\hidal\Desktop\Classes\GRAD\Data Sci Mech of Mat\HW2 Deliverable\External Dataset\data_all_stress-controlled.xls"

# Load the entire master Excel file
df_master = pd.read_excel(excel_path, sheet_name=0)

# Create an empty list to store all processed dataframes
all_data = []

# Loop through each row in the master Excel file
for index, row in df_master.iterrows():
    # Extract material properties from the row
    E_GPa = row['E(Gpa)']
    TS_MPa = row['TS(Mpa)']
    ss_MPa = row['ss£¨Mpa£©']
    m = row['m']
    Nf_label = row['Nf(label)']
    material_name = row['load']  # Material name is in the 'load' column

    # Construct the CSV file name based on the material name
    csv_file = f"{material_name}"  # The CSV file should be named exactly as the material name
    csv_file_path = os.path.join(csv_path, csv_file)

    # Check if the CSV file exists
    if os.path.exists(csv_file_path):
        try:
            # Read the CSV file (stress and strain columns only)
            df_csv = pd.read_csv(csv_file_path, header=None, usecols=[0, 1])  # Read only the first two columns
            df_csv.columns = ['Stress', 'Strain']  # Assign column names for stress and strain

            # Add a new column for the material name at the beginning
            df_csv.insert(0, 'Material_Name', material_name)  # Insert at the first column position

            # Add material properties to the dataframe
            df_csv['Youngs_Modulus'] = E_GPa
            df_csv['Tensile_Strength'] = TS_MPa
            df_csv['ss(MPa)'] = ss_MPa
            df_csv['Poisson_Ratio'] = m
            df_csv['Fatigue_Life'] = Nf_label

            # Append the processed dataframe to the list
            all_data.append(df_csv)

            print(f"Processed: {material_name}")

        except Exception as e:
            print(f"Error reading CSV file {csv_file}: {e}")
    else:
        print(f"File {csv_file} does not exist at the path: {csv_file_path}")

# Check if any data has been processed
if all_data:
    # Combine all data into one big dataframe
    final_df = pd.concat(all_data, ignore_index=True)

    # Save the combined data into one Excel file
    output_file_path = os.path.join(os.getcwd(), "Processed_Material_Data_All_Loads.xlsx")
    final_df.to_excel(output_file_path, index=False)

    print(f"All data has been processed and saved into: {output_file_path}")
else:
    print("No data was processed. Please check for errors or missing files.")
