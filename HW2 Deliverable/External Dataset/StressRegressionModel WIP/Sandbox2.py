import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# Paths
csv_path = r"C:\Users\hidal\Desktop\Classes\GRAD\Data Sci Mech of Mat\HW2 Deliverable\External Dataset\All data_Strain"
file_path = r"C:\Users\hidal\Desktop\Classes\GRAD\Data Sci Mech of Mat\HW2 Deliverable\External Dataset\data_all_strain-controlled.xls"

# Load material properties dataset
pd_train = pd.read_excel(file_path, nrows=100000)
pd_train.drop(columns=['Unnamed: 5'], inplace=True, errors='ignore')  # Drop unnecessary column

# Extract CSV file names from dataset and remove NaNs
csv_files = pd_train['load'].dropna().values  

# Read CSV files and flatten data
value_list = []
for csv_file in csv_files:
    try:
        file_path = os.path.join(csv_path, csv_file)
        one_df = pd.read_csv(file_path, header=None).iloc[:, :2]  # Reading only the first two columns

        # Flatten data and store it in the list
        value_list.append(one_df.values.flatten())  # Flatten for consistency

    except Exception as e:
        print(f"Skipping {csv_file}: {e}")

# Convert to numpy array
csv_value_array = np.array(value_list)

# Flatten the entire array (if it's multi-dimensional)
flattened_data = csv_value_array.flatten()

# Check for data range and outliers before plotting
print(f"Data range: {flattened_data.min()} to {flattened_data.max()}")

# Plotting the histogram
plt.figure(figsize=(10, 6))
plt.hist(flattened_data, bins=50, color='skyblue', edgecolor='black')
plt.title('Histogram of Flattened CSV Data')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
