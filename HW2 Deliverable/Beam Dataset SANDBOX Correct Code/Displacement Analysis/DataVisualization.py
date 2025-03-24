import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
from matplotlib.patches import Patch

# Define file path for the transformed dataset
transformed_data_file = r"C:\Users\hidal\Desktop\Classes\GRAD\Data Sci Mech of Mat\HW2 Deliverable\Beam Dataset SANDBOX\Displacement Analysis\Transformed_Displ_Data.xlsx"

# Load the dataset
df = pd.read_excel(transformed_data_file, sheet_name="Sheet1")

# Ensure Load values are converted properly
df["Load"] = df["Load"].astype(str)  # Convert to string
df["Load"] = df["Load"].replace({"100": "100N", "500": "500N"})  # Ensure correct labeling

# Define color mapping for load labels
color_map = {"100N": "blue", "500N": "red"}

# 2️⃣ **Histogram & KDE for Displacement Values (100N vs. 500N)**
plt.figure(figsize=(7, 5))
ax = sns.histplot(data=df, x="Displacement_Values", hue="Load", kde=True, bins=50, 
                  palette=color_map, alpha=0.6)

# **Manually Add Custom Legend**
legend_labels = [Patch(color="blue", label="100N Load"), 
                 Patch(color="red", label="500N Load")]
plt.legend(handles=legend_labels, title="Load", loc="upper right")

plt.title("Distribution of Displacement Values for 100N and 500N Load")
plt.xlabel("Displacement")
plt.ylabel("Frequency")
plt.show()

# 3️⃣ **Box Plot - Detect Outliers in Displacement Values**
plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x="Load", y="Displacement_Values", palette=["blue", "red"])
plt.title("Box Plot of Displacement Values for 100N and 500N Load")
plt.xlabel("Load")
plt.ylabel("Displacement")
plt.show()

# 4️⃣ **Violin Plot - Displacement Distribution**
plt.figure(figsize=(7, 5))
sns.violinplot(data=df, x="Load", y="Displacement_Values", palette=["blue", "red"])
plt.title("Violin Plot of Displacement Distributions for 100N and 500N")
plt.xlabel("Load")
plt.ylabel("Displacement")
plt.show()

# 5️⃣ **Pair Plot - Fixed Label Formatting**
pairplot = sns.pairplot(df, hue="Load", palette=color_map, diag_kind="kde", height=2.5)

# Adjust spacing to prevent labels from being cut off
plt.subplots_adjust(bottom=0.2, left=0.2, right=0.95, top=0.95)

# Rotate labels for better visibility
for ax in pairplot.axes[-1, :]:  # Target bottom row labels
    ax.set_xlabel(ax.get_xlabel(), fontsize=10, rotation=45)

for ax in pairplot.axes[:, 0]:  # Target left column labels
    ax.set_ylabel(ax.get_ylabel(), fontsize=10, rotation=45)

plt.suptitle("Pairwise Feature Relationships", y=1.02, fontsize=12)
plt.show()

# 6️⃣ **Skewness & Kurtosis - Check for Normality**
displacement_skewness = skew(df["Displacement_Values"])
displacement_kurtosis = kurtosis(df["Displacement_Values"])

print(f"Skewness of Displacement Values: {displacement_skewness:.3f}")
print(f"Kurtosis of Displacement Values: {displacement_kurtosis:.3f}")

# Interpretation:
# - Skewness > 1 or < -1 suggests strong asymmetry (log transformation might help).
# - Kurtosis > 3 suggests heavy tails (outliers present).
