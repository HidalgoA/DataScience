import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
from matplotlib.patches import Patch

# Define file path for the transformed dataset
transformed_data_file = r"C:\Users\hidal\Desktop\Classes\GRAD\Data Sci Mech of Mat\HW2 Deliverable\Beam Dataset SANDBOX\Stress Analysis\Transformed_Stress_Data.xlsx"

# Load the dataset
df = pd.read_excel(transformed_data_file, sheet_name="Sheet1")

# Ensure Displacement is numeric for calculations
df["Displacement"] = df["Displacement"].astype(float)

# Convert back to string format for visualization
df["Displacement"] = df["Displacement"].astype(int).astype(str) + "mm"

# Define color mapping for displacement labels
color_map = {"1mm": "blue", "5mm": "red"}

# 2️⃣ **Histogram & KDE for Stress Values (1mm vs. 5mm)**
plt.figure(figsize=(7, 5))
ax = sns.histplot(data=df, x="Stress_Values", hue="Displacement", kde=True, bins=50, 
                  palette=color_map, alpha=0.6)

# **Manually Add Custom Legend**
legend_labels = [Patch(color="blue", label="1mm Displacement"), 
                 Patch(color="red", label="5mm Displacement")]
plt.legend(handles=legend_labels, title="Displacement", loc="upper right")

plt.title("Distribution of Stress Values for 1mm and 5mm Displacement")
plt.xlabel("Stress")
plt.ylabel("Frequency")
plt.show()

# 3️⃣ **Box Plot - Detect Outliers in Stress Values**
plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x="Displacement", y="Stress_Values", palette=["blue", "red"])
plt.title("Box Plot of Stress Values for 1mm and 5mm Displacement")
plt.xlabel("Displacement")
plt.ylabel("Stress")
plt.show()

# 4️⃣ **Violin Plot - Stress Distribution**
plt.figure(figsize=(7, 5))
sns.violinplot(data=df, x="Displacement", y="Stress_Values", palette=["blue", "red"])
plt.title("Violin Plot of Stress Distributions for 1mm and 5mm")
plt.xlabel("Displacement")
plt.ylabel("Stress")
plt.show()

# 5️⃣ **Pair Plot - Fixed Label Formatting**
pairplot = sns.pairplot(df, hue="Displacement", palette=color_map, diag_kind="kde", height=2.5)

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
stress_skewness = skew(df["Stress_Values"])
stress_kurtosis = kurtosis(df["Stress_Values"])

print(f"Skewness of Stress Values: {stress_skewness:.3f}")
print(f"Kurtosis of Stress Values: {stress_kurtosis:.3f}")

# Interpretation:
# - Skewness > 1 or < -1 suggests strong asymmetry (log transformation might help).
# - Kurtosis > 3 suggests heavy tails (outliers present).
