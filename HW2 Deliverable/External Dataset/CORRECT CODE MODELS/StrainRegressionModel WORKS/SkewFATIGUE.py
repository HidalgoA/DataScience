import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import boxcox

# ---------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------
master_file = r"C:\Users\hidal\Desktop\Classes\GRAD\Data Sci Mech of Mat\HW2 Deliverable\External Dataset\data_all_stress-controlled.xls"

# ---------------------------------------------------------
# 2. Load Excel and Extract Fatigue Life
# ---------------------------------------------------------
df_master = pd.read_excel(master_file)
df_master.drop(columns=['Unnamed: 5'], inplace=True, errors='ignore')

# Extract fatigue life values (Nf(label)) and drop any NaNs
fatigue_life = df_master['Nf(label)'].dropna()

# ---------------------------------------------------------
# 3. Original Skewness
# ---------------------------------------------------------
skew_original = fatigue_life.skew()
print(f"Skewness (Original): {skew_original:.3f}")

# ---------------------------------------------------------
# 4. Shift Data if Needed
#    (Box-Cox and Log require strictly positive values)
# ---------------------------------------------------------
if (fatigue_life <= 0).any():
    shift = abs(fatigue_life.min()) + 1e-6
    fatigue_life_shifted = fatigue_life + shift
else:
    fatigue_life_shifted = fatigue_life.copy()

# ---------------------------------------------------------
# 5. Log Transformation
# ---------------------------------------------------------
fatigue_life_log = np.log(fatigue_life_shifted)
skew_log = fatigue_life_log.skew()
print(f"Skewness (Log-Transformed): {skew_log:.3f}")

# ---------------------------------------------------------
# 6. Box-Cox Transformation
# ---------------------------------------------------------
# Box-Cox requires strictly positive data, so we use 'fatigue_life_shifted'
# which we already ensured is > 0.
fatigue_life_boxcox, lambda_ = boxcox(fatigue_life_shifted)
skew_boxcox = pd.Series(fatigue_life_boxcox).skew()
print(f"Skewness (Box-Cox): {skew_boxcox:.3f}")
print(f"Box-Cox Lambda: {lambda_:.3f}")

# ---------------------------------------------------------
# 7. Visualize All Three Distributions
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Original Distribution
sns.histplot(fatigue_life, kde=True, bins=30, ax=axes[0])
axes[0].set_title(f"Original Distribution\n(skew={skew_original:.3f})")
axes[0].set_xlabel("Fatigue Life")

# Log-Transformed Distribution
sns.histplot(fatigue_life_log, kde=True, bins=30, ax=axes[1])
axes[1].set_title(f"Log-Transformed\n(skew={skew_log:.3f})")
axes[1].set_xlabel("Log(Fatigue Life)")

# Box-Cox Distribution
sns.histplot(fatigue_life_boxcox, kde=True, bins=30, ax=axes[2])
axes[2].set_title(f"Box-Cox (λ={lambda_:.3f})\n(skew={skew_boxcox:.3f})")
axes[2].set_xlabel("Box-Cox(Fatigue Life)")

plt.tight_layout()
plt.show()
