import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Generate skewed and normal distributions
skewed_dist = stats.loggamma.rvs(5, size=10000) + 5
normal_dist = np.random.normal(0, 1, 10000)

# Create a figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot normal distribution
sns.histplot(normal_dist, bins=50, kde=True, ax=axes[0])
axes[0].set_title("Normal Distribution")

# Plot skewed distribution
sns.histplot(skewed_dist, bins=50, kde=True, ax=axes[1])
axes[1].set_title("Skewed Distribution")

# Print skewness values
print("Skewness for the normal distribution: ", stats.skew(normal_dist))
print("Skewness for the skewed distribution: ", stats.skew(skewed_dist))

# Show the plot
plt.tight_layout()
plt.show() 






# Probability plot: Assess if the dataset follows an exponential distribution- Qauntile Quantile Q Q Plot
fig, ax = plt.subplots()
stats.probplot(normal_dist, dist="norm", plot=ax)
ax.set_title("Q-Q Plot for Normal Distribution (Exponential Fit)")
plt.show()

fig, ax = plt.subplots()
stats.probplot(skewed_dist, dist="norm", plot=ax)
ax.set_title("Q-Q Plot for Skewed Distribution (Exponential Fit)")
plt.show()

# Box-Cox Transformation (requires positive values, positivly skewed)
skewed_box_cox, lmda = stats.boxcox(skewed_dist)

# Plot transformed data
sns.histplot(skewed_box_cox, bins=50, kde=True)
plt.title("Box-Cox Transformed Distribution")
plt.show()

# Probability plot after Box-Cox Transformation
fig, ax = plt.subplots()
stats.probplot(skewed_box_cox, dist="norm", plot=ax)
ax.set_title("Q-Q Plot After Box-Cox Transformation")
plt.show()

# Print lambda value for Box-Cox transformation
print("Lambda parameter for Box-Cox Transformation is:", lmda)
