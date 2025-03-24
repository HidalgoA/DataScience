# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
# Sample dataset (Position vs. Salary)
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]) # Position level
y = np.array([45000, 50000, 60000, 80000, 110000, 150000, 200000, 300000, 500000, 1000000]) # Salary
# Feature scaling
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
# Train SVR model with RBF kernel
regressor = SVR(kernel='rbf', C=1e3, gamma=0.1)
regressor.fit(X_scaled, y_scaled)
# Generate smooth curve for visualization
X_grid = np.arange(min(X_scaled), max(X_scaled), 0.01).reshape(-1, 1)
y_pred = regressor.predict(X_grid)
# Inverse scaling for interpretability
X_plot = scaler_X.inverse_transform(X_grid)
y_plot = scaler_y.inverse_transform(y_pred.reshape(-1, 1))
# Plot results
plt.scatter(scaler_X.inverse_transform(X_scaled), scaler_y.inverse_transform(y_scaled.reshape(-1, 1)),
color='red', label='Data')
plt.plot(X_plot, y_plot, color='blue', label='SVR (RBF Kernel)')
plt.title('Support Vector Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.legend()
plt.show()