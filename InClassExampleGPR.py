import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


#Sample data uses Gaussian Process Regression to predict yield strength based on alloy composition. 
X = np.array([[0.1, 0.5, 0.2, 0.1], #C , Mn, Si, Cr
              [0.2, 0.6, 0.3, 0.2],
              [0.15, 0.55, 0.25, 0.15]]) 

y= np.array([500, 550, 525])

#Define Kernel
kernel = C(1.0, (1e-3, 1e3)) * RBF([1.0, 1.0, 1.0, 1.0], (1e-2, 1e2))

#Create and fit the GP model
gp = GaussianProcessRegressor(kernel= kernel, n_restarts_optimizer=14)
gp.fit(X, y)

#Predict for a new composition 
X_new = np.array([[0.18, 0.58 ,0.28 , 0.18]])
y_pred , sigma = gp.predict(X_new, return_std = True)

print(f"Predicted YS: {y_pred[0]:.2f}")
print(f"Uncertainty: {sigma[0]:.2f}")