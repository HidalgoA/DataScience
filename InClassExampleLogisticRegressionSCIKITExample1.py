import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
"""https://realpython.com/logistic-regression-python/"""
x = np.arange(10).reshape(10, 1) #creates one column, and 10 rows
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1]) # y is 1D with 10 items. 

#defaults to L2 Regression, which is ridge regression. regularization- preventing overfitting 

#can improve model bychanging regularization stregnth c to 10.0 , INSTEAD OF DEFAULT 1. 

"""
High CC (e.g., 10, 100, 1000)	Less regularization → Fits the data more closely (risk of overfitting).
Default C=1.0C=1.0	Balanced regularization (prevents overfitting but still learns well).
Low CC (e.g., 0.1, 0.01, 0.001)	More regularization → Simplifies the model (risk of underfitting).
"""
model = LogisticRegression(solver = 'liblinear', C= 10.0, random_state= 0)

#FIT AND TRAINS MODEL, determines coefficents correspond to best value of function 
model.fit(x,y)

#his is the example of binary classification, and y can be 0 or 1, as indicated above.
print(f"Binary Classification {model.classes_}")

#You can also get the value of the slope 𝑏₁ and the intercept 𝑏₀ of the linear function 𝑓 like so:
print(f"the intercept{model.intercept_}")
print(f"the value of the slope {model.coef_}")

print(f"the matrix of probabilities that the predicted output is equal to zero or one {model.predict_proba(x)}")

# Get probabilities
probs = model.predict_proba(x)  # Returns probability of class 0 and class 1

"""
Each row in the output corresponds to one input data point (one value of x).

    First column → Probability that the model predicts 0
    Second column → Probability that the model predicts 1
    The two probabilities always add up to 1

"""

# Plot
plt.figure(figsize=(8, 6))
plt.plot(x, probs[:, 0], 'bo--', label="P(Y=0 | X)", alpha=0.7)  # Probability of class 0
plt.plot(x, probs[:, 1], 'ro-', label="P(Y=1 | X)", alpha=0.7)  # Probability of class 1
plt.axhline(0.5, color='gray', linestyle='dashed', label="Decision Threshold (0.5)")
plt.xlabel("X Value")
plt.ylabel("Probability")
plt.title("Logistic Regression Predictions")
plt.legend()
#plt.show()

print(f"score/accuracy of the model {model.score(x, y)}")


# Can create confusion matrix
"""

    True negatives in the upper-left position
    False negatives in the lower-left position
    False positives in the upper-right position
    True positives in the lower-right position

"""

print(f"Confusion Matrix  {confusion_matrix(y, model.predict(x))}")

"""

    Three true negative predictions: The first three observations are zeros predicted correctly.
    No false negative predictions: These are the ones wrongly predicted as zeros.
    One false positive prediction: The fourth observation is a zero that was wrongly predicted as one.
    Six true positive predictions: The last six observations are ones predicted correctly.

"""
