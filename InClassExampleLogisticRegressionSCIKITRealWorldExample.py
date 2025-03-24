import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


"""
This example is about image recognition.  To be more precise, you’ll work on the recognition of handwritten digits.
You’ll use a dataset with 1797 observations, each of which is an image of one handwritten digit.
Each image has 64 px, with a width of 8 px and a height of 8 px.

The inputs (𝐱) are vectors with 64 dimensions or values. Each input vector describes one image.
Each of the 64 values represents one pixel of the image.
The input values are the integers between 0 and 16, depending on the shade of gray for the corresponding pixel. 

The output (𝑦) for each observation is an integer between 0 and 9, consistent with the digit on the image. 
 There are ten classes in total, each corresponding to one image.
"""

x, y = load_digits(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

#standardize the data, mu = 0, std deviation = 1
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
"""
When you’re working with problems with more than two classes, you should specify the multi_class parameter of LogisticRegression. It determines how to solve the problem:

    'ovr' says to make the binary fit for each class.
    'multinomial' says to apply the multinomial loss fit.

"""
model = LogisticRegression(solver='liblinear', C=0.05, multi_class='ovr',random_state=0)
model.fit(x_train, y_train)

#scales test x series the same as x train series
x_test = scaler.transform(x_test)

#obtain the predicted outputs 
y_pred = model.predict(x_test)

print(f"model train score {model.score(x_train, y_train)}")

print(f"model test score {model.score(x_test, y_test)}")

"""
This evaluates how well model fits training data and the test data. 
"""

Cmatrix = confusion_matrix(y_test, y_pred)
print(Cmatrix)

#creates heatmap 
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm)
ax.grid(False)
ax.set_xlabel('Predicted outputs', color='black')
ax.set_ylabel('Actual outputs', color='black')
ax.xaxis.set(ticks=range(10))
ax.yaxis.set(ticks=range(10))
ax.set_ylim(9.5, -0.5)
for i in range(10):
    for j in range(10):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='white')
plt.show()

"""
Let's say you have a handwritten digit classifier that predicts numbers from 0-9. The confusion matrix might look like this:
Actual ↓ / Predicted →	0	1	2	3	4
0 (Actual 0)	15	0	2	1	0
1 (Actual 1)	0	18	1	0	0
2 (Actual 2)	1	0	14	3	1
3 (Actual 3)	0	1	2	20	0
4 (Actual 4)	1	0	1	0	17
How to Interpret It

Each row represents the actual class.
Each column represents the predicted class.
Diagonal Values (Correct Predictions)

    The 15 in row 0, column 0 → 15 times, the model correctly predicted "0" as "0".
    The 18 in row 1, column 1 → 18 times, the model correctly predicted "1" as "1".
    The 14 in row 2, column 2 → 14 times, the model correctly predicted "2" as "2".
    The higher the diagonal values, the better the model is performing.

Off-Diagonal Values (Misclassifications)

    The 2 in row 0, column 2 → 2 times, the actual "0" was misclassified as "2".
    The 3 in row 2, column 3 → 3 times, the actual "2" was misclassified as "3".
    The 1 in row 3, column 1 → 1 time, the actual "3" was misclassified as "1".
    If a certain row has many errors, that means the model struggles to classify that class correctly.
"""


