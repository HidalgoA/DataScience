import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, RidgeCV, Lasso
from sklearn.preprocessing import StandardScaler

# URL for the original Boston dataset
data_url = "http://lib.stat.cmu.edu/datasets/boston"

# Load the dataset (fixing the separator issue with r"\s+")
raw_df = pd.read_csv(data_url, sep=r"\s+", skiprows=22, header=None)

# Reshape the data to match the original format
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

# Define feature names (since they aren't included in the raw dataset)
feature_names = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", 
    "PTRATIO", "B", "LSTAT"
]

# Create a DataFrame to match the original boston dataset
boston_df = pd.DataFrame(data, columns=feature_names)

# Add target variable (house price)
boston_df["Price"] = target

# Preview the DataFrame
print(boston_df.head())

# Data Exploration: Correlation Heatmap
plt.figure(figsize=(10, 10))
sns.heatmap(boston_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Boston Housing Dataset")
#plt.show()

boston_df.drop(columns = ["INDUS", "NOX"], inplace = True)

#pairplot
sns.pairplot(boston_df)

#we will log the LSTAT Column, and visualize the pairplot relations, scatter plots of all points
boston_df.LSTAT = np.log(boston_df.LSTAT)
#plt.show()

#prevSELECTS FIRST 11 COLUMNS AS FEATURES
features = boston_df.columns[0:11]

#SELECTS LAST COLUMN "PRICE" AS TARGET VARIABLE
target = boston_df.columns[-1]

#X and y values; EXTRACTS feature values as NUmpy array by .values. converts dataframe into NumpY array
X = boston_df[features].values
y = boston_df[target].values

#splits dta in training and test data. training 70% and testing 30%. 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=17)

print("The dimension of X_train is {}".format(X_train.shape))
print("The dimension of X_test is {}".format(X_test.shape))

#creates an instance of the standardscaler function, Z score, mean 0 and staandard deviation of 1. 
scaler = StandardScaler()  
"""
Data Leakage- info from outside training dataset is used to create model. 

For both training and test set, we want to use same mean and stnd deviation. 
So initally on training set, we use fit_transform to find mean/stnd deviation
But on test set, we only use transform to keep the same mean/stnd deviation.
Otherwise if we do fit_transform on test, wed get new mean/stnd deviation. 
"""
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#PERFORMS TEST 

#Creates instance of LinearRegression function, lr
lr = LinearRegression()

#Fit model
lr.fit(X_train, y_train)

#predict
#prediction = lr.predict(X_test)

#actual
actual = y_test

#Measures how well the model fits training data using the R² score.
train_score_lr = lr.score(X_train, y_train)

#Measures how well the model generalizes to new (test) data.
test_score_lr = lr.score(X_test, y_test)

print("The train score for lr model is {}".format(train_score_lr))
print("The test score for lr model is {}".format(test_score_lr))


#Ridge Regression Model TEST Create a Ridge Regression model with regularization strength α=10. strong ridge regression
ridgeReg = Ridge(alpha=10)

ridgeReg.fit(X_train,y_train) #trains model

#train and test scorefor ridge regression
train_score_ridge = ridgeReg.score(X_train, y_train) # R² on training data
test_score_ridge = ridgeReg.score(X_test, y_test) # R² on test data

print("\nRidge Model............................................\n")
print("The train score for ridge model is {}".format(train_score_ridge))
print("The test score for ridge model is {}".format(test_score_ridge))

'''
R² = 1 → Perfect fit (model explains all variance in data).
R² close to 1 → Good fit.
R² close to 0 → Model explains very little of the data’s variance.
R² < 0 → Model performs worse than a naive mean-based predictor.

TRAIN SCORE - How well model fits training data (data it learned from)
TEST SCORE- How well model generalizes to unseen data (new data)

If the train score is much higher than the test score, your model is likely overfitting 
(memorizing training data but performing poorly on new data).
If the train and test scores are similar, the model generalizes well, which is ideal.

LASSO REGRESSION BELOW
'''

#Lasso regression model
print("\nLasso Model............................................\n")
lasso = Lasso(alpha = 10)
lasso.fit(X_train,y_train)
train_score_ls =lasso.score(X_train,y_train)
test_score_ls =lasso.score(X_test,y_test)

print("The train score for ls model is {}".format(train_score_ls))
print("The test score for ls model is {}".format(test_score_ls))

"""
We get zero coefficents since alpha = 10 is too rigid and cancels out all coefficents. NOT WANTED. 

Selecting Optimal Alpha Values Using Cross-Validation in Sklearn
"""

#Using the linear CV model
from sklearn.linear_model import LassoCV

#Lasso Cross validation
lasso_cv = LassoCV(alphas = [0.0001, 0.001,0.01, 0.1, 1, 10], random_state=0).fit(X_train, y_train)
"""
Tries each alpha value from list, performs cross validation for each alpha, and choosen best alpha.

Can choose a range of alphas too
alphas = np.logspace(-4, 1, 50)  # Generates 50 values between 10^(-4) and 10^(1)

lasso_cv = LassoCV(alphas=alphas, random_state=0).fit(X_train, y_train)

"""

#score
print(lasso_cv.score(X_train, y_train))
print(lasso_cv.score(X_test, y_test))