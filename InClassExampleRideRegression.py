# load and summarize housing data sheet
from numpy import mean
from numpy import std
from numpy import absolute
from pandas import read_csv
from matplotlib import pyplot

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import Ridge

#Load datasheet
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
dataframe = read_csv(url, header = None)
# summarize shape but it acts like excel sheet
print(dataframe.shape)
# summarize first few lines
print(dataframe.head())


#Inititally dataframe isnt an arrary, but we want to read the data easily so we convert to an array. 
data = dataframe.values

#Pick all rows, and all columns but the last column 
X, y = data[:, :-1], data[:,-1]

#Define model
model = Ridge(alpha=0.5)

#define model evaluation method. 
#Sets up test; tests model 10 times on different slices of data. SPLIT BETWEEN TEST AND TRAINING DATA. 
#Perform 3 times to make sure results are consistent . CV = cross validation 
cv = RepeatedKFold(n_splits= 10, n_repeats=3, random_state=1)

#evaluates model. trains model multiple times, and tests how much error it has
#error measured using mean absoulte error MAE; how far off predictions are from real answers. 
scores = cross_val_score(model, X, y, scoring = 'neg_mean_absolute_error', cv = cv, n_jobs= -1)

#force scores to be positive
scores = absolute(scores)
#prints out error from the model. can modify lambda for example to get low std. this is to just test how well
#the model works
print('Mean MAE: %.3f (%.3f)' % (mean(scores), std(scores)))


"""
Using Ridge Regression as final model, make prediction on new data. 
"""
# fit model
model.fit(X, y)
# define new data
row = [0.00632,18.00,2.310,0,0.5380,6.5750,65.20,4.0900,1,296.0,15.30,396.90,4.98]
# make a prediction
yhat = model.predict([row])
# summarize prediction
print('Predicted: %.3f' % yhat)


"""
How do we know that the default hyperparameters of alpha=1.0 is appropriate for our dataset?
We dont. 
Instead, it is good practice to test a suite of different configurations and discover what works best for our dataset.
One approach would be to grid search alpha values from perhaps 1e-5 to 100 on a log scale and discover what works best for a dataset.
Another approach would be to test values between 0.0 and 1.0 with a grid separation of 0.01. We will try the latter in this case
"""