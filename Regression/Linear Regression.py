# Linear Regression Example using only the first feature from the Diabetes Sklearn Dataset in order to illustrate a two-dimensional plot of this regression technique

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model

# Load the diabetes dataset:
diabetes = datasets.load_diabetes()

# Use only one feature ... we do this in order to illustrate a two-dim. plot of this regression technique
diabetes_X = diabetes.data[:, np.newaxis, 2]

# Split the data into training/test sets:
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/test sets:
diabetes_Y_train = diabetes.target[:-20]
diabetes_Y_test = diabetes.target[-20:]

# Create linear regression object:
regr = linear_model.LinearRegression()

# Train the model using the training sets:
regr.fit(diabetes_X_train, diabetes_Y_train)

# The coefficients:
print ('Coefficients: \n', regr.coef_)
# The Mean Squared Error:
print ("Mean Squared Error: %.2f"               # "print" treats the % as a special character you need to add, so it can know, that when you type "f", the number (result) that will be printed will be a floating point type, and the ".2" tells your "print" to print only the first 2 digits after the point.
       % np.mean((regr.predict(diabetes_X_test) - diabetes_Y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print ("Variance Score: %.2f"
       % regr.score(diabetes_X_test, diabetes_Y_test))

# Plot Outputs:
plt.scatter(diabetes_X_test, diabetes_Y_test, color='black')
plt.plot(diabetes_X_test, regr.predict(diabetes_X_test), color='green', linewidth=3)

plt.title("Linear Regression Example")
plt.xticks()
plt.yticks()

plt.show()
