import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

# Create a random dataset:
rng = np.random.RandomState(1)
X = np.sort(5 * rng.rand(80,1), axis=0)
Y = np.sin(X).ravel()                                   # ravel() returns a 1-D array containing the elements of the input (here the input is sin(x)) 
Y[::5] += 3 * (0.5 - rng.rand(16))

# Fit the regression model:
regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=5)
regr_1.fit(X, Y)
regr_2.fit(X, Y)

# Predict:
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
Y_1 = regr_1.predict(X_test)
Y_2 = regr_2.predict(X_test)

# Plot the results
plt.figure()
plt.scatter(X, Y, c="darkorange", label='data')
plt.scatter(X_test, Y_1, color="cornflowerblue", label="max_depth=2", linewidth="1.")
plt.scatter(X_test, Y_2, color="yellowgreen", label="max_depth=5", linewidth=".1")
plt.xlabel("Data")
plt.ylabel("Target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()
