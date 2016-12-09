import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

# the decision tree is used to predict simultaneously the noisy x and y observations of a circle given a single underlying feature
# as a result, it learns local linear regressions approximating the circle

# Create a random dataset:
rng = np.random.RandomState(1)
X = np.sort(200 * rng.rand(100, 1) - 100, axis = 0)
Y = np.array([np.pi * np.sin(X).ravel(), np.pi * np.cos(X).ravel()]).T
Y[::5, :] += (0.5 - rng.rand(20, 2))

# Fit regression model:
regr_1 = DecisionTreeRegressor(max_depth = 2)
regr_2 = DecisionTreeRegressor(max_depth = 5)
regr_3 = DecisionTreeRegressor(max_depth = 8)
regr_1.fit(X, Y)
regr_2.fit(X, Y)
regr_3.fit(X, Y)

# Predict:
x_test = np.arange(-100.0, 100.0, 0.01)[:, np.newaxis]
y_1 = regr_1.predict(x_test)
y_2 = regr_2.predict(x_test)
y_3 = regr_3.predict(x_test)

# Plot the results
plt.figure()
s = 50
plt.scatter(Y[:, 0], Y[:, 1], c="navy", s=s, label="Data")
plt.scatter(y_1[:, 0], y_1[:, 1], c="cornflowerblue", s=s, label="max_depth=2")
plt.scatter(y_2[:, 0], y_2[:, 1], c="c", s=s, label="max_depth=5")
plt.scatter(y_3[:, 0], y_3[:, 1], c="orange", s=s, label="max_depth=8")
plt.xlim([-6, 6])
plt.ylim([-6, 6])
plt.xlabel("target 1")
plt.ylabel("target 2")
plt.title("Multi-Output Decision Tree Regression")
plt.legend()
plt.show()
