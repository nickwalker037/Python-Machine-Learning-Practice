import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors


# Generate Sample Data
np.random.seed(0)
X = np.sort(5 * np.random.rand(40, 1), axis = 0)
T = np.linspace(0, 5, 500)[:, np.newaxis]
Y = np.sin(X).ravel()

# Add noise to the targets
Y[::5] += 1 * (0.5 - np.random.rand(8))

# Fit regression model
n_neighbors = 5

for i, weights in enumerate(['uniform', 'distance']):
    knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
    Y_ = knn.fit(X, Y).predict(T)

    plt.subplot(2, 1, i + 1)
    plt.scatter(X, Y, c='b', label="Data")
    plt.plot(T, Y_, c='g', label='Prediction')
    plt.axis('tight')
    plt.legend()
    plt.title("KNeighborsRegressor (k = %i, weights = '%s')" %(n_neighbors, weights))

plt.show()
