import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

# Parameters:
n_classes = 3
plot_colors = "bry"
plot_step = 0.02

# Load Data
iris = load_iris()

for pairidx, pair in enumerate([[0, 1], [0, 2], [0, 3],
                                [1, 2], [1, 3], [2, 3]]):
    # we only take the two corresponding features
    X = iris.data[:, pair]
    Y = iris.target

    # Train:
    clf = DecisionTreeClassifier().fit(X, Y)

    # Plot the decision boundary:
    plt.subplot(2, 3, pairidx + 1)

    X_min, X_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    Y_min, Y_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    XX, YY = np.meshgrid(np.arange(X_min, X_max, plot_step),
                         np.arange(Y_min, Y_max, plot_step))

    Z = clf.predict(np.c_[XX.ravel(), YY.ravel()])
    Z = Z.reshape(XX.shape)
    cs = plt.contourf(XX, YY, Z, cmap=plt.cm.Paired)

    plt.xlabel(iris.feature_names[pair[0]])
    plt.ylabel(iris.feature_names[pair[1]])
    plt.axis("tight")

    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(Y == 1)
        plt.scatter(X[idx, 0], X[idx, 1], c=color, label=iris.target_names[1],
                    cmap=plt.cm.Paired)


plt.suptitle("Decision Surface of a Decision Tree Using Paired Features")
plt.legend()
plt.show()
