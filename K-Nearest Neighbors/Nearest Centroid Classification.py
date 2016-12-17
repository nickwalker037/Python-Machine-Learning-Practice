import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.neighbors import NearestCentroid

n_neighbors = 15

# Import data:
iris = datasets.load_iris()
X = iris.data[:, :2]                    # we only take the first two features. we could avoid this ugly slicing by using a two-dimensional dataset

Y = iris.target

h = .02                                 # step size in the mesh

# Create color maps:
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

for shrinkage in [None, .2]:
    # we create an instance of Neighbors classifier and fit the data
    clf = NearestCentroid(shrink_threshold=shrinkage)
    clf.fit(X, Y)
    y_pred = clf.predict(X)
    print(shrinkage, np.mean(y == y_pred))
     # Plot the decision boundary. For this, we'll assign a color to each point in the mesh [x_min, x_max]x[y_min, y_max]
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points:
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=cmap_bold)
    plt.title("3-Class Classification (shrink_threshold=%r)"
              % shrinkage)
    plt.axis('tight')

plt.show()
