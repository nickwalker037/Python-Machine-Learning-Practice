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

h = 0.2                                 # step size in the mesh

# Create color maps:
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['FF0000', '#00FF00', '#0000FF'])

for shrinkage in [None, .2]:
    # we create an instance of Neighbors classifier and fit the data
    clf = NearestCentroid(shrink_threshold=shrinkage)
    clf.fit(X, Y)
    y_pred = clf.predict(X)
    print(shrinkage, np.mean(y == y_pred))
    
