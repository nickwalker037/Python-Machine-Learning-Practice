import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

xx, yy = np.meshgrid(np.linspace(-3, 3, 500),
                     np.linspace(-3, 3, 500))           # sets up an array of 500 evenly spaced points between -3, 3 that represent the coordinates of a grid

np.random.seed(0)
X = np.random.randn(300, 2)
Y = np.logical_xor(X[:, 0] > 0, X[:,1] > 0)             # xor function inputs true when the inputs differ (one is true, the other is false)

# fit the model:
clf = svm.NuSVC()
clf.fit(X, Y)

# plot the decision function for each datapoint on the grid
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])    # decision fn tells us which side of the hyperplane generated by the classifier we are (and how far away we are from it)
Z = Z.reshape(xx.shape)                                

plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           aspect='auto',
           origin='lower',
           cmap=plt.cm.PuOr_r)
contours = plt.contour(xx, yy, Z, levels=[0], linewidths=2, linetypes='--')

plt.scatter(X[:, 0],X[:, 1], s=30, c=Y, cmap=plt.cm.Paired)
plt.xticks(())
plt.yticks(())
plt.axis([-3, 3, -3, 3])
plt.show()
