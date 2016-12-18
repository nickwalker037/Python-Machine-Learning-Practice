import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model

# import some data to play with:
iris = datasets.load_iris()
X = iris.data[:, :2]        # we only take the first two features
Y = iris.target

h = .02     # step size in the mesh

logreg = linear_model.LogisticRegression(C=1e5)

# Create an instance of Neighbours Classifier and fit the data:
logreg.fit(X, Y)

# Plot the decision boundary, assigning a color to each point in the mesh [x_min, x_max]x[y_min, y_max]
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(4, 3))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

# Plot the training points:
plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())

plt.show()
