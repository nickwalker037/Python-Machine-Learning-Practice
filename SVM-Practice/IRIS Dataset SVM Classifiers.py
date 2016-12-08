import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets

iris = datasets.load_iris()
X = iris.data[:, :2]                # We only take the first two features. We could avoid this ugly slicing by using a two-dim dataset
Y = iris.target

h = .02                             # step size in the mesh

# We create an instance of SVM and fit our data. We do not scale our data since we want to plot the support vectors
C = 1.0                           # SVM regularization parameter
svc = svm.SVC(kernel='linear', C=C).fit(X, Y)
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, Y)
ply_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, Y)
lin_svc = svm.LinearSVC(C=C).fit(X, Y)

# create a mesh to plot in:
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() -1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),            # np.arange - returns evenly spaced values within a given interval, with step size h
                     np.arange(y_min, y_max, h))

# Titles for the plots:
titles = ['SVC with linear kernel',
          'LinearSVC (Linear Kernel)',
          'SVC with RBF Kernel',
          'SVC with Polynomial Kernel']

for i, clf in enumerate((svc, lin_svc, rbf_svc, ply_svc)):
    # plot the decision boundary. For that, we'll assign a color to each point in the mesh [x_min, x_max]x[y_min, y_max]
    plt.subplot(2, 2, i + 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot:
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.coolwarm)
    plt.xlabel('Sepal Length')
    plt.ylabel('Sepal Width')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(titles[i])

plt.show()
    
