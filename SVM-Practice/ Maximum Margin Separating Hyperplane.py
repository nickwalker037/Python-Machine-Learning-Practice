import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

# create 40 separable points:
np.random.seed(0)                               # gives predictable random values with seed value = 0
X = np.r_[np.random.randn(20, 2) - [2,2], np.random.randn(20,2) + [2,2]]
          # gives a value from the st. normal dist.
          # Ex. normally written as ---- sigma * np.random.randn(dimensions) + mu
Y = [0] * 20 + [1] * 20
    # gives you an array of twenty 0's and then twenty 1's --- i.e. the desired output

# fit the model:
clf = svm.SVC(kernel='linear')
clf.fit(X,Y)

# get the separating hyperplane:
w = clf.coef_[0]
a = -w[0] /w[1]
xx = np.linspace(-5,5)
yy = a * xx - (clf.intercept_[0]) / w[1]

# plot the parallels to the separating hyperplane that pass through the support vectors:
b = clf.support_vectors_[0]
yy_down = a * xx + (b[1] - a * b[0])
b = clf.support_vectors_[-1]
yy_up = a * xx + (b[1] - a * b[0])

plt.plot(xx, yy, 'k-')
plt.plot(xx, yy_down, 'k--')
plt.plot(xx, yy_up, 'k--')

plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80, facecolors='none')
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)

plt.axis('tight')
plt.show()
