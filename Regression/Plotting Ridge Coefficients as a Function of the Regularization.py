import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

# X is the 10x10 Hilbert Matrix
X = 1. / (np.arange(1, 11) + np.arange(0, 10)[:, np.newaxis])
Y = np.ones(10)

# Compute paths:
n_alphas = 10
alphas = np.logspace(-10, -2, n_alphas)
clf = linear_model.Ridge(fit_intercept=False)

coefs = []
for a in alphas:
    clf.set_params(alpha=a)
    clf.fit(X, Y)
    coefs.append(clf.coef_)

# Display the results:
ax = plt.gca()

ax.plot(alphas, coefs)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])    # reverse axis
plt.xlabel('Alpha')
plt.ylabel('Weights')
plt.title('Ridge Coefficients as a Function of the Regularization')
plt.axis('tight')
plt.show()


# Each color represents a different feature of the coefficient vector
# This example shows the usefulness of applying Ridge regression to highly ill-conditioned matrices.
    # For such matrices, a slight change in the target variable can cause huge variances in the calculated weights. 
    # In such cases, it is useful to set a certain regularization (alpha) to reduce this variation (noise).

# When alpha is very large, the regularization effect dominates the squared loss function and the coefficients tend to zero. 
# As alpha tends toward zero and the solution tends towards the ordinary least squares, coefficients exhibit big oscillations
