import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR

# generate sample data:
X = np.sort(5 * np.random.rand(40,1), axis=0)
Y = np.cos(X).ravel()
   # putting sin or cos here makes it fit to the rbf model

# Add noise to targets:
Y[::5] += 3 * (0.5 - np.random.rand(8))

# Fit regression models
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_lin = SVR(kernel='linear', C=1e3)
svr_poly = SVR(kernel='poly', C=1e3, degree=2)
Y_rbf = svr_rbf.fit(X, Y).predict(X)
Y_lin = svr_lin.fit(X, Y).predict(X)
Y_poly = svr_poly.fit(X, Y).predict(X)

# Look at results:
lw = 2
plt.scatter(X, Y, color='darkorange', label='Data')
plt.hold('on')
plt.plot(X, Y_rbf, color='navy', lw=lw, label='RBF model')
plt.plot(X, Y_lin, color='c', lw=lw, label='Linear model')
plt.plot(X, Y_poly, color='cornflowerblue', lw=lw, label='Polynomial Model')
plt.xlabel('Data')
plt.ylabel('Target')
plt.title('Support Vector Regression')
plt.legend()
plt.show()
