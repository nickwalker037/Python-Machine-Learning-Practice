# This plot shows the training and validation scores of an SVM for different values of the kernel parameter gamma
# You want high values of both to show that the classifier is performing well

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import validation_curve

digits = load_digits()
X, Y= digits.data, digits.target

param_range = np.logspace(-6, -1, 5)        # (-6, -1 ) = range, 5 = number of samples to take
train_scores, test_scores = validation_curve(SVC(), X, Y, param_name="gamma", param_range=param_range, cv=10, scoring="accuracy", n_jobs=1)
    # in above line, SVC() is the estimator, param_range = values of the parameter that will be evaluated, cv = determines the cross-validation splitting strategy, n_jobs = number of jobs to run in parallel (default is 1)
train_scores_mean = np.mean(train_scores, axis=1) # axis = axis over which the mean is performed. 1 = a flattened array
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title("Validation Curve with SVM")
plt.xlabel("$\gamma$")
plt.ylabel("Score")
plt.ylim(0.0, 1.1)
lw=2

plt.semilogx(param_range, train_scores_mean, label="Training Score", color="darkorange", lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.2)

plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
                         color="navy", lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)

plt.show()
