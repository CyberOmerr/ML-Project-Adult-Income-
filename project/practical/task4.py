# Task 4 – Logistic Regression from Scratch

import numpy as np
from sklearn.datasets import make_classification

class ScratchLogistic:
    def __init__(self, lr=0.01, n_iter=1000, reg=0.0):
        self.lr = lr
        self.n_iter = n_iter
        self.reg = reg

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        X = np.c_[np.ones(X.shape[0]), X]
        self.theta = np.zeros(X.shape[1])
        for _ in range(self.n_iter):
            h = self.sigmoid(X @ self.theta)
            grad = (X.T @ (h - y)) / y.size + self.reg * self.theta
            self.theta -= self.lr * grad
        return self

    def predict(self, X):
        X = np.c_[np.ones(X.shape[0]), X]
        return (self.sigmoid(X @ self.theta) >= 0.5).astype(int)

# Test on toy dataset
Xt, yt = make_classification(n_samples=500, n_features=5, random_state=42)
sl = ScratchLogistic(lr=0.1, n_iter=1000).fit(Xt, yt)
ypred = sl.predict(Xt)

from sklearn.metrics import accuracy_score
print("Scratch Logistic Accuracy:", accuracy_score(yt, ypred))
