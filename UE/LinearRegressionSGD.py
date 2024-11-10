import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.linear_model import Lasso, Ridge, SGDRegressor


class LinearRegressionSGD:
    def __init__(self, n_iter, rand_key=42, learning_rate=0.001):
        self.n_iter = n_iter
        self.rand_key = rand_key
        self.lr = learning_rate
        self.params_hist = []

    def set_train_test(self, X_train, X_test, y_train, y_test):
        """
        Set the training and testing data.
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        # Ensure the dimensions of the data are correct. Throw an error if not.
        if self.X_train.shape[0] != self.y_train.shape[0] or self.X_test.shape[0] != self.y_test.shape[0]:
            raise ValueError("The number of samples in X and y must be the same.")
    
    def trainSGD(self, res=False):
        """
        Train the model using stochastic gradient descent.
        """
        self.model = SGDRegressor(max_iter=self.n_iter,
                                  learning_rate=self.lr,
                                  random_state=self.rand_key)
        self.model.fit(self.X_train, self.y_train)
        # Save the parameters
        self.w = self.model.coef_
        self.b = self.model.intercept_
    
    def train_ridge(self, lambda_):
        self.model = Ridge(alpha=lambda_,
                            random_state=self.rand_key,
                            max_iter=self.n_iter)
        self.model.fit(self.X_train, self.y_train)
        self.w = self.model.coef_
        self.b = self.model.intercept_

    def train_lasso(self, lambda_):
        self.model = Lasso(alpha=lambda_,
                            random_state=self.rand_key,
                            max_iter=self.n_iter)
        self.model.fit(self.X_train, self.y_train)
        self.w = self.model.coef_
        self.b = self.model.intercept_

    def eval

