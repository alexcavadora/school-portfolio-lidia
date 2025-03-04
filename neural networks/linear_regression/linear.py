import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

test = datasets.make_regression(n_samples= 100, n_features = 1, noise= 20)

#y = mx + b
class LinearRegression:
    def __init__(self, iterations = 100, learning_rate = 0.1):
        self.iterations = iterations
        self.learning_rate = 0.1
    
    def fit(self, inputs, outputs):
        n_samples, n_features = input.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.iterations):
            prediction = self.predict(inputs)
            dw = (1/n_samples) * np.dot(inputs.T, prediction - outputs)
            db = (1/n_samples) * np.sum(result - outputs) * 2
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(X):
        return np.dot(self.weights, x) + bias

    def mse(X, y, theta):
        y_hat = theta_hat * X
        mse = np.mean((y - y_hat)**2)

        return mse


