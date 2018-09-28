from statsmodels.api import datasets
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import numpy as np

# Dataset for solarradiation
File_name1 = ""
# Dataset for air pollution
File_name2 = ""

solar_radiation = pd.read_csv(File_name1)
air_pollution  =pd.read_csv(File_name2)

features = ["date", "location", "WeekdayOrNot", "solar_radiation", "air_pollution"]


# def pca(X):
#     m, n = X.shape
#     U = np.zeros((n))
#     S = np.zeros((n))
#     Sigma = 1/m * X.T * X
#     U, S, V = np.linalg.svd(Sigma)
#     return U, S

def featureNormalize(X):
    mu = np.mean(0, X)
    X_norm = X - mu
    sigma = np.std(0, X_norm)
    X_norm = X_norm / sigma
    return X_norm, mu, sigma

X = solar_radiation + air_pollution

X_norm, mu, sigma = featureNormalize(X)

X_pca = X_norm.pca()

m, n = X_pca.shape()

theta = np.array([[1],[1]])

_lambda = 0




def linearRegCostFunction(X, y, theta, _lambda):
    [a, b] = y.shape()
    [c, d] = theta.shape()
    
    J = 0
    grad = np.zeros(theta.shape())
    [e, f] = grad.shape()

    Hx = np.dot(X, theta)
    special_part_theta = theta[1:c, :]
    regularization = 1 / (2*b) * np.dot((sum(np.power(special_part_theta, 2))), _lambda)
    J = sum(np.power((Hx - y), 2)) / (2*b) + regularization

    grad_1 = X[:, 1].T * (Hx - y) /b
    grad_2 = np.dot(X[:, 2:e].T, (Hx - y)) / m + np.power(_lambda, theta[2:c]) / b
    
    grad = np.concatenate((grad_1, grad_2), axis=0)

    return J, grad

def gradientDescent(X, y, theta, alpha, num_iter):
    [a, b] = y.shape()
    [c, d] = X[:, 2].shape()
    J_history = np.zeros((num_iter, 1))

    for iter in range(num_iter):
        T = np.zeros((c, 1))
        H = np.dot(X, theta)
        for i in range(a):
            T = T + np.dot((H[i,:] - y[i,:]), X[i, :].T)
        theta = theta - T * alpha / a
        # This is a mix of linear regression and polynomial regression
        # so further modification and clarification is needed
        J_history[iter, :] = linearRegCostFunction(X, y, theta, _lambda)
    
    return theta, J_history

def trainLinearRegression(X, y, _lambda):
    [a, b] = X.shape()
    initial_theta = np.zeros((a, 1))
    costFunction = linearRegCostFunction(X, y, theta, _lambda)
    # options = optimse()
