# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 12:39:45 2024

@author: lich5
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from scipy.io import loadmat

# Set random seed and simulate data
np.random.seed(300)
X = np.random.randn(100, 5)  # 100 samples, 5 features
beta = np.array([1, 2, 3, 4, 5])  # True coefficients
Y = X @ beta + np.random.randn(100)  # Dependent variable

# Linear regression (to mimic fitlm in MATLAB)
mdl1 = LinearRegression()
mdl1.fit(X, Y)
Y_pred = mdl1.predict(X)

# Plot the fit
plt.figure()
plt.scatter(Y, Y_pred, label="Predicted vs Actual")
plt.plot([min(Y), max(Y)], [min(Y), max(Y)], 'r--', label="Ideal fit")
plt.xlabel('Actual Y')
plt.ylabel('Predicted Y')
plt.legend()
plt.title('Linear Regression Fit')
plt.show()

# Lasso regression with a range of lambdas
lambdas = np.logspace(0, 1, 50)
lasso_model = Lasso(max_iter=10000)
coefficients = []

for l in lambdas:
    lasso_model.set_params(alpha=l)
    lasso_model.fit(X, Y)
    coefficients.append(lasso_model.coef_)

coefficients = np.array(coefficients)

# Plot the coefficient path
plt.figure()
for i in range(coefficients.shape[1]):
    plt.plot(lambdas, coefficients[:, i], label=f'x{i+1}')
plt.xscale('log')
plt.xlabel('Lambda')
plt.ylabel('Coefficient Estimate')
plt.title('Lasso Regression: Coefficient Path')
plt.legend()
plt.show()

# Load acetylene-like data
dataset = loadmat('acetylene.mat')
x1 = dataset['x1'] # np.random.randn(100)
x2 = dataset['x2'] # np.random.randn(100)
x3 = dataset['x3'] # np.random.randn(100)
y = np.squeeze(dataset['y']) # np.random.randn(100)
X_poly = np.column_stack((x1, x2, x3))
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
D = poly.fit_transform(X_poly)

# Lasso regression on acetylene-like data
lambda_range = np.linspace(0, 0.005, 100)
coefficients_poly = []

for l in lambda_range:
    lasso_model.set_params(alpha=l)
    lasso_model.fit(D, y)
    coefficients_poly.append(lasso_model.coef_)

coefficients_poly = np.array(coefficients_poly)

# Plot the ridge trace
plt.figure()
for i in range(coefficients_poly.shape[1]):
    plt.plot(lambda_range, coefficients_poly[:, i], label=f'x{i+1}')
plt.xlabel('Ridge Parameter')
plt.ylabel('Standardized Coefficient')
plt.title('Ridge Trace')
plt.legend()
plt.grid(True)
plt.show()

# [EOF]
