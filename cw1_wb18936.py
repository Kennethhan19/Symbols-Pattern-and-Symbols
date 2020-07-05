from __future__ import print_function

import os
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

#from utilities.py
#to load points from csv files
def load_points_from_file(filename):
    points = pd.read_csv(filename, header=None)
    return points[0].values, points[1].values

#split points into segments of 20
def split (xs, ys):
    x, y = [], []
    for i in range (len(xs)//20):
        x.append(xs[i*20: (i+1)*20])
        y.append(ys[i*20: (i+1)*20])
    return x, y

#least square linear
def least_squares_linear(x, y):
    ones = np.ones(x.shape)
    x_e = np.column_stack((ones, x))
    v = np.linalg.inv(x_e.T.dot(x_e)).dot(x_e.T).dot(y)
    return v

#least square cubic
def least_squares_polynomial(x,y):
    ones = np.ones(x.shape)
    x_e = np.column_stack((ones, x, x**2, x**3))
    v = np.linalg.inv(x_e.T.dot(x_e)).dot(x_e.T).dot(y)
    return v

#least square sinusoidal
def least_squares_sinusoidal(x,y):
    ones = np.ones(x.shape)
    x_e = np.column_stack((ones, np.sin(x)))
    v = np.linalg.inv(x_e.T.dot(x_e)).dot(x_e.T).dot(y)
    return v

# k - fold cross validation used to prevent overfitting
def cross_validation(x, y):
    cvError_linear, cvError_polynomial, cvError_sinusoidal = [], [], []
    for i in range (len(x)):
        error_linear, error_polynomial, error_sinusoidal = [], [], []
        x_cv, y_cv = x[i], y[i]
        for i in range (len(x_cv)//5):
            train_x, test_x = np.append(x_cv[:(i)*5] , x_cv[(i+1)*5:]), x_cv[(i)*5:(i+1)*5]
            train_y, test_y = np.append(y_cv[:(i)*5] , y_cv[(i+1)*5:]), y_cv[(i)*5:(i+1)*5]

            v1 = least_squares_linear(train_x, train_y)
            y_hat_linear = (v1[0] + np.multiply(v1[1], test_x))
            error_linear.append(np.sum((test_y - y_hat_linear)**2))

            v2 = least_squares_polynomial(train_x, train_y)
            y_hat_polynomial = (v2[0] + np.multiply(v2[1],test_x) + np.multiply(v2[2],(test_x)**2) + np.multiply(v2[3],(test_x)**3))
            error_polynomial.append(np.sum((test_y - y_hat_polynomial)**2))

            v3 = least_squares_sinusoidal(train_x, train_y)
            y_hat_sinusoidal = (v3[0] + np.multiply(v3[1],np.sin(test_x)))
            error_sinusoidal.append(np.sum((test_y - y_hat_sinusoidal)**2))

        cvError_linear.append(np.mean(error_linear))
        cvError_polynomial.append(np.mean(error_polynomial))
        cvError_sinusoidal.append(np.mean(error_sinusoidal))

    return (cvError_linear, cvError_polynomial, cvError_sinusoidal)

#pick function by using min cvError
def min_cv_error (x,y):
    cvError_linear, cvError_polynomial, cvError_sinusoidal = cross_validation(x,y)
    error = []
    for i in range (len(x)):
        error.append(min(cvError_linear[i], cvError_polynomial[i], cvError_sinusoidal[i]))
    return (error, cvError_linear, cvError_polynomial, cvError_sinusoidal)

#calculate y hat value after picking function
def y_hat(x, y, xs):
    yHat = []
    error, cvError_linear, cvError_polynomial, cvError_sinusoidal = min_cv_error(x,y)
    for i in range (len(x)):
        if error[i] == cvError_linear[i]:
            v1 = least_squares_linear(x[i], y[i])
            yHat.append(v1[0] + np.multiply(v1[1],x[i]))

        elif error[i] == cvError_polynomial[i]:
            v2 = least_squares_polynomial(x[i], y[i])
            yHat.append(v2[0] + np.multiply(v2[1],x[i]) + np.multiply(v2[2],(x[i])**2) + np.multiply(v2[3],(x[i])**3))

        elif error[i] == cvError_sinusoidal[i]:
            v3 = least_squares_sinusoidal(x[i], y[i])
            yHat.append(v3[0] + np.multiply(v3[1],np.sin(x[i])))
    return yHat

#calculate squared error
def squared_error (x, y, xs):
    yHat = y_hat(x, y, xs)
    error = []
    for i in range (len(x)):
        error.append(np.sum((y[i] - (yHat[i]))**2))
    return (error)

#plot function
def plot (x, y, xs, ys):
    yHat = y_hat(x,y,xs)
    yhat = np.reshape(yHat, (len(xs),1))
    error, cvError_linear, cvError_polynomial, cvError_sinusoidal = min_cv_error(x,y)
    fig, ax = plt.subplots()
    ax.scatter(xs, ys, c ='k')
    for i in range (len(xs)//20):
        if error[i] == cvError_linear[i] :
            ax.plot(x[i], yHat[i], 'r-', label='linear', lw=4)

        elif error[i] == cvError_polynomial[i]:
            ax.plot(x[i], yHat[i], 'c-', label='cube', lw=4)

        elif error[i] == cvError_sinusoidal[i]:
            ax.plot(x[i], yHat[i], 'b-', label = 'sinusoidal', lw=4)
    plt.legend(loc='best')
    plt.show()

def main():
    filename = sys.argv[1]
    xs, ys = load_points_from_file(filename)
    x, y = split(xs, ys)
    error = squared_error(x, y, xs)
    print(sum(error))

    if len(sys.argv) == 3:
        if sys.argv[2] == "--plot":
            plot(x, y, xs, ys,)
        else:
            print("error, please type '--plot' to view graph")

main()
