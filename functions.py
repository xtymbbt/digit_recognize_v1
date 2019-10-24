# import required packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.python.framework import ops


# define the sigmoid function


def sigmoid(z):
    g = 1.0 / (1.0 + np.exp(-z))
    return g


# compute the cost using theta, x, y, lambda


def cost_function(theta, x, y, lambda_):

    # initialize the parameters
    m = y.shape[0]
    j = 0
    grad = np.zeros(theta.shape)
    # print(m, j, grad)

    # compute the cost
    h = sigmoid(np.dot(x, theta))
    j = (-np.dot(y.T, np.log(h)) - np.dot((1 - y).T, np.log(1 - h))) / m
    # print(h, j)

    theta_g = theta
    theta_g[0] = 0
    # print(theta, theta_g)

    j = j + np.dot(theta_g.T, theta_g) * lambda_ / (2*m)

    grad = np.dot(x.T, (h - y)) / m
    grad = grad + theta_g * lambda_ / m

    return j, grad


def cost_function2d(theta, x, y, lambda_):

    # initialize the parameters
    m = y.shape[0]
    j = 0
    grad = np.zeros(theta.shape)
    # print(m, j, grad)

    # compute the cost
    h = sigmoid(np.dot(x, theta))
    # j = (-np.dot(y.T, np.log(h)) - np.dot((1 - y).T, np.log(1 - h))) / m
    j = np.sum((-y * np.log(h) - (1 - y) * np.log(1 - h))) / m
    # print(h, j)

    theta_g = theta
    theta_g[0, :] = 0
    # print(theta, theta_g)

    # j = j + np.dot(theta_g.T, theta_g) * lambda_ / (2*m)
    j = j + np.sum(theta_g * theta_g) * lambda_ / (2*m)

    grad = np.dot(x.T, (h - y)) / m
    grad = grad + theta_g * lambda_ / m

    return j, grad

