import math
import numpy as np


def deviation(X):
    """

    :param X:
    :return:
    """
    return math.sqrt(variance(X))


def variance(X):
    """

    :param X:
    :return:
    """
    x_ = mean(X)
    inv = (1.0 / (len(X) - 1))
    return inv * sum([(x - x_) ** 2 for x in X])


def mean(X):
    """

    :param X:
    :return:
    """
    inv = (1.0 / len(X))
    return inv * sum(X)


def cov(x1, x2):
    nx1 = len(x1)
    nx2 = len(x2)
    if nx2 != nx1:
        raise ValueError("x1 and x2 must be equal in size, got ({}, {})".format(nx1, nx2))

    # Compute mean.
    x1m = mean(x1)
    x2m = mean(x2)

    return (1.0 / (nx1 - 1)) * sum([(x - x1m) * (y - x2m) for x, y in zip(x1, x2)])


def corr(x1, x2):
    return cov(x1, x2) / (deviation(x1) * deviation(x2))


def SE(X, Y, predictY):
    n = len(Y)
    nx = len(X)
    

    # Compute the mean square.
    nu = np.sum((predictY - Y) ** 2)
    xm = mean(X)

    de = (n - 2) * sum([(_x - xm) ** 2 for _x in X])
    return math.sqrt(nu / de)


def normalizeFeature(featuresSet):
    """

    :param featuresSet:
    :return:
    """
    normalized = []
    for X in featuresSet:
        om = deviation(X)
        m = mean(X)
        normalized.append([(x - m) / om for x in X])

    return normalized


def sigmoid(z):
    """
    Compute sigmoid
    :param z:
    :return:
    """
    return 1.0 / (1.0 + math.e ** (-z))


def softmax(z):
    """

    :param z:
    :return:
    """
    d = sum([math.exp(z_i) for z_i in z])
    return [math.exp(zi) / (d + 1e-6) for zi in z]
