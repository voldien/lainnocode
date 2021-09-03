import math
import numpy as np

from statistics import sigmoid, softmax


def check_len():
    pass


def Sigmod(M):
    """
    Compute sigmoid on a matrix of any dimensions.
    :param M:
    :return:
    """
    if not hasattr(M, "__len__"):
        raise ValueError("Requires a array object!")

    array = []
    for i, x in enumerate(M):
        if hasattr(x, "__len__"):
            array.append(Sigmod(x))
        else:
            x = sigmoid(x)
            array.append(x)
    return array


def vectorCost(beta, Xe, y):
    """
    Compute cost function with vectices.
    :param beta:
    :param Xe:
    :param y:
    :return:
    """
    n = len(y)
    xn = len(Xe)

    if xn != n:
        raise ValueError("Theta and Extended matrix element must be equal size. ({},{})".format(n, xn))

    nInv = (1.0 / n)
    j = (Xe.dot(beta) - y)
    return nInv * (j.T.dot(j))


def extendMatrix(list, pow=False, tranpose=True):
    s = len(list[0])

    # All most be equal size.
    for v in list[1:]:
        if s != len(v):
            raise ValueError("All vector must be equal dimension")

    #
    vectors = []

    # Add ethe one column.
    ones = np.ones((s, 1))
    vectors.append([i[0] for i in ones])

    if pow is False:
        for i, x in enumerate(list):
            vectors.append(x)
    elif pow is True:
        for i, x in enumerate(list):
            vectors.append(x ** (1 + i))
    else:
        raise ValueError("Invalid!")

    # nparray = [[x[j] for j in range(0, s + 1, 1)] for x in vectors]
    assert len(vectors) == len(list) + 1
    nparray = []
    if tranpose:
        for j in range(0, s, 1):
            vt = []
            for v in vectors:
                vt.append(v[j])
            nparray.append(vt)

        # assert len(nparray) == len(list[0])
    else:
        for v in list:
            ext = [1.0]
            ext.extend(v)
            nparray.append(ext)
    return np.array(nparray)


#    nparray = [[x[j] for x in vectors] for j in range(0, s, 1)]


def logisticCost_func(X, beta, y, polyfunc):
    """

    :param X:
    :param beta:
    :param y:
    :param polyfunc:
    :return:
    """
    g = sigmoid(polyfunc(beta, X))
    return y * math.log(g) + (1.0 - y) * math.log(1.0 - g + 0.0000000000000001)


# TODO improve!
def logisticCost(Xe, beta, y, polyfunc):
    """

    :param Xe:
    :param beta:
    :param y:
    :param polyfunc:
    :return:
    """
    nInv = -(1.0 / len(y))

    #
    li = [logisticCost_func(x_[1:], beta, float(y_), polyfunc) for x_, y_ in zip(Xe, y)]
    logi = sum(li)

    return nInv * logi


def thetaSize(X):
    """
    Compute the size of the theta vector.
    :param Xe:
    :return:
    """
    return len(X) + 1


def thetaShape(Xe):
    return np.zeros(thetaSize(Xe))


def normalEquation(Xe, y):
    """

    :param Xe:
    :param y:
    :return:
    """
    Xen = len(Xe)
    bn = len(y)
    if Xen != bn:
        raise ValueError("Xe and y expected to be equal. However got ({}, {})".format(Xen, bn))

    beta = np.linalg.inv(Xe.T.dot(Xe)).dot(Xe.T).dot(y)
    return beta


def _grad_vectorized_linear(Xe, theta):
    return np.dot(Xe, theta)


def _grad_vectorized_logistic(Xe, theta):
    return np.array(Sigmod(Xe.dot(theta)))


def _grad_vectorized_logistic_p(Xe, theta, func):
    g = [func(x, theta) for x in Xe]
    return np.array(Sigmod(g))


def gradientVecorized(Xe, beta, y, alpha, it, pred, func=None):
    """

    :param Xe: Extended matrix
    :param beta:
    :param y:
    :param alpha:
    :param it:
    :param pred:
    :param func:
    :return:
    """
    # Validate arguments
    Xen = len(Xe[0])
    bn = len(beta)
    if Xen != bn:
        raise ValueError("Xe and theta expected to be equal. However got ({}, {})".format(Xen, bn))

    if it <= 0:
        raise ValueError("Iterator must be greater than 0, current {}".format(it))

    if pred is None:
        raise ValueError("The function can not be null")

    theta = np.ones(beta.shape)

    m = float(len(y))
    # Compute gradient decent for n iterations.
    for i in range(0, it):
        if func is None:
            prediction = pred(Xe, theta)
        else:
            prediction = pred(Xe, theta, func)

        theta = theta - ((1.0 / m) * alpha) * (Xe.T.dot(prediction - y)).reshape(theta.shape)

    return theta


def gradient(Xe, beta, y, alpha=0.001, it=1000):
    return gradientVecorized(Xe, beta, y, alpha, it, _grad_vectorized_linear)


def logisticGradient(Xe, beta, y, alpha, it=10000):
    return gradientVecorized(Xe, beta, y, alpha, it, _grad_vectorized_logistic)


def linear(beta, X):
    """
    Compute linear equation.
    :param beta:
    :param X:
    :return:
    """
    bl = len(beta)
    xl = len(X)
    if bl - 1 != xl:
        raise ValueError("theta:{} is not less than X:{} by one".format(bl, xl))
    base = beta[0]

    return base + sum([b * x for b, x in zip(beta[1:], X)])


def poly(beta, X):
    """
    Compute polynomial equation.
    :param beta:
    :param X:
    :return:
    """
    if len(beta) - 1 != len(X):
        raise ValueError("beta and X must have one element in different!")
    base = beta[0]

    return base + sum([b * (x ** (i + 1)) for b, (i, x) in zip(beta[1:], enumerate(X))])


def logisticClamp(x, threshold):
    return x > threshold


def mapFeature(X1, X2, D):
    if D < 1:
        raise ValueError("Degree must be at least 1")
    one = np.ones([len(X1), 1])
    Xe = np.c_[one, X1, X2]  # Start with [1,X1,X2]
    for i in range(2, D + 1):
        for j in range(0, i + 1):
            Xnew = X1 ** (i - j) * X2 ** j  # type (N)
    Xnew = Xnew.reshape(-1, 1)  # type (N,1) required by append
    Xe = np.append(Xe, Xnew, 1)  # axis = 1 ==> append column
    return Xe


def one_hot(size, one):
    """

    :param size:
    :param one:
    :return:
    """
    hot = np.zeros(size)
    hot[one] = 1
    return hot


def net(X, W, b):
    """

    :param X:
    :param W:
    :param b:
    :return:
    """
    if len(W) != len(b):
        raise
    y_linear = np.dot(X, W) + b
    yhat = softmax(y_linear)
    return yhat


def multiLogisticPredict(X, bias, weights):
    # Test for the size.

    predicts = []
    for _x in X:
        predict = np.array(net(_x, weights, bias))
        # s = np.sum(_x)
        # res = weights * s + bias
        # predict = np.array(softmax(res))
        predicts.append(predict)
    return predicts


def cross_entropy(yhat, y):
    """

    :param yhat:
    :param y:
    :return:
    """
    return - np.sum(y * np.log(yhat + 1e-6))


def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad


def multiLogisticFit(X, y, nrClasses):
    num_inputs = len(X[0])
    num_outputs = nrClasses

    W = np.zeros((num_inputs, num_outputs))
    bias = np.random.rand(nrClasses)

    lr = 1.2

    params = [W, bias]
    for _x, _y in zip(X, y):
        for _ in range(0, 50):
            predict = np.array(net(_x, W, bias))
            hot = one_hot(num_inputs, _y - 1)
            loss = cross_entropy(predict, hot)
            grad = np.array(hot - predict)
            # SGD(params, lr)
            W += lr * (1.0 / num_inputs) * grad

    return W, bias


def logicError(zs, labels):
    acc = 0
    for z, l in zip(zs, labels):
        ind = np.argsort(z)
        predlabel = ind[-1] + 1
        if l == predlabel:
            acc = acc + 1

    return acc / len(labels)

# def evaluate_accuracy(data_iterator, net):
#     numerator = 0.
#     denominator = 0.
#     for i, (datasets, label) in enumerate(data_iterator):
#         datasets = datasets.as_in_context(model_ctx).reshape((-1,784))
#         label = label.as_in_context(model_ctx)
#         label_one_hot = nd.one_hot(label, 10)
#         output = net(datasets)
#         predictions = nd.argmax(output, axis=1)
#         numerator += nd.sum(predictions == label)
#         denominator += datasets.shape[0]
#     return (numerator / denominator).asscalar()
