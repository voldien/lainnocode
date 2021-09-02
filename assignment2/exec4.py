from operator import itemgetter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import style
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

from graph import meshgrid, mesh2Dgrid
from regression import logisticGradient, logisticCost, extendMatrix, normalEquation, gradientVecorized, thetaShape, \
    linear, _grad_vectorized_linear, Sigmod, mapFeature
from statistics import normalizeFeature, sigmoid

style.use('fivethirtyeight')
np.set_printoptions(precision=4)
degress = [i for i in range(1, 10)]
h = .05  # step size in the mesh
threshold = 0.5
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])  # mesh plot
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])  # colors


def getColor(y):
    return ['r', 'b'][y]


def mapExpandedList(X):
    return [X[0], X[1], X[1] ** 2, X[0] * X[1]]


df = pd.read_csv("datasets/microchips.csv")

# Train Data.
X = np.array(normalizeFeature(np.array(df.drop(['id'], axis=1).T)))
y = np.array(df['id'])

# Linear logistic function.
Xe = extendMatrix(mapExpandedList(X), False)
# Get the shape of the beta.
linearBeta = np.zeros(5)

# Compute the best theta with gradient descent (alpha, n)
cost_ai = []
for alpha in [1.0 / i for i in range(10, 5000, 1000)]:
    for n in range(10, 1000, 100):
        optimized = logisticGradient(Xe, linearBeta, y, alpha, n)
        combinationCost = logisticCost(Xe, optimized, y, linear)

        # Results.
        print("alpha {}, iterations: {} J(B) : {}".format(alpha, n, round(combinationCost, 7)))

        cost_ai.append([[n, alpha], combinationCost])

# Create graph of cost vs iterations
# Extract the best fit beta
XeB = sorted(cost_ai, key=itemgetter(1))
error = XeB[0][1]
n, j = XeB[0][0]
print("The best cost {} with ({},{})".format(error, n, j))
lineX = []
lineY = []
for n in range(1, 1000, 10):
    optimized = logisticGradient(Xe, linearBeta, y, j, n)
    combinationCost = logisticCost(Xe, optimized, y, linear)
    lineX.append(n)
    lineY.append(combinationCost)

plti = plt.subplot(1, 2, 1)
plti.plot(lineX, lineY)
plti.set_xlabel("Iterations")
plti.set_ylabel("Cost")

x_train, x_test, y_train, y_test = train_test_split(X.T, y, test_size=0.25, random_state=0, shuffle=True)

# Create graph of decision boundary.
plti = plt.subplot(1, 2, 2)

X1 = X[0]
X2 = X[1]

#
xx, yy, x1, x2 = mesh2Dgrid((-2, -2), (2, 2), h)
poly = PolynomialFeatures(2, include_bias=False)
Xdeg = poly.fit_transform(x_train)

#
map = poly.fit_transform([[_x, _y] for _x, _y in zip(x1, x2)])
linearBeta = np.ones(len(Xdeg[0]))
optimized = logisticGradient(Xdeg, linearBeta, y_train, j, n)
p = np.array(Sigmod(np.dot(map, optimized)))  # classify mesh ==> probabilities
classes = p > threshold  # round off probabilities
clz_mesh = classes.reshape(xx.shape)  # return to mesh format

# Draw mesh and points.
plti.pcolormesh(xx, yy, clz_mesh, cmap=cmap_light)
for x_, y_ in zip(X.T, y):
    plti.scatter(x_[0], x_[1], color=getColor(y_), cmap=cmap_bold)

plti.set_xlabel("X")
plti.set_ylabel("Y")

plt.suptitle("microchips.csv - Logistic Regression (Redefined)")
plt.show()

# Repeat 2
lineX = []
lineY = []
for n in range(1, 1000, 5):
    regression = LogisticRegression(max_iter=n, n_jobs=4, C=j, verbose=1, solver='lbfgs', random_state=0)
    regression.fit(x_train, y_train)
    cost = regression.score(x_test, y_test)
    lineX.append(n)
    lineY.append(cost)

plti = plt.subplot(1, 2, 1)
plti.plot(lineX, lineY)
plti.set_xlabel("Iterations")
plti.set_ylabel("Cost")

# Create graph of decision boundary.
plti = plt.subplot(1, 2, 2)
xx, yy, x1, x2 = mesh2Dgrid((-2, -2), (2, 2), h)

XXe = mapFeature(x_train.T[0], x_train.T[1], 2)  # Extend matrix for degree 2
regression = LogisticRegression(max_iter=n, n_jobs=4, C=j, verbose=1, solver='lbfgs', random_state=0)
regression.fit(XXe, y_train)

XXe = mapFeature(x1, x2, 2)  # Extend matrix for degree 2

p = regression.predict(XXe)
clz_mesh = p.reshape(xx.shape)  # return to mesh format

# Draw mesh and points.
plti.pcolormesh(xx, yy, clz_mesh, cmap=cmap_light)
for x_, y_ in zip(X.T, y):
    plti.scatter(x_[0], x_[1], color=getColor(y_), cmap=cmap_bold)

plti.set_xlabel("Iterations")
plti.set_ylabel("Cost")

plt.suptitle("microchips.csv - Logistic Regression (Predefined)")
plt.show()

x_train, x_test, y_train, y_test = train_test_split(X.T, y, test_size=0.25, random_state=0)

for d in degress:
    poly = PolynomialFeatures(d, include_bias=False)
    x1 = x_train.T[0]
    x2 = x_train.T[1]
    XXe = poly.fit_transform([[_x, _y] for _x, _y in zip(x1, x2)])
    x1 = x_test.T[0]
    x2 = x_test.T[1]
    XXeT = poly.fit_transform([[_x, _y] for _x, _y in zip(x1, x2)])
    regression = LogisticRegression()

    regression.fit(XXe, y_train)
    predictions = regression.predict(XXeT)
    score = regression.score(XXeT, y_test)

    print("alpha {}, iterations: {} J(B) : {}".format(alpha, n, round(combinationCost, 7)))

    plti = plt.subplot(3, 3, d)
    plti.set_title("Degree {} - score {}".format(d, round(score, 4)))

    # Decision boundry
    xx, yy, x1, x2 = mesh2Dgrid((-2, -2), (2, 2), h)

    poly = PolynomialFeatures(d, include_bias=False)
    map = poly.fit_transform([[_x, _y] for _x, _y in zip(x1, x2)])

    predictedMap = regression.predict(map)

    clz_mesh = predictedMap.reshape(xx.shape)  # return to mesh format
    plti.pcolormesh(xx, yy, clz_mesh, cmap=cmap_light)

    for _x, _y in zip(x_train, y_train):
        plti.scatter(_x[0], _x[1], color=getColor(_y), s=5)
    for _x, _y in zip(x_test, y_test):
        plti.scatter(_x[0], _x[1], color=getColor(_y), s=10)


plt.show()
