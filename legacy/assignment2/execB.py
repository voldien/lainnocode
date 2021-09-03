import pandas as pd
from matplotlib import style
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np

from regression import logisticCost, \
    Sigmod, linear, extendMatrix, logisticGradient
from statistics import normalizeFeature, deviation

style.use('fivethirtyeight')
colorSet = ['r', 'b']

df = pd.read_csv('admission.csv', header=None)

X = np.array(df.drop(columns=2, axis=1).T)
y = np.array(df.loc[:, 2])

normalX = np.array(normalizeFeature(X))
normalY = np.array(normalizeFeature([y]))

for i, x in enumerate(normalX):
    print("nth", i, "deviation:", deviation(x))

# Plot normalized
for x in normalX:
    for x_, _y in zip(x, y):
        plt.scatter(x_, _y, color=colorSet[_y])

plt.show()

# Use sigmod matrix.
m = [[0, 1, ], [2, 3]]
sig = Sigmod(m)
print(sig)

# Create Extended X matrix.
Xe = extendMatrix([x for x in normalX])

# logistic cost function
testBeta = np.array([0, 0, 0])
cost = logisticCost(Xe, testBeta, y, linear)
print("expected cost : {}, computed cost {}".format(0.6931, cost))

# Simple gradient descent
beta = logisticGradient(Xe, testBeta, y, 0.5, 1)
cost = logisticCost(Xe, beta, y, linear)
expectedBeta = [0.05, 0.141, 0.125]
print("expected beta : {}, computed beta {}".format(expectedBeta, beta.tolist()))

# gradient descent
alpha = 0.2035
for n in range(10, 10000, 500):
    optimized = logisticGradient(Xe, testBeta, y, alpha, n)

    combinationCost = logisticCost(Xe, optimized, y, linear)

    print("alpha {}, iterations: {} cost : {}".format(float(alpha), n, round(combinationCost, 10)))

# gradient descent
