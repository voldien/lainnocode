from operator import itemgetter

import math
import pandas as pd
from matplotlib import style
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np

from regression import normalEquation, extendMatrix, vectorCost, linear, gradient, poly
from statistics import normalizeFeature, variance

style.use('fivethirtyeight')
np.set_printoptions(precision=4)

columns = ["CudaCores", "BaseClock", "BoostClock", "MemorySpeed", "MemoryConfig", "MemoryBandwidth", "BenchmarkSpeed"
           ]

df = pd.read_csv('datasets/GPUbenchmark.csv', header=None)

X = np.array(df.drop(columns=6, axis=1).T)
y = np.array(df.loc[:, 6])

# Compute normalized 
normalX = np.array(normalizeFeature(X))
normalY = np.array(normalizeFeature([y]))
print("Y normalized variance {}".format(variance(normalY[0])))
print("X normalized variance {}".format(variance(normalX[0])))

plt.suptitle("Linear Regression {}".format("GPUbenchmark.csv"))
for i, X_ in enumerate(normalX):
    for x_, y_ in zip(X_, normalY[0]):
        pltii = plt.subplot(2, 3, i + 1)
        pltii.scatter(x_, y_, color='r')

        # Set title and labels.
        title = columns[i]
        pltii.set_xlabel(title)
        pltii.set_ylabel(columns[-1])

plt.show()

# Compute beta
Xe = extendMatrix(([xi for xi in normalX]), False)
beta = normalEquation(Xe, y)

# Compute and predict.
testX = normalizeFeature([[2432.0, 1607.0, 1683.0, 8.0, 8.0, 256.0]])
testy = [114.0]
predictY = linear(beta, testX[0])
print("expected: {}, predicated {}".format(testy[0], predictY))

# Compute the cost function
NEcost = vectorCost(beta, Xe, y)
print("cost from normal equation {}".format(NEcost))

# Solve with gradient descent.
errorMargin = NEcost / 100.0    # Compute 1 % of the original cost.
XeFit = []
Xe = extendMatrix(([xi for xi in normalX]), False)
for j in [1.0 / i for i in range(10, 10000, 500)]:
    for n in range(10, 10000, 500):
        optimized = gradient(Xe, beta, y, j, n)

        combinationCost = vectorCost(optimized, Xe, y)

        # Check if within 1 % error margin.
        error = math.fabs(combinationCost - NEcost)
        print("alpha {}, iterations: {} cost : {}".format(float(j), n, round(combinationCost, 5)))
        if error < errorMargin:
            XeFit.append([[n, j], combinationCost])

# Extract the best fit beta
XeB = sorted(XeFit, key=itemgetter(1))
error = XeB[0][1]
n, j = XeB[0][0]
print("Best fit with cost of {} for alpha and iteration ({}, {}) .".format(error, j, n))

# Solve gradient decent
beta = gradient(Xe, beta, y, j, n)
predictY = linear(beta, testX[0])
print("expected: {}, predicted {}".format(testy[0], predictY))
